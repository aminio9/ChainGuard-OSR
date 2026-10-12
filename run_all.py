"""Unified reproducible experiment runner for FedTROS-MC (E1-E8 and A1-A6).

Usage:
    python run_all.py --mode debug --seeds 42 --output_root results/runs_v2
    python run_all.py --mode full --seeds 42,43,44 --output_root results/runs_v2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.base import FlowDataset
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.tabular_p2p import load_benchmark_dataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import (
    compute_binary_auroc,
    compute_confusion_matrix as confusion_matrix,
    compute_fpr95,
    compute_osr_metrics,
)
from src.tracking.reproducibility import set_seed
from src.utils.artifact_logger import (
    aggregate_MultiSeed_Summary,
    save_seed_run_artifacts,
    export_round_telemetry_jsonl,
    export_plot_structured_artifacts,
)
from src.utils.device import get_device
from src.utils.suite_runners import run_suite_extensions
from experiments.run_wire_speed_profiling import run_wire_speed_experiment

try:
    from src.osr.conformal import (
        compute_finite_sample_quantile,
        evaluate_conformal_sensitivity_sweep,
    )
except ImportError:
    def compute_finite_sample_quantile(
        scores: np.ndarray | torch.Tensor,
        alpha: float = 0.05,
    ) -> float:
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy().flatten()
        else:
            scores_np = np.asarray(scores).flatten()
        n = len(scores_np)
        if n == 0:
            return float("inf")
        target_rank = np.ceil((n + 1) * (1.0 - alpha))
        if target_rank > n:
            return float(max(float(np.max(scores_np)), 1e-12))
        q_level = float(min(1.0, max(0.0, target_rank / float(n))))
        return float(np.quantile(scores_np, q_level, method="higher"))

    def evaluate_conformal_sensitivity_sweep(
        cal_scores_per_class: Dict[int, np.ndarray | torch.Tensor],
        test_dist_matrix: np.ndarray | torch.Tensor,
        test_labels: np.ndarray | torch.Tensor,
        num_known_classes: int,
        alphas: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if alphas is None:
            alphas = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        if isinstance(test_dist_matrix, torch.Tensor):
            dist_mat = test_dist_matrix.detach().cpu().numpy()
        else:
            dist_mat = np.asarray(test_dist_matrix)
        if isinstance(test_labels, torch.Tensor):
            labels = test_labels.detach().cpu().numpy()
        else:
            labels = np.asarray(test_labels)

        known_mask = (labels < num_known_classes)
        unknown_mask = (labels == num_known_classes)
        n_known = int(known_mask.sum())
        n_unknown = int(unknown_mask.sum())
        results = []
        for a in alphas:
            taus = {}
            for c in range(num_known_classes):
                if c in cal_scores_per_class and len(cal_scores_per_class[c]) > 0:
                    taus[c] = compute_finite_sample_quantile(cal_scores_per_class[c], alpha=a)
                else:
                    taus[c] = float("inf")
            tau_arr = np.array([taus[c] for c in range(num_known_classes)], dtype=np.float64).reshape(1, -1)
            accepted_mask = dist_mat <= tau_arr
            is_unknown = ~np.any(accepted_mask, axis=1)
            kfr = float(is_unknown[known_mask].mean()) if n_known > 0 else 0.0
            cov = 1.0 - kfr
            rec = float(is_unknown[unknown_mask].mean()) if n_unknown > 0 else 0.0
            n_rej = int(is_unknown.sum())
            tp = int((is_unknown & unknown_mask).sum())
            prec = float(tp / n_rej) if n_rej > 0 else 0.0
            f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
            results.append({
                "alpha": float(a),
                "nominal_coverage": float(1.0 - a),
                "empirical_coverage": float(cov),
                "empirical_kfr": float(kfr),
                "zero_day_recall": float(rec),
                "zero_day_precision": float(prec),
                "zero_day_f1": float(f1),
                "thresholds": {int(c): float(taus[c]) for c in range(num_known_classes)},
                "status": "GUARANTEED" if kfr <= (a + 0.005) else "DRIFT",
            })
        return results


def run_fedtros_mc_full_diagnostics(
    dataset_name: str,
    seed: int,
    rounds: int,
    beta_dirichlet: float,
    samples_per_class: Optional[int],
    output_root: Path,
    experiment_id: str = "E2_main_osr",
    num_clients: int = 10,
    num_blocks: Optional[int] = None,
    block_dim: Optional[int] = None,
    lambda_vib: Optional[float] = None,
    use_bla: bool = True,
    tau_bla: Optional[float] = None,
    enable_conformal: bool = True,
    use_decoupled: bool = True,
    batch_size: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run FedTROS-MC end-to-end and save full per-seed diagnostics for plotting."""
    cfg = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
    cfg.data.dataset_name = dataset_name
    cfg.experiment.seed = seed
    cfg.federated.num_rounds = rounds
    cfg.federated.num_clients = num_clients
    if num_blocks is not None:
        cfg.federated.num_blocks = num_blocks
        cfg.osr.num_blocks = num_blocks
    else:
        num_blocks = getattr(cfg.osr, "num_blocks", 1)
    if block_dim is not None:
        cfg.federated.block_dim = block_dim
        cfg.osr.block_dim = block_dim
    else:
        block_dim = getattr(cfg.osr, "block_dim", 64)
    cfg.federated.partition.beta_dirichlet = beta_dirichlet
    if batch_size is not None:
        cfg.federated.batch_size = batch_size
    if tau_bla is not None:
        cfg.loss.tau_bla = float(tau_bla)
        cfg.loss.gamma_bla = float(tau_bla)
    else:
        cfg.loss.tau_bla = 1.0 if use_bla else 0.0
        cfg.loss.gamma_bla = 1.0 if use_bla else 0.0
    cfg.osr.enable_conformal = enable_conformal
    cfg.federated.aggregation_strategy = "decoupled" if use_decoupled else "fedavg"
    if lambda_vib is not None:
        cfg.loss.lambda_vib = lambda_vib
        cfg.model.stochastic_encoder = bool(lambda_vib > 0.0)

    set_seed(seed)
    device = get_device(cfg.experiment.device)

    raw_dataset, meta = load_benchmark_dataset(
        dataset_name=dataset_name,
        data_dir="data/raw",
        samples_per_class=samples_per_class,
    )
    cfg.data.known_classes = meta["known_classes"]
    cfg.data.unknown_classes = meta["unknown_classes"]
    cfg.data.in_features = meta["in_features"]

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=cfg.data.known_classes,
        unknown_classes=cfg.data.unknown_classes,
        test_ratio=cfg.data.test_split_ratio,
        cal_ratio=cfg.data.cal_split_ratio,
        seed=seed,
        protocol="full",
    )
    cfg.data.in_features = int(train_ds.features.shape[1])
    cfg.model.in_features = int(train_ds.features.shape[1])

    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=num_clients,
        beta_dirichlet=beta_dirichlet,
        seed=seed,
    )

    clients = []
    for cid in range(num_clients):
        idx_t = torch.tensor(client_indices[cid], dtype=torch.long)
        sub_ds = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        clients.append(FederatedClient(cid, sub_ds, cfg, device, cal_dataset=cal_ds))

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    server = FederatedServer(config=cfg, clients=clients, device=device)

    round_history: List[Dict[str, Any]] = []
    num_known = len(cfg.data.known_classes)
    t_p1_start = time.perf_counter()
    log_interval = max(1, rounds // 10) if rounds >= 20 else max(1, rounds // 5)
    
    test_closed_x = test_closed_ds.features.to(device)
    test_closed_y = test_closed_ds.labels.numpy()
    c_acc = 0.0

    for r in range(1, rounds + 1):
        t_r0 = time.perf_counter()
        r_loss = server.run_round(r)
        if r == 1 or r % log_interval == 0 or r == rounds:
            with torch.no_grad():
                logits_c = server.global_model.classify(test_closed_x)
                preds_c = torch.argmax(logits_c, dim=-1).cpu().numpy()
                c_acc = float(np.mean(preds_c == test_closed_y))
        r_entry = dict(r_loss)
        r_entry["val_closed_accuracy"] = c_acc
        round_history.append(r_entry)
        if r == 1 or r % log_interval == 0 or r == rounds:
            elapsed = time.perf_counter() - t_p1_start
            r_sec = time.perf_counter() - t_r0
            avg_sec = elapsed / r
            eta_sec = avg_sec * (rounds - r)
            loss_val = r_loss.get("train_loss", r_loss.get("loss_bla", 0.0)) if isinstance(r_loss, dict) else 0.0
            print(
                f"    [FedTROS-MC] Round {r:3d}/{rounds} | Loss: {loss_val:.4f} | Val Acc: {c_acc * 100:.2f}% | "
                f"Speed: {r_sec:.2f}s/rnd | Elapsed: {elapsed:.1f}s | ETA: {eta_sec:.1f}s",
                flush=True,
            )

    t_phase1 = time.perf_counter() - t0
    t1 = time.perf_counter()
    server.run_phase_two_profiling(
        num_blocks=num_blocks,
        block_dim=block_dim,
        alpha=cfg.osr.conformal_alpha,
        cal_dataset=cal_ds,
    )
    t_phase2 = time.perf_counter() - t1
    peak_mem_bytes = (
        torch.cuda.max_memory_allocated(device)
        if (torch.cuda.is_available() and device.type == "cuda")
        else 0
    )

    detector = server.get_edge_detector(
        unknown_label=num_known, num_blocks=num_blocks, block_dim=block_dim
    )
    det = detector.predict_batch_detailed(test_open_ds.features)
    y_open_true = test_open_ds.labels.numpy()
    ranking_scores = det["raw_scores"]
    osr_metrics = compute_osr_metrics(
        y_true=y_open_true,
        y_pred=det["predictions"],
        anomaly_scores=ranking_scores,
        num_known_classes=num_known,
        unknown_label=num_known,
    )
    y_bin = (y_open_true == num_known).astype(int)
    osr_metrics["auroc_raw"] = float(compute_binary_auroc(y_bin, det["raw_scores"]))
    osr_metrics["fpr95_raw"] = float(compute_fpr95(y_bin, det["raw_scores"]))
    osr_metrics["auroc_normalized"] = float(compute_binary_auroc(y_bin, det["normalized_scores"]))
    osr_metrics["fpr95_normalized"] = float(compute_fpr95(y_bin, det["normalized_scores"]))

    with torch.no_grad():
        logits_closed = server.global_model.classify(test_closed_ds.features.to(device))
        preds_closed = torch.argmax(logits_closed, dim=-1).cpu().numpy()
    closed_metrics = compute_classification_metrics(
        y_true=test_closed_ds.labels.numpy(),
        y_pred=preds_closed,
        num_classes=num_known,
    )

    num_params = sum(p.numel() for p in server.global_model.parameters())
    p1_bytes = int(2 * num_params * 4 * rounds)
    p2_bytes = int(num_known * (cfg.model.latent_dim + num_blocks * (block_dim ** 2) + 2) * 4)

    sweep_results = None
    cal_scores = getattr(server, "cal_scores_per_class", None)
    if cal_scores is not None and "dist_matrix" in det:
        sweep_results = evaluate_conformal_sensitivity_sweep(
            cal_scores_per_class=cal_scores,
            test_dist_matrix=det["dist_matrix"],
            test_labels=y_open_true,
            num_known_classes=num_known,
            alphas=[0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        )

    all_metrics = {
        **osr_metrics,
        "conformal_sensitivity_sweep": sweep_results or [],
        "accuracy": closed_metrics["accuracy"],
        "closed_set_accuracy": closed_metrics["accuracy"],
        "closed_set_balanced_accuracy": closed_metrics["balanced_accuracy"],
        "closed_set_macro_f1": closed_metrics["macro_f1"],
        "phase1_wall_time_sec": float(t_phase1),
        "phase2_wall_time_sec": float(t_phase2),
        "total_wall_time_sec": float(t_phase1 + t_phase2),
        "peak_memory_mb": float(peak_mem_bytes / (1024.0 * 1024.0)),
        "phase1_comm_bytes_per_client": p1_bytes,
        "phase2_comm_bytes_per_client": p2_bytes,
        "total_comm_bytes_per_client": p1_bytes + p2_bytes,
    }

    cm_before = confusion_matrix(
        y_open_true, det["candidate_classes"], labels=list(range(num_known + 1))
    )
    cm_after = confusion_matrix(
        y_open_true, det["predictions"], labels=list(range(num_known + 1))
    )

    scores_table = {
        "y_true": y_open_true,
        "y_pred": det["predictions"],
        "candidate_class": det["candidate_classes"],
        "is_unknown_true": (y_open_true == num_known).astype(np.int32),
        "is_rejected": det["is_unknown"].astype(np.int32),
        "raw_mahalanobis": det["raw_scores"],
        "normalized_score": det["normalized_scores"],
        "threshold_used": det["thresholds_used"],
    }

    split_meta = {
        "n_train_known": len(train_ds),
        "n_cal_known": len(cal_ds),
        "n_test_closed": len(test_closed_ds),
        "n_test_open": len(test_open_ds),
        "known_classes": cfg.data.known_classes,
        "unknown_classes": cfg.data.unknown_classes,
        "beta_dirichlet": beta_dirichlet,
    }

    save_seed_run_artifacts(
        output_root=output_root,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        seed=seed,
        config=cfg,
        metrics=all_metrics,
        split_metadata=split_meta,
        round_history=round_history,
        thresholds=server.calibration_thresholds,
        calibration_counts=server.calibration_counts,
        per_class_kfr=osr_metrics.get("per_class_kfr"),
        per_class_coverage=osr_metrics.get("per_class_coverage"),
        scores_table=scores_table,
        confusion_before=cm_before,
        confusion_after=cm_after,
        extra_arrays={
            "latent_z": det["latent_z"][:2000],
            "y_open_subset": y_open_true[:2000],
        },
    )

    # Section 12 Hook 1: Export round telemetry JSONL
    export_round_telemetry_jsonl(
        output_root=output_root,
        experiment_id=experiment_id,
        seed=seed,
        round_history=round_history,
        beta=beta_dirichlet,
    )

    # Section 12 Hook 2: Export Tier C structured artifacts (.npz) for OSR figures (E3/E4)
    is_unk = (y_open_true == num_known)
    known_scores_arr = ranking_scores[~is_unk]
    unknown_scores_arr = ranking_scores[is_unk]
    export_plot_structured_artifacts(
        output_root=output_root,
        artifact_name=f"{experiment_id}_score_distributions_seed{seed}",
        known_scores=known_scores_arr,
        unknown_scores=unknown_scores_arr,
        nominal_alpha=float(cfg.osr.conformal_alpha),
        calibrated_quantile=float(np.mean(list(server.calibration_thresholds.values()))) if server.calibration_thresholds else 0.0,
        empirical_kfr=float(all_metrics.get("kfr", 0.0)),
        roc_fpr=np.asarray(all_metrics.get("roc_fpr", [])),
        roc_tpr=np.asarray(all_metrics.get("roc_tpr", [])),
        pr_recall=np.asarray(all_metrics.get("pr_recall", [])),
        pr_precision=np.asarray(all_metrics.get("pr_precision", [])),
    )

    return {
        **all_metrics,
        "metrics": all_metrics,
        "extra_arrays": {
            "latent_z": det["latent_z"][:2000],
            "y_open_subset": y_open_true[:2000],
        },
        "det": det,
        "server": server,
        "test_open_ds": test_open_ds,
        "test_closed_ds": test_closed_ds,
        "cal_ds": cal_ds,
        "train_ds": train_ds,
        "num_known": num_known,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all FedTROS-MC experiments and save structured results.")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--rounds", type=int, default=None, help="Optional override for communication rounds")
    parser.add_argument("--spc", type=int, default=None, help="Optional override for samples per class")
    parser.add_argument("--output_root", type=str, default="results/runs_v2")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rounds = args.rounds if args.rounds is not None else (2 if args.mode == "debug" else 5)
    spc = args.spc if args.spc is not None else (100 if args.mode == "debug" else None)
    out_root = Path(args.output_root)

    for seed in seeds:
        for skew_name, beta in [
            ("E1_iid", 100.0),
            ("E1_dirichlet_05", 0.5),
            ("E1_dirichlet_01", 0.1),
            ("E1_pathological_c2", 0.05),
        ]:
            run_fedtros_mc_full_diagnostics(
                dataset_name="b_nat",
                seed=seed,
                rounds=rounds,
                beta_dirichlet=beta,
                samples_per_class=spc,
                output_root=out_root,
                experiment_id=skew_name,
            )

        for ds in ["b_nat", "b_tat", "ton_iot", "cic_ids2017"]:
            run_fedtros_mc_full_diagnostics(
                dataset_name=ds,
                seed=seed,
                rounds=rounds,
                beta_dirichlet=0.5,
                samples_per_class=spc,
                output_root=out_root,
                experiment_id="E2_E3_benchmarks",
            )

        for b_vib in [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
            run_fedtros_mc_full_diagnostics(
                dataset_name="b_nat",
                seed=seed,
                rounds=rounds,
                beta_dirichlet=0.5,
                samples_per_class=spc,
                output_root=out_root,
                experiment_id=f"A2_vib_beta_{b_vib}",
                lambda_vib=b_vib,
            )

        for nb, bd in [(64, 1), (8, 8), (4, 16), (2, 32), (1, 64)]:
            run_fedtros_mc_full_diagnostics(
                dataset_name="b_nat",
                seed=seed,
                rounds=rounds,
                beta_dirichlet=0.5,
                samples_per_class=spc,
                output_root=out_root,
                experiment_id=f"A5_blocks_{nb}x{bd}",
                num_blocks=nb,
                block_dim=bd,
            )

    run_suite_extensions(
        project_root=PROJECT_ROOT,
        output_root=out_root,
        seeds=seeds,
        rounds=rounds,
        spc=spc,
    )

    for exp_dir in sorted(out_root.iterdir()):
        if exp_dir.is_dir():
            for ds_dir in sorted(exp_dir.iterdir()):
                if ds_dir.is_dir():
                    aggregate_MultiSeed_Summary(out_root, exp_dir.name, ds_dir.name)

    run_wire_speed_experiment(output_dir=str(out_root / "E8_wire_speed"))

    from src.ref_plots.contract_exporter_part3 import finalize_real_plot_contract

    contract_dir = out_root / "plot_contract"
    finalize_real_plot_contract(
        results_root=out_root,
        contract_dir=contract_dir,
        rounds=max(rounds, 4),
        spc=150 if args.mode == "debug" else spc,
        seed=seeds[0],
    )
    print(f"[SUCCESS] Structured experiment outputs and plot contract saved under: {out_root}")


if __name__ == "__main__":
    main()
