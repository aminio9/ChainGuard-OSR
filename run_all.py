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
from src.metrics.osr_metrics import compute_confusion_matrix as confusion_matrix, compute_osr_metrics
from src.tracking.reproducibility import set_seed
from src.utils.artifact_logger import aggregate_MultiSeed_Summary, save_seed_run_artifacts
from src.utils.device import get_device
from src.utils.suite_runners import run_suite_extensions
from experiments.run_wire_speed_profiling import run_wire_speed_experiment


def run_fedtros_mc_full_diagnostics(
    dataset_name: str,
    seed: int,
    rounds: int,
    beta_dirichlet: float,
    samples_per_class: Optional[int],
    output_root: Path,
    experiment_id: str = "E2_main_osr",
    num_clients: int = 10,
    num_blocks: int = 4,
    block_dim: int = 16,
    lambda_vib: Optional[float] = None,
    use_bla: bool = True,
    enable_conformal: bool = True,
    batch_size: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run FedTROS-MC end-to-end and save full per-seed diagnostics for plotting."""
    cfg = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
    cfg.data.dataset_name = dataset_name
    cfg.experiment.seed = seed
    cfg.federated.num_rounds = rounds
    cfg.federated.num_clients = num_clients
    cfg.federated.num_blocks = num_blocks
    cfg.federated.block_dim = block_dim
    cfg.federated.partition.beta_dirichlet = beta_dirichlet
    if batch_size is not None:
        cfg.federated.batch_size = batch_size
    cfg.loss.tau_bla = 1.0 if use_bla else 0.0
    cfg.osr.enable_conformal = enable_conformal
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
    osr_metrics = compute_osr_metrics(
        y_true=y_open_true,
        y_pred=det["predictions"],
        anomaly_scores=det["normalized_scores"],
        num_known_classes=num_known,
        unknown_label=num_known,
    )
    from src.metrics.osr_metrics import compute_binary_auroc, compute_fpr95
    y_bin = (y_open_true == num_known).astype(int)
    osr_metrics["auroc_normalized"] = float(osr_metrics["auroc"])
    osr_metrics["fpr95_normalized"] = float(osr_metrics["fpr95"])
    osr_metrics["auroc_raw"] = float(compute_binary_auroc(y_bin, det["raw_scores"]))
    osr_metrics["fpr95_raw"] = float(compute_fpr95(y_bin, det["raw_scores"]))

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

    all_metrics = {
        **osr_metrics,
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
    return all_metrics


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
