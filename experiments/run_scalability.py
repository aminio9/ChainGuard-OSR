"""Experiment E7: Federation Scalability and Timing Breakdown across K in {10, 50, 100} clients.

Aligned with LaTeX Section 5.1 (e), Section 6.7 (Table 11 & Figure 9).
Evaluates:
  - Sublinear wall-clock computational scaling (Stage I vs Stage II decomposition).
  - Open-world detection efficacy (AUROC, Unknown-F1, Open-Set Macro-F1).
  - Conformal calibration guarantee (KFR <= alpha = 5%) across expanding federation tiers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.tracking.logger import setup_logger
from src.utils.device import get_device
from src.utils.io import save_json
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.tabular_p2p import load_benchmark_dataset, BENCHMARK_METADATA
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics


def run_scalability_experiment(
    dataset_name: str = "b_nat",
    client_counts: Optional[List[int]] = None,
    rounds: int = 10,
    beta_dirichlet: float = 0.5,
    max_samples: Optional[int] = None,
    output_dir: str = "results/scalability",
    seed: int = 42,
    batch_size: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Execute scalability benchmark across federation scale K in {10, 50, 100}."""
    logger = setup_logger(name="scalability_runner")
    if client_counts is None:
        client_counts = [10, 50, 100]

    logger.info(
        f"Starting Federation Scalability Experiment (E7) on {dataset_name} "
        f"with K in {client_counts}, rounds={rounds}, beta={beta_dirichlet}..."
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark dataset
    if dataset_name in BENCHMARK_METADATA:
        raw_dataset, meta = load_benchmark_dataset(
            dataset_name=dataset_name,
            data_dir="data/raw",
            samples_per_class=max_samples,
            cache_dir="data/processed",
        )
        known_classes = meta["known_classes"]
        unknown_classes = meta["unknown_classes"]
    else:
        logger.warning(f"Dataset {dataset_name} not found in metadata; generating synthetic data.")
        raw_dataset = generate_synthetic_p2p_data(
            num_samples=max_samples or 3000,
            in_features=34,
            num_classes=5,
            seed=seed,
        )
        known_classes = [0, 1, 2, 3]
        unknown_classes = [4]

    # 2. Strict Zero-Leakage Split (Train, Cal, Test Known, Test Unknown)
    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=0.2,
        cal_ratio=0.15,
        seed=seed,
        protocol=os.environ.get("OSR_PROTOCOL", "full"),
    )

    results: Dict[str, Dict[str, Any]] = {}

    for K in client_counts:
        logger.info(f"\n=======================================================")
        logger.info(f"--> Evaluating Federation Scale: K = {K} clients")
        logger.info(f"=======================================================")
        set_seed(seed)

        # Load default configuration
        config = load_config("configs/default.yaml")
        config.federated.num_clients = K
        config.federated.num_rounds = rounds
        if batch_size is not None:
            config.federated.batch_size = batch_size
        config.federated.partition.beta_dirichlet = beta_dirichlet
        config.data.known_classes = known_classes
        config.data.unknown_classes = unknown_classes
        config.model.in_features = train_ds.features.shape[1]

        device = get_device(config.experiment.device)

        # Dirichlet Partitioning across K clients
        client_indices = partition_dirichlet_non_iid(
            dataset=train_ds,
            num_clients=K,
            beta_dirichlet=beta_dirichlet,
            min_samples_per_client=max(2, min(10, len(train_ds) // (2 * K))),
            seed=seed,
        )

        clients = []
        for client_id in range(K):
            sub_indices = client_indices[client_id]
            idx_t = torch.tensor(sub_indices, dtype=torch.long)
            sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
            clients.append(FederatedClient(client_id, sub_dataset, config, device, cal_dataset=cal_ds))

        server = FederatedServer(config, clients, device)

        # --- Stage I: Federated Collaborative Training ---
        stage1_start = time.perf_counter()
        log_interval = max(1, rounds // 10) if rounds >= 20 else max(1, rounds // 5)
        for r in range(1, rounds + 1):
            t_r0 = time.perf_counter()
            r_loss = server.run_round(r)
            if r == 1 or r % log_interval == 0 or r == rounds:
                elapsed = time.perf_counter() - stage1_start
                r_sec = time.perf_counter() - t_r0
                avg_sec = elapsed / r
                eta_sec = avg_sec * (rounds - r)
                loss_val = r_loss.get("train_loss", r_loss.get("loss_bla", 0.0)) if isinstance(r_loss, dict) else 0.0
                print(
                    f"    [Scalability-K={K}] Round {r:3d}/{rounds} | Loss: {loss_val:.4f} | "
                    f"Speed: {r_sec:.2f}s/rnd | Elapsed: {elapsed:.1f}s | ETA: {eta_sec:.1f}s",
                    flush=True,
                )
        server.freeze_global_encoder()
        stage1_duration = time.perf_counter() - stage1_start

        # --- Stage II: Moment Profiling & Conformal Calibration ---
        stage2_start = time.perf_counter()
        num_known = len(known_classes)

        server.run_phase_two_profiling(
            num_blocks=config.federated.num_blocks,
            block_dim=config.federated.block_dim,
            alpha=config.osr.conformal_alpha,
        )

        edge_detector = server.get_edge_detector(
            unknown_label=num_known,
            num_blocks=config.federated.num_blocks,
            block_dim=config.federated.block_dim,
        )

        preds_open, is_unk, cand, anomaly_scores = edge_detector.predict_batch(test_open_ds.features)
        stage2_duration = time.perf_counter() - stage2_start
        total_duration = stage1_duration + stage2_duration

        # --- Compute Metrics ---
        y_open_true = test_open_ds.labels.numpy()
        osr_metrics = compute_osr_metrics(
            y_true=y_open_true,
            y_pred=preds_open,
            anomaly_scores=anomaly_scores,
            num_known_classes=num_known,
            unknown_label=num_known,
        )

        with torch.no_grad():
            x_closed = test_closed_ds.features.to(device)
            logits_closed = server.global_model.classify(x_closed)
            preds_closed = torch.argmax(logits_closed, dim=-1).cpu().numpy()
        y_closed_true = test_closed_ds.labels.numpy()
        closed_metrics = compute_classification_metrics(
            y_true=y_closed_true,
            y_pred=preds_closed,
            num_classes=num_known,
        )

        # Compute exact communication payload
        num_params = sum(p.numel() for p in server.global_model.parameters())
        phase1_comm_bytes = int(2 * num_params * 4 * rounds)  # Uplink + downlink
        num_blocks = config.federated.num_blocks
        block_dim = config.federated.block_dim
        latent_dim = config.model.latent_dim
        phase2_comm_bytes = int(num_known * (latent_dim + num_blocks * (block_dim ** 2) + 2) * 4)

        k_results = {
            "experiment_id": f"E7_scalability_K{K}",
            "client_count": K,
            "num_clients": K,
            "phase1_wall_time_sec": float(stage1_duration),
            "phase2_wall_time_sec": float(stage2_duration),
            "stage1_time_sec": float(stage1_duration),
            "stage2_time_sec": float(stage2_duration),
            "total_time_sec": float(total_duration),
            "total_wall_time_sec": float(total_duration),
            "phase1_comm_bytes_per_client": phase1_comm_bytes,
            "phase2_comm_bytes_per_client": phase2_comm_bytes,
            "realized_kfr": float(osr_metrics["kfr"]),
            "kfr": float(osr_metrics["kfr"]),
            "macro_f1_closed": float(closed_metrics["macro_f1"]),
            "closed_accuracy": float(closed_metrics["accuracy"]),
            "auroc_open": float(osr_metrics["auroc"]),
            "auroc": float(osr_metrics["auroc"]),
            "aupr": float(osr_metrics["aupr"]),
            "unknown_recall": float(osr_metrics["unknown_recall"]),
            "unknown_f1": float(osr_metrics["unknown_f1"]),
            "open_set_macro_f1": float(osr_metrics["open_set_macro_f1"]),
        }

        results[f"K_{K}"] = k_results
        logger.info(
            f"--> [K={K}] AUROC: {k_results['auroc']*100:.2f}% | "
            f"Unk-F1: {k_results['unknown_f1']*100:.2f}% | "
            f"OS Macro-F1: {k_results['open_set_macro_f1']*100:.2f}% | "
            f"Realized KFR: {k_results['realized_kfr']*100:.2f}% | "
            f"Phase I: {stage1_duration:.2f}s ({phase1_comm_bytes/1e6:.2f} MB) | "
            f"Phase II: {stage2_duration:.2f}s ({phase2_comm_bytes/1e3:.1f} KB)"
        )

    # Save JSON summary & Q1 structured audit log
    save_json(results, out_p / "client_scalability_summary.json")
    save_json(results, out_p / "q1_scalability_log.json")

    # Generate LaTeX table matching Table 11
    latex_tbl = generate_scalability_latex_table(results)
    with open(out_p / "table_client_scalability.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Scalability experiment complete. Saved to {out_p}")
    return results


def generate_scalability_latex_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Format scalability results into publication-grade LaTeX tabular environment (Table 11)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Client scaling and stage-wise runtime decomposition on B-NAT at $\alpha_D=0.5$.}",
        r"\label{tab:client_scaling}",
        r"\small",
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"Clients & AUROC (\%) & Unknown-F1 (\%) & OS Macro-F1 (\%) & KFR (\%) & Runtime (s) \\",
        r"\midrule",
    ]

    for k_key, v in results.items():
        k = v["num_clients"]
        auroc = f"{v['auroc']*100:.3f}"
        unk_f1 = f"{v['unknown_f1']*100:.3f}"
        os_f1 = f"{v['open_set_macro_f1']*100:.3f}"
        kfr = f"{v['kfr']*100:.3f}"
        runtime = f"{v['total_time_sec']:.1f}"
        lines.append(f"{k} & {auroc} & {unk_f1} & {os_f1} & {kfr} & {runtime} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federation Scalability Experiment (E7)")
    parser.add_argument("--dataset", type=str, default="b_nat", help="Dataset name")
    parser.add_argument("--clients", type=str, default="10,50,100", help="Comma-separated client counts")
    parser.add_argument("--rounds", type=int, default=10, help="Federated training rounds")
    parser.add_argument("--beta", type=float, default=0.5, help="Dirichlet concentration parameter")
    parser.add_argument("--max_samples", type=int, default=None, help="Sample count limit for fast testing")
    parser.add_argument("--output_dir", type=str, default="results/scalability")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client_list = [int(c.strip()) for c in args.clients.split(",") if c.strip()]
    run_scalability_experiment(
        dataset_name=args.dataset,
        client_counts=client_list,
        rounds=args.rounds,
        beta_dirichlet=args.beta,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
