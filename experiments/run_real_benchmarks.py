"""Unified experiment runner for real intrusion detection benchmarks.

Executes 3-Phase Open-Set Recognition and evaluates both Closed-Set and Open-Set performance
across all 4 real benchmarks from the paper (Section 5.1 & Section 6.4, Table 7):
  1. B-NAT (Blockchain Network Attack Traffic)
  2. B-TAT (Blockchain Transaction and Attack Traffic)
  3. ToN-IoT (Industrial IoT Telemetry Benchmark)
  4. CIC-IDS2017 (Enterprise Network Intrusion Benchmark)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict
import numpy as np
import torch

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.tracking.logger import setup_logger
from src.utils.device import get_device
from src.utils.io import save_json
from src.data.tabular_p2p import load_benchmark_dataset, BENCHMARK_METADATA
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics


def run_benchmark_trial(
    dataset_name: str,
    rounds: int = 3,
    samples_per_class: int = 200,
    beta_dirichlet: float = 0.5,
    config_path: str = "configs/default.yaml",
) -> Dict[str, float]:
    """Execute complete 3-phase training and closed/open-set evaluation on a real benchmark."""
    logger = setup_logger(name=f"bench_{dataset_name}")
    logger.info(f"Loading real benchmark dataset: {dataset_name.upper()} (samples_per_class={samples_per_class})...")

    dataset, meta = load_benchmark_dataset(
        dataset_name=dataset_name,
        data_dir="data/raw",
        samples_per_class=samples_per_class,
    )

    in_features = meta["in_features"]
    known_classes = meta["known_classes"]
    unknown_classes = meta["unknown_classes"]
    num_known = len(known_classes)

    logger.info(
        f"[{dataset_name.upper()}] Features: {in_features} | Known classes: {meta['known_class_names']} | "
        f"Unknown classes: {meta['unknown_class_names']} | Total samples: {len(dataset)}"
    )

    overrides = {
        "data": {
            "dataset_name": dataset_name,
            "in_features": in_features,
            "num_classes_total": num_known + len(unknown_classes),
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
        },
        "model": {
            "in_features": in_features,
        },
        "federated": {
            "num_rounds": rounds,
            "partition": {"beta_dirichlet": beta_dirichlet},
            # Ensure block_dim * num_blocks == latent_dim (64)
            "num_blocks": 8,
            "block_dim": 8,
        },
    }

    config = load_config(config_path, overrides=overrides)
    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)

    # 1. Zero-Leakage Dataset Splitting (Section 5.1 Discipline)
    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=config.data.test_split_ratio,
        cal_ratio=config.data.cal_split_ratio,
        seed=config.experiment.seed,
    )

    logger.info(
        f"Splits -> Train (known only): {len(train_ds)}, Cal (known only): {len(cal_ds)}, "
        f"Test Closed: {len(test_closed_ds)}, Test Open (with zero-day): {len(test_open_ds)}"
    )

    # 2. Non-IID Dirichlet Partition
    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=config.federated.num_clients,
        beta_dirichlet=beta_dirichlet,
        seed=config.experiment.seed,
    )

    clients = []
    for client_id in range(config.federated.num_clients):
        sub_indices = client_indices[client_id]
        idx_t = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        clients.append(FederatedClient(client_id, sub_dataset, config, device, cal_dataset=cal_ds))

    server = FederatedServer(config, clients, device)

    # 3. Phase I: Collaborative Representation Learning
    logger.info(f"--- [Phase I] Collaborative Training across {config.federated.num_clients} clients ({rounds} rounds) ---")
    for r in range(1, rounds + 1):
        round_metrics = server.run_round(r)
        logger.info(f"Round [{r}/{rounds}] | Loss: {round_metrics.get('loss_total', 0.0):.4f}")

    server.freeze_global_encoder()

    # 4. Phase II: Byzantine-Robust Profiling via Law of Total Variance
    logger.info("--- [Phase II] Sub-Block Moment Extraction & Byzantine MAD Aggregation ---")
    server.run_phase_two_profiling(
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
        alpha=config.osr.conformal_alpha,
        apply_mad_filter=True,
        cal_dataset=cal_ds,
    )

    # 5. Phase III: Wire-Speed Block-Diagonal Mahalanobis Edge Detector
    logger.info("--- [Phase III] Evaluating Wire-Speed Conformal Edge Detector ---")
    edge_detector = server.get_edge_detector(
        unknown_label=num_known,
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
    )

    # --- A) Closed-Set Evaluation (on test_closed_ds) ---
    preds_closed, _, _, _ = edge_detector.predict_batch(test_closed_ds.features)
    y_closed_true = test_closed_ds.labels.numpy()
    closed_metrics = compute_classification_metrics(
        y_true=y_closed_true,
        y_pred=preds_closed,
        num_classes=num_known,
    )

    # Balanced accuracy for closed-set
    class_accs = []
    for c in range(num_known):
        c_mask = (y_closed_true == c)
        if np.sum(c_mask) > 0:
            class_accs.append(float(np.mean(preds_closed[c_mask] == c)))
    balanced_acc = float(np.mean(class_accs)) if class_accs else 0.0

    # --- B) Open-Set Evaluation (on test_open_ds) ---
    preds_open, is_unk, cand, anomaly_scores = edge_detector.predict_batch(test_open_ds.features)
    y_open_true = test_open_ds.labels.numpy()
    osr_metrics = compute_osr_metrics(
        y_true=y_open_true,
        y_pred=preds_open,
        anomaly_scores=anomaly_scores,
        num_known_classes=num_known,
        unknown_label=num_known,
    )

    final_results = {
        # Metadata
        "dataset": dataset_name.upper(),
        "in_features": in_features,
        "num_known": num_known,
        "num_unknown": len(unknown_classes),
        # Closed-Set Metrics
        "closed_accuracy": closed_metrics["accuracy"],
        "closed_balanced_accuracy": balanced_acc,
        "closed_macro_f1": closed_metrics["macro_f1"],
        # Open-Set Ranking Metrics
        "osr_auroc": osr_metrics["auroc"],
        "osr_aupr": osr_metrics["aupr"],
        "osr_fpr95": osr_metrics["fpr95"],
        # Open-Set Operational Metrics
        "osr_unknown_recall": osr_metrics["unknown_recall"],
        "osr_unknown_precision": osr_metrics["unknown_precision"],
        "osr_unknown_f1": osr_metrics["unknown_f1"],
        "osr_kfr": osr_metrics["kfr"],
        "osr_macro_f1": osr_metrics["open_set_macro_f1"],
    }

    logger.info(
        f"=== [{dataset_name.upper()}] Evaluation Results ===\n"
        f"  [Closed-Set] Accuracy: {final_results['closed_accuracy']*100:.2f}% | "
        f"Balanced Acc: {final_results['closed_balanced_accuracy']*100:.2f}% | "
        f"Macro-F1: {final_results['closed_macro_f1']*100:.2f}%\n"
        f"  [Open-Set]   AUROC: {final_results['osr_auroc']*100:.2f}% | "
        f"AUPR: {final_results['osr_aupr']*100:.2f}% | "
        f"Unk-Recall: {final_results['osr_unknown_recall']*100:.2f}% | "
        f"Unk-F1: {final_results['osr_unknown_f1']*100:.2f}% | "
        f"KFR: {final_results['osr_kfr']*100:.2f}% | "
        f"OS Macro-F1: {final_results['osr_macro_f1']*100:.2f}%"
    )

    return final_results


def run_all_four_benchmarks(
    rounds: int = 3,
    samples_per_class: int = 200,
    beta_dirichlet: float = 0.5,
    output_dir: str = "results/real_benchmarks",
) -> None:
    """Run both closed-set and open-set training/eval across all 4 real benchmarks."""
    logger = setup_logger(name="benchmark_suite")
    logger.info("==========================================================================")
    logger.info("Executing 3-Phase Evaluation across all 4 Real Benchmarks (Table 7)")
    logger.info(f"Rounds: {rounds} | Samples/Class: {samples_per_class} | Dirichlet Beta: {beta_dirichlet}")
    logger.info("==========================================================================")

    all_benchmarks = ["b_nat", "b_tat", "ton_iot", "cic_ids2017"]
    summary: Dict[str, Dict[str, float]] = {}

    for ds_name in all_benchmarks:
        res = run_benchmark_trial(
            dataset_name=ds_name,
            rounds=rounds,
            samples_per_class=samples_per_class,
            beta_dirichlet=beta_dirichlet,
        )
        summary[ds_name] = res

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_p / "four_benchmarks_summary.json")

    # Generate LaTeX table matching Table 7
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-Dataset Open-Set Recognition Performance across Four Real Intrusion Benchmarks (Table 7)}",
        r"\label{tab:cross_dataset_results_real}",
        r"\small",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"Dataset & $d$ & Closed Acc (\%) & AUROC (\%) & AUPR (\%) & Unknown-F1 (\%) & OS Macro-F1 (\%) & KFR (\%) \\",
        r"\midrule",
    ]

    for ds_key, m in summary.items():
        ds_name = m["dataset"]
        d = int(m["in_features"])
        c_acc = m["closed_accuracy"] * 100.0
        auroc = m["osr_auroc"] * 100.0
        aupr = m["osr_aupr"] * 100.0
        unk_f1 = m["osr_unknown_f1"] * 100.0
        os_f1 = m["osr_macro_f1"] * 100.0
        kfr = m["osr_kfr"] * 100.0

        latex.append(
            f"{ds_name} & {d} & {c_acc:.2f} & {auroc:.2f} & {aupr:.2f} & {unk_f1:.2f} & {os_f1:.2f} & {kfr:.2f} \\\\"
        )

    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    latex_str = "\n".join(latex)

    with open(out_p / "table_cross_dataset.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    logger.info(f"All 4 benchmark evaluations completed! Results saved to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4 Real Intrusion Benchmarks")
    parser.add_argument("--rounds", type=int, default=3, help="Federation rounds per benchmark")
    parser.add_argument("--samples_per_class", type=int, default=200, help="Samples per class for stratified balance")
    parser.add_argument("--beta", type=float, default=0.5, help="Dirichlet concentration alpha_D (default: 0.5)")
    parser.add_argument("--output_dir", type=str, default="results/real_benchmarks")
    args = parser.parse_args()

    run_all_four_benchmarks(
        rounds=args.rounds,
        samples_per_class=args.samples_per_class,
        beta_dirichlet=args.beta,
        output_dir=args.output_dir,
    )
