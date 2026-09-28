"""Benchmark evaluation suite comparing canonical federated baselines against proposed FedTROS-MC.

Aligned with LaTeX Section 5.3 & Section 6.2 (Tables 4 & 5).
Baselines:
  - FedAvg (McMahan et al., 2017): Coordinate-wise averaging, empirical cross-entropy.
  - FedProx (Li et al., 2020): Proximal parameter-regularized federated learning (mu = 0.01).
  - FedProto + Energy: Baseline prototype aggregation with energy-based OSR.
  - FedTROS-MC (Proposed): 3-Phase protocol with dynamic anchor, VIB, Byzantine-MAD, Total Variance,
    and Wire-Speed Block-Diagonal Mahalanobis Edge Detector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, Optional
import numpy as np
import torch

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.tracking.logger import setup_logger
from src.utils.device import get_device
from src.utils.io import save_json
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.tabular_p2p import load_p2p_tabular_dataset
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics
from src.visualization.latex_tables import generate_latex_benchmark_table


def evaluate_method_configuration(
    method_name: str,
    config_path: str,
    rounds: int = 15,
    beta_dirichlet: Optional[float] = None,
) -> Dict[str, float]:
    """Execute complete training and 3-phase open-set evaluation for a given method configuration."""
    overrides: Dict[str, Any] = {}
    if beta_dirichlet is not None:
        overrides.setdefault("federated", {})["partition"] = {"beta_dirichlet": beta_dirichlet}

    config = load_config(config_path, overrides=overrides if overrides else None)
    config.federated.num_rounds = rounds
    if beta_dirichlet is not None:
        config.federated.partition.beta_dirichlet = beta_dirichlet

    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)

    # 1. Prepare Data
    if config.data.dataset_name == "synthetic_p2p":
        raw_dataset = generate_synthetic_p2p_data(
            num_samples=config.data.num_samples or 3000,
            in_features=config.data.in_features,
            num_classes=config.data.num_classes_total,
            seed=config.experiment.seed,
        )
    else:
        csv_file = Path(config.data.data_dir) / f"{config.data.dataset_name}.csv"
        if csv_file.exists():
            raw_dataset = load_p2p_tabular_dataset(
                csv_path=csv_file,
                target_column=config.data.target_column or "label",
                normalize=config.data.normalize,
            )
        else:
            raw_dataset = generate_synthetic_p2p_data(
                num_samples=config.data.num_samples or 3000,
                in_features=config.data.in_features,
                num_classes=config.data.num_classes_total,
                seed=config.experiment.seed,
            )

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=config.data.known_classes,
        unknown_classes=config.data.unknown_classes,
        test_ratio=config.data.test_split_ratio,
        cal_ratio=config.data.cal_split_ratio,
        seed=config.experiment.seed,
    )

    # 2. Dirichlet Non-IID Partition
    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=config.federated.num_clients,
        beta_dirichlet=config.federated.partition.beta_dirichlet,
        seed=config.experiment.seed,
    )

    clients = []
    for client_id in range(config.federated.num_clients):
        sub_indices = client_indices[client_id]
        idx_t = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        clients.append(FederatedClient(client_id, sub_dataset, config, device, cal_dataset=cal_ds))

    server = FederatedServer(config, clients, device)

    # 3. Phase I: Collaborative Federated Training
    for r in range(1, rounds + 1):
        server.run_round(r)

    # Freeze encoder permanently (Algorithm 1 Line 7)
    server.freeze_global_encoder()

    num_known = len(config.data.known_classes)

    # 4. Phase II: Byzantine-Robust Profiling via Law of Total Variance
    server.run_phase_two_profiling(
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
        alpha=config.osr.conformal_alpha,
        apply_mad_filter=True,
        mad_gamma_0=config.federated.mad_gamma_0,
        mad_zeta=config.federated.mad_zeta,
    )

    # 5. Phase III: Wire-Speed Block-Diagonal Mahalanobis Edge Detector
    edge_detector = server.get_edge_detector(
        unknown_label=num_known,
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
    )

    # Evaluate on open-set test data
    preds_open, is_unk, cand, anomaly_scores = edge_detector.predict_batch(test_open_ds.features)
    y_open_true = test_open_ds.labels.numpy()

    metrics = compute_osr_metrics(
        y_true=y_open_true,
        y_pred=preds_open,
        anomaly_scores=anomaly_scores,
        num_known_classes=num_known,
        unknown_label=num_known,
    )

    # Evaluate on closed-set test data for closed-world classification fidelity
    preds_closed, _, _, _ = edge_detector.predict_batch(test_closed_ds.features)
    y_closed_true = test_closed_ds.labels.numpy()
    closed_metrics = compute_classification_metrics(
        y_true=y_closed_true,
        y_pred=preds_closed,
        num_classes=num_known,
    )

    metrics["closed_set_accuracy"] = closed_metrics["accuracy"]
    metrics["accuracy"] = closed_metrics["accuracy"]
    metrics["closed_set_macro_f1"] = closed_metrics["macro_f1"]

    return metrics


def run_all_baselines(
    rounds: int = 10,
    beta_dirichlet: float = 0.1,
    output_dir: str = "results/baselines",
) -> None:
    """Run comparative baseline suite across canonical algorithms under identical data splits."""
    logger = setup_logger(name="baseline_runner")
    logger.info(
        f"Running complete baseline comparison suite (rounds={rounds}, beta={beta_dirichlet})..."
    )

    methods = {
        "FedAvg": "configs/baselines/fedavg_msp.yaml",
        "FedProx": "configs/baselines/fedprox_openmax.yaml",
        "FedProto + Energy": "configs/baselines/fedproto_energy.yaml",
        "FedTROS-MC (Proposed)": "configs/default.yaml",
    }

    all_results: Dict[str, Dict[str, float]] = {}
    for name, cfg_path in methods.items():
        logger.info(f"--> Evaluating: {name} (config: {cfg_path})")
        metrics = evaluate_method_configuration(
            name, cfg_path, rounds=rounds, beta_dirichlet=beta_dirichlet
        )
        all_results[name] = metrics
        logger.info(
            f"    [{name}] AUROC: {metrics['auroc']*100:.2f}% | "
            f"AUPR: {metrics['aupr']*100:.2f}% | "
            f"Unk-Recall: {metrics['unknown_recall']*100:.2f}% | "
            f"Unk-F1: {metrics['unknown_f1']*100:.2f}% | "
            f"OS Macro-F1: {metrics['open_set_macro_f1']*100:.2f}% | "
            f"KFR: {metrics['kfr']*100:.2f}%"
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(all_results, out_p / f"baseline_comparison_beta_{beta_dirichlet}.json")

    latex_tbl = generate_latex_benchmark_table(
        all_results,
        caption=f"Baseline Comparison under Non-IID Dirichlet Skew ($\\alpha_D = {beta_dirichlet}$)",
        label=f"tab:baseline_comparison_{str(beta_dirichlet).replace('.', '_')}",
    )
    with open(out_p / f"table_baselines_beta_{beta_dirichlet}.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Baseline benchmarking completed. Table written to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Baselines Suite")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per baseline")
    parser.add_argument("--beta", type=float, default=0.1, help="Dirichlet concentration alpha_D")
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    args = parser.parse_args()

    run_all_baselines(rounds=args.rounds, beta_dirichlet=args.beta, output_dir=args.output_dir)
