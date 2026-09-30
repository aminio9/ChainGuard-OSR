"""Complete 3-Phase Federated Open-Set Recognition Experiment (Algorithms 1 & 2).

Executes:
  - Phase I: Representation Learning with Drift Control & Cosine Annealing (Equations 1-11)
  - Phase II: Zero-Leakage Federated Class Profiling via Law of Total Variance
  - Phase III: Wire-Speed Conformal Edge Detection with Block-Diagonal Mahalanobis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.tracking.logger import setup_logger
from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.device import get_device
from src.utils.checkpoint import CheckpointManager
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.tabular_p2p import load_p2p_tabular_dataset
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.osr.block_mahalanobis import WireSpeedEdgeDetector
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics
from src.metrics.conformal_metrics import compute_conformal_metrics
from src.visualization.publication_plots import (
    plot_convergence_curves,
    plot_conformal_set_distribution,
)


def run_experiment(config_path: str = "configs/default.yaml", override_rounds: int | None = None) -> None:
    # 1. Load Configuration
    config = load_config(config_path)
    if override_rounds is not None:
        config.federated.num_rounds = override_rounds

    # 2. Setup Seed, Device, and Tracking
    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)

    tracker = ExperimentTracker(config.experiment.name, base_dir=config.experiment.output_dir)
    logger = setup_logger(name="fl_osr", log_dir=tracker.run_dir)
    checkpoint_mgr = CheckpointManager(checkpoint_dir=config.experiment.checkpoint_dir)

    logger.info("Initializing 3-Phase Federated OSR Experiment (Algorithms 1 & 2)")
    logger.info(f"Target Device: {device} | Dirichlet Beta: {config.federated.partition.beta_dirichlet}")
    tracker.initialize_run(config.model_dump())

    # 3. Prepare Dataset & Open-Set Splits
    if config.data.dataset_name == "synthetic_p2p":
        raw_dataset = generate_synthetic_p2p_data(
            num_samples=config.data.num_samples or 5000,
            in_features=config.data.in_features,
            num_classes=config.data.num_classes_total,
            seed=config.experiment.seed,
        )
    else:
        raw_dataset = load_p2p_tabular_dataset(
            csv_path=Path(config.data.data_dir) / f"{config.data.dataset_name}.csv",
            target_column=config.data.target_column or "label",
            normalize=config.data.normalize,
        )

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=config.data.known_classes,
        unknown_classes=config.data.unknown_classes,
        test_ratio=config.data.test_split_ratio,
        cal_ratio=config.data.cal_split_ratio,
        seed=config.experiment.seed,
    )

    logger.info(
        f"Data Partitioned: Train={len(train_ds)}, Cal={len(cal_ds)}, "
        f"Test Closed={len(test_closed_ds)}, Test Open={len(test_open_ds)}"
    )

    # 4. Dirichlet Non-IID Partition across clients
    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=config.federated.num_clients,
        beta_dirichlet=config.federated.partition.beta_dirichlet,
        seed=config.experiment.seed,
    )

    clients = []
    for client_id in range(config.federated.num_clients):
        sub_indices = client_indices[client_id]
        idx_tensor = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_tensor], train_ds.labels[idx_tensor])
        client = FederatedClient(client_id, sub_dataset, config, device, cal_dataset=cal_ds)
        clients.append(client)

    server = FederatedServer(config, clients, device)

    # =========================================================================
    # PHASE I: Representation Learning with Drift Control (Equations 1-11)
    # =========================================================================
    num_rounds = config.federated.num_rounds
    logger.info(f"--- [Phase I] Starting {num_rounds} communication rounds with Cosine Annealed SGD & Decoupled Aggregation ---")

    for r in range(1, num_rounds + 1):
        round_loss = server.run_round(r)
        tracker.log_round(r, round_loss)

        if r % 10 == 0 or r == num_rounds:
            logger.info(
                f"Round [{r:03d}/{num_rounds:03d}] | Total Loss: {round_loss.get('loss_total', 0.0):.4f} "
                f"| BLA Loss: {round_loss.get('loss_bla', 0.0):.4f} | VIB Loss: {round_loss.get('loss_vib', 0.0):.4f} "
                f"| LR: {round_loss.get('lr_scheduled', 0.0):.6f}"
            )
            if config.experiment.save_checkpoints:
                checkpoint_mgr.save_checkpoint(
                    round_idx=r,
                    model_state_dict=server.global_model.state_dict(),
                    global_prototypes=getattr(server, "global_prototypes", None),
                )

    # Freeze Global Encoder permanently: theta_g^* = theta_g^T (Algorithm 1 Line 7)
    server.freeze_global_encoder()
    logger.info("--- [Phase I Completed] Global encoder theta_g^* frozen permanently ---")

    # =========================================================================
    # PHASE II: Zero-Leakage Federated Class Profiling via Law of Total Variance
    # =========================================================================
    logger.info("--- [Phase II] Extracting sub-block moments and running Law of Total Variance aggregation ---")
    global_profiles = server.run_phase_two_profiling(
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
        alpha=config.osr.conformal_alpha,
    )
    logger.info(f"Synthesized exact global profiles for {len(global_profiles)} known classes via Theorem 1")

    # =========================================================================
    # PHASE III: Wire-Speed Conformal Edge Detection (Algorithm 2)
    # =========================================================================
    logger.info("--- [Phase III] Evaluating Wire-Speed Block-Diagonal Mahalanobis Edge Detector (Algorithm 2) ---")
    num_known = len(config.data.known_classes)
    edge_detector = server.get_edge_detector(
        unknown_label=num_known,
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
    )

    preds_wire, is_unk_wire, cand_wire, dist_scores_wire = edge_detector.predict_batch(test_open_ds.features)
    y_test_true = test_open_ds.labels.numpy()

    # Metrics for Wire-Speed Block-Diagonal Mahalanobis
    mahalanobis_osr_metrics = compute_osr_metrics(
        y_true=y_test_true,
        y_pred=preds_wire,
        anomaly_scores=dist_scores_wire,
        num_known_classes=num_known,
        unknown_label=num_known,
    )

    logger.info("=== Wire-Speed Block-Diagonal Mahalanobis Evaluation (Phase III) ===")
    logger.info(f"Open-Set AUROC (Mahalanobis): {mahalanobis_osr_metrics['auroc'] * 100.0:.2f}%")
    logger.info(f"Open-Set AUPR: {mahalanobis_osr_metrics['aupr'] * 100.0:.2f}%")
    logger.info(f"Open-Set Macro F1: {mahalanobis_osr_metrics['open_set_macro_f1'] * 100.0:.2f}%")
    logger.info(f"Known False Rejection Rate (KFR): {mahalanobis_osr_metrics['kfr'] * 100.0:.2f}% (Target <= {config.osr.conformal_alpha * 100:.1f}%)")

    final_results = {
        "mahalanobis_auroc": mahalanobis_osr_metrics["auroc"],
        "mahalanobis_aupr": mahalanobis_osr_metrics["aupr"],
        "mahalanobis_f1": mahalanobis_osr_metrics["open_set_macro_f1"],
        "mahalanobis_kfr": mahalanobis_osr_metrics["kfr"],
    }

    # Persist Artifacts and Plots
    tracker.finish_run(final_results=final_results)
    plot_convergence_curves(tracker.history, tracker.run_dir / "convergence.png")
    # plot_conformal_set_distribution(conf_sets, tracker.run_dir / "conformal_distribution.png")
    logger.info(f"Experiment finished successfully! Results saved to: {tracker.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-Phase Federated OSR Experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    args = parser.parse_args()

    run_experiment(config_path=args.config, override_rounds=args.rounds)
