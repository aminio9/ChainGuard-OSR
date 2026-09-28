"""Leave-One-Attack-Out (LOAO) Semantic Robustness Suite.

Aligned with LaTeX Section 6.5 (Table 6 & Figure 7):
  - Sequentially sequesters each attack identity as the zero-day proxy C_U:
      1. BP (Block Parser buffer overflow exploits)
      2. DoS (high-rate Denial of Service floods)
      3. FoT (Flooding of Transactions / memory exhaustion)
      4. MitM (Man-in-the-Middle routing exploits)
  - Evaluates: AUROC, AUPR, FPR95, Unknown Recall, and Unknown-F1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict
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
from src.metrics.osr_metrics import compute_osr_metrics
from src.visualization.latex_tables import generate_latex_loao_table


def evaluate_loao_attack(
    attack_name: str,
    known_classes: list[int],
    unknown_classes: list[int],
    config_path: str = "configs/default.yaml",
    rounds: int = 15,
    beta_dirichlet: float = 0.5,
) -> Dict[str, float]:
    """Execute complete 3-phase training and evaluation for a specific LOAO attack holdout."""
    config = load_config(config_path)
    config.federated.num_rounds = rounds
    config.federated.partition.beta_dirichlet = beta_dirichlet
    config.data.known_classes = known_classes
    config.data.unknown_classes = unknown_classes

    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)

    # 1. Dataset loading / generation
    raw_dataset = generate_synthetic_p2p_data(
        num_samples=config.data.num_samples or 3000,
        in_features=config.data.in_features,
        num_classes=max(known_classes + unknown_classes) + 1,
        seed=config.experiment.seed,
    )

    train_ds, cal_ds, _, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=config.data.test_split_ratio,
        cal_ratio=config.data.cal_split_ratio,
        seed=config.experiment.seed,
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
    for r in range(1, rounds + 1):
        server.run_round(r)

    # Freeze encoder permanently (Algorithm 1 Line 7)
    server.freeze_global_encoder()

    num_known = len(known_classes)

    # 4. Phase II: Byzantine-Robust Profiling via Law of Total Variance
    server.run_phase_two_profiling(
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
        alpha=config.osr.conformal_alpha,
        apply_mad_filter=True,
    )

    # 5. Phase III: Wire-Speed Block-Diagonal Mahalanobis Edge Detector
    edge_detector = server.get_edge_detector(
        unknown_label=num_known,
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
    )

    preds_open, is_unk, cand, anomaly_scores = edge_detector.predict_batch(test_open_ds.features)
    y_open_true = test_open_ds.labels.numpy()

    metrics = compute_osr_metrics(
        y_true=y_open_true,
        y_pred=preds_open,
        anomaly_scores=anomaly_scores,
        num_known_classes=num_known,
        unknown_label=num_known,
    )
    return metrics


def run_loao_suite(
    rounds: int = 10,
    beta_dirichlet: float = 0.5,
    output_dir: str = "results/loao",
) -> None:
    """Run Leave-One-Attack-Out across BP, DoS, FoT, MitM matching Table 6."""
    logger = setup_logger(name="loao_runner")
    logger.info(f"Starting Leave-One-Attack-Out (LOAO) suite (rounds={rounds}, beta={beta_dirichlet})...")

    # B-NAT classes: 0: Normal, 1: BP, 2: DoS, 3: FoT, 4: MitM
    # Each run sequesters one attack as C_U and retains the remaining 4 as C_K
    attacks = {
        "BP": {"known": [0, 2, 3, 4], "unknown": [1]},
        "DoS": {"known": [0, 1, 3, 4], "unknown": [2]},
        "FoT": {"known": [0, 1, 2, 4], "unknown": [3]},
        "MitM": {"known": [0, 1, 2, 3], "unknown": [4]},
    }

    results: Dict[str, Dict[str, float]] = {}

    for attack_name, splits in attacks.items():
        logger.info(f"--> Evaluating LOAO Attack: {attack_name} as held-out C_U")
        metrics = evaluate_loao_attack(
            attack_name=attack_name,
            known_classes=splits["known"],
            unknown_classes=splits["unknown"],
            rounds=rounds,
            beta_dirichlet=beta_dirichlet,
        )
        results[attack_name] = metrics
        logger.info(
            f"    [{attack_name}] AUROC: {metrics['auroc']*100:.2f}% | "
            f"AUPR: {metrics['aupr']*100:.2f}% | "
            f"FPR95: {metrics['fpr95']*100:.2f}% | "
            f"Unk-Recall: {metrics['unknown_recall']*100:.2f}% | "
            f"Unk-F1: {metrics['unknown_f1']*100:.2f}%"
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(results, out_p / f"loao_results_beta_{beta_dirichlet}.json")

    latex_tbl = generate_latex_loao_table(
        results,
        caption=f"B-NAT Leave-One-Attack-Out Evaluation at $\\alpha_D = {beta_dirichlet}$ across Distinct Held-Out Attacks (Table 6)",
        label=f"tab:loao_results_{str(beta_dirichlet).replace('.', '_')}",
    )
    with open(out_p / f"table_loao_beta_{beta_dirichlet}.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"LOAO benchmarking completed. Table written to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Leave-One-Attack-Out Evaluation")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per attack holdout")
    parser.add_argument("--beta", type=float, default=0.5, help="Dirichlet concentration alpha_D (paper default: 0.5)")
    parser.add_argument("--output_dir", type=str, default="results/loao")
    args = parser.parse_args()

    run_loao_suite(rounds=args.rounds, beta_dirichlet=args.beta, output_dir=args.output_dir)
