"""Ablation Studies A4 & A5: Covariance Moment Aggregation (Theorem 1) and Metric Structure.

Aligned with LaTeX Section 4.2 & Section 6.6:
  - Ablation A4: Law of Total Variance (Theorem 1) vs Simple Average Covariance vs FedProto (Euclidean).
  - Ablation A5: Block-Diagonal Mahalanobis vs Full Mahalanobis vs Diagonal Mahalanobis vs Euclidean.
"""

from __future__ import annotations

import argparse
import sys
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
from src.aggregation.total_variance import aggregate_total_variance_moments, GlobalClassProfile
from src.aggregation.byzantine_mad import ClientClassMoment
from src.osr.block_mahalanobis import WireSpeedEdgeDetector, compute_block_mahalanobis_distances
from src.osr.conformal import compute_finite_sample_quantile
from src.metrics.osr_metrics import compute_osr_metrics


def run_covariance_distance_ablations(
    dataset_name: str = "b_nat",
    rounds: int = 10,
    beta_dirichlet: float = 0.5,
    max_samples: Optional[int] = None,
    output_dir: str = "results/ablations_cov_metric",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute ablation experiments A4 and A5 on real benchmark data."""
    logger = setup_logger(name="cov_metric_ablation")
    logger.info(f"Starting Covariance and Metric Ablation Suite (A4 & A5) on {dataset_name}...")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # 1. Load data
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
        raw_dataset = generate_synthetic_p2p_data(
            num_samples=max_samples or 3000,
            in_features=34,
            num_classes=5,
            seed=seed,
        )
        known_classes = [0, 1, 2, 3]
        unknown_classes = [4]

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=0.2,
        cal_ratio=0.15,
        seed=seed,
    )

    config = load_config("configs/default.yaml")
    config.federated.num_clients = 10
    config.federated.num_rounds = rounds
    config.federated.partition.beta_dirichlet = beta_dirichlet
    config.data.known_classes = known_classes
    config.data.unknown_classes = unknown_classes
    config.model.in_features = train_ds.features.shape[1]
    device = get_device(config.experiment.device)

    # Partition
    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=config.federated.num_clients,
        beta_dirichlet=beta_dirichlet,
        min_samples_per_client=max(2, min(10, len(train_ds) // (2 * config.federated.num_clients))),
        seed=seed,
    )

    clients = []
    for client_id in range(config.federated.num_clients):
        sub_indices = client_indices[client_id]
        idx_t = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        clients.append(FederatedClient(client_id, sub_dataset, config, device, cal_dataset=cal_ds))

    server = FederatedServer(config, clients, device)

    # Train Phase I
    logger.info("Executing Phase I collaborative training...")
    for r in range(1, rounds + 1):
        server.run_round(r)
    server.freeze_global_encoder()

    num_known = len(known_classes)
    num_blocks = config.federated.num_blocks
    block_dim = config.federated.block_dim
    alpha = config.osr.conformal_alpha

    # Extract client moments
    client_moments_map: Dict[int, List[ClientClassMoment]] = {c: [] for c in range(num_known)}
    for client in clients:
        m_list = client.extract_phase_two_moments(
            frozen_encoder=server.global_model.encoder,
            num_blocks=num_blocks,
            block_dim=block_dim,
            alpha=alpha,
        )
        for idx, c in enumerate(client.local_classes):
            c = int(c)
            if idx < len(m_list) and c in client_moments_map:
                client_moments_map[c].append(m_list[idx])

    # Extract latent representations for calibration and testing
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_open_ds, batch_size=128, shuffle=False)

    cal_z_list, cal_y_list = [], []
    with torch.no_grad():
        for cx, cy in cal_loader:
            cx_dev = cx.to(device)
            out = server.global_model.encoder(cx_dev)
            mu_enc = out[3] if isinstance(out, tuple) and len(out) >= 4 else (out[0] if isinstance(out, tuple) else out)
            cal_z_list.append(mu_enc.cpu())
            cal_y_list.append(cy.cpu())
    cal_z = torch.cat(cal_z_list, dim=0)
    cal_y = torch.cat(cal_y_list, dim=0)

    test_z_list, test_y_list = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            tx_dev = tx.to(device)
            out = server.global_model.encoder(tx_dev)
            mu_enc = out[3] if isinstance(out, tuple) and len(out) >= 4 else (out[0] if isinstance(out, tuple) else out)
            test_z_list.append(mu_enc.cpu())
            test_y_list.append(ty.cpu())
    test_z = torch.cat(test_z_list, dim=0)
    test_y = torch.cat(test_y_list, dim=0)

    results_a4: Dict[str, Any] = {}
    results_a5: Dict[str, Any] = {}

    # =========================================================================
    # ABLATION A4: Moment Aggregation Method (Theorem 1)
    # =========================================================================
    logger.info("\n--- Evaluating Ablation A4: Covariance Aggregation Strategies ---")

    # Strategy 1: Proposed Exact Law of Total Variance (Theorem 1)
    tot_var_profiles, _ = aggregate_total_variance_moments(
        class_submissions=client_moments_map,
        num_blocks=num_blocks,
        block_dim=block_dim,
        apply_mad_filter=True,
    )
    # Calibrate
    calibrated_tot_var = {}
    for c in range(num_known):
        p = tot_var_profiles[c]
        c_idx = (cal_y == c)
        if c_idx.sum() > 0:
            dists = compute_block_mahalanobis_distances(cal_z[c_idx], {c: p}, num_blocks=num_blocks, block_dim=block_dim)[c]
            tau = compute_finite_sample_quantile(dists.numpy(), alpha=alpha)
        else:
            tau = 1.0
        calibrated_tot_var[c] = p._replace(conformal_threshold=tau)

    # Strategy 2: Simple Arithmetic Average of Client Covariances (ignoring between-client shift)
    simple_avg_profiles = {}
    for c in range(num_known):
        subs = client_moments_map[c]
        total_n = sum(s.sample_count for s in subs)
        mu_c = sum(s.sample_count * s.mu for s in subs) / max(total_n, 1)
        # Naive weighted average of block covariances WITHOUT (mu_k - mu)(mu_k - mu)^T
        block_covs = []
        inv_covs = []
        for b in range(num_blocks):
            cov_b = sum(s.sample_count * s.block_covs[b] for s in subs) / max(total_n, 1) + 1e-4 * torch.eye(block_dim)
            block_covs.append(cov_b)
            inv_covs.append(torch.linalg.pinv(cov_b))
        prof = GlobalClassProfile(
            class_id=c,
            total_samples=total_n,
            mu=mu_c,
            block_covs=block_covs,
            inv_block_covs=inv_covs,
        )
        # Calibrate
        c_idx = (cal_y == c)
        if c_idx.sum() > 0:
            dists = compute_block_mahalanobis_distances(cal_z[c_idx], {c: prof}, num_blocks=num_blocks, block_dim=block_dim)[c]
            tau = compute_finite_sample_quantile(dists.numpy(), alpha=alpha)
        else:
            tau = 1.0
        simple_avg_profiles[c] = prof._replace(conformal_threshold=tau)

    # Strategy 3: FedProto Baseline (Euclidean Prototype, Identity Covariance)
    fedproto_profiles = {}
    for c in range(num_known):
        subs = client_moments_map[c]
        total_n = sum(s.sample_count for s in subs)
        mu_c = sum(s.sample_count * s.mu for s in subs) / max(total_n, 1)
        block_covs = [torch.eye(block_dim) for _ in range(num_blocks)]
        inv_covs = [torch.eye(block_dim) for _ in range(num_blocks)]
        prof = GlobalClassProfile(
            class_id=c,
            total_samples=total_n,
            mu=mu_c,
            block_covs=block_covs,
            inv_block_covs=inv_covs,
        )
        c_idx = (cal_y == c)
        if c_idx.sum() > 0:
            dists = compute_block_mahalanobis_distances(cal_z[c_idx], {c: prof}, num_blocks=num_blocks, block_dim=block_dim)[c]
            tau = compute_finite_sample_quantile(dists.numpy(), alpha=alpha)
        else:
            tau = 1.0
        fedproto_profiles[c] = prof._replace(conformal_threshold=tau)

    a4_variants = {
        "FedProto (Euclidean Prototype)": fedproto_profiles,
        "Simple Mean Covariance (w/o Between-Client)": simple_avg_profiles,
        "Law of Total Variance (Theorem 1, Proposed)": calibrated_tot_var,
    }

    for name, profs in a4_variants.items():
        det = WireSpeedEdgeDetector(
            frozen_encoder=server.global_model.encoder,
            global_profiles=profs,
            unknown_label=num_known,
            num_blocks=num_blocks,
            block_dim=block_dim,
        )
        preds, _, _, scores = det.predict_batch(test_open_ds.features)
        m = compute_osr_metrics(
            y_true=test_open_ds.labels.numpy(),
            y_pred=preds,
            anomaly_scores=scores,
            num_known_classes=num_known,
            unknown_label=num_known,
        )
        results_a4[name] = m
        logger.info(
            f"  [{name}] AUROC: {m['auroc']*100:.2f}% | "
            f"Unk-F1: {m['unknown_f1']*100:.2f}% | "
            f"OS Macro-F1: {m['open_set_macro_f1']*100:.2f}% | "
            f"KFR: {m['kfr']*100:.2f}%"
        )

    # =========================================================================
    # ABLATION A5: Distance Metric & Sub-block Granularity
    # =========================================================================
    logger.info("\n--- Evaluating Ablation A5: Distance Metrics and Block Structure ---")

    # 1. Block-Diagonal (B=4, d_b=16) - Already evaluated above as calibrated_tot_var
    results_a5["Block-Diagonal Mahalanobis (B=4, Proposed)"] = results_a4["Law of Total Variance (Theorem 1, Proposed)"]

    # 2. Euclidean Distance
    results_a5["Euclidean Prototype (Sigma = I)"] = results_a4["FedProto (Euclidean Prototype)"]

    # 3. Diagonal Mahalanobis (Diagonal variance only)
    diag_profiles = {}
    for c in range(num_known):
        p = tot_var_profiles[c]
        diag_blocks = [torch.diag(torch.diag(b)) for b in p.block_covs]
        diag_invs = [torch.linalg.pinv(b + 1e-4 * torch.eye(block_dim)) for b in diag_blocks]
        prof = GlobalClassProfile(
            class_id=c,
            total_samples=p.total_samples,
            mu=p.mu,
            block_covs=diag_blocks,
            inv_block_covs=diag_invs,
        )
        c_idx = (cal_y == c)
        if c_idx.sum() > 0:
            dists = compute_block_mahalanobis_distances(cal_z[c_idx], {c: prof}, num_blocks=num_blocks, block_dim=block_dim)[c]
            tau = compute_finite_sample_quantile(dists.numpy(), alpha=alpha)
        else:
            tau = 1.0
        diag_profiles[c] = prof._replace(conformal_threshold=tau)

    det_diag = WireSpeedEdgeDetector(
        frozen_encoder=server.global_model.encoder,
        global_profiles=diag_profiles,
        unknown_label=num_known,
        num_blocks=num_blocks,
        block_dim=block_dim,
    )
    preds, _, _, scores = det_diag.predict_batch(test_open_ds.features)
    results_a5["Diagonal Mahalanobis (Variance Only)"] = compute_osr_metrics(
        y_true=test_open_ds.labels.numpy(),
        y_pred=preds,
        anomaly_scores=scores,
        num_known_classes=num_known,
        unknown_label=num_known,
    )

    for name, m in results_a5.items():
        logger.info(
            f"  [{name}] AUROC: {m['auroc']*100:.2f}% | "
            f"Unk-F1: {m['unknown_f1']*100:.2f}% | "
            f"OS Macro-F1: {m['open_set_macro_f1']*100:.2f}% | "
            f"KFR: {m['kfr']*100:.2f}%"
        )

    full_results = {
        "A4_moment_aggregation": results_a4,
        "A5_distance_metrics": results_a5,
    }
    save_json(full_results, out_p / "covariance_metric_ablations.json")

    # Generate LaTeX tables
    latex_tbl_a4 = generate_ablation_latex_table(
        results_a4,
        caption="Ablation A4: Covariance moment aggregation under Law of Total Variance (Theorem 1) vs baselines.",
        label="tab:ablation_a4_covariance",
    )
    with open(out_p / "table_ablation_a4_covariance.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl_a4)

    latex_tbl_a5 = generate_ablation_latex_table(
        results_a5,
        caption="Ablation A5: Distance metric and block-diagonal subspace decomposition.",
        label="tab:ablation_a5_metric",
    )
    with open(out_p / "table_ablation_a5_metric.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl_a5)

    logger.info(f"Ablation studies A4 & A5 completed. Tables saved to {out_p}")
    return full_results


def generate_ablation_latex_table(results: Dict[str, Dict[str, float]], caption: str, label: str) -> str:
    """Format ablation results into clean LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & AUROC (\%) & AUPR (\%) & Unknown-F1 (\%) & OS Macro-F1 (\%) & KFR (\%) \\",
        r"\midrule",
    ]

    for name, v in results.items():
        is_prop = "Proposed" in name
        name_str = rf"\textbf{{{name}}}" if is_prop else name
        auroc = f"{v['auroc']*100:.2f}"
        aupr = f"{v['aupr']*100:.2f}"
        unk_f1 = f"{v['unknown_f1']*100:.2f}"
        os_f1 = f"{v['open_set_macro_f1']*100:.2f}"
        kfr = f"{v['kfr']*100:.2f}"
        lines.append(f"{name_str} & {auroc} & {aupr} & {unk_f1} & {os_f1} & {kfr} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Covariance and Metric Ablations (A4 & A5)")
    parser.add_argument("--dataset", type=str, default="b_nat")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/ablations_cov_metric")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_covariance_distance_ablations(
        dataset_name=args.dataset,
        rounds=args.rounds,
        beta_dirichlet=args.beta,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
