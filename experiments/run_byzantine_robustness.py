"""Experiment E9: Byzantine Poisoning Robustness and Moment Safety.

Aligned with LaTeX Section 4.2 (Equations 14-16 & Theorem 1).
Evaluates:
  - Resistance against malicious Byzantine clients injecting corrupted moments (f in {0.10, 0.20, 0.30}).
  - Comparison: Sample-Weighted MAD Filter vs. Unfiltered Simple Averaging.
  - Metrics: Moment Frobenius error (||Delta mu||_F, ||Delta Sigma||_F), AUROC, Unknown-F1, KFR.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, List, Optional, Tuple
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
from src.aggregation.byzantine_mad import ClientClassMoment, filter_byzantine_moments
from src.aggregation.total_variance import aggregate_total_variance_moments, GlobalClassProfile
from src.osr.block_mahalanobis import WireSpeedEdgeDetector, compute_block_mahalanobis_distances
from src.osr.conformal import compute_finite_sample_quantile
from src.metrics.osr_metrics import compute_osr_metrics


def run_byzantine_robustness_experiment(
    dataset_name: str = "b_nat",
    num_clients: int = 20,
    rounds: int = 10,
    poison_fractions: Optional[List[float]] = None,
    perturbation_scale: float = 50.0,
    max_samples: Optional[int] = None,
    output_dir: str = "results/byzantine",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate Byzantine robustness under moment poisoning attacks across malicious fractions f."""
    logger = setup_logger(name="byzantine_runner")
    if poison_fractions is None:
        poison_fractions = [0.0, 0.10, 0.20, 0.30]

    logger.info(
        f"Starting Byzantine Poisoning Robustness Experiment (E9) on {dataset_name} "
        f"with K={num_clients}, f in {poison_fractions}, scale={perturbation_scale}..."
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

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
    config.federated.num_clients = num_clients
    config.federated.num_rounds = rounds
    config.data.known_classes = known_classes
    config.data.unknown_classes = unknown_classes
    config.model.in_features = train_ds.features.shape[1]
    device = get_device(config.experiment.device)

    # Partition across K clients
    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=num_clients,
        beta_dirichlet=0.5,
        min_samples_per_client=max(2, min(10, len(train_ds) // (2 * num_clients))),
        seed=seed,
    )

    clients = []
    for client_id in range(num_clients):
        sub_indices = client_indices[client_id]
        idx_t = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        clients.append(FederatedClient(client_id, sub_dataset, config, device, cal_dataset=cal_ds))

    server = FederatedServer(config, clients, device)

    # Phase I: Train clean representation
    logger.info("Executing Phase I collaborative federated training...")
    for r in range(1, rounds + 1):
        server.run_round(r)
    server.freeze_global_encoder()

    # Collect clean honest moments from clients
    logger.info("Extracting honest client moments for Phase II...")
    num_blocks = config.federated.num_blocks
    block_dim = config.federated.block_dim
    alpha = config.osr.conformal_alpha
    num_known = len(known_classes)

    honest_submissions_per_class: Dict[int, List[ClientClassMoment]] = {c: [] for c in range(num_known)}
    for client in clients:
        m_list = client.extract_phase_two_moments(
            frozen_encoder=server.global_model.encoder,
            num_blocks=num_blocks,
            block_dim=block_dim,
            alpha=alpha,
        )
        for idx, c in enumerate(client.local_classes):
            c = int(c)
            if idx < len(m_list) and c in honest_submissions_per_class:
                honest_submissions_per_class[c].append(m_list[idx])

    # Compute clean ground-truth global profiles (reference)
    clean_profiles, _ = aggregate_total_variance_moments(
        class_submissions=honest_submissions_per_class,
        num_blocks=num_blocks,
        block_dim=block_dim,
        apply_mad_filter=False,
    )

    results: Dict[str, Any] = {}

    for f in poison_fractions:
        num_byzantine = int(round(f * num_clients))
        logger.info(f"\n---> Evaluating Malicious Fraction f = {f*100:.0f}% ({num_byzantine}/{num_clients} Byzantine nodes)")

        # Create poisoned submissions
        rng = np.random.default_rng(seed + int(f * 1000))
        byzantine_client_ids = set(rng.choice(num_clients, size=num_byzantine, replace=False))

        poisoned_submissions_per_class: Dict[int, List[ClientClassMoment]] = {c: [] for c in range(num_known)}

        for c in range(num_known):
            for mom in honest_submissions_per_class[c]:
                if mom.client_id in byzantine_client_ids:
                    # Byzantine client poisons centroid with extreme shift and explodes variance
                    noise_dir = torch.randn_like(mom.mu)
                    noise_dir = noise_dir / (torch.linalg.norm(noise_dir) + 1e-8)
                    poisoned_mu = mom.mu + perturbation_scale * noise_dir
                    poisoned_covs = [cov * 100.0 for cov in mom.block_covs]
                    poisoned_mom = ClientClassMoment(
                        client_id=mom.client_id,
                        sample_count=mom.sample_count,
                        mu=poisoned_mu,
                        block_covs=poisoned_covs,
                        cal_quantile=mom.cal_quantile * 10.0 if mom.cal_quantile else None,
                    )
                    poisoned_submissions_per_class[c].append(poisoned_mom)
                else:
                    poisoned_submissions_per_class[c].append(mom)

        # Regimes to test:
        # a) Unfiltered Simple Aggregation (Baseline)
        # b) Sample-Weighted MAD Filter (Proposed)
        for filter_mode in ["unfiltered", "mad_filtered"]:
            mode_name = "Proposed (MAD Filtered)" if filter_mode == "mad_filtered" else "Unfiltered Baseline"
            use_mad = (filter_mode == "mad_filtered")

            agg_profiles, _ = aggregate_total_variance_moments(
                class_submissions=poisoned_submissions_per_class,
                num_blocks=num_blocks,
                block_dim=block_dim,
                apply_mad_filter=use_mad,
                mad_gamma_0=config.federated.mad_gamma_0,
                mad_zeta=config.federated.mad_zeta,
            )

            mu_errors: List[float] = []
            cov_errors: List[float] = []

            for c in range(num_known):
                prof = agg_profiles[c]
                ref_prof = clean_profiles[c]
                mu_err = float(torch.linalg.norm(prof.mu - ref_prof.mu).item())
                cov_err = sum(
                    float(torch.linalg.norm(prof.block_covs[b] - ref_prof.block_covs[b]).item())
                    for b in range(num_blocks)
                )
                mu_errors.append(mu_err)
                cov_errors.append(cov_err)

            # Calibrate conformal thresholds using calibration dataset
            cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=128, shuffle=False)
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

            calibrated_profiles: Dict[int, GlobalClassProfile] = {}
            for c in range(num_known):
                prof = agg_profiles[c]
                c_idx = (cal_y == c)
                if c_idx.sum() > 0:
                    z_c = cal_z[c_idx]
                    dists = compute_block_mahalanobis_distances(z_c, {c: prof}, num_blocks=num_blocks, block_dim=block_dim)[c]
                    scores = dists.numpy().tolist()
                    tau = compute_finite_sample_quantile(scores, alpha=alpha)
                else:
                    tau = 1.0
                calibrated_profiles[c] = prof._replace(conformal_threshold=tau)

            # Build edge detector with calibrated profiles
            detector = WireSpeedEdgeDetector(
                frozen_encoder=server.global_model.encoder,
                global_profiles=calibrated_profiles,
                unknown_label=num_known,
                num_blocks=num_blocks,
                block_dim=block_dim,
            )

            preds_open, _, _, anomaly_scores = detector.predict_batch(test_open_ds.features)
            y_open_true = test_open_ds.labels.numpy()
            osr_metrics = compute_osr_metrics(
                y_true=y_open_true,
                y_pred=preds_open,
                anomaly_scores=anomaly_scores,
                num_known_classes=num_known,
                unknown_label=num_known,
            )

            key = f"f_{int(f*100)}_{filter_mode}"
            results[key] = {
                "poison_fraction": f,
                "filter_mode": filter_mode,
                "label": mode_name,
                "mean_mu_error_frobenius": float(np.mean(mu_errors)),
                "mean_cov_error_frobenius": float(np.mean(cov_errors)),
                "auroc": float(osr_metrics["auroc"]),
                "aupr": float(osr_metrics["aupr"]),
                "unknown_f1": float(osr_metrics["unknown_f1"]),
                "kfr": float(osr_metrics["kfr"]),
                "open_set_macro_f1": float(osr_metrics["open_set_macro_f1"]),
            }

            logger.info(
                f"   [{mode_name}] Mu-Err: {np.mean(mu_errors):.3f} | "
                f"Cov-Err: {np.mean(cov_errors):.3f} | "
                f"AUROC: {osr_metrics['auroc']*100:.2f}% | "
                f"Unk-F1: {osr_metrics['unknown_f1']*100:.2f}% | "
                f"KFR: {osr_metrics['kfr']*100:.2f}%"
            )

    save_json(results, out_p / "byzantine_robustness_summary.json")

    # Generate LaTeX table
    latex_tbl = generate_byzantine_latex_table(results)
    with open(out_p / "table_byzantine_robustness.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Byzantine robustness experiment complete. Output written to {out_p}")
    return results


def generate_byzantine_latex_table(results: Dict[str, Any]) -> str:
    """Format Byzantine robustness findings into LaTeX tabular environment."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Byzantine moment poisoning robustness ($f \in \{10\%, 20\%, 30\%\}$) on B-NAT.}",
        r"\label{tab:byzantine_robustness}",
        r"\small",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"Poison ($f$) & Aggregation Strategy & $\|\Delta \boldsymbol{\mu}\|_F$ & $\|\Delta \boldsymbol{\Sigma}\|_F$ & AUROC (\%) & Unk-F1 (\%) & KFR (\%) \\",
        r"\midrule",
    ]

    fractions = sorted(list(set(v["poison_fraction"] for v in results.values())))
    for f in fractions:
        for mode in ["unfiltered", "mad_filtered"]:
            k = f"f_{int(f*100)}_{mode}"
            if k not in results:
                continue
            entry = results[k]
            f_str = f"{int(f*100)}\\%"
            strat = "Sample-Weighted MAD (Ours)" if mode == "mad_filtered" else "Standard Linear Average"
            if mode == "mad_filtered":
                strat = r"\textbf{" + strat + "}"
            mu_err = f"{entry['mean_mu_error_frobenius']:.2f}"
            cov_err = f"{entry['mean_cov_error_frobenius']:.2f}"
            auroc = f"{entry['auroc']*100:.2f}"
            unk_f1 = f"{entry['unknown_f1']*100:.2f}"
            kfr = f"{entry['kfr']*100:.2f}"
            lines.append(f"{f_str} & {strat} & {mu_err} & {cov_err} & {auroc} & {unk_f1} & {kfr} \\\\")
        lines.append(r"\midrule")

    # Remove the trailing \midrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Byzantine Robustness Experiment (E9)")
    parser.add_argument("--dataset", type=str, default="b_nat")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--fractions", type=str, default="0.0,0.10,0.20,0.30")
    parser.add_argument("--scale", type=float, default=50.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/byzantine")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    f_list = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    run_byzantine_robustness_experiment(
        dataset_name=args.dataset,
        num_clients=args.clients,
        rounds=args.rounds,
        poison_fractions=f_list,
        perturbation_scale=args.scale,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
