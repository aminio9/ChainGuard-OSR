"""Exact Pooled Covariance Aggregation via Law of Total Variance (Theorem 1, Equations 17-18).

Reconstructs the exact centralized block-diagonal covariance matrices without transmitting
any raw feature vectors, mathematically preserving between-client dispersion under label skew.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple
import torch
from src.aggregation.byzantine_mad import ClientClassMoment, filter_byzantine_moments


class GlobalClassProfile(NamedTuple):
    """Container for aggregated global class profile."""

    class_id: int
    total_samples: int
    mu: torch.Tensor  # Global centroid of shape (d_z,)
    block_covs: List[torch.Tensor]  # B block-diagonal covariance tensors of shape (d_b, d_b)
    inv_block_covs: List[torch.Tensor]  # Precomputed inverted block matrices
    conformal_threshold: float = 0.0  # Mondrian conformal threshold tau_{alpha, c}


def aggregate_total_variance_moments(
    class_submissions: Dict[int, List[ClientClassMoment]],
    num_blocks: int = 4,
    block_dim: int = 16,
    tikhonov_eps: float = 1e-5,
    apply_mad_filter: bool = True,
    mad_gamma_0: float = 3.0,
    mad_zeta: float = 1.5,
) -> Tuple[Dict[int, GlobalClassProfile], Dict[int, List[int]]]:
    """Aggregate local moments into exact global class profiles via Law of Total Variance (Theorem 1).

    Mathematical Formulation (LaTeX Eq. 157-167):
        1. Clean Set (Byzantine Filter, Eq. 151-155):
           K_c^{clean} = FilterMAD(K_c)

        2. Global Centroid (Eq. 159):
           mu_c = (1 / N_c) * sum_{k in K_c^{clean}} n_c^{(k)} * mu_c^{(k)}
           where N_c = sum_{k in K_c^{clean}} n_c^{(k)}

        3. Global Pooled Block-Covariance (Eq. 164 / Theorem 1):
           Sigma_{c, G_b} = (1 / N_c) * sum_{k in K_c^{clean}} n_c^{(k)} * [
               Sigma_{c, G_b}^{(k)} + (mu_c^{(k)}[G_b] - mu_c[G_b])(mu_c^{(k)}[G_b] - mu_c[G_b])^T
           ] + eps_v * I_{d_b}

    Args:
        class_submissions: Dict mapping class_id -> list of ClientClassMoment.
        num_blocks: Number of diagonal sub-blocks B (default: 4 or 8).
        block_dim: Dimensionality of each sub-block d_b = d_z / B.
        tikhonov_eps: Tikhonov regularization parameter eps_v = 1e-5 (default: 1e-5).
        apply_mad_filter: Whether to run Byzantine MAD outlier filter.
        mad_gamma_0: Base multiplier for MAD threshold.
        mad_zeta: Finite-sample adaptation coefficient for MAD.

    Returns:
        (global_profiles, rejected_clients_per_class)
    """
    profiles: Dict[int, GlobalClassProfile] = {}
    rejections: Dict[int, List[int]] = {}

    for c, subs in class_submissions.items():
        if not subs:
            continue

        actual_latent = subs[0].mu.size(0)
        curr_b = num_blocks
        curr_db = block_dim
        if curr_b * curr_db != actual_latent:
            curr_b = min(curr_b, actual_latent)
            curr_db = actual_latent // curr_b

        # Step 1: Byzantine-Robust Outlier Filter (Equations 14-16)
        if apply_mad_filter and len(subs) > 2:
            clean_subs, rejected_ids = filter_byzantine_moments(
                subs, gamma_0=mad_gamma_0, zeta=mad_zeta
            )
            rejections[c] = rejected_ids
        else:
            clean_subs = subs
            rejections[c] = []

        if not clean_subs:
            continue

        device = clean_subs[0].mu.device
        dtype = torch.float32

        # Step 2: Global Sample Count and Centroid (Equation 17 / LaTeX Eq. 159)
        total_samples = sum(s.sample_count for s in clean_subs)
        n_c = max(total_samples, 1)

        weighted_mu_sum = torch.zeros(actual_latent, dtype=dtype, device=device)
        for s in clean_subs:
            weighted_mu_sum += float(s.sample_count) * s.mu.to(dtype=dtype, device=device)

        global_mu = weighted_mu_sum / n_c  # (d_z,)

        # Step 3: Exact Pooled Block Covariance via Law of Total Variance (Equation 18 / LaTeX Eq. 164)
        reg_eye = tikhonov_eps * torch.eye(curr_db, dtype=dtype, device=device)
        global_block_covs: List[torch.Tensor] = []
        inv_block_covs: List[torch.Tensor] = []

        for b in range(curr_b):
            start_idx = b * curr_db
            end_idx = start_idx + curr_db
            mu_global_b = global_mu[start_idx:end_idx]

            pooled_b = torch.zeros((curr_db, curr_db), dtype=dtype, device=device)

            for s in clean_subs:
                n_k = float(s.sample_count)
                local_block_cov = s.block_covs[b].to(dtype=dtype, device=device)
                local_mu_b = s.mu[start_idx:end_idx].to(dtype=dtype, device=device)

                # Between-client dispersion term: (mu_k[G_b] - mu[G_b]) * (mu_k[G_b] - mu[G_b])^T
                mean_diff = (local_mu_b - mu_global_b).unsqueeze(1)  # (d_b, 1)
                dispersion_b = torch.matmul(mean_diff, mean_diff.t())  # (d_b, d_b)

                # Law of Total Variance inner term (Equation 18)
                pooled_b += n_k * (local_block_cov + dispersion_b)

            # Normalize by total sample count and add Tikhonov stabilizer
            sigma_c_b = (pooled_b / n_c) + reg_eye

            # Symmetrize numerically
            sigma_c_b = 0.5 * (sigma_c_b + sigma_c_b.t())

            # Invert for wire-speed Mahalanobis inference
            try:
                inv_sigma_c_b = torch.linalg.inv(sigma_c_b)
            except RuntimeError:
                inv_sigma_c_b = torch.linalg.pinv(sigma_c_b)

            global_block_covs.append(sigma_c_b)
            inv_block_covs.append(inv_sigma_c_b)

        # Step 4: Quantile Aggregation (Equation 23) if calibration quantiles provided
        weighted_q_sum = 0.0
        cal_sample_total = 0
        for s in clean_subs:
            if s.cal_quantile is not None:
                weighted_q_sum += float(s.sample_count) * float(s.cal_quantile)
                cal_sample_total += s.sample_count

        conformal_tau = (
            weighted_q_sum / max(cal_sample_total, 1) if cal_sample_total > 0 else 0.0
        )

        profiles[c] = GlobalClassProfile(
            class_id=c,
            total_samples=total_samples,
            mu=global_mu,
            block_covs=global_block_covs,
            inv_block_covs=inv_block_covs,
            conformal_threshold=conformal_tau,
        )

    return profiles, rejections
