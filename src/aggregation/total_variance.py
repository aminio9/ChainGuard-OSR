"""Exact Pooled Covariance Aggregation via Law of Total Variance (Theorem 1).

Reconstructs the exact centralized block-diagonal covariance matrices without transmitting
any raw feature vectors, mathematically preserving between-client dispersion under label skew.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional
import torch


class ClientClassMoment(NamedTuple):
    """Container for client-submitted class moments from eligible nodes k in K_c^{prof} (N_{k,c} >= N_min)."""

    client_id: int
    sample_count: int
    mu: torch.Tensor  # Shape: (d_z,)
    block_covs: List[torch.Tensor]  # B tensors of shape (d_b, d_b)
    cal_quantile: Optional[float] = None  # Local (1 - alpha) calibration quantile
    class_id: Optional[int] = None  # Known class index c in Y_K


class GlobalClassProfile(NamedTuple):
    """Container for aggregated global class profile."""

    class_id: int
    total_samples: int
    mu: torch.Tensor  # Global centroid of shape (d_z,)
    block_covs: List[torch.Tensor]  # B block-diagonal covariance tensors of shape (d_b, d_b)
    inv_block_covs: List[torch.Tensor]  # Precomputed inverted block matrices
    conformal_threshold: float = 0.0  # Mondrian conformal threshold tau_{alpha, c}


def build_global_profile_from_moments(
    class_id: int,
    total_samples: int,
    mu: torch.Tensor,
    block_covs: List[torch.Tensor],
    tikhonov_eps: float = 1e-5,
    conformal_threshold: float = 0.0,
) -> GlobalClassProfile:
    """Construct a GlobalClassProfile with symmetrized and regularized inverse block covariances."""
    device = mu.device
    dtype = torch.float32
    reg_covs: List[torch.Tensor] = []
    inv_covs: List[torch.Tensor] = []

    for cov_b in block_covs:
        d_b = cov_b.size(0)
        reg_eye = tikhonov_eps * torch.eye(d_b, dtype=dtype, device=device)
        sigma_b = cov_b.to(dtype=dtype, device=device) + reg_eye
        sigma_b = 0.5 * (sigma_b + sigma_b.t())
        try:
            inv_b = torch.linalg.inv(sigma_b)
        except RuntimeError:
            inv_b = torch.linalg.pinv(sigma_b)
        reg_covs.append(sigma_b)
        inv_covs.append(inv_b)

    return GlobalClassProfile(
        class_id=class_id,
        total_samples=total_samples,
        mu=mu.to(dtype=dtype, device=device),
        block_covs=reg_covs,
        inv_block_covs=inv_covs,
        conformal_threshold=conformal_threshold,
    )


def aggregate_total_variance_moments(
    class_submissions: Dict[int, List[ClientClassMoment]],
    num_blocks: int = 4,
    block_dim: int = 16,
    tikhonov_eps: float = 1e-5,
    apply_between_client_dispersion: bool = True,
    min_cov_samples: int = 3,
) -> Dict[int, GlobalClassProfile]:
    """Aggregate local moments into global class profiles via the Law of Total Variance (Section 4.3, Prop. 1).

    Mathematical Formulation (LaTeX Section 4.3):
        1. Global Centroid over eligible profiling nodes K_c^{prof} = {k in M : N_{k,c} >= N_min} (Eq. eq:global_mean):
           mu_c = (1 / N_c^{prof}) * sum_{k in K_c^{prof}} N_{k,c} * mu_{k,c},   where N_c^{prof} = sum_{k in K_c^{prof}} N_{k,c}

        2. Global Pooled Block-Covariance over the same eligible nodes K_c^{prof} = {k in M : N_{k,c} >= N_min}
           (Eq. eq:global_block_cov):
           Sigma_{c, G_b} = (1 / N_c^{prof}) * sum_{k in K_c^{prof}} N_{k,c} * [
               Sigma_{k, c, G_b} + (mu_{k,c}[G_b] - mu_c[G_b])(mu_{k,c}[G_b] - mu_c[G_b])^T
           ] + eps_v * I_{d_b}
           The outer-product term captures between-node dispersion of local sample means mu_{k,c} around mu_c
           (arising from finite local sample counts N_{k,c} under Dirichlet skew as well as any node-level
           traffic variation), ensuring exact centralized pooled covariance recovery while keeping class-conditional
           calibration samples exchangeable when pooled across K_{cal, c}.

    Args:
        class_submissions: Dict mapping class_id -> list of ClientClassMoment from eligible nodes K_c^{prof}.
        num_blocks: Number of diagonal sub-blocks B (default: 4).
        block_dim: Dimensionality of each sub-block d_b = d_z / B.
        tikhonov_eps: Tikhonov regularization parameter eps_v > 0 (default: 1e-5).
        apply_between_client_dispersion: If True (default), includes the Law of Total
            Variance outer-product dispersion term. If False, computes naive within-client
            average covariance (Ablation).
        min_cov_samples: Minimum local sample count N_min >= 3 required for a node to
            contribute to profiling (K_c^{prof}).

    Returns:
        Dict mapping class_id -> GlobalClassProfile.
    """
    profiles: Dict[int, GlobalClassProfile] = {}

    for c, subs in class_submissions.items():
        active_subs = [s for s in subs if s.sample_count > 0]
        if not active_subs:
            continue

        # Eligible profiling nodes K_c^{prof} = {k in M : N_{k,c} >= N_min}
        prof_subs = [s for s in active_subs if s.sample_count >= min_cov_samples]
        if not prof_subs:
            prof_subs = active_subs

        actual_latent = prof_subs[0].mu.size(0)
        curr_b = num_blocks
        curr_db = block_dim
        if curr_b * curr_db != actual_latent:
            curr_b = min(curr_b, actual_latent)
            curr_db = actual_latent // curr_b

        device = prof_subs[0].mu.device
        dtype = torch.float32

        # Step 1: Pooled Sample Count N_c^{prof} and Centroid mu_c over K_c^{prof} (Eq. eq:global_mean)
        total_samples = sum(s.sample_count for s in prof_subs)
        n_c_prof = float(max(total_samples, 1))

        weighted_mu_sum = torch.zeros(actual_latent, dtype=dtype, device=device)
        for s in prof_subs:
            weighted_mu_sum += float(s.sample_count) * s.mu.to(dtype=dtype, device=device)

        global_mu = weighted_mu_sum / n_c_prof  # (d_z,)

        # Step 2: Global Block-Covariance over the exact same set K_c^{prof} (Eq. eq:global_block_cov)
        cov_subs = [
            s for s in prof_subs
            if s.block_covs is not None and len(s.block_covs) >= curr_b
        ]
        n_c_cov = float(max(sum(s.sample_count for s in cov_subs), 1))

        reg_eye = tikhonov_eps * torch.eye(curr_db, dtype=dtype, device=device)
        global_block_covs: List[torch.Tensor] = []
        inv_block_covs: List[torch.Tensor] = []

        for b in range(curr_b):
            start_idx = b * curr_db
            end_idx = start_idx + curr_db
            mu_global_b = global_mu[start_idx:end_idx]

            if cov_subs:
                pooled_b = torch.zeros((curr_db, curr_db), dtype=dtype, device=device)
                for s in cov_subs:
                    n_k = float(s.sample_count)
                    local_block_cov = s.block_covs[b].to(dtype=dtype, device=device)
                    local_mu_b = s.mu[start_idx:end_idx].to(dtype=dtype, device=device)

                    if apply_between_client_dispersion:
                        # Between-client dispersion term: (mu_{k,c}[G_b] - mu_c[G_b])(mu_{k,c}[G_b] - mu_c[G_b])^T
                        mean_diff = (local_mu_b - mu_global_b).unsqueeze(1)  # (d_b, 1)
                        dispersion_b = torch.matmul(mean_diff, mean_diff.t())  # (d_b, d_b)
                        pooled_b += n_k * (local_block_cov + dispersion_b)
                    else:
                        # Naive within-client mean covariance (omits between-client dispersion)
                        pooled_b += n_k * local_block_cov

                # Normalize by N_c^{cov} and add Tikhonov stabilizer eps_v * I_{d_b}
                sigma_c_b = (pooled_b / n_c_cov) + reg_eye
            else:
                sigma_c_b = torch.eye(curr_db, dtype=dtype, device=device) + reg_eye

            # Symmetrize numerically
            sigma_c_b = 0.5 * (sigma_c_b + sigma_c_b.t())

            # Invert for wire-speed Mahalanobis inference
            try:
                inv_sigma_c_b = torch.linalg.inv(sigma_c_b)
            except RuntimeError:
                inv_sigma_c_b = torch.linalg.pinv(sigma_c_b)

            global_block_covs.append(sigma_c_b)
            inv_block_covs.append(inv_sigma_c_b)

        # Step 3: Legacy Quantile Fallback (overridden by Stage 3 Pooled Empirical CDF Calibration)
        weighted_q_sum = 0.0
        cal_sample_total = 0
        for s in active_subs:
            if s.cal_quantile is not None and math.isfinite(float(s.cal_quantile)):
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

    return profiles
