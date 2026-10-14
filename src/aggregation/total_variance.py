"""Exact Pooled Covariance Aggregation via Law of Total Variance (Theorem 1).

Reconstructs the exact centralized block-diagonal covariance matrices without transmitting
any raw feature vectors, mathematically preserving between-client dispersion under label skew.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import torch


class ClientClassMoment(NamedTuple):
    """Container for client-submitted class moments from eligible nodes k in K_c^{prof} (N_{k,c} >= N_min)."""

    client_id: int
    sample_count: int
    mu: torch.Tensor  # Shape: (d_z,)
    block_covs: List[torch.Tensor]  # B tensors of shape (d_b, d_b)
    cal_quantile: Optional[float] = None  # Local (1 - alpha) calibration quantile
    class_id: Optional[int] = None  # Known class index c in Y_K
    local_prototypes: Optional[torch.Tensor] = None  # Local multi-prototype centroids of shape (M_c, d_z)
    local_proto_counts: Optional[List[int]] = None  # Sample counts per local prototype cluster


class GlobalClassProfile(NamedTuple):
    """Container for aggregated global class profile."""

    class_id: int
    total_samples: int
    mu: torch.Tensor  # Global centroid of shape (d_z,)
    block_covs: List[torch.Tensor]  # B block-diagonal covariance tensors of shape (d_b, d_b)
    inv_block_covs: List[torch.Tensor]  # Precomputed inverted block matrices
    conformal_threshold: float = 0.0  # Mondrian conformal threshold tau_{alpha, c}
    prototypes: Optional[torch.Tensor] = None  # Optional multi-prototypes tensor of shape (M_c, d_z)
    # Per-prototype per-block precision matrices: list of M_c lists, each containing B (d_b, d_b) tensors.
    # proto_inv_block_covs[m][b] = (Sigma_{c,m,G_b}^*)^{-1}, covariance around the *normalized* prototype mu_{c,m}.
    # None when single-centroid (legacy path).
    proto_inv_block_covs: Optional[List[List[torch.Tensor]]] = None


def compute_server_side_oas_shrinkage(
    cov_sym: torch.Tensor,
    total_samples: int,
    tikhonov_eps: float = 1e-4,
) -> Tuple[torch.Tensor, float]:
    """Oracle Approximating Shrinkage (OAS, Chen et al. 2010) on pooled covariance.
    
    Computes optimal shrinkage intensity in closed form from Tr(Sigma) and Tr(Sigma^2),
    preserving federated privacy without requiring client raw data or 4th-order moments.
    """
    d_b = cov_sym.size(0)
    device = cov_sym.device
    dtype = cov_sym.dtype

    tr_s = torch.trace(cov_sym)
    tr_s2 = torch.trace(torch.matmul(cov_sym, cov_sym))
    mu = tr_s / float(d_b)

    num = (1.0 - 2.0 / float(d_b)) * tr_s2 + (tr_s ** 2)
    den = (float(total_samples) + 1.0 - 2.0 / float(d_b)) * (tr_s2 - (tr_s ** 2) / float(d_b))

    if den.item() > 0:
        rho = float(torch.clamp(num / den, min=0.0, max=1.0).item())
    else:
        rho = 1.0

    sigma_shrunk = (1.0 - rho) * cov_sym + (rho * mu + tikhonov_eps) * torch.eye(d_b, dtype=dtype).to(device)
    return sigma_shrunk, rho


def build_global_profile_from_moments(
    class_id: int,
    total_samples: int,
    mu: torch.Tensor,
    block_covs: List[torch.Tensor],
    tikhonov_eps: float = 1e-4,
    conformal_threshold: float = 0.0,
    use_oas: bool = True,
    prototypes: Optional[torch.Tensor] = None,
) -> GlobalClassProfile:
    """Construct a GlobalClassProfile with symmetrized and regularized inverse block covariances."""
    device = mu.device
    dtype = torch.float32
    reg_covs: List[torch.Tensor] = []
    inv_covs: List[torch.Tensor] = []

    for cov_b in block_covs:
        d_b = cov_b.size(0)
        cov_dev = cov_b.to(dtype=dtype, device=device)
        cov_sym = 0.5 * (cov_dev + cov_dev.t())
        
        if use_oas:
            sigma_b, _ = compute_server_side_oas_shrinkage(cov_sym, total_samples, tikhonov_eps=tikhonov_eps)
        else:
            tr_b = torch.trace(cov_sym) / float(d_b)
            rho = max(1e-4, min(0.2, float(d_b) / max(float(total_samples), 1.0)))
            sigma_b = (1.0 - rho) * cov_sym + (rho * tr_b + tikhonov_eps) * torch.eye(d_b, dtype=dtype).to(device)

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
        prototypes=prototypes.to(dtype=dtype, device=device) if prototypes is not None else None,
    )


def aggregate_total_variance_moments(
    class_submissions: Dict[int, List[ClientClassMoment]],
    num_blocks: int = 4,
    block_dim: int = 16,
    tikhonov_eps: float = 1e-3,
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

        reg_eye = tikhonov_eps * torch.eye(curr_db, dtype=dtype).to(device)
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

                # Normalize by N_c^{cov} and apply Server-Side OAS Shrinkage (if regularized)
                raw_cov = pooled_b / n_c_cov
                raw_cov = 0.5 * (raw_cov + raw_cov.t())
                if tikhonov_eps > 0.0:
                    sigma_c_b, _ = compute_server_side_oas_shrinkage(raw_cov, int(n_c_cov), tikhonov_eps=tikhonov_eps)
                else:
                    sigma_c_b = raw_cov
            else:
                sigma_c_b = torch.eye(curr_db, dtype=dtype).to(device) + reg_eye

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


def extract_spherical_class_prototypes(
    features: torch.Tensor,
    num_prototypes: int,
    random_state: int = 42,
) -> torch.Tensor:
    """Extract M_c spherical prototype centroids from class features on S^{d-1} (MPHIL 2026 / AEGIS-Net 2026).

    Args:
        features: (N, d) feature tensor.
        num_prototypes: Target number of prototypes M_c.
        random_state: Random seed for deterministic clustering.

    Returns:
        (M_c, d) Tensor of L2-normalized prototype centroids.
    """
    device = features.device
    dtype = features.dtype
    n_samples = features.size(0)

    # Normalize features onto unit sphere
    norms = torch.norm(features, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    feat_norm = (features / norms).cpu().numpy()

    k_actual = min(num_prototypes, n_samples)
    if k_actual <= 1:
        mean_v = features.mean(dim=0, keepdim=True)
        mean_v = mean_v / torch.norm(mean_v, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        return mean_v.to(dtype=dtype, device=device)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k_actual, random_state=random_state, n_init=5)
    km.fit(feat_norm)
    centers = torch.from_numpy(km.cluster_centers_).to(dtype=dtype, device=device)
    centers_norm = centers / torch.norm(centers, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    return centers_norm


def extract_spherical_class_prototypes_with_covs(
    features: torch.Tensor,
    num_prototypes: int,
    num_blocks: int,
    block_dim: int,
    tikhonov_eps: float = 1e-4,
    random_state: int = 42,
    pooled_covariance: bool = True,
    normalize_features: bool = False,
) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """Extract M_c spherical or Euclidean prototypes AND per-prototype per-block OAS-regularized inverse covariances.

    Args:
        features: (N, d_z) feature tensor.
        num_prototypes: Target number of prototypes M_c.
        num_blocks: Number of block-diagonal sub-blocks B.
        block_dim: Dimension of each sub-block d_b = d_z / B.
        tikhonov_eps: Tikhonov ridge applied after OAS shrinkage for numerical safety.
        random_state: RNG seed for reproducibility.
        pooled_covariance: If True (default), pools within-prototype scatter across all class samples.
        normalize_features: If True, projects to S^{d-1}. If False (default), preserves anisotropic R^{d_z}.

    Returns:
        proto_centroids: (M_c, d_z) prototype centroids.
        proto_inv_block_covs: list of M_c lists of inverse block covariances.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    device = features.device
    dtype = torch.float32
    n_samples = features.size(0)

    # --- Step 1: Feature preparation (Spherical or Euclidean) ---
    f32 = features.to(dtype=dtype)
    if normalize_features:
        norms = torch.norm(f32, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        feat_norm = f32 / norms  # (N, d_z), each row on unit sphere
    else:
        feat_norm = f32
    feat_np = feat_norm.cpu().numpy()

    k_actual = min(num_prototypes, n_samples)

    # --- Step 2: Cluster features ---
    if k_actual <= 1:
        mean_v = feat_norm.mean(dim=0)
        if normalize_features:
            mean_v = mean_v / torch.norm(mean_v, p=2).clamp(min=1e-8)
        proto_centroids = mean_v.unsqueeze(0)  # (1, d_z)
        labels_np = np.zeros(n_samples, dtype=np.int32)
    else:
        km = KMeans(n_clusters=k_actual, random_state=random_state, n_init=5)
        km.fit(feat_np)
        labels_np = km.labels_  # (N,)
        centers_raw = torch.from_numpy(km.cluster_centers_).to(dtype=dtype, device=device)
        if normalize_features:
            centers_norm = centers_raw / torch.norm(centers_raw, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            proto_centroids = centers_norm  # (M_c, d_z)
        else:
            proto_centroids = centers_raw

    M_c = proto_centroids.size(0)
    d_z = feat_norm.size(1)

    # Reconcile block geometry
    curr_B = num_blocks
    curr_db = block_dim
    if curr_B * curr_db != d_z:
        curr_B = min(curr_B, d_z)
        curr_db = d_z // curr_B

    labels_t = torch.from_numpy(labels_np.astype(np.int64)).to(device)

    proto_inv_block_covs: List[List[torch.Tensor]] = []

    if pooled_covariance:
        # Pooled within-prototype covariance: pool deviations around nearest prototype
        # across all N_c class samples. Guarantees uniform metric manifold across prototypes.
        xc_within = torch.zeros_like(feat_norm)
        for m in range(M_c):
            cluster_mask = (labels_t == m)
            if int(cluster_mask.sum().item()) > 0:
                xc_within[cluster_mask] = feat_norm[cluster_mask] - proto_centroids[m]

        pooled_inv_blocks: List[torch.Tensor] = []
        for b in range(curr_B):
            start = b * curr_db
            end = start + curr_db
            xc_b = xc_within[:, start:end]
            if n_samples > 1:
                raw_cov = torch.matmul(xc_b.t(), xc_b) / float(n_samples)
            else:
                raw_cov = torch.eye(curr_db, dtype=dtype).to(device)
            raw_cov = 0.5 * (raw_cov + raw_cov.t())
            sigma_b, _ = compute_server_side_oas_shrinkage(raw_cov, n_samples, tikhonov_eps=tikhonov_eps)
            try:
                inv_b = torch.linalg.inv(sigma_b)
            except RuntimeError:
                inv_b = torch.linalg.pinv(sigma_b)
            pooled_inv_blocks.append(inv_b)

        proto_inv_block_covs = [pooled_inv_blocks for _ in range(M_c)]
    else:
        for m in range(M_c):
            cluster_mask = (labels_t == m)
            z_m = feat_norm[cluster_mask]       # (N_m, d_z) — already on sphere
            N_m = z_m.size(0)
            mu_m = proto_centroids[m]           # (d_z,) — normalized prototype

            inv_blocks_m: List[torch.Tensor] = []
            for b in range(curr_B):
                start = b * curr_db
                end = start + curr_db
                z_m_b = z_m[:, start:end]       # (N_m, d_b)
                mu_m_b = mu_m[start:end]        # (d_b,)

                diff_b = z_m_b - mu_m_b.unsqueeze(0)   # (N_m, d_b)
                if N_m > 1:
                    raw_cov = torch.matmul(diff_b.t(), diff_b) / float(N_m)
                else:
                    raw_cov = torch.eye(curr_db, dtype=dtype).to(device)

                # Symmetrize and apply OAS
                raw_cov = 0.5 * (raw_cov + raw_cov.t())
                sigma_b, _ = compute_server_side_oas_shrinkage(raw_cov, N_m, tikhonov_eps=tikhonov_eps)

                try:
                    inv_b = torch.linalg.inv(sigma_b)
                except RuntimeError:
                    inv_b = torch.linalg.pinv(sigma_b)

                inv_blocks_m.append(inv_b)

            proto_inv_block_covs.append(inv_blocks_m)

    return proto_centroids, proto_inv_block_covs


def federated_align_and_average_prototypes(
    local_prototypes_list: List[Tuple[torch.Tensor, List[int]]],
    target_num_prototypes: int,
    normalize_features: bool = False,
    fallback_mean: Optional[torch.Tensor] = None,
    random_state: int = 42,
) -> torch.Tensor:
    r"""Align and aggregate local client prototype centroids using Hungarian bipartite matching.

    Problem Addressed (Permutation Ambiguity / Label Switching):
        Independent local k-means on non-IID client nodes generates arbitrary cluster index
        permutations. For benign blockchain traffic with M_1 = 4 behavioral modes (handshake,
        peer discovery, transaction propagation, block announcement), client k may assign mode 1
        to index 0 while client k' assigns it to index 2. Naive slot-wise averaging blends distinct
        operational regimes, corrupts global prototypes \bar{\mu}_{1,m}, and artificially inflates
        the between-node dispersion term (\mu_{k,1,m} - \bar{\mu}_{1,m}) in the Law of Total Variance.

    Algorithmic Resolution (Hungarian Algorithm / Linear Bipartite Matching):
        1. Establish reference template R in R^{M_1 x d_z} across local candidate centers.
        2. Construct pairwise Euclidean distance cost matrix C_{i, j} = ||R_i - \mu_{k, 1, j}||_2^2.
        3. Solve optimal one-to-one assignment \pi_k = \arg\min_\pi \sum_{m=1}^{M_1} ||R_m - \mu_{k, 1, \pi(m)}||_2^2
           using the Hungarian algorithm (scipy.optimize.linear_sum_assignment) with complexity
           O(M_1^3) = 4^3 = 64 operations (sub-millisecond execution on the coordinator).
        4. Reconstruct sample-weighted global prototypes:
               \bar{\mu}_{1, m} = \frac{\sum_{k \in \mathcal{K}_1} N_{k, 1, \pi_k(m)} \mu_{k, 1, \pi_k(m)}}{\sum_{k \in \mathcal{K}_1} N_{k, 1, \pi_k(m)}}

    Two-Subscript Law & Shared Covariance:
        All M_1 = 4 prototypes share a single pooled block-diagonal precision matrix
        \tilde{\Sigma}_{1,b}^{-1} aggregated via the Law of Total Variance from intra-cluster
        deviations. Notation is strictly bounded to two subscripts (\bar{\mu}_{1,m} and
        \tilde{\Sigma}_{1,b}^{-1}), eliminating three-subscript matrix proliferation.

    Privacy Preservation (Zero Raw Data Leakage):
        Nodes transmit ONLY 4 local prototype centroid vectors and cluster scalar counts N_{k,1,m}.
        No raw feature embeddings z leave any client.

    Args:
        local_prototypes_list: List of (local_protos, local_counts) tuples from active clients.
            local_protos has shape (M_{k,c}, d_z); local_counts has length M_{k,c}.
        target_num_prototypes: Desired number of global prototypes M_c (e.g., M_1 = 4).
        normalize_features: If True, projects aggregated prototypes onto unit hypersphere S^{d-1}.
        fallback_mean: Global centroid mu_c to fall back to if no valid local prototypes exist.
        random_state: Seed for deterministic execution.

    Returns:
        (M_c, d_z) Tensor of aligned and aggregated global prototypes.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    valid_subs = [
        (p.cpu().to(torch.float32), (counts if counts is not None else [1] * p.size(0)))
        for p, counts in local_prototypes_list
        if p is not None and p.ndim == 2 and p.size(0) > 0
    ]

    if not valid_subs:
        if fallback_mean is not None:
            f_mean = fallback_mean.cpu().to(torch.float32)
            if normalize_features:
                f_mean = f_mean / torch.norm(f_mean, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            return f_mean.unsqueeze(0).repeat(target_num_prototypes, 1)
        return torch.zeros((target_num_prototypes, 64), dtype=torch.float32)

    d_z = valid_subs[0][0].size(1)

    if target_num_prototypes <= 1:
        total_w = 0.0
        sum_vec = torch.zeros(d_z, dtype=torch.float32)
        for p, counts in valid_subs:
            for i in range(p.size(0)):
                w = float(counts[i]) if i < len(counts) else 1.0
                sum_vec += w * p[i]
                total_w += w
        mean_p = sum_vec / max(total_w, 1e-8)
        if normalize_features:
            mean_p = mean_p / torch.norm(mean_p, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        return mean_p.unsqueeze(0)

    # Gather all candidate centers across clients
    all_centers = []
    all_weights = []
    for p, counts in valid_subs:
        for i in range(p.size(0)):
            w = float(counts[i]) if i < len(counts) else 1.0
            all_centers.append(p[i].numpy())
            all_weights.append(w)

    all_centers_np = np.asarray(all_centers, dtype=np.float32)
    all_weights_np = np.asarray(all_weights, dtype=np.float32)

    total_pool_size = len(all_centers_np)
    k_actual = min(target_num_prototypes, total_pool_size)

    # Establish reference anchor template R (k_actual, d_z)
    if k_actual > 1:
        km = KMeans(n_clusters=k_actual, random_state=random_state, n_init=5)
        km.fit(all_centers_np, sample_weight=all_weights_np)
        reference_R = torch.from_numpy(km.cluster_centers_).to(torch.float32)
    else:
        reference_R = torch.from_numpy(all_centers_np[0:1]).to(torch.float32)

    # Bipartite matching alignment against reference template
    slot_sums = torch.zeros((k_actual, d_z), dtype=torch.float32)
    slot_weights = torch.zeros(k_actual, dtype=torch.float32)

    ref_np = reference_R.numpy()

    for p, counts in valid_subs:
        p_np = p.numpy()
        n_p = len(p_np)
        # Compute cost matrix (k_actual x n_p)
        diff = ref_np[:, np.newaxis, :] - p_np[np.newaxis, :, :]
        cost_mat = np.sum(diff ** 2, axis=-1)  # (k_actual, n_p)

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        for r_slot, c_idx in zip(row_ind, col_ind):
            w = float(counts[c_idx]) if c_idx < len(counts) else 1.0
            slot_sums[r_slot] += w * p[c_idx]
            slot_weights[r_slot] += w

    # Calculate weighted average per slot
    aligned_prototypes = []
    for s in range(k_actual):
        if slot_weights[s].item() > 0:
            proto_s = slot_sums[s] / slot_weights[s]
        else:
            proto_s = reference_R[s]
        if normalize_features:
            proto_s = proto_s / torch.norm(proto_s, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        aligned_prototypes.append(proto_s)

    # Pad if k_actual < target_num_prototypes
    while len(aligned_prototypes) < target_num_prototypes:
        aligned_prototypes.append(aligned_prototypes[-1].clone())

    return torch.stack(aligned_prototypes[:target_num_prototypes], dim=0)

