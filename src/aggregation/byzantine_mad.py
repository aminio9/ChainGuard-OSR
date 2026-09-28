"""Sample-Weighted Median Absolute Deviation (MAD) Byzantine Filter (Equations 14-16).

Filters out malicious/poisoned clients injecting corrupt class moments in Phase II,
while gracefully accommodating natural finite-sample variance from small honest clients.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Tuple
import torch


class ClientClassMoment(NamedTuple):
    """Container for client-submitted class moments."""

    client_id: int
    sample_count: int
    mu: torch.Tensor  # Shape: (d_z,)
    block_covs: List[torch.Tensor]  # B tensors of shape (d_b, d_b)
    cal_quantile: Optional[float] = None  # Local (1 - alpha) calibration quantile


def filter_byzantine_moments(
    submissions: List[ClientClassMoment],
    gamma_0: float = 3.0,
    zeta: float = 1.5,
    mad_eps: float = 1e-6,
) -> Tuple[List[ClientClassMoment], List[int]]:
    """Filter Byzantine outliers via sample-size-aware MAD (Equations 14-16).

    Mathematical Formulation:
        mu_c^{med} = coordinate_median_{k in K_c} { mu_c^{(k)} }
        MAD_c = median_{k in K_c} { || mu_c^{(k)} - mu_c^{med} ||_2 }
        gamma_c(n_c^{(k)}) = gamma_0 * ( 1 + zeta / sqrt(n_c^{(k)}) )
        K_c^{clean} = { k in K_c | || mu_c^{(k)} - mu_c^{med} ||_2 <= gamma_c(n_c^{(k)}) * MAD_c }

    Args:
        submissions: List of ClientClassMoment objects for a single class c.
        gamma_0: Base multiplier for MAD distance threshold (default: 3.0).
        zeta: Finite sample-size adaptation coefficient (default: 1.5).
        mad_eps: Floor for MAD to avoid division by zero or trivial rejection (default: 1e-6).

    Returns:
        (clean_submissions, rejected_client_ids)
    """
    if len(submissions) <= 2:
        # With 1 or 2 clients, statistical median outlier detection is under-determined.
        return submissions, []

    # Stack means: shape (K, d_z)
    all_mus = torch.stack([s.mu for s in submissions], dim=0)

    # 1. Coordinate-wise median mu_c^{med} (Equation 14)
    med_mu = torch.median(all_mus, dim=0).values

    # 2. Euclidean distances to coordinate-wise median: shape (K,)
    diffs = all_mus - med_mu.unsqueeze(0)
    dists = torch.linalg.norm(diffs, ord=2, dim=1)  # (K,)

    # 3. Median Absolute Deviation: MAD_c (Equation 14)
    mad_c = float(torch.median(dists).item())
    effective_mad = max(mad_c, mad_eps)

    clean_submissions: List[ClientClassMoment] = []
    rejected_client_ids: List[int] = []

    # 4. Check adaptive threshold for each client (Equations 15-16)
    for idx, sub in enumerate(submissions):
        dist_k = float(dists[idx].item())
        n_k = max(sub.sample_count, 1)
        gamma_k = gamma_0 * (1.0 + zeta / math.sqrt(n_k))
        threshold_k = gamma_k * effective_mad

        if dist_k <= threshold_k:
            clean_submissions.append(sub)
        else:
            rejected_client_ids.append(sub.client_id)

    # Safety: If all were rejected due to extreme variance, fall back to the closest client
    if not clean_submissions:
        min_idx = int(torch.argmin(dists).item())
        clean_submissions.append(submissions[min_idx])
        rejected_client_ids.remove(submissions[min_idx].client_id)

    return clean_submissions, rejected_client_ids
