"""Statistical Class Profiling via Law of Total Variance (LOTV) and OAS Shrinkage.

Corresponds to the statistical profiling step:
    1. Multi-modal normal traffic alignment via One-Shot Federated K-Means.
    2. Global covariance matrix reconstruction via the Law of Total Variance (LOTV).
    3. Analytical regularized covariance stabilization via Oracle Approximating Shrinkage (OAS).
"""

from __future__ import annotations

from src.aggregation.total_variance import (
    ClientClassMoment,
    GlobalClassProfile,
    aggregate_total_variance_moments,
    compute_server_side_oas_shrinkage,
    extract_one_shot_federated_reference_centers,
)

__all__ = [
    "ClientClassMoment",
    "GlobalClassProfile",
    "aggregate_total_variance_moments",
    "compute_server_side_oas_shrinkage",
    "extract_one_shot_federated_reference_centers",
]
