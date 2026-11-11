"""Multi-Centroid Mahalanobis Distance Metric.

Corresponds to the multi-centroid Mahalanobis distance formulation:
    - Attack Classes: Single-centroid Mahalanobis distance using the inverted regularized covariance.
    - Normal Traffic: Minimum Mahalanobis distance across all four operational subclusters
      (session handshake, ledger synchronization, transaction propagation, block announcement).
"""

from __future__ import annotations

import torch
from typing import Dict
from src.aggregation.total_variance import GlobalClassProfile
from src.osr.block_mahalanobis import (
    compute_single_block_distance,
    compute_block_mahalanobis_distances,
)

# Alias matching the thesis terminology
compute_multimodal_mahalanobis_distances = compute_block_mahalanobis_distances

__all__ = [
    "compute_single_block_distance",
    "compute_block_mahalanobis_distances",
    "compute_multimodal_mahalanobis_distances",
]
