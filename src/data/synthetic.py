"""Synthetic P2P blockchain network flow dataset generator for benchmarking and tests."""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from src.data.base import FlowDataset


def generate_synthetic_p2p_data(
    num_samples: int = 5000,
    in_features: int = 43,
    num_classes: int = 10,
    normal_ratio: float = 0.5,
    noise_level: float = 0.05,
    seed: int = 42,
) -> FlowDataset:
    """Generate synthetic network flow feature vectors mimicking P2P blockchain traffic.

    Class 0 represents Normal P2P traffic (dominant volume).
    Classes 1..(C-1) represent various known and unknown attack classes.
    """
    rng = np.random.default_rng(seed)

    # Allocate number of samples per class
    # Normal traffic takes normal_ratio of samples; rest is divided among attack classes
    num_normal = int(num_samples * normal_ratio)
    num_attacks_total = num_samples - num_normal
    num_attack_classes = num_classes - 1

    # Exponentially distributed attack frequencies to simulate real-world long-tail imbalance
    weights = np.exp(-0.3 * np.arange(num_attack_classes))
    weights /= weights.sum()
    attack_counts = np.round(weights * num_attacks_total).astype(int)

    # Adjust counts to match total
    diff = num_samples - (num_normal + attack_counts.sum())
    attack_counts[0] += diff

    all_features = []
    all_labels = []

    # Distinct center per class in high-dimensional feature space
    centers = rng.normal(loc=0.0, scale=3.0, size=(num_classes, in_features))

    # Generate Normal Traffic (Class 0)
    normal_feat = rng.normal(loc=centers[0], scale=1.0, size=(num_normal, in_features))
    all_features.append(normal_feat)
    all_labels.append(np.zeros(num_normal, dtype=np.int64))

    # Generate Attacks (Classes 1..num_classes-1)
    for c_idx, count in enumerate(attack_counts, start=1):
        if count <= 0:
            continue
        feat = rng.normal(loc=centers[c_idx], scale=1.0 + noise_level, size=(count, in_features))
        all_features.append(feat)
        all_labels.append(np.full(count, c_idx, dtype=np.int64))

    X = np.vstack(all_features).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int64)

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # Standard Z-Score Normalization
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-7
    X_norm = (X - mean) / std

    return FlowDataset(torch.from_numpy(X_norm), torch.from_numpy(y))
