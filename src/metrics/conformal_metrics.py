"""Conformal prediction metrics: empirical coverage, efficiency set size, and empty set rate."""

from __future__ import annotations

from typing import Dict, List
import numpy as np


def compute_conformal_metrics(
    prediction_sets: List[List[int]],
    y_true: np.ndarray,
    target_coverage: float = 0.95,
) -> Dict[str, float]:
    """Compute empirical coverage rate, mean set size, and empty set rate.

    Args:
        prediction_sets: List of conformal prediction sets per sample.
        y_true: Ground truth known class labels.
        target_coverage: 1 - alpha nominal guarantee level.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    total_samples = len(y_true)
    if total_samples == 0:
        return {"empirical_coverage": 0.0, "average_set_size": 0.0, "empty_set_rate": 0.0}

    covered_count = 0
    empty_count = 0
    sizes = []

    for i, pset in enumerate(prediction_sets):
        s_len = len(pset)
        sizes.append(s_len)
        if s_len == 0:
            empty_count += 1
        elif y_true[i] in pset:
            covered_count += 1

    empirical_coverage = float(covered_count / total_samples)
    avg_set_size = float(np.mean(sizes))
    empty_set_rate = float(empty_count / total_samples)

    return {
        "target_coverage": float(target_coverage),
        "empirical_coverage": empirical_coverage,
        "coverage_violation": max(0.0, target_coverage - empirical_coverage),
        "average_set_size": avg_set_size,
        "empty_set_rate": empty_set_rate,
    }
