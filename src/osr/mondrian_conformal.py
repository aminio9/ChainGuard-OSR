"""Distributed Mondrian Conformal Calibration.

Corresponds to the distributed conformal calibration step:
    1. Evaluates nonconformity scores on local calibration data.
    2. Builds local discrete count histograms to preserve privacy.
    3. Aggregates histograms globally at coordinator.
    4. Computes class-specific finite-sample corrected quantile thresholds.
"""

from __future__ import annotations

from src.osr.conformal import (
    compute_right_closed_histogram,
    compute_finite_sample_quantile,
    aggregate_distributed_cdfs,
    MondrianConformalCalibrator,
)

__all__ = [
    "compute_right_closed_histogram",
    "compute_finite_sample_quantile",
    "aggregate_distributed_cdfs",
    "MondrianConformalCalibrator",
]
