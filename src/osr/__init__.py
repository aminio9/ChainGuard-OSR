"""Open-Set Recognition (OSR) and Conformal Prediction: Block-Diagonal Mahalanobis and Mondrian Conformal (LaTeX Section 4.2-4.4)."""

from src.osr.weibull_evt import WeibullEVTFitter, compute_evt_probabilities
from src.osr.energy_ood import compute_free_energy_score
from src.osr.conformal import (
    MondrianConformalCalibrator,
    compute_finite_sample_quantile,
    aggregate_distributed_quantiles,
)
from src.osr.block_mahalanobis import (
    WireSpeedEdgeDetector,
    MahalanobisInferenceResult,
    compute_block_mahalanobis_distances,
)

__all__ = [
    "WeibullEVTFitter",
    "compute_evt_probabilities",
    "compute_free_energy_score",
    "MondrianConformalCalibrator",
    "compute_finite_sample_quantile",
    "aggregate_distributed_quantiles",
    "WireSpeedEdgeDetector",
    "MahalanobisInferenceResult",
    "compute_block_mahalanobis_distances",
]
