"""Open-Set Recognition (OSR) and Conformal Prediction: Weibull EVT, Free Energy, and Mondrian Conformal."""

from src.osr.weibull_evt import WeibullEVTFitter, compute_evt_probabilities
from src.osr.energy_ood import compute_free_energy_score
from src.osr.conformal import MondrianConformalCalibrator
from src.osr.edge_inference import EdgeInferenceEngine

__all__ = [
    "WeibullEVTFitter",
    "compute_evt_probabilities",
    "compute_free_energy_score",
    "MondrianConformalCalibrator",
    "EdgeInferenceEngine",
]
