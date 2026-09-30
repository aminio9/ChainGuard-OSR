"""Evaluation metrics: standard classification, open-set AUROC/F1, and conformal coverage."""

from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics, compute_binary_auroc, compute_fpr95
from src.metrics.conformal_metrics import compute_conformal_metrics
from src.metrics.statistical_tests import (
    CANONICAL_TEN_SEEDS,
    CANONICAL_FIVE_SEEDS,
    compute_cliffs_delta,
    compute_wilcoxon_signed_rank,
    compute_bootstrap_ci,
    compute_curve_bootstrap_ci,
    summarize_multi_seed_Comparison,
    summarize_multi_seed_comparison,
)

__all__ = [
    "compute_classification_metrics",
    "compute_osr_metrics",
    "compute_binary_auroc",
    "compute_fpr95",
    "compute_conformal_metrics",
    "CANONICAL_TEN_SEEDS",
    "CANONICAL_FIVE_SEEDS",
    "compute_cliffs_delta",
    "compute_wilcoxon_signed_rank",
    "compute_bootstrap_ci",
    "compute_curve_bootstrap_ci",
    "summarize_multi_seed_Comparison",
    "summarize_multi_seed_comparison",
]

