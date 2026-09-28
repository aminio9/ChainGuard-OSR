"""Unit tests for OSR evaluation metrics (Section 5.5 of paper)."""

import numpy as np
import pytest
from src.metrics.osr_metrics import (
    compute_binary_aupr,
    compute_binary_auroc,
    compute_fpr95,
    compute_osr_metrics,
)


def test_binary_auroc_and_aupr_ideal():
    # Perfectly separated scores: positives (unknown) have high scores, negatives (known) have low scores
    y_binary = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.05, 0.1, 0.8, 0.9, 0.85, 0.95, 0.75])

    auroc = compute_binary_auroc(y_binary, scores)
    aupr = compute_binary_aupr(y_binary, scores)
    fpr95 = compute_fpr95(y_binary, scores)

    assert auroc == pytest.approx(1.0, abs=1e-4)
    assert aupr == pytest.approx(1.0, abs=1e-2)
    assert fpr95 == pytest.approx(0.0, abs=1e-4)


def test_comprehensive_osr_metrics():
    # 4 known samples (class 0, 1) and 2 unknown samples (class 2)
    y_true = np.array([0, 0, 1, 1, 2, 2])
    # Predictions: 1 known misclassified as unknown, both unknowns correctly caught as unknown (label -1)
    y_pred = np.array([0, -1, 1, 1, -1, -1])
    anomaly_scores = np.array([1.0, 5.0, 1.2, 0.8, 10.0, 12.0])

    metrics = compute_osr_metrics(
        y_true=y_true,
        y_pred=y_pred,
        anomaly_scores=anomaly_scores,
        num_known_classes=2,
        unknown_label=-1,
    )

    # 1 out of 4 known falsely rejected as unknown -> KFR = 0.25
    assert metrics["kfr"] == pytest.approx(0.25, abs=1e-3)
    assert metrics["false_alarm_rate"] == pytest.approx(0.25, abs=1e-3)

    # 2 out of 2 unknowns detected -> unknown_recall = 1.0
    assert metrics["unknown_recall"] == pytest.approx(1.0, abs=1e-3)

    # Ranking metrics
    assert metrics["auroc"] > 0.8
    assert metrics["aupr"] > 0.8
    assert "open_set_macro_f1" in metrics
