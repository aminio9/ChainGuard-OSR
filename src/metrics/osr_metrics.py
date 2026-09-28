"""Open-Set Recognition (OSR) metrics: AUROC, Open-Set F1, and False Alarm Rate (FAR)."""

from __future__ import annotations

from typing import Dict
import numpy as np


def compute_binary_auroc(y_binary: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic curve (AUROC) from scratch.

    y_binary: 1 for Positive (Unknown Zero-Day), 0 for Negative (Known).
    anomaly_scores: Higher values indicate higher probability of being Unknown.
    """
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(anomaly_scores, dtype=float)

    n_pos = np.sum(y_binary == 1)
    n_neg = np.sum(y_binary == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort descending by score
    order = np.argsort(scores)[::-1]
    y_sorted = y_binary[order]

    # Calculate Mann-Whitney U statistic
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = tps / float(n_pos)
    fpr = fps / float(n_neg)

    # Trapezoidal integration
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return auroc


def compute_osr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    num_known_classes: int,
    unknown_label: int = -1,
) -> Dict[str, float]:
    """Compute comprehensive Open-Set Recognition evaluation metrics.

    Args:
        y_true: Ground truth labels (known are 0..C-1, unknown is >= C).
        y_pred: Predicted class (-1 or >= C for unknown, 0..C-1 for known).
        anomaly_scores: Scores where higher values indicate unknown anomaly.
        num_known_classes: Total known classes C.
        unknown_label: Value used in y_pred to designate unknown.

    Returns:
        Dict containing:
            - auroc: Area under ROC curve for known vs unknown discrimination
            - false_alarm_rate: Fraction of known samples incorrectly rejected as unknown
            - unknown_detection_rate: Recall on unknown zero-day attacks
            - open_set_macro_f1: Macro-F1 across known classes plus the unknown class
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    is_unknown_true = (y_true >= num_known_classes).astype(int)
    is_unknown_pred = (y_pred == unknown_label) | (y_pred >= num_known_classes)

    # 1. Binary AUROC
    auroc = compute_binary_auroc(is_unknown_true, anomaly_scores)

    # 2. False Alarm Rate on known traffic (FAR <= alpha = 0.05 by Theorem 2)
    known_mask = (is_unknown_true == 0)
    if np.sum(known_mask) > 0:
        far = float(np.mean(is_unknown_pred[known_mask]))
    else:
        far = 0.0

    # 3. Detection Rate on unknown zero-day attacks (Recall_U)
    unknown_mask = (is_unknown_true == 1)
    if np.sum(unknown_mask) > 0:
        dr = float(np.mean(is_unknown_pred[unknown_mask]))
    else:
        dr = 0.0

    # 4. Open-Set Macro F1 (C known classes + 1 unknown class)
    # Remap unknown to class index C for both y_true and y_pred
    remapped_true = np.where(y_true >= num_known_classes, num_known_classes, y_true)
    remapped_pred = np.where(is_unknown_pred, num_known_classes, y_pred)

    f1_list = []
    for c in range(num_known_classes + 1):
        tp = int(np.sum((remapped_true == c) & (remapped_pred == c)))
        fp = int(np.sum((remapped_true != c) & (remapped_pred == c)))
        fn = int(np.sum((remapped_true == c) & (remapped_pred != c)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    return {
        "auroc": auroc,
        "false_alarm_rate": far,
        "unknown_detection_rate": dr,
        "open_set_macro_f1": float(np.mean(f1_list)),
    }
