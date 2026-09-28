"""Open-Set Recognition (OSR) metrics: AUROC, AUPR, FPR95, Open-Set F1, and Known False-Rejection Rate (KFR)."""

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

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = tps / float(n_pos)
    fpr = fps / float(n_neg)

    # Trapezoidal integration
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return float(np.clip(auroc, 0.0, 1.0))


def compute_binary_aupr(y_binary: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall curve (AUPR) for unknown attack detection.

    y_binary: 1 for Positive (Unknown Zero-Day), 0 for Negative (Known).
    anomaly_scores: Higher values indicate higher probability of being Unknown.
    """
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(anomaly_scores, dtype=float)

    n_pos = np.sum(y_binary == 1)
    if n_pos == 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    y_sorted = y_binary[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    precisions = tps / (tps + fps)
    recalls = tps / float(n_pos)

    # Precision at recall = 0 is precision of first point
    precisions = np.concatenate([[precisions[0] if len(precisions) > 0 else 1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])

    aupr = float(np.trapezoid(precisions, recalls)) if hasattr(np, "trapezoid") else float(np.trapz(precisions, recalls))
    return float(np.clip(aupr, 0.0, 1.0))


def compute_fpr95(y_binary: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Compute False Positive Rate on known traffic when True Positive Rate on unknown attacks is 95%.

    Measures score-overlap regimes where AUROC alone can be deceptive.
    """
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(anomaly_scores, dtype=float)

    pos_scores = scores[y_binary == 1]
    neg_scores = scores[y_binary == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    # 95% TPR threshold: 5th percentile of positive scores
    threshold = float(np.percentile(pos_scores, 5.0))

    # Fraction of negative (known) instances exceeding this threshold
    fpr = float(np.mean(neg_scores >= threshold))
    return fpr


def compute_osr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    num_known_classes: int,
    unknown_label: int = -1,
) -> Dict[str, float]:
    """Compute comprehensive Open-Set Recognition evaluation metrics per Section 5.5 of paper.

    Args:
        y_true: Ground truth labels (known are 0..C-1, unknown is >= C).
        y_pred: Predicted class (-1 or >= C for unknown, 0..C-1 for known).
        anomaly_scores: Scores where higher values indicate unknown anomaly.
        num_known_classes: Total known classes C.
        unknown_label: Value used in y_pred to designate unknown.

    Returns:
        Dict containing:
            - auroc: Area under ROC curve for known vs unknown discrimination
            - aupr: Area under Precision-Recall curve
            - fpr95: False Positive Rate on known traffic at 95% TPR
            - kfr: Known False-Rejection Rate (Equation 109: false alarms on known traffic)
            - false_alarm_rate: Alias for kfr
            - unknown_recall: Recall on unknown zero-day attacks (Section 5.5)
            - unknown_precision: Precision on unknown zero-day attacks
            - unknown_f1: F1 score for the unknown class
            - open_set_macro_f1: Macro-F1 across all |C_K| + 1 decision classes
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    is_unknown_true = (y_true >= num_known_classes).astype(int)
    is_unknown_pred = (y_pred == unknown_label) | (y_pred >= num_known_classes)

    # 1. Ranking Metrics
    auroc = compute_binary_auroc(is_unknown_true, anomaly_scores)
    aupr = compute_binary_aupr(is_unknown_true, anomaly_scores)
    fpr95 = compute_fpr95(is_unknown_true, anomaly_scores)

    # 2. Known False-Rejection Rate (KFR / FAR <= alpha = 0.05 by Theorem 2)
    known_mask = (is_unknown_true == 0)
    if np.sum(known_mask) > 0:
        kfr = float(np.mean(is_unknown_pred[known_mask]))
    else:
        kfr = 0.0

    # 3. Detection Rate / Recall on unknown zero-day attacks
    unknown_mask = (is_unknown_true == 1)
    if np.sum(unknown_mask) > 0:
        unk_recall = float(np.mean(is_unknown_pred[unknown_mask]))
    else:
        unk_recall = 0.0

    # Unknown Precision and F1
    pred_unk_count = int(np.sum(is_unknown_pred))
    tp_unk = int(np.sum(is_unknown_pred & (is_unknown_true == 1)))
    unk_precision = float(tp_unk / pred_unk_count) if pred_unk_count > 0 else 0.0
    if unk_precision + unk_recall > 0:
        unk_f1 = float(2.0 * unk_precision * unk_recall / (unk_precision + unk_recall))
    else:
        unk_f1 = 0.0

    # 4. Open-Set Macro F1 (C known classes + 1 unknown class)
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
        "aupr": aupr,
        "fpr95": fpr95,
        "kfr": kfr,
        "false_alarm_rate": kfr,
        "unknown_recall": unk_recall,
        "unknown_detection_rate": unk_recall,
        "unknown_precision": unk_precision,
        "unknown_f1": unk_f1,
        "open_set_macro_f1": float(np.mean(f1_list)),
    }
