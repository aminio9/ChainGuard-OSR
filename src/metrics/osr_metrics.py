"""Open-Set Recognition (OSR) metrics: AUROC, AUPR, FPR95, Open-Set F1, and Known False-Rejection Rate (KFR)."""

from __future__ import annotations

from typing import Any, Dict
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


def compute_oscr(
    y_true: np.ndarray,
    closed_preds: np.ndarray,
    anomaly_scores: np.ndarray,
    num_known_classes: int,
) -> float:
    """Compute Open-Set Classification Rate (OSCR: area under CCR vs FPR curve).

    CCR(tau) = P(y_hat_closed(x) == y_true AND S(x) <= tau | y_true in C_K)
    FPR(tau) = P(S(x) <= tau | y_true in C_U)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    closed_preds = np.asarray(closed_preds, dtype=np.int64)
    scores = np.asarray(anomaly_scores, dtype=float)

    known_mask = y_true < num_known_classes
    unknown_mask = y_true >= num_known_classes
    n_k = int(np.sum(known_mask))
    n_u = int(np.sum(unknown_mask))
    if n_k == 0 or n_u == 0:
        return 0.0

    k_scores = scores[known_mask]
    k_correct = (closed_preds[known_mask] == y_true[known_mask])
    u_scores = scores[unknown_mask]

    thresholds = np.linspace(min(k_scores.min(), u_scores.min()), max(k_scores.max(), u_scores.max()), 1000)
    k_sort_idx = np.argsort(k_scores)
    k_scores_sorted = k_scores[k_sort_idx]
    k_correct_sorted = k_correct[k_sort_idx]
    cum_correct = np.cumsum(k_correct_sorted)

    indices_k = np.searchsorted(k_scores_sorted, thresholds, side="right") - 1
    ccr = np.where(indices_k >= 0, cum_correct[np.maximum(indices_k, 0)], 0) / float(n_k)

    indices_u = np.searchsorted(np.sort(u_scores), thresholds, side="right")
    fpr = indices_u / float(n_u)

    fpr = np.concatenate([[0.0], fpr, [1.0]])
    ccr = np.concatenate([[0.0], ccr, [ccr[-1] if len(ccr) > 0 else 0.0]])
    order = np.argsort(fpr, kind="mergesort")
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    oscr = float(trap(ccr[order], fpr[order]))
    return float(np.clip(oscr, 0.0, 1.0))


def compute_osr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    num_known_classes: int,
    unknown_label: int = -1,
    closed_preds: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute comprehensive Open-Set Recognition evaluation metrics per Section 5.5 of paper."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    is_unknown_true = (y_true >= num_known_classes).astype(int)
    is_unknown_pred = (y_pred == unknown_label) | (y_pred >= num_known_classes)

    # 1. Ranking Metrics
    auroc = compute_binary_auroc(is_unknown_true, anomaly_scores)
    aupr = compute_binary_aupr(is_unknown_true, anomaly_scores)
    fpr95 = compute_fpr95(is_unknown_true, anomaly_scores)

    cand_preds = closed_preds if closed_preds is not None else np.where(is_unknown_pred, 0, y_pred)
    oscr = compute_oscr(y_true, cand_preds, anomaly_scores, num_known_classes)

    # 2. Known False-Rejection Rate (KFR / FAR <= alpha = 0.05 by Theorem 2) & Per-Class Coverage
    known_mask = (is_unknown_true == 0)
    if np.sum(known_mask) > 0:
        kfr = float(np.mean(is_unknown_pred[known_mask]))
    else:
        kfr = 0.0

    per_class_kfr: Dict[int, float] = {}
    per_class_coverage: Dict[int, float] = {}
    for c in range(num_known_classes):
        c_mask = (y_true == c)
        if np.sum(c_mask) > 0:
            c_kfr = float(np.mean(is_unknown_pred[c_mask]))
            per_class_kfr[c] = c_kfr
            per_class_coverage[c] = 1.0 - c_kfr
        else:
            per_class_kfr[c] = 0.0
            per_class_coverage[c] = 1.0

    # 3. Detection Rate / Recall (TPR) on unknown zero-day attacks
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
        "oscr": oscr,
        "fpr95": fpr95,
        "kfr": kfr,
        "empirical_coverage": 1.0 - kfr,
        "false_alarm_rate": kfr,
        "per_class_kfr": per_class_kfr,
        "per_class_coverage": per_class_coverage,
        "unknown_recall": unk_recall,
        "unknown_tpr": unk_recall,
        "unknown_detection_rate": unk_recall,
        "unknown_precision": unk_precision,
        "unknown_f1": unk_f1,
        "open_set_macro_f1": float(np.mean(f1_list)),
        "accuracy": float(np.mean(remapped_true == remapped_pred)),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> np.ndarray:
    """Compute exact integer confusion matrix using pure NumPy."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if labels is None:
        all_lbls = sorted(set(yt.tolist()) | set(yp.tolist()))
    else:
        all_lbls = list(labels)
    lbl_to_idx = {int(lbl): idx for idx, lbl in enumerate(all_lbls)}
    n = len(all_lbls)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(yt, yp):
        ti = lbl_to_idx.get(int(t))
        pi = lbl_to_idx.get(int(p))
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


def compute_roc_curve_points(
    y_binary: np.ndarray,
    anomaly_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for ROC curve plotting in pure NumPy."""
    yb = np.asarray(y_binary, dtype=int)
    sc = np.asarray(anomaly_scores, dtype=float)
    n_pos = int(np.sum(yb == 1))
    n_neg = int(np.sum(yb == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(sc)[::-1]
    y_sorted = yb[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / float(n_pos), [1.0]])
    fpr = np.concatenate([[0.0], fps / float(n_neg), [1.0]])
    return fpr, tpr


def compute_pr_curve_points(
    y_binary: np.ndarray,
    anomaly_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (precision, recall) arrays for PR curve plotting in pure NumPy."""
    yb = np.asarray(y_binary, dtype=int)
    sc = np.asarray(anomaly_scores, dtype=float)
    n_pos = int(np.sum(yb == 1))
    if n_pos == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    order = np.argsort(sc)[::-1]
    y_sorted = yb[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / float(n_pos)
    first_p = float(prec[0]) if len(prec) > 0 else 1.0
    precisions = np.concatenate([[first_p], prec])
    recalls = np.concatenate([[0.0], rec])
    return precisions, recalls


