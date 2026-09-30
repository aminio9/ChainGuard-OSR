"""Standard classification metrics: Accuracy, Macro F1, Precision, and Recall."""

from __future__ import annotations

from typing import Dict
import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """Calculate multiclass accuracy, balanced accuracy, macro F1, and per-class audit metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    total_samples = len(y_true)
    if total_samples == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "class_wise_f1": [],
            "class_wise_precision": [],
            "class_wise_recall": [],
            "confusion_matrix": [],
        }

    accuracy = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(y_true == c) - tp)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": macro_rec,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": macro_rec,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "class_wise_f1": f1s,
        "class_wise_precision": precisions,
        "class_wise_recall": recalls,
        "confusion_matrix": cm.tolist(),
    }

