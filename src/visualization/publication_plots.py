"""Publication-grade vector and high-DPI raster visualization generators."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.io import ensure_dir

# Set publication style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)


def plot_convergence_curves(history: List[Dict[str, float]], output_path: str | Path) -> None:
    """Plot training loss curves across federated rounds."""
    p = Path(output_path)
    ensure_dir(p.parent)

    rounds = [entry["round"] for entry in history]
    losses = [entry.get("loss_total", 0.0) for entry in history]
    cb_losses = [entry.get("loss_cb", 0.0) for entry in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, losses, label="Total Loss", color="#1f77b4", linewidth=2.0)
    if "loss_cb" in history[0]:
        ax.plot(rounds, cb_losses, label="CB Loss", color="#ff7f0e", linestyle="--")

    ax.set_xlabel("Federated Communication Round")
    ax.set_ylabel("Loss")
    ax.set_title("Federated Training Convergence")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)


def plot_auroc_vs_openness(
    openness_values: List[float],
    method_results: Dict[str, List[float]],
    output_path: str | Path,
) -> None:
    """Plot OSR AUROC performance as a function of Openness metric (Figure 4 in paper)."""
    p = Path(output_path)
    ensure_dir(p.parent)

    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ["o", "s", "^", "D", "v"]

    for idx, (method_name, aurocs) in enumerate(method_results.items()):
        marker = markers[idx % len(markers)]
        ax.plot(
            openness_values,
            aurocs,
            label=method_name,
            marker=marker,
            markersize=6,
            linewidth=2.0,
        )

    ax.set_xlabel(r"Openness Metric ($\mathcal{O}$)")
    ax.set_ylabel("Open-Set AUROC")
    ax.set_title("Robustness Across Increasing Openness")
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
) -> None:
    """Generate normalized confusion matrix heatmap."""
    p = Path(output_path)
    ensure_dir(p.parent)

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, pred in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= pred < num_classes:
            cm[t, pred] += 1

    # Row-normalized percentages
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Accuracy"},
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Open-Set Confusion Matrix")
    plt.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)


def plot_conformal_set_distribution(
    prediction_sets: List[List[int]], output_path: str | Path
) -> None:
    """Plot distribution of conformal prediction set sizes (singleton, empty, multi-class)."""
    p = Path(output_path)
    ensure_dir(p.parent)

    sizes = [len(s) for s in prediction_sets]
    unique, counts = np.unique(sizes, return_counts=True)
    percentages = counts / len(sizes) * 100.0

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar([str(u) for u in unique], percentages, color="#2ca02c", edgecolor="black", width=0.6)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Prediction Set Cardinality |C(x)|")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Conformal Prediction Set Cardinality Distribution")
    ax.set_ylim([0, max(percentages) * 1.15])
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
