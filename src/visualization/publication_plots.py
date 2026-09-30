"""Publication-grade IEEE Transactions vector (.pdf) and high-DPI raster (.png) visualization generators.

Strictly adheres to IEEE TIFS/TDSC visual standards:
  - Colorblind-safe Okabe-Ito palette (`OKABE_ITO`).
  - 8pt font sizing matching IEEE double-column (3.5 in) and full-width (7.16 in) layouts.
  - Simultaneous vector `.pdf` (`pdf.fonttype = 42`) and 300-DPI `.png` export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.io import ensure_dir

# Colorblind-safe Okabe-Ito Palette (IEEE TIFS/TDSC Standard)
OKABE_ITO: Dict[str, str] = {
    "blue": "#0072B2",      # Proposed Method
    "red": "#D55E00",       # Key Competitor 1 (e.g., FedPD++)
    "green": "#009E73",     # Key Competitor 2 (e.g., FedProto / FLTrust)
    "purple": "#CC79A7",    # Competitor 3 (e.g., FedDyn / OpenMax)
    "orange": "#E69F00",    # Competitor 4 (e.g., FedRoD / Krum)
    "sky": "#56B4E9",       # Competitor 5 (e.g., SCAFFOLD / Trimmed Mean)
    "yellow": "#F0E442",    # Competitor 6
    "black": "#000000",     # Theoretical Bound / Oracle
}

OKABE_ITO_LIST: List[str] = [
    OKABE_ITO["blue"],
    OKABE_ITO["red"],
    OKABE_ITO["green"],
    OKABE_ITO["purple"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky"],
    OKABE_ITO["black"],
]

MARKERS: List[str] = ["o", "s", "^", "D", "v", "P", "*"]


def apply_ieee_tifs_style() -> None:
    """Configure Matplotlib rcParams for IEEE Transactions 8pt publication figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.titlesize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


apply_ieee_tifs_style()


def _save_dual_format(fig: plt.Figure, output_path: str | Path) -> None:
    """Save both the requested file path and its corresponding vector .pdf / .png counterpart."""
    p = Path(output_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    if p.suffix.lower() == ".png":
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    elif p.suffix.lower() == ".pdf":
        fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_convergence_curves(history: List[Dict[str, float]], output_path: str | Path) -> None:
    """Plot training loss curves across federated rounds."""
    apply_ieee_tifs_style()
    rounds = [entry["round"] for entry in history]
    losses = [entry.get("loss_total", 0.0) for entry in history]
    cb_losses = [entry.get("loss_cb", entry.get("loss_bla", 0.0)) for entry in history]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(rounds, losses, label="Total Loss", color=OKABE_ITO["blue"], linewidth=1.8, marker="o", markersize=3)
    if any(v > 0 for v in cb_losses):
        ax.plot(
            rounds,
            cb_losses,
            label="Logit-Adjusted Risk",
            color=OKABE_ITO["red"],
            linestyle="--",
            linewidth=1.5,
        )

    ax.set_xlabel("Federated Communication Round")
    ax.set_ylabel("Objective Loss")
    ax.set_title("Phase I Federated Training Convergence")
    ax.legend(frameon=True)
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)


def plot_auroc_vs_openness(
    openness_values: Sequence[float],
    method_results: Dict[str, Sequence[float]],
    output_path: str | Path,
    ci_bounds: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
) -> None:
    """Plot OSR AUROC performance as a function of Openness metric with optional 95% CI bands."""
    apply_ieee_tifs_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for idx, (method_name, aurocs) in enumerate(method_results.items()):
        color = OKABE_ITO_LIST[idx % len(OKABE_ITO_LIST)]
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(
            openness_values,
            aurocs,
            label=method_name,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.8 if "Proposed" in method_name else 1.3,
        )
        if ci_bounds and method_name in ci_bounds:
            lo = ci_bounds[method_name].get("lower", aurocs)
            hi = ci_bounds[method_name].get("upper", aurocs)
            ax.fill_between(openness_values, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel(r"Openness Metric ($\mathcal{O}$)")
    ax.set_ylabel("Open-Set AUROC")
    ax.set_title("Open-Set Detection Across Increasing Openness")
    ax.set_ylim([0.4, 1.02])
    ax.legend(frameon=True)
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
) -> None:
    """Generate row-normalized confusion matrix heatmap."""
    apply_ieee_tifs_style()
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, pred in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= pred < num_classes:
            cm[t, pred] += 1

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Recall"},
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Open-Set Confusion Matrix")
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)


def plot_conformal_set_distribution(
    prediction_sets: List[List[int]], output_path: str | Path
) -> None:
    """Plot distribution of conformal prediction set sizes (singleton, empty, multi-class)."""
    apply_ieee_tifs_style()
    sizes = [len(s) for s in prediction_sets]
    unique, counts = np.unique(sizes, return_counts=True)
    percentages = counts / max(len(sizes), 1) * 100.0

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(
        [str(u) for u in unique],
        percentages,
        color=OKABE_ITO["blue"],
        edgecolor=OKABE_ITO["black"],
        width=0.55,
    )

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xlabel("Prediction Set Cardinality $|\\mathcal{C}(x)|$")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Conformal Set Cardinality Distribution")
    ax.set_ylim([0, max(percentages) * 1.18 if len(percentages) > 0 else 100])
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)


def plot_svd_singular_value_spectrum(
    spectra: Dict[str, Sequence[float]],
    output_path: str | Path,
) -> None:
    """Plot normalized singular value spectrum of latent covariance (Ablation A1: Dimensional Collapse)."""
    apply_ieee_tifs_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for idx, (label, sv_vals) in enumerate(spectra.items()):
        arr = np.asarray(sv_vals, dtype=np.float64)
        indices = np.arange(1, len(arr) + 1)
        color = OKABE_ITO_LIST[idx % len(OKABE_ITO_LIST)]
        ax.semilogy(
            indices,
            np.clip(arr, 1e-8, None),
            label=label,
            color=color,
            linewidth=1.6,
        )

    ax.set_xlabel("Singular Value Index $i$")
    ax.set_ylabel(r"Normalized Singular Value $\sigma_i / \sigma_1$")
    ax.set_title("Latent Covariance SVD Spectrum (A1)")
    ax.legend(frameon=True)
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)


def plot_subblock_pareto_curve(
    pareto_data: Dict[str, Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Plot dual-axis sub-block size d_b Pareto curve (AUROC vs Latency & FLOPs, Ablation A5)."""
    apply_ieee_tifs_style()
    keys = sorted(pareto_data.keys(), key=lambda k: int(pareto_data[k]["block_dim"]))
    block_dims = [int(pareto_data[k]["block_dim"]) for k in keys]
    aurocs = [float(pareto_data[k]["auroc"]) * 100.0 for k in keys]
    latencies = [float(pareto_data[k]["median_latency_us"]) for k in keys]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    ax2 = ax1.twinx()

    x_pos = np.arange(len(block_dims))
    l1 = ax1.plot(
        x_pos,
        aurocs,
        color=OKABE_ITO["blue"],
        marker="o",
        linewidth=1.8,
        label="Open-Set AUROC (%)",
    )
    l2 = ax2.plot(
        x_pos,
        latencies,
        color=OKABE_ITO["red"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label=r"Median Latency ($\mu$s)",
    )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(d) for d in block_dims])
    ax1.set_xlabel(r"Block-Diagonal Sub-Block Dimension $d_b$")
    ax1.set_ylabel("Open-Set AUROC (%)", color=OKABE_ITO["blue"])
    ax2.set_ylabel(r"Single-Packet Latency ($\mu$s)", color=OKABE_ITO["red"])
    ax2.grid(False)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)
    ax1.set_title("Sub-Block Granularity Pareto Front (A5)")
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)


def plot_conformal_calibration_diagonal(
    alphas: Sequence[float],
    mondrian_kfr: Sequence[float],
    global_kfr: Sequence[float],
    output_path: str | Path,
) -> None:
    """Plot Conformal Nominal Target alpha vs Empirical Known False Rejection Rate (Ablation A6)."""
    apply_ieee_tifs_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.plot(
        alphas,
        alphas,
        color=OKABE_ITO["black"],
        linestyle=":",
        linewidth=1.5,
        label=r"Ideal Diagonal ($\text{KFR} = \alpha$)",
    )
    ax.plot(
        alphas,
        mondrian_kfr,
        color=OKABE_ITO["blue"],
        marker="o",
        linewidth=1.8,
        label="Mondrian Class-Conditional (Ours)",
    )
    ax.plot(
        alphas,
        global_kfr,
        color=OKABE_ITO["red"],
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label="Global Single-Threshold",
    )

    ax.set_xlabel(r"Nominal Significance Level $\alpha$")
    ax.set_ylabel("Empirical Known False Rejection (KFR)")
    ax.set_title("Finite-Sample Conformal Calibration (A6)")
    ax.legend(frameon=True)
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)


def plot_ldp_privacy_utility_mia(
    epsilons: Sequence[float],
    aurocs: Sequence[float],
    mia_aucs: Sequence[float],
    output_path: str | Path,
) -> None:
    """Plot Local Differential Privacy budget epsilon vs Open-Set AUROC and MIA Attack AUC."""
    apply_ieee_tifs_style()
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        epsilons,
        [a * 100.0 for a in aurocs],
        color=OKABE_ITO["blue"],
        marker="o",
        linewidth=1.8,
        label="Open-Set AUROC (%)",
    )
    l2 = ax2.plot(
        epsilons,
        [m * 100.0 for m in mia_aucs],
        color=OKABE_ITO["red"],
        marker="^",
        linestyle="--",
        linewidth=1.6,
        label="MIA Attack AUC (%)",
    )
    ax2.axhline(50.0, color=OKABE_ITO["black"], linestyle=":", linewidth=1.0, label="Random Guess (50%)")

    ax1.set_xlabel(r"Local DP Privacy Budget $\epsilon$")
    ax1.set_ylabel("Open-Set AUROC (%)", color=OKABE_ITO["blue"])
    ax2.set_ylabel("Membership Inference AUC (%)", color=OKABE_ITO["red"])
    ax2.set_ylim([45.0, 100.0])
    ax2.grid(False)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)
    ax1.set_title("LDP Moment Perturbation & MIA Audit")
    plt.tight_layout()
    _save_dual_format(fig, output_path)
    plt.close(fig)
