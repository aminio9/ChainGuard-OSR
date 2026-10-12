"""Unified Publication Plotting and Styling Utilities for FedTROS-MC inside D:\\PAPER\\Code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "Proposed": "#7B1FA2",
    "ChainGuard-OSR": "#7B1FA2",
    "ChainGuard": "#7B1FA2",
    "FedTROS-MC": "#7B1FA2",
    "FedTROS": "#7B1FA2",
    "FedAvg": "#1976D2",
    "FedProx": "#E64A19",
    "FedProto": "#00897B",
    "Known": "#2E7D32",
    "Unknown": "#C2185B",
    "Overall": "#00897B",
    "Error": "#D32F2F",
    "Neutral": "#546E7A",
    "Light": "#ECEFF1",
    "Before": "#607D8B",
    "After": "#7B1FA2",
    "ours": "#7B1FA2",
    "fedavg": "#1976D2",
    "fedprox": "#E64A19",
    "fedproto": "#00897B",
    "known": "#2E7D32",
    "unknown": "#C2185B",
    "threshold": "#D32F2F",
}

CATEGORY_COLORS = [
    "#00897B",
    "#1976D2",
    "#F57C00",
    "#C2185B",
    "#7B1FA2",
    "#2E7D32",
    "#0288D1",
    "#D84315",
]

HATCHES = ["///", "\\\\\\", "xxx", "...", "+++", "---", "OOO", "***"]

METHOD_STYLES = {
    "Proposed": {
        "color": "#7B1FA2",
        "marker": "o",
        "linestyle": "-",
        "hatch": "///",
        "label": "ChainGuard-OSR (Proposed)",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "ChainGuard-OSR": {
        "color": "#7B1FA2",
        "marker": "o",
        "linestyle": "-",
        "hatch": "///",
        "label": "ChainGuard-OSR (Proposed)",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedTROS-MC": {
        "color": "#7B1FA2",
        "marker": "o",
        "linestyle": "-",
        "hatch": "///",
        "label": "FedTROS-MC (Proposed)",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedAvg": {
        "color": "#1976D2",
        "marker": "s",
        "linestyle": "--",
        "hatch": "\\\\\\",
        "label": "FedAvg",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedProx": {
        "color": "#E64A19",
        "marker": "^",
        "linestyle": "-.",
        "hatch": "xxx",
        "label": "FedProx",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedProto": {
        "color": "#00897B",
        "marker": "D",
        "linestyle": ":",
        "hatch": "...",
        "label": "FedProto",
        "linewidth": 2.2,
        "markersize": 7.0,
    },
}

CLIENT_STYLES = {
    10: {"color": "#00897B", "marker": "o", "linestyle": "-", "hatch": "///", "label": "10 clients", "linewidth": 2.4, "markersize": 7.5},
    50: {"color": "#1976D2", "marker": "s", "linestyle": "--", "hatch": "\\\\\\", "label": "50 clients", "linewidth": 2.4, "markersize": 7.5},
    100: {"color": "#C2185B", "marker": "^", "linestyle": "-.", "hatch": "xxx", "label": "100 clients", "linewidth": 2.4, "markersize": 7.5},
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "academic_sequential",
    ["#FFFFFF", "#E1F5FE", "#4FC3F7", "#0288D1", "#1565C0", "#0D47A1", "#051B38"],
    N=256,
)

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "academic_diverging",
    ["#0D47A1", "#1976D2", "#BBDEFB", "#FFFFFF", "#FFCDD2", "#E53935", "#B71C1C"],
    N=256,
)


def configure_academic_style() -> None:
    """Set global rcParams for Q1 journal-grade 600 DPI plotting with deep true-black typography."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
            "font.size": 11.0,
            "font.weight": "medium",
            "text.color": "#000000",
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.2,
            "axes.labelcolor": "#000000",
            "axes.labelsize": 12.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.titlecolor": "#000000",
            "axes.titlepad": 10.0,
            "axes.axisbelow": True,
            "axes.grid": True,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "xtick.labelcolor": "#000000",
            "ytick.labelcolor": "#000000",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.70,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#000000",
            "legend.fancybox": False,
            "legend.fontsize": 10.0,
            "legend.title_fontsize": 10.5,
            "legend.labelcolor": "#000000",
            "lines.linewidth": 2.3,
            "lines.markersize": 7.0,
            "lines.markeredgewidth": 1.0,
            "lines.solid_capstyle": "round",
            "figure.facecolor": "#FFFFFF",
            "figure.edgecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apply_academic_spines(ax: plt.Axes, top: bool = False, right: bool = False) -> None:
    """Ensure clear, bold black spines with true-black ticks, labels, and title."""
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    if top:
        ax.spines["top"].set_color("#000000")
        ax.spines["top"].set_linewidth(1.2)
    if right:
        ax.spines["right"].set_color("#000000")
        ax.spines["right"].set_linewidth(1.2)
    ax.tick_params(colors="#000000", which="both", labelcolor="#000000")
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")
    ax.title.set_color("#000000")


def save_dual_figure(fig: plt.Figure, output_dir: Path | str, stem: str | None = None, dpi: int = 600) -> list[str]:
    """Save figure as both 600 DPI PNG and vector PDF."""
    out_p = Path(output_dir)
    if stem is None:
        stem = out_p.stem
        out_p = out_p.parent
    out_p.mkdir(parents=True, exist_ok=True)
    png_path = out_p / f"{stem}.png"
    pdf_path = out_p / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def auto_contrast_text_color(value: float, max_val: float = 100.0, min_val: float = 0.0) -> str:
    rng = max_val - min_val
    if rng <= 0:
        return "#000000"
    norm = (value - min_val) / rng
    return "#FFFFFF" if norm > 0.55 else "#000000"


def row_percent(values: np.ndarray) -> np.ndarray:
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values, dtype=float), where=row_sums > 0) * 100.0
