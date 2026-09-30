"""Unified Publication Plotting and Styling Utilities for Academic Journals (FedTROS)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Professional Publication Color Palettes
# ---------------------------------------------------------------------------

COLORS = {
    "Proposed": "#7B1FA2",      # Deep Royal Purple (Proposed Method in paper)
    "FedTROS-MC": "#7B1FA2",    # Canonical code alias
    "FedTROS-PR": "#7B1FA2",    # Legacy prototype-rank alias
    "FedTROS": "#7B1FA2",       # General project alias
    "FedAvg": "#1976D2",     # Ocean Blue
    "FedProx": "#E64A19",    # Deep Amber/Coral
    "Known": "#2E7D32",      # Forest Green
    "Unknown": "#C2185B",    # Vivid Crimson
    "Overall": "#00897B",    # Deep Teal
    "Error": "#D32F2F",      # Cardinal Red
    "Neutral": "#546E7A",    # Slate Blue-Gray
    "Light": "#ECEFF1",      # Pale Ice Gray
    "Before": "#607D8B",     # Slate Gray
    "After": "#7B1FA2",      # Royal Purple
}

# Categorical Color Palette for Multi-group distributions
CATEGORY_COLORS = [
    "#00897B",  # Teal
    "#1976D2",  # Ocean Blue
    "#F57C00",  # Amber
    "#C2185B",  # Crimson
    "#7B1FA2",  # Purple
    "#2E7D32",  # Green
    "#0288D1",  # Sky Blue
    "#D84315",  # Coral
]

# Standard distinguishable hatch patterns for black-and-white print compatibility
HATCHES = ["///", "\\\\\\", "xxx", "...", "+++", "---", "OOO", "***"]

# Method-specific signature styles
METHOD_STYLES = {
    "Proposed": {
        "color": "#7B1FA2",
        "marker": "o",
        "linestyle": "-",
        "hatch": "///",
        "label": "Proposed",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedTROS-MC": {
        "color": "#7B1FA2",
        "marker": "o",
        "linestyle": "-",
        "hatch": "///",
        "label": "Proposed",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedTROS-PR": {
        "color": "#7B1FA2",
        "marker": "o",
        "linestyle": "-",
        "hatch": "///",
        "label": "Proposed",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedAvg-Student": {
        "color": "#1976D2",
        "marker": "s",
        "linestyle": "--",
        "hatch": "\\\\\\",
        "label": "FedAvg",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    "FedProx-Student": {
        "color": "#E64A19",
        "marker": "^",
        "linestyle": "-.",
        "hatch": "xxx",
        "label": "FedProx",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
}

# The established renderer uses concise paper labels, whereas the canonical
# experiment store uses architecture-qualified method names.  Both resolve to
# one visual identity so old plot modules and new exports remain compatible.
METHOD_STYLES["FedTROS"] = {**METHOD_STYLES["Proposed"], "label": "Proposed"}
METHOD_STYLES["FedAvg"] = {**METHOD_STYLES["FedAvg-Student"], "label": "FedAvg"}
METHOD_STYLES["FedProx"] = {**METHOD_STYLES["FedProx-Student"], "label": "FedProx"}

# Client-scale styles (10, 50, 100 clients)
CLIENT_STYLES = {
    10: {
        "color": "#00897B",
        "marker": "o",
        "linestyle": "-",
        "hatch": "///",
        "label": "10 clients",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    50: {
        "color": "#1976D2",
        "marker": "s",
        "linestyle": "--",
        "hatch": "\\\\\\",
        "label": "50 clients",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
    100: {
        "color": "#C2185B",
        "marker": "^",
        "linestyle": "-.",
        "hatch": "xxx",
        "label": "100 clients",
        "linewidth": 2.4,
        "markersize": 7.5,
    },
}

# Sequential colormap with monotonic luminance progression
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "academic_sequential",
    ["#FFFFFF", "#E1F5FE", "#4FC3F7", "#0288D1", "#1565C0", "#0D47A1", "#051B38"],
    N=256,
)

# Diverging colormap for delta matrices
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "academic_diverging",
    ["#0D47A1", "#1976D2", "#BBDEFB", "#FFFFFF", "#FFCDD2", "#E53935", "#B71C1C"],
    N=256,
)


# ---------------------------------------------------------------------------
# Global Matplotlib Publication Style
# ---------------------------------------------------------------------------

def configure_academic_style() -> None:
    """Set global rcParams for Q1 journal-grade 600 DPI plotting with deep true-black typography."""
    plt.rcParams.update(
        {
            # Typography: true deep black with crisp sans-serif and matching math
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Nimbus Sans", "Liberation Sans"],
            "font.size": 11.0,
            "font.weight": "medium",
            "text.color": "#000000",
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            # Axes and Spines: solid black edges and bold labels
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
            # Ticks: solid black outward ticks
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
            # Grid: subtle light gray behind data
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.70,
            # Legend: sharp black border and pure black labels
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#000000",
            "legend.fancybox": False,
            "legend.fontsize": 10.0,
            "legend.title_fontsize": 10.5,
            "legend.labelcolor": "#000000",
            # Lines and Markers
            "lines.linewidth": 2.3,
            "lines.markersize": 7.0,
            "lines.markeredgewidth": 1.0,
            "lines.solid_capstyle": "round",
            # Figure and Export defaults
            "figure.facecolor": "#FFFFFF",
            "figure.edgecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


# ---------------------------------------------------------------------------
# Spines & Frame Utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PNG-Only 600 DPI Exporter
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = 600) -> Path:
    """Save as a high-resolution 600 DPI PNG and vector PDF in publication color."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Matrix & Auto-Contrast Utilities
# ---------------------------------------------------------------------------

def auto_contrast_text_color(value: float, max_val: float, min_val: float = 0.0) -> str:
    """Return '#FFFFFF' or '#000000' based on normalized background intensity."""
    rng = max_val - min_val
    if rng <= 0:
        return "#000000"
    norm = (value - min_val) / rng
    return "#FFFFFF" if norm > 0.55 else "#000000"


def row_percent(values: np.ndarray) -> np.ndarray:
    """Row-normalize a 2D confusion matrix into percentages summing to 100%."""
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values, dtype=float), where=row_sums > 0) * 100.0


def style_axes(ax: plt.Axes, top: bool = False, right: bool = False) -> None:
    """Apply academic spines and tick styling to an Axes instance."""
    apply_academic_spines(ax, top=top, right=right)


def apply_grid(ax: plt.Axes, axis: str = "both") -> None:
    """Enable subtle dashed academic grid on the specified axis."""
    ax.grid(True, axis=axis, color="#E5E5E5", linestyle="--", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)


def add_bar_labels(ax: plt.Axes, bars: Any, fmt: str = "{:.1f}%", fontsize: float = 8.5, dy: float = 3.0) -> None:
    """Annotate bar tops with formatted numeric values."""
    for bar in bars:
        h = float(bar.get_height())
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color="#000000",
        )


def add_Crosshair(ax: plt.Axes, x_val: float, y_val: float, label: str, color: str = "#7B1FA2") -> None:
    """Draw operating point crosshair and callout annotation."""
    ax.axvline(x_val, color=color, linestyle=":", linewidth=1.4, alpha=0.8)
    ax.axhline(y_val, color=color, linestyle=":", linewidth=1.4, alpha=0.8)
    ax.scatter([x_val], [y_val], s=110, color=color, edgecolors="black", linewidths=1.2, zorder=6)
    ax.annotate(
        label,
        xy=(x_val, y_val),
        xytext=(max(5.0, x_val - 36.0), max(32.0, y_val - 22.0)),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=color),
        fontsize=9.0,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, lw=1.2),
        zorder=7,
    )


def save_academic_figure(fig: plt.Figure, stem_path: Path, dpi: int = 600, close: bool = True) -> None:
    """Save figure as both 600 DPI PNG and vector PDF at stem_path."""
    stem_path = Path(stem_path)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(stem_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06, facecolor="white")
    if close:
        plt.close(fig)

