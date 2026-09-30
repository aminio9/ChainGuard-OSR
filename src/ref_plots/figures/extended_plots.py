"""Extended publication analysis figures 27–31 for FedTROS-PR Q1 submission.

Every single figure is generated strictly from measured experiment data exported
from FedTROS runs; zero synthetic/fallback data is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.ref_plots.config import FIGURES_OUTPUT_DIR, TABLES_OUTPUT_DIR, PROCESSED_DATA_DIR
from src.ref_plots.data.loader import resolve_data_path
from src.ref_plots.style import (
    CATEGORY_COLORS,
    COLORS,
    HATCHES,
    HEATMAP_CMAP,
    METHOD_STYLES,
    apply_academic_spines,
    configure_academic_style,
    save_figure,
)


def plot_27_multidataset_generalization(data_dir: Path, out: Path) -> None:
    """Figure 27: Cross-Dataset Benchmark Comparison across multi-domain datasets."""
    path = resolve_data_path("multidataset_benchmark.csv", data_dir)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("multidataset_benchmark.csv contains no rows.")

    df["method"] = df["method"].replace({
        "FedTROS": "Proposed", "FedTROS-MC": "Proposed", "FedTROS-PR": "Proposed"
    })
    datasets = sorted(df["dataset"].unique())
    methods = [m for m in ["FedAvg", "FedProx", "Proposed"] if m in set(df["method"].unique())] or sorted(df["method"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 2.5 * len(datasets)), 4.8), layout="constrained")
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)

    # Panel A: Closed-Set Macro F1 (%)
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset").reindex(datasets)
        st = METHOD_STYLES.get(method, METHOD_STYLES["Proposed"])
        pos = x + (i - (len(methods) - 1) / 2) * width
        vals = pd.to_numeric(sub["macro_f1"], errors="coerce").fillna(0.0).to_numpy(float)
        bars = axes[0].bar(
            pos,
            vals,
            width=width,
            label=st["label"],
            color=st["color"],
            edgecolor="#000000",
            linewidth=0.9,
            hatch=st["hatch"],
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    weight="bold",
                    color="#000000",
                )

    axes[0].set_title("(a) Macro-F1", fontsize=12, weight="bold")
    axes[0].set_xticks(x, datasets)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[0].set_ylim(0, 105)
    apply_academic_spines(axes[0])

    # Panel B: Open-Set Detection AUROC (%)
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset").reindex(datasets)
        st = METHOD_STYLES.get(method, METHOD_STYLES["Proposed"])
        pos = x + (i - (len(methods) - 1) / 2) * width
        vals = pd.to_numeric(sub["auroc"], errors="coerce").fillna(0.0).to_numpy(float)
        bars = axes[1].bar(
            pos,
            vals,
            width=width,
            label=st["label"],
            color=st["color"],
            edgecolor="#000000",
            linewidth=0.9,
            hatch=st["hatch"],
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    weight="bold",
                    color="#000000",
                )

    axes[1].set_title("(b) Open-Set AUROC", fontsize=12, weight="bold")
    axes[1].set_xticks(x, datasets)
    axes[1].set_ylabel("AUROC (%)")
    axes[1].set_ylim(0, 105)
    apply_academic_spines(axes[1])

    # Shared Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=max(len(methods), 1),
        framealpha=0.95,
    )
    save_figure(fig, out, "27_multidataset_generalization")


def plot_28_leave_one_attack_out_breakdown(data_dir: Path, out: Path) -> None:
    """Figure 28: Leave-One-Attack-Out (LOAO) Per-Attack Detection Breakdown."""
    path = resolve_data_path("loao_breakdown.csv", data_dir)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("loao_breakdown.csv contains no rows.")

    attacks = df["attack"].tolist()
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(attacks)), 5.2), layout="constrained")
    y = np.arange(len(attacks))
    height = 0.26

    # 1. Unknown Recall
    bars1 = ax.barh(
        y - height,
        df["unknown_recall"],
        height=height,
        label="Unknown Recall",
        color=COLORS["Unknown"],
        edgecolor="#000000",
        linewidth=0.8,
        hatch="///",
    )
    # 2. Known Retained Accuracy
    bars2 = ax.barh(
        y,
        df["known_acc"],
        height=height,
        label="Known Accuracy",
        color=COLORS["Known"],
        edgecolor="#000000",
        linewidth=0.8,
        hatch="...",
    )
    # 3. False Unknown Alarm Rate
    bars3 = ax.barh(
        y + height,
        df["false_unknown"],
        height=height,
        label="Known False Unknown Rate",
        color=COLORS["Error"],
        edgecolor="#000000",
        linewidth=0.8,
        hatch="xxx",
    )

    for b in bars1:
        ax.text(b.get_width() + 1.0, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8.0, color="#000000", weight="bold")
    for b in bars2:
        ax.text(b.get_width() + 1.0, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8.0, color="#000000", weight="bold")
    for b in bars3:
        ax.text(b.get_width() + 1.0, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8.0, color="#000000", weight="bold")

    ax.set_yticks(y, attacks)
    ax.invert_yaxis()
    ax.set_xlabel("Rate / Percentage (%)")
    ax.set_xlim(0, 114)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), framealpha=0.95, fontsize=9.2)
    apply_academic_spines(ax)
    save_figure(fig, out, "28_leave_one_attack_out_breakdown")


def plot_29_training_dynamics_gating_temperature(data_dir: Path, out: Path) -> None:
    """Figure 29: Dynamic regularization evolution over rounds."""
    path = resolve_data_path("training_dynamics.csv", data_dir)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("training_dynamics.csv contains no rows.")

    rounds = df["round"].to_numpy()
    if "loss_bla" in df.columns and "loss_vib" in df.columns:
        y1 = df["loss_bla"].to_numpy()
        y2 = df["loss_vib"].to_numpy()
        lbl1 = r"Balanced Logit-Adjusted Loss ($\mathcal{L}_{\mathrm{BLA}}$)"
        lbl2 = r"VIB Information Bottleneck ($\mathcal{L}_{\mathrm{VIB}}$)"
        yl1 = r"Logit-Adjusted Classification Loss ($\mathcal{L}_{\mathrm{BLA}}$)"
        yl2 = r"KL Divergence Regularization ($\mathcal{L}_{\mathrm{VIB}}$)"
        title_str = "Phase I Training Loss & Information Bottleneck Dynamics"
    else:
        y1 = df["disagreement"].to_numpy()
        y2 = df["temperature"].to_numpy()
        lbl1 = "Disagreement Gate (1 - A)"
        lbl2 = "KD Temperature (τ_t)"
        yl1 = "Teacher–Student Disagreement (1 - A)"
        yl2 = "Distillation Temperature (τ)"
        title_str = "Distillation Gating & Temperature Cooling Dynamics"

    fig, ax1 = plt.subplots(figsize=(7.8, 4.6), layout="constrained")
    ax1_twin = ax1.twinx()

    l1 = ax1.plot(
        rounds,
        y1,
        color=COLORS["Proposed"],
        linewidth=2.4,
        marker="o",
        markevery=max(1, len(rounds) // 8),
        label=lbl1,
    )
    l2 = ax1_twin.plot(
        rounds,
        y2,
        color=COLORS["FedProx"],
        linestyle="--",
        linewidth=2.4,
        marker="s",
        markevery=max(1, len(rounds) // 8),
        label=lbl2,
    )

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel(yl1, color="#000000", weight="bold")
    ax1_twin.set_ylabel(yl2, color="#000000", weight="bold")
    ax1.tick_params(axis="y", labelcolor="#000000")
    ax1_twin.tick_params(axis="y", labelcolor="#000000")
    ax1.set_title(title_str, fontsize=11.5, weight="bold", color="#000000")
    apply_academic_spines(ax1)
    apply_academic_spines(ax1_twin, right=True)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.95)

    save_figure(fig, out, "29_training_dynamics_gating_temperature")


def plot_30_hyperparameter_sensitivity(data_dir: Path, out: Path) -> None:
    """Figure 30: Hyperparameter Sensitivity & Stability across variants."""
    path = resolve_data_path("hyperparameter_sensitivity.csv", data_dir)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("hyperparameter_sensitivity.csv contains no rows.")

    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    x = np.arange(len(df))
    width = 0.35

    mf1 = pd.to_numeric(df["macro_f1"], errors="coerce").fillna(0.0).to_numpy()
    auc = pd.to_numeric(df["auroc"], errors="coerce").fillna(0.0).to_numpy()

    b1 = ax.bar(x - width/2, mf1, width=width, label="Macro-F1 (%)", color=COLORS["Overall"], edgecolor="#000000", linewidth=0.8, hatch="///")
    b2 = ax.bar(x + width/2, auc, width=width, label="AUROC (%)", color=COLORS["Proposed"], edgecolor="#000000", linewidth=0.8, hatch="xxx")

    for b in b1:
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.2, weight="bold")
    for b in b2:
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.2, weight="bold")

    ax.set_xticks(x, df["variant"], rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Hyperparameter Sensitivity Benchmark ($S_1$)", fontsize=12.0, weight="bold")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.2)
    apply_academic_spines(ax)

    save_figure(fig, out, "30_hyperparameter_sensitivity")


def plot_31_client_forgetting_mitigation(data_dir: Path, out: Path) -> None:
    """Figure 31: Catastrophic Forgetting Mitigation on Incomplete Non-IID Clients."""
    path = resolve_data_path("client_forgetting_mitigation.csv", data_dir)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("client_forgetting_mitigation.csv contains no rows.")

    methods = [m.replace("FedTROS", "Proposed").replace("FedPROTEUS", "Proposed") for m in df["method"].tolist()]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
    x = np.arange(len(methods))
    width = 0.35

    pres = pd.to_numeric(df["present_accuracy"], errors="coerce").fillna(0.0).to_numpy()
    absent = pd.to_numeric(df["absent_accuracy"], errors="coerce").fillna(0.0).to_numpy()

    b1 = ax.bar(x - width/2, pres, width=width, label="Locally Present Classes", color=COLORS["Known"], edgecolor="#000000", linewidth=0.8, hatch="...")
    b2 = ax.bar(x + width/2, absent, width=width, label="Locally Absent Classes", color=COLORS["Error"], edgecolor="#000000", linewidth=0.8, hatch="xxx")

    for b in b1:
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")
    for b in b2:
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")

    ax.set_xticks(x, methods)
    ax.set_ylabel("Client Evaluation Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Non-IID Absent-Class Forgetting Mitigation ($q_i \\leq 0.40$)", fontsize=12.0, weight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.2)
    apply_academic_spines(ax)
    save_figure(fig, out, "31_client_forgetting_mitigation")


def generate_extended_plots(
    data_dir: Path | None = None,
    figures_out: Path | None = None,
    tables_out: Path | None = None,
) -> None:
    """Generate all 5 extended Q1 publication figures (27-31) strictly from measured data."""
    configure_academic_style()
    d_dir = Path(data_dir or PROCESSED_DATA_DIR)
    f_out = Path(figures_out or FIGURES_OUTPUT_DIR)
    t_out = Path(tables_out or TABLES_OUTPUT_DIR)
    f_out.mkdir(parents=True, exist_ok=True)
    t_out.mkdir(parents=True, exist_ok=True)

    plot_27_multidataset_generalization(d_dir, f_out)
    plot_28_leave_one_attack_out_breakdown(d_dir, f_out)
    plot_29_training_dynamics_gating_temperature(d_dir, f_out)
    plot_30_hyperparameter_sensitivity(d_dir, f_out)
    plot_31_client_forgetting_mitigation(d_dir, f_out)
