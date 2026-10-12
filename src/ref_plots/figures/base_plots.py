"""Base figures 01–15 for FedTROS publication.

Legacy method aliases in this module are import-only compatibility for archived
pre-bundle result files; canonical publication rendering uses FedTROS-PR names.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle, Patch
from matplotlib.ticker import PercentFormatter

from src.ref_plots.config import BASE_PLOT_STEMS, DEFAULT_RESULTS_JSON, FIGURES_OUTPUT_DIR
from src.ref_plots.data.loader import (
    load_confusion_matrix,
    load_json_data,
    load_score_arrays,
    resolve_data_path,
    using_data_dir,
)
from src.ref_plots.style import (
    CATEGORY_COLORS,
    COLORS,
    HATCHES,
    HEATMAP_CMAP,
    METHOD_STYLES,
    apply_academic_spines,
    auto_contrast_text_color,
    configure_academic_style,
    row_percent,
    save_figure,
)


def method_style(method: str) -> dict[str, Any]:
    # Normalize legacy aliases if encountered
    if method in ("FedPROTEUS", "DKD-FedOS", "dkd_fedos", "FedTROS", "FedTROS-MC", "FedTROS-PR"):
        method = "Proposed"
    return METHOD_STYLES.get(
        method,
        {
            "color": COLORS["Neutral"],
            "marker": "D",
            "linestyle": ":",
            "hatch": "...",
            "label": method,
            "linewidth": 2.4,
            "markersize": 7.5,
        },
    )


def convergence_rows(block: dict[str, Any]) -> pd.DataFrame:
    replace_map = {
        "FedPROTEUS": "Proposed",
        "DKD-FedOS": "Proposed",
        "FedTROS": "Proposed",
        "FedTROS-MC": "Proposed",
        "FedTROS-PR": "Proposed",
    }
    if block.get("data_file"):
        path = resolve_data_path(block["data_file"])
        df = pd.read_csv(path)
        df["method"] = df["method"].replace(replace_map)
        return df
    if block.get("rows"):
        df = pd.DataFrame(block["rows"])
        df["method"] = df["method"].replace(replace_map)
        return df
    raise FileNotFoundError("Convergence plot requires real measured data_file or rows; synthetic generation is disabled")


def score_arrays(block: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if block.get("data_file"):
        path = resolve_data_path(block["data_file"])
        return load_score_arrays(path)
    if block.get("known_scores") and block.get("unknown_scores"):
        return np.asarray(block["known_scores"], float), np.asarray(block["unknown_scores"], float)
    raise FileNotFoundError("Canonical publication score distribution requires measured test_scores.csv with nonconformity_score")


def roc_points(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.r_[0, tp / max(int(y.sum()), 1), 1]
    fpr = np.r_[0, fp / max(int((y == 0).sum()), 1), 1]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def pr_points(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = np.r_[1, tp / np.maximum(tp + fp, 1)]
    recall = np.r_[0, tp / max(int(y.sum()), 1)]
    return recall, precision, float(np.trapezoid(precision, recall))


# ---------------------------------------------------------------------------
# Individual Clean Publication Plots
# ---------------------------------------------------------------------------

def plot_01_scalability(block: dict[str, Any], out: Path) -> None:
    df = pd.DataFrame(block["rows"]).sort_values("clients")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    specs = [
        ("known_accuracy", "Known Accuracy", COLORS["Known"], "o", "-", 8.0),
        ("overall_accuracy", "Overall Accuracy", COLORS["FedAvg"], "s", "--", 7.5),
        ("auroc", "Open-Set AUROC", COLORS["Proposed"], "^", "-.", 8.0),
        ("unknown_f1", "Unknown F1", COLORS["Unknown"], "D", ":", 7.0),
    ]
    for key, label, color, marker, ls, ms in specs:
        ax.plot(
            df["clients"],
            df[key],
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.4,
            markersize=ms,
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            label=label,
        )
        for x, y in zip(df["clients"], df[key]):
            if y is None or (isinstance(y, float) and not np.isfinite(y)):
                continue
            ax.annotate(
                f"{y:.1f}%",
                (x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8.8,
                weight="semibold",
                color="#000000",
            )
    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("Performance (%)")
    ax.set_xticks(df["clients"])
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, loc="lower left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_01_scalability"][0])


def plot_02_non_iid_distribution(block: dict[str, Any], out: Path) -> None:
    if block.get("data_file"):
        path = resolve_data_path(block["data_file"])
        df = pd.read_csv(path)
    elif block.get("rows"):
        df = pd.read_csv(resolve_data_path(block["data_file"])) if block.get("data_file") else pd.DataFrame(block["rows"])
    else:
        raise FileNotFoundError("Non-IID distribution plot requires real measured client_distribution CSV")

    pivot = df.pivot_table(index="client", columns="traffic_group", values="proportion", fill_value=0)
    pivot = pivot.reindex(sorted(pivot.index, key=lambda value: int(str(value).split()[-1])))
    ordered_groups = [group for group in block.get("groups", []) if group in pivot.columns]
    pivot = pivot.reindex(columns=ordered_groups or list(pivot.columns), fill_value=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    left = np.zeros(len(pivot))
    
    for i, group in enumerate(pivot.columns):
        values = pivot[group].to_numpy()
        fill_color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        hatch = HATCHES[i % len(HATCHES)]
        ax.barh(
            np.arange(len(pivot)),
            values,
            left=left,
            color=fill_color,
            edgecolor="#000000",
            linewidth=0.9,
            hatch=hatch,
            label=group,
        )
        left += values

    ax.set_yticks(np.arange(len(pivot)), pivot.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Local Traffic Mixture Proportion")
    ax.set_ylabel("Federated Client")
    ax.legend(ncol=min(len(pivot.columns), 5), loc="upper center", bbox_to_anchor=(0.5, -0.11), framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_02_non_iid_distribution"][0])


def plot_02_partition_comparison(block: dict[str, Any], out: Path) -> None:
    path = resolve_data_path(block["comparison_data_file"])
    df = pd.read_csv(path)
    settings = (
        df[["setting", "setting_order", "panel_title"]]
        .drop_duplicates()
        .sort_values("setting_order")
    )
    unique_groups = list(dict.fromkeys(df["traffic_group"]))
    preset_groups = block.get("groups", [])
    groups = [group for group in preset_groups if group in unique_groups] + [group for group in unique_groups if group not in preset_groups]

    client_counts = df.groupby("setting")["client_id"].nunique()
    max_clients = int(client_counts.max()) if not client_counts.empty else 1
    fig_height = max(5.8, min(10.5, 4.4 + 0.32 * max_clients))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, fig_height),
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    legend_handles = []

    for ax, setting in zip(axes.flat, settings.itertuples(index=False)):
        subset = df[df["setting"] == setting.setting]
        pivot = subset.pivot_table(index="client_id", columns="traffic_group", values="proportion", fill_value=0)
        pivot = pivot.sort_index()
        pivot = pivot.reindex(columns=groups, fill_value=0)
        left = np.zeros(len(pivot))

        for index, group in enumerate(groups):
            fill_color = CATEGORY_COLORS[index % len(CATEGORY_COLORS)]
            hatch = HATCHES[index % len(HATCHES)]
            bars = ax.barh(
                np.arange(len(pivot)),
                pivot[group].to_numpy(float),
                left=left,
                color=fill_color,
                edgecolor="#000000",
                linewidth=0.75,
                hatch=hatch,
                label=group,
            )
            if len(legend_handles) < len(groups):
                legend_handles.append(bars[0])
            left += pivot[group].to_numpy(float)

        regime_label = str(setting.panel_title).replace("alpha", "\u03b1")
        ax.set_title(
            regime_label,
            fontsize=11.0,
            weight="bold",
            color="#000000",
            pad=6,
        )
        ax.set_yticks(np.arange(len(pivot)), [f"Client {client}" for client in pivot.index])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        apply_academic_spines(ax)

    for ax in axes.flat[len(settings):]:
        ax.set_visible(False)

    fig.supxlabel("Local Traffic Mixture Proportion", fontsize=11.5)
    fig.supylabel("Federated Client", fontsize=11.5)
    fig.legend(
        legend_handles,
        groups,
        ncol=len(groups),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        framealpha=0.95,
    )
    save_figure(fig, out, BASE_PLOT_STEMS["plot_02_non_iid_distribution"][1])


def _plot_convergence(block: dict[str, Any], out: Path, stem: str) -> None:
    df = convergence_rows(block)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")

    plotted_values: list[float] = []
    for method_index, method in enumerate(("FedAvg", "FedProx", "Proposed")):
        group = df[df["method"] == method].sort_values("round")
        if group.empty:
            continue
        st = method_style(method)
        x = group["round"].to_numpy(float)
        y_column = "accuracy_percent" if "accuracy_percent" in group.columns else "macro_f1_percent"
        y = group[y_column].to_numpy(float)
        plotted_values.extend(y[np.isfinite(y)].tolist())
        band = group.get("band", pd.Series(np.zeros(len(group)))).to_numpy(float)

        if np.any(band > 0):
            ax.fill_between(
                x,
                y - band,
                y + band,
                color=st["color"],
                alpha=0.15,
                linewidth=0.5,
                edgecolor=st["color"],
            )

        ax.plot(
            x,
            y,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            marker=st["marker"],
            markevery=max(len(group) // 8, 1),
            markersize=st["markersize"],
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            label=st["label"],
        )
        ax.annotate(
            f"{y[-1]:.1f}%",
            (x[-1], y[-1]),
            xytext=(6, (-10, 0, 10)[method_index]),
            textcoords="offset points",
            va="center",
            fontsize=9.0,
            weight="bold",
            color="#000000",
        )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel(f"{block.get('metric_label', 'Validation Accuracy')} (%)")
    if plotted_values:
        low = min(plotted_values)
        high = max(plotted_values)
        margin = max(2.0, 0.12 * max(high - low, 1.0))
        ax.set_ylim(max(0.0, low - margin), min(102.0, high + margin))
    ax.legend(loc="lower right", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, stem)


def plot_03_convergence_mild(block: dict[str, Any], out: Path) -> None:
    _plot_convergence(block, out, BASE_PLOT_STEMS["plot_03_convergence_mild"][0])


def plot_04_convergence_hard(block: dict[str, Any], out: Path) -> None:
    _plot_convergence(block, out, BASE_PLOT_STEMS["plot_04_convergence_hard"][0])


def plot_05_mahalanobis_nonconformity_distribution(block: dict[str, Any], out: Path) -> None:
    known, unknown = score_arrays(block)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    combined = np.concatenate([known, unknown])
    if combined.size == 0:
        raise ValueError("No nonconformity scores available")
    lo = float(max(0.0, np.nanmin(combined)))
    p995 = float(np.nanpercentile(combined, 99.5))
    raw_hi = float(np.nanmax(combined))
    hi = p995 * 1.15 if raw_hi > p995 * 1.8 else raw_hi * 1.05
    if hi <= lo:
        hi = lo + 1.0
    known_clipped = known[known <= hi]
    unknown_clipped = unknown[unknown <= hi]
    bins = np.linspace(lo, hi, 36)

    ax.hist(
        known_clipped,
        bins=bins,
        density=True,
        color=COLORS["FedAvg"],
        hatch="///",
        edgecolor="#000000",
        linewidth=0.9,
        alpha=0.75,
        label="Known Test ($\\mathcal{C}_K$)",
    )
    ax.hist(
        unknown_clipped,
        bins=bins,
        density=True,
        color=COLORS["Unknown"],
        hatch="\\\\\\",
        edgecolor="#000000",
        linewidth=0.9,
        alpha=0.75,
        label="Unknown Zero-Day ($\\mathcal{C}_U$)",
    )

    calibration_ref = block.get("calibration_file") or block.get("data_file")
    if calibration_ref:
        score_path = resolve_data_path(calibration_ref)
        score_frame = pd.read_csv(score_path)
        if "tau_alpha" not in score_frame:
            raise ValueError("Conformal calibration artifact must contain tau_alpha")
        threshold = float(pd.to_numeric(score_frame["tau_alpha"], errors="coerce").dropna().median())
    else:
        threshold = float(block["threshold"])

    ax.axvline(
        threshold,
        color="#000000",
        linestyle="--",
        linewidth=2.4,
        label=f"Conformal Cutoff ($\\epsilon=0.05, \\tau_{{0.05}}={threshold:.3f}$)",
    )
    ax.set_xlabel("Block-Diagonal Mahalanobis Nonconformity Score $S(x)$")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(lo, hi)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), framealpha=0.95, fontsize=9.0)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_05_score_distribution"][0])
    save_figure(fig, out, "05_prototype_rank_score_distribution")


def plot_06_openness(block: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    if block.get("rows"):
        rows = pd.DataFrame(block["rows"])
        rows["method"] = rows["method"].replace({
            "FedTROS-PR": "Proposed", "FedTROS": "Proposed", "FedTROS-MC": "Proposed",
            "FedAvg-Student": "FedAvg", "FedProx-Student": "FedProx"
        })
        rows = rows.groupby(["openness", "method"], as_index=False)["auroc"].mean()
        for method, group in rows.groupby("method", sort=True):
            group = group.sort_values("openness")
            style = method_style(str(method))
            ax.plot(
                group["openness"], group["auroc"], color=style["color"],
                marker=style["marker"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], markersize=style["markersize"],
                markeredgecolor="#000000", markeredgewidth=0.8, label=style["label"],
            )
    else:
        x = np.asarray(block["openness"], float)
        fedtros_auroc = block.get("fedtros_auroc_percent", block.get("fedproteus_auroc_percent"))
        ax.plot(x, fedtros_auroc, color=COLORS["Proposed"], marker="o", linestyle="-",
                linewidth=2.5, markersize=8.0, markeredgecolor="#000000",
                markeredgewidth=0.8, label="Proposed")
        ax.plot(x, block["baseline_auroc_percent"], color=COLORS["FedAvg"], marker="s",
                linestyle="--", linewidth=2.4, markersize=7.5,
                markeredgecolor="#000000", markeredgewidth=0.8, label="Open-Set Baseline")
    ax.set_xlabel("Openness Index")
    ax.set_ylabel("AUROC (%)")
    ax.set_ylim(55, 102)
    ax.legend(loc="lower left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_06_openness"][0])


def plot_07_roc(block: dict[str, Any], score_block: dict[str, Any], out: Path) -> None:
    if block.get("data_file"):
        path = resolve_data_path(block["data_file"])
        curve = pd.read_csv(path).dropna(subset=["fpr", "tpr"]).sort_values("fpr")
        fpr, tpr = curve["fpr"].to_numpy(float), curve["tpr"].to_numpy(float)
        auc = float(np.trapezoid(tpr, fpr))
    else:
        known, unknown = score_arrays(score_block)
        y = np.r_[np.zeros(len(known), int), np.ones(len(unknown), int)]
        fpr, tpr, auc = roc_points(y, np.r_[known, unknown])

    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.plot(
        fpr,
        tpr,
        color=COLORS["Proposed"],
        linewidth=2.8,
        label=f"Proposed Method ROC (AUC = {auc:.3f})",
    )
    ax.fill_between(fpr, tpr, color=COLORS["Proposed"], alpha=0.12)
    ax.plot([0, 1], [0, 1], color="#000000", linestyle=":", linewidth=1.6, label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_07_roc"][0])


def plot_08_pr(block: dict[str, Any], score_block: dict[str, Any], out: Path) -> None:
    known, unknown = score_arrays(score_block) if score_block else (np.array([]), np.array([]))
    if block.get("data_file"):
        path = resolve_data_path(block["data_file"])
        curve = pd.read_csv(path).dropna(subset=["recall", "precision"]).sort_values("recall")
        recall, precision = curve["recall"].to_numpy(float), curve["precision"].to_numpy(float)
        auprc = float(np.sum(np.diff(recall) * precision[1:]))
        total_samples = len(known) + len(unknown)
        prevalence = (len(unknown) / total_samples) if total_samples > 0 else 0.0
    else:
        y = np.r_[np.zeros(len(known), int), np.ones(len(unknown), int)]
        recall, precision, auprc = pr_points(y, np.r_[known, unknown])
        prevalence = float(y.mean())

    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.plot(
        recall,
        precision,
        color=COLORS["Unknown"],
        linewidth=2.8,
        label=f"Proposed Method Precision-Recall (AUPRC = {auprc:.3f})",
    )
    ax.fill_between(recall, precision, color=COLORS["Unknown"], alpha=0.12)
    ax.axhline(
        prevalence,
        color="#000000",
        linestyle=":",
        linewidth=1.6,
        label=f"Prevalence Baseline ({prevalence:.1%})",
    )
    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_08_pr"][0])


def plot_09_10_confusion(block: dict[str, Any], out: Path) -> None:
    def load_matrix(key: str) -> tuple[np.ndarray, list[str]]:
        path = resolve_data_path(block[key])
        frame = pd.read_csv(path)
        label_column = frame.columns[0]
        labels = frame[label_column].astype(str).tolist()
        values = frame.drop(columns=[label_column]).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        return values, labels

    if block.get("before_data_file") and block.get("after_data_file"):
        before, labels = load_matrix("before_data_file")
        after, after_labels = load_matrix("after_data_file")
    else:
        after = np.asarray(block["matrix"], float)
        before = after.copy()
        labels = list(block.get("labels", [str(i) for i in range(after.shape[0])]))
    column_labels = labels
    maximum = max(float(before.max()), float(after.max()), 1.0)

    def draw_single(values: np.ndarray, stem: str) -> None:
        fig, ax = plt.subplots(figsize=(6.8, 5.5), layout="constrained")
        image = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=maximum)
        ax.set_xticks(range(len(column_labels)), column_labels, rotation=25, ha="right", fontsize=10.0)
        ax.set_yticks(range(len(labels)), labels, fontsize=10.0)
        ax.set_xlabel("Predicted Class", fontsize=11.5, weight="medium")
        ax.set_ylabel("True Class", fontsize=11.5, weight="medium")

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                txt_color = auto_contrast_text_color(val, maximum)
                ax.text(
                    j,
                    i,
                    f"{int(val):,}",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    weight="bold",
                    color=txt_color,
                )
        cbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.03)
        cbar.set_label("Sample Count", fontsize=10.5, color="#000000", weight="bold")
        cbar.outline.set_edgecolor("#000000")
        cbar.outline.set_linewidth(1.0)
        cbar.ax.tick_params(colors="#000000")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#000000")
            spine.set_linewidth(1.0)
        save_figure(fig, out, stem)

    draw_single(before, BASE_PLOT_STEMS["plot_09_10_confusion"][0])
    draw_single(after, BASE_PLOT_STEMS["plot_09_10_confusion"][1])

    # Side-by-side count comparison
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), layout="constrained", sharey=True)
    for ax, values, panel_label in zip(axes, (before, after), ("(a) Before Rejection", "(b) After Rejection")):
        image = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=maximum)
        ax.set_xticks(range(len(column_labels)), column_labels, rotation=25, ha="right", fontsize=9.8)
        ax.set_yticks(range(len(labels)), labels, fontsize=9.8)
        ax.set_xlabel("Predicted Class", fontsize=11.0)
        ax.set_title(panel_label, fontsize=12.0, pad=8, weight="bold")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                txt_color = auto_contrast_text_color(val, maximum)
                ax.text(
                    j,
                    i,
                    f"{int(val):,}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    weight="bold",
                    color=txt_color,
                )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#000000")
            spine.set_linewidth(1.0)
    axes[0].set_ylabel("True Class", fontsize=11.0)
    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Sample Count", fontsize=10.5, color="#000000", weight="bold")
    cbar.outline.set_edgecolor("#000000")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(colors="#000000")
    save_figure(fig, out, BASE_PLOT_STEMS["plot_09_10_confusion"][2])

    # Side-by-side row-normalized percentage comparison
    before_pct = row_percent(before)
    after_pct = row_percent(after)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), layout="constrained", sharey=True)
    for ax, values, panel_label in zip(axes, (before_pct, after_pct), ("(a) Before Rejection", "(b) After Rejection")):
        image = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=100)
        ax.set_xticks(range(len(column_labels)), column_labels, rotation=25, ha="right", fontsize=9.8)
        ax.set_yticks(range(len(labels)), labels, fontsize=9.8)
        ax.set_xlabel("Predicted Class", fontsize=11.0)
        ax.set_title(panel_label, fontsize=12.0, pad=8, weight="bold")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                txt_color = auto_contrast_text_color(val, 100.0)
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    weight="bold",
                    color=txt_color,
                )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#000000")
            spine.set_linewidth(1.0)
    axes[0].set_ylabel("True Class", fontsize=11.0)
    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("True Class Proportion (%)", fontsize=10.5, color="#000000", weight="bold")
    cbar.outline.set_edgecolor("#000000")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(colors="#000000")
    save_figure(fig, out, BASE_PLOT_STEMS["plot_09_10_confusion"][3])


def plot_11_robustness(block: dict[str, Any], out: Path) -> None:
    df = pd.DataFrame(block["rows"])
    df["method"] = df["method"].replace({
        "FedPROTEUS": "Proposed", "DKD-FedOS": "Proposed", "FedTROS": "Proposed", "FedTROS-MC": "Proposed", "FedTROS-PR": "Proposed"
    })
    alphas = [0.1, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    
    all_y: list[float] = []
    for method_idx, method in enumerate(("FedAvg", "FedProx", "Proposed")):
        points = []
        for alpha in alphas:
            match = df[
                (df["method"] == method)
                & (df["alpha_requested"] == alpha)
                & (df["config_mismatch"] != True)
                & df["status"].str.startswith("available")
            ]
            if len(match) and pd.notna(match.iloc[-1]["accuracy"]):
                points.append((alpha, float(match.iloc[-1]["accuracy"])))
        st = method_style(method)
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        all_y.extend(y_values)
        ax.plot(
            x_values,
            y_values,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            markersize=st["markersize"],
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            label=st["label"],
        )
        y_offset = (-12, 6, 8)[method_idx]
        for x, y in points:
            ax.annotate(
                f"{y:.1f}%",
                (x, y),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.6,
                weight="bold",
                color="#000000",
            )

    ax.set_xlabel("Dirichlet Parameter α (Lower = Higher Non-IID)")
    ax.set_ylabel("Performance (%)")
    ax.set_xticks(alphas)
    low_y = max(45.0, min(all_y) - 6.0) if all_y else 65.0
    ax.set_ylim(low_y, 102)
    ax.legend(loc="lower right", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_11_robustness"][0])


def confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    covariance = np.cov(x, y)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(np.arctan2(*vectors[:, 0][::-1]))
    width, height = 4.4 * np.sqrt(np.maximum(values, 1e-9))
    ax.add_patch(
        Ellipse(
            (x.mean(), y.mean()),
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor="#000000",
            alpha=0.10,
            linewidth=1.0,
        )
    )


def latent_data(block: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if block.get("rows"):
        frame = pd.DataFrame(block["rows"])
        if "point_type" in frame and frame["point_type"].astype(str).str.contains("negative|boundary|synthetic", case=False, regex=True).any():
            raise ValueError("Canonical v2 latent geometry cannot contain negative/boundary/synthetic prototypes")
        if block.get("negative_prototypes") is not None and len(block.get("negative_prototypes")):
            raise ValueError("Canonical v2 latent geometry cannot contain negative/boundary prototypes")
        return frame, np.asarray(block.get("positive_prototypes", []), float), np.asarray(block.get("negative_prototypes", []), float)
    raise FileNotFoundError("Canonical latent geometry requires measured latent_projection.csv; synthetic generation is disabled")


def plot_12_latent_geometry(block: dict[str, Any], out: Path) -> None:
    if block.get("data_file"):
        path = resolve_data_path(block["data_file"])
        projected = pd.read_csv(path)
        df = projected[projected["point_type"] == "sample"].copy()
        df["group"] = df["label"]
        df["final_reject"] = df["final_reject"].astype(bool)
        positive_frame = projected[projected["point_type"] == "prototype"]
        if (projected["point_type"].astype(str).str.contains("negative|boundary|synthetic", case=False, regex=True)).any():
            raise ValueError("Canonical v2 latent projection contains prohibited boundary/negative prototypes")
        positive = positive_frame[["x", "y"]].to_numpy(float)
        negative = np.empty((0, 2))
    else:
        df, positive, negative = latent_data(block)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    known, unknown = df[df["is_unknown"] == 0], df[df["is_unknown"] == 1]
    
    if not block.get("data_file"):
        for _, group in known.groupby("group"):
            confidence_ellipse(ax, group["x"].to_numpy(), group["y"].to_numpy(), COLORS["FedAvg"])

    ax.scatter(
        known.loc[~known["final_reject"], "x"],
        known.loc[~known["final_reject"], "y"],
        s=24,
        alpha=0.55,
        color="#42A5F5",
        edgecolors="none",
        label="Known Accepted",
        rasterized=True,
    )
    ax.scatter(
        unknown.loc[unknown["final_reject"], "x"],
        unknown.loc[unknown["final_reject"], "y"],
        s=42,
        alpha=0.80,
        color=COLORS["Unknown"],
        marker="^",
        edgecolors="#000000",
        linewidths=0.6,
        label="Unknown Rejected",
        rasterized=True,
    )
    ax.scatter(
        unknown.loc[~unknown["final_reject"], "x"],
        unknown.loc[~unknown["final_reject"], "y"],
        s=50,
        facecolors="none",
        edgecolors=COLORS["Error"],
        linewidths=1.8,
        label="Unknown Missed",
    )
    if positive.size:
        ax.scatter(
            positive[:, 0],
            positive[:, 1],
            s=250,
            color="#FFD600",
            marker="*",
            edgecolors="#000000",
            linewidths=1.3,
            label="Learned prototype centers",
            zorder=10,
        )
    # Canonical v2 has no negative or boundary prototypes.  Legacy geometry
    # remains available only through explicitly archived renderers.

    ax.set_xlabel("Latent Projection Dimension 1")
    ax.set_ylabel("Latent Projection Dimension 2")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_12_latent_geometry"][0])
    save_figure(fig, out, "12_latent_prototype_geometry")


def plot_13_communication(block: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    if block.get("data_file"):
        path = resolve_data_path(block["data_file"])
        frame = pd.read_csv(path)
        frame["method"] = frame["method"].replace({
            "FedPROTEUS": "Proposed", "DKD-FedOS": "Proposed", "FedTROS": "Proposed", "FedTROS-MC": "Proposed", "FedTROS-PR": "Proposed"
        })
        for method in ("FedAvg", "FedProx", "Proposed"):
            group = frame[frame["method"] == method].sort_values("round")
            if group.empty:
                continue
            st = method_style(method)
            ax.plot(
                group["cumulative_mb"],
                group["accuracy_percent"],
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                marker=st["marker"],
                markevery=max(len(group) // 10, 1),
                markersize=st["markersize"],
                markeredgecolor="#000000",
                markeredgewidth=0.8,
                label=st["label"],
            )
            final = group.iloc[-1]
            ax.annotate(
                f"{final['accuracy_percent']:.1f}%",
                (final["cumulative_mb"], final["accuracy_percent"]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8.8,
                weight="bold",
                color="#000000",
            )
    else:
        rounds = np.arange(1, int(block["rounds"]) + 1)
        fedtros_spec = block.get("fedtros", block.get("fedproteus"))
        for method, spec in (("Proposed", fedtros_spec), ("FedAvg", block["fedavg"])):
            communication = rounds * float(spec["communication_per_round"])
            accuracy = spec["final"] - (spec["final"] - spec["start"]) * np.exp(-rounds / spec["tau"])
            st = method_style(method)
            ax.plot(
                communication,
                accuracy,
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                marker=st["marker"],
                markevery=10,
                markersize=st["markersize"],
                markeredgecolor="#000000",
                markeredgewidth=0.8,
                label=st["label"],
            )

    ax.set_xlabel("Estimated Parameter Payload (MB)")
    ax.set_ylabel("Validation Accuracy (%)")
    if block.get("data_file"):
        ax.set_xscale("log")
    ax.legend(loc="lower right", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_13_communication"][0])


def plot_14_ablation(block: dict[str, Any], out: Path) -> None:
    rows = block.get("rows", [])
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "macro_f1_percent" not in df.columns:
        return
    df = df.sort_values("macro_f1_percent")
    fig, ax = plt.subplots(figsize=(7.8, 4.8), layout="constrained")
    
    for i, (_, row) in enumerate(df.iterrows()):
        variant = (
            str(row["variant"])
            .replace("A4-PR:", "A4-Rejection:")
            .replace("FedTROS-MC", "Proposed")
            .replace("FedTROS-PR", "Proposed")
            .replace("FedTROS", "Proposed")
        )
        val = row["macro_f1_percent"]
        hatch = HATCHES[i % len(HATCHES)]
        if variant in (
            "Full method",
            "Full method (Proposed)",
            "Full method (FedTROS)",
            "FedTROS",
            "Proposed",
            "Proposed (Full)",
            "Full Proposed-MC",
            "Proposed-MC (Full)",
        ) or "Full" in variant:
            fill_color = COLORS["Proposed"]
            hatch = "///"
        elif "FedAvg" in variant:
            fill_color = COLORS["FedAvg"]
            hatch = "\\\\\\"
        else:
            fill_color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]

        bar = ax.barh(
            variant,
            val,
            color=fill_color,
            edgecolor="#000000",
            linewidth=1.1,
            hatch=hatch,
        )
        ax.text(
            val + 0.4,
            bar[0].get_y() + bar[0].get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9.2,
            weight="bold",
            color="#000000",
        )

    ax.set_xlabel("Open-Set Macro-F1 / AUROC (%)")
    min_val = float(df["macro_f1_percent"].min())
    max_val = float(df["macro_f1_percent"].max())
    ax.set_xlim(max(0.0, min(65.0, min_val - 6.0)), min(105.0, max_val + 8.0))
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_14_ablation"][0])


def plot_15_method_comparison(block: dict[str, Any], out: Path) -> None:
    df = pd.DataFrame(block["rows"])
    df["method"] = df["method"].replace({
        "FedPROTEUS": "Proposed", "DKD-FedOS": "Proposed", "FedTROS": "Proposed", "FedTROS-MC": "Proposed", "FedTROS-PR": "Proposed"
    })
    order = ["FedAvg", "FedProx", "Proposed"]
    df = df.set_index("method").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    
    x = np.arange(len(df))
    values = pd.to_numeric(df["accuracy"], errors="raise").to_numpy(float)
    value_range = float(values.max() - values.min())
    margin = max(2.0, 0.30 * max(value_range, 1.0))
    lower = max(0.0, float(values.min()) - margin)
    upper = min(102.0, float(values.max()) + margin)
    text_offset = max(0.25, 0.06 * max(upper - lower, 1.0))
    for i, (_, row) in enumerate(df.iterrows()):
        m = row["method"]
        val = float(row["accuracy"])
        st = method_style(m)
        ax.bar(
            i,
            val,
            width=0.55,
            color=st["color"],
            edgecolor="#000000",
            linewidth=1.2,
            hatch=st["hatch"],
            label=st["label"],
        )
        ax.text(
            i,
            val + text_offset,
            f"{val:.2f}%",
            ha="center",
            fontsize=9.5,
            weight="bold",
            color="#000000",
        )

    ax.set_xticks(x, df["method"], fontsize=11.0, weight="medium")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(lower, upper)
    apply_academic_spines(ax)
    save_figure(fig, out, BASE_PLOT_STEMS["plot_15_method_comparison"][0])


def generate_base_plots(
    results_json_path: Path | None = None,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
) -> None:
    """Generate all 18 base figures (01-15, 02_2x2, 09_10 comparison count & %)."""
    configure_academic_style()
    with using_data_dir(data_dir):
        data = load_json_data(results_json_path)
        out = Path(output_dir or FIGURES_OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        p = data["plots"]

        plot_01_scalability(p["plot_01_scalability"], out)
        plot_02_non_iid_distribution(p["plot_02_non_iid_distribution"], out)
        plot_02_partition_comparison(p["plot_02_non_iid_distribution"], out)
        plot_03_convergence_mild(p["plot_03_convergence_mild"], out)
        plot_04_convergence_hard(p["plot_04_convergence_hard"], out)
        plot_05_mahalanobis_nonconformity_distribution(p["plot_05_score_distribution"], out)
        plot_06_openness(p["plot_06_openness"], out)
        plot_07_roc(p["plot_07_roc"], p["plot_05_score_distribution"], out)
        plot_08_pr(p["plot_08_pr"], p["plot_05_score_distribution"], out)
        plot_09_10_confusion(p["plot_09_10_confusion"], out)
        plot_11_robustness(p["plot_11_robustness"], out)
        plot_12_latent_geometry(p["plot_12_latent_geometry"], out)
        plot_13_communication(p["plot_13_communication"], out)
        plot_14_ablation(p["plot_14_ablation"], out)
        plot_15_method_comparison(p["plot_15_method_comparison"], out)
