"""Non-scalability analysis figures 23–26 for FedTROS publication.

Legacy method aliases in this module are import-only compatibility for archived
pre-bundle result files; canonical publication rendering uses FedTROS-PR names.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from src.ref_plots.config import FIGURES_OUTPUT_DIR, TABLES_OUTPUT_DIR, PROCESSED_DATA_DIR
from src.ref_plots.data.loader import load_confusion_matrix, resolve_data_path
from src.ref_plots.style import (
    COLORS,
    DIVERGING_CMAP,
    METHOD_STYLES,
    apply_academic_spines,
    configure_academic_style,
    save_figure,
)


def binary_detection_metrics(scores: pd.DataFrame) -> dict[str, float]:
    y = (scores["known_or_unknown"].astype(str) == "unknown").astype(int).to_numpy()
    score_col = "nonconformity_score" if "nonconformity_score" in scores.columns else "prototype_rank_score"
    thresh_col = "tau_alpha" if "tau_alpha" in scores.columns else "selected_threshold_used"
    s = pd.to_numeric(scores[score_col], errors="raise").to_numpy(float)
    threshold = float(pd.to_numeric(scores[thresh_col], errors="coerce").dropna().median())
    pred = s >= threshold
    pos = y == 1
    neg = ~pos
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum(~pred & neg))
    fn = int(np.sum(~pred & pos))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    known_accept = tn / (tn + fp) if tn + fp else 0.0
    balanced = 0.5 * (recall + known_accept)

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    p = max(int(y.sum()), 1)
    n = max(int((1 - y).sum()), 1)
    tpr = np.r_[0.0, tp_cum / p]
    fpr = np.r_[0.0, fp_cum / n]
    auroc = float(np.trapezoid(tpr, fpr))
    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall_curve = tp_cum / p
    auprc = float(np.sum((recall_curve - np.r_[0.0, recall_curve[:-1]]) * precision_curve))
    return {
        "threshold": threshold,
        "auroc": auroc,
        "auprc": auprc,
        "known_acceptance": known_accept,
        "known_false_reject": 1.0 - known_accept,
        "unknown_precision": precision,
        "unknown_recall": recall,
        "unknown_f1": f1,
        "balanced_accuracy": balanced,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "known_samples": int(neg.sum()),
        "unknown_samples": int(pos.sum()),
    }


def threshold_curve(scores: pd.DataFrame, points: int = 350) -> pd.DataFrame:
    y = (scores["known_or_unknown"].astype(str) == "unknown").astype(int).to_numpy()
    score_col = "nonconformity_score" if "nonconformity_score" in scores.columns else "prototype_rank_score"
    s = pd.to_numeric(scores[score_col], errors="raise").to_numpy(float)
    thresholds = np.unique(np.quantile(s, np.linspace(0.0, 1.0, points)))
    rows = []
    pos = y == 1
    neg = ~pos
    for threshold in thresholds:
        pred = s >= threshold
        tp = np.sum(pred & pos)
        fp = np.sum(pred & neg)
        tn = np.sum(~pred & neg)
        fn = np.sum(~pred & pos)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        known_accept = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "unknown_recall": float(recall),
                "known_acceptance": float(known_accept),
                "unknown_precision": float(precision),
                "unknown_f1": float(f1),
                "balanced_accuracy": float(0.5 * (recall + known_accept)),
            }
        )
    return pd.DataFrame(rows)


def class_metrics(cm: pd.DataFrame, stage: str) -> pd.DataFrame:
    values = cm.to_numpy(float)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    tp = np.diag(values)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return pd.DataFrame(
        {
            "stage": stage,
            "class": cm.index,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def aggregate_confusion_metrics(cm: pd.DataFrame, stage: str) -> dict[str, float | str]:
    per = class_metrics(cm, stage)
    values = cm.to_numpy(float)
    support = values.sum(axis=1)
    total = values.sum()
    diagonal = np.diag(values)
    labels = list(cm.index)
    unknown_index = labels.index("Unknown")
    known_indices = [i for i in range(len(labels)) if i != unknown_index]
    known_support = support[known_indices].sum()
    return {
        "stage": stage,
        "overall_accuracy": float(diagonal.sum() / total),
        "macro_precision": float(per["precision"].mean()),
        "macro_recall": float(per["recall"].mean()),
        "macro_f1": float(per["f1"].mean()),
        "weighted_f1": float(np.average(per["f1"], weights=per["support"])),
        "known_accuracy": float(diagonal[known_indices].sum() / known_support),
        "unknown_recall": float(diagonal[unknown_index] / support[unknown_index]),
        "known_to_unknown_rate": float(values[known_indices, unknown_index].sum() / known_support),
    }


def convergence_summary(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    frame["method"] = frame["method"].replace({
        "FedPROTEUS": "Proposed", "DKD-FedOS": "Proposed", "FedTROS": "Proposed", "FedTROS-MC": "Proposed", "FedTROS-PR": "Proposed"
    })
    metric_column = (
        "accuracy_percent"
        if "accuracy_percent" in frame.columns
        else "macro_f1_percent"
    )
    metric_name = "accuracy" if metric_column == "accuracy_percent" else "macro_f1"
    rows = []
    for method, group in frame.groupby("method"):
        group = group.sort_values("round")
        rounds = group["round"].to_numpy(float)
        performance = group[metric_column].to_numpy(float)
        row: dict[str, float | str] = {
            "alpha": alpha,
            "method": str(method),
            "metric": metric_name,
            "final_metric_percent": float(performance[-1]),
            "best_metric_percent": float(performance.max()),
            "last10_mean_percent": float(performance[-10:].mean()),
            "last10_std_percent": float(performance[-10:].std(ddof=1)),
            "learning_curve_auc_percent": float(
                np.trapezoid(performance, rounds) / (rounds[-1] - rounds[0])
            ),
        }
        for target in (90, 92, 94, 95):
            hits = np.flatnonzero(performance >= target)
            row[f"round_to_{target}_percent"] = float(rounds[hits[0]]) if hits.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def communication_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame["method"] = frame["method"].replace({
        "FedPROTEUS": "Proposed", "DKD-FedOS": "Proposed", "FedTROS": "Proposed", "FedTROS-MC": "Proposed", "FedTROS-PR": "Proposed"
    })
    rows = []
    for method, group in frame.groupby("method"):
        group = group.sort_values("round")
        row: dict[str, float | str] = {
            "method": str(method),
            "final_accuracy_percent": float(group.iloc[-1]["accuracy_percent"]),
            "total_communication_mb": float(group.iloc[-1]["cumulative_mb"]),
        }
        for target in (90, 92, 94, 95):
            hit = group[group["accuracy_percent"] >= target]
            row[f"round_to_{target}_percent"] = float(hit.iloc[0]["round"]) if not hit.empty else np.nan
            row[f"mb_to_{target}_percent"] = float(hit.iloc[0]["cumulative_mb"]) if not hit.empty else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    fedavg_total = float(result.loc[result["method"] == "FedAvg", "total_communication_mb"].iloc[0])
    result["communication_reduction_vs_fedavg_x"] = fedavg_total / result["total_communication_mb"]
    return result


def plot_threshold_operating_curve(curve: pd.DataFrame, selected: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    ax.plot(
        curve["threshold"],
        curve["known_acceptance"] * 100,
        lw=2.4,
        ls="-",
        color=COLORS["Known"],
        label="Known Acceptance",
    )
    ax.plot(
        curve["threshold"],
        curve["unknown_recall"] * 100,
        lw=2.4,
        ls="--",
        color=COLORS["Unknown"],
        label="Unknown Recall",
    )
    ax.plot(
        curve["threshold"],
        curve["unknown_f1"] * 100,
        lw=2.2,
        ls="-.",
        color=COLORS["Proposed"],
        label="Unknown F1",
    )
    ax.plot(
        curve["threshold"],
        curve["balanced_accuracy"] * 100,
        lw=2.2,
        ls=":",
        color=COLORS["FedAvg"],
        label="Balanced Accuracy",
    )
    ax.axvline(
        selected,
        color="#000000",
        ls="--",
        lw=2.2,
        label=f"Selected Threshold ({selected:.3f})",
    )
    nearest = curve.iloc[(curve["threshold"] - selected).abs().argmin()]
    ax.scatter(
        [nearest["threshold"]] * 2,
        [nearest["known_acceptance"] * 100, nearest["unknown_recall"] * 100],
        s=60,
        color=[COLORS["Known"], COLORS["Unknown"]],
        edgecolors="#000000",
        linewidths=1.0,
        zorder=5,
    )
    ax.set_xlabel(r"Proposed Conformal Rejection Threshold $\tau_\alpha$")
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, loc="lower left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "23_conformal_calibration_validation")
    save_figure(fig, out, "23_open_set_threshold_operating_curve")


def plot_classwise_tradeoff(per_class: pd.DataFrame, out: Path) -> None:
    classes = per_class["class"].drop_duplicates().tolist()
    before = per_class[per_class["stage"] == "Before rejection"].set_index("class")
    after = per_class[per_class["stage"] == "After rejection"].set_index("class")
    x = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained", sharey=True)
    
    for ax, metric, panel_label in zip(axes, ("recall", "f1"), ("(a) Class-Wise Recall", "(b) Class-Wise F1-Score")):
        b = before.loc[classes, metric].to_numpy() * 100
        a = after.loc[classes, metric].to_numpy() * 100
        ax.plot(
            x,
            b,
            marker="s",
            ls="--",
            lw=2.2,
            markersize=7.5,
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            color=COLORS["Before"],
            label="Before Rejection",
        )
        ax.plot(
            x,
            a,
            marker="o",
            ls="-",
            lw=2.4,
            markersize=8.0,
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            color=COLORS["After"],
            label="After Rejection",
        )
        for i, (bv, av) in enumerate(zip(b, a)):
            ax.vlines(i, min(bv, av), max(bv, av), color="#000000", lw=1.2, linestyle=":", zorder=0)
            if abs(av - bv) >= 4:
                ax.annotate(
                    f"{av-bv:+.1f}",
                    (i, av),
                    xytext=(0, 8 if av >= bv else -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    weight="bold",
                    color="#000000",
                )
        ax.set_xticks(x, classes, rotation=25, ha="right", fontsize=10.0)
        ax.set_title(panel_label, fontsize=12.0, weight="bold", pad=8)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y")
        apply_academic_spines(ax)

    axes[0].set_ylabel("Metric (%)", fontsize=11.5)
    axes[1].legend(loc="lower right", framealpha=0.95)
    save_figure(fig, out, "24_classwise_open_set_tradeoff")


def plot_non_iid_retention(convergence: pd.DataFrame, out: Path) -> None:
    convergence["method"] = convergence["method"].replace({
        "FedPROTEUS": "Proposed", "DKD-FedOS": "Proposed", "FedTROS": "Proposed", "FedTROS-MC": "Proposed", "FedTROS-PR": "Proposed"
    })
    pivot = convergence.pivot(index="method", columns="alpha", values="last10_mean_percent")
    available = [m for m in ("FedAvg", "FedProx", "Proposed") if m in pivot.index and 0.1 in pivot.columns and 1.0 in pivot.columns]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    
    for method in available:
        mild = float(pivot.loc[method, 1.0])
        hard = float(pivot.loc[method, 0.1])
        st = METHOD_STYLES.get(method, {"color": COLORS["Neutral"], "marker": "o", "linestyle": "-"})
        ax.plot(
            [1.0, 0.1],
            [mild, hard],
            marker=st["marker"],
            linestyle=st.get("linestyle", "-"),
            markersize=8.5,
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            lw=2.6,
            color=st["color"],
            label=st.get("label", method),
        )
        ax.annotate(
            f"{mild:.2f}%",
            (1.0, mild),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            fontsize=9.2,
            weight="bold",
            color="#000000",
        )
        ax.annotate(
            f"{hard:.2f}%",
            (0.1, hard),
            xytext=(8, 8),
            textcoords="offset points",
            ha="left",
            fontsize=9.2,
            weight="bold",
            color="#000000",
        )
        ax.text(
            0.55,
            (mild + hard) / 2 + 0.4,
            f"Δ {hard-mild:+.2f} pp",
            color="#000000",
            ha="center",
            va="bottom",
            fontsize=9.0,
            weight="bold",
        )

    ax.set_xticks([1.0, 0.1], ["Mild Non-IID (α = 1.0)", "Hard Non-IID (α = 0.1)"], fontsize=10.5, weight="medium")
    ax.set_ylabel("Last-10-Round Accuracy (%)")
    ax.set_xlim(1.08, 0.02)
    ax.set_ylim(max(0, pivot.loc[available].min().min() - 5), 102)
    ax.grid(True, axis="y")
    ax.legend(loc="lower left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "25_non_iid_accuracy_retention")


def plot_confusion_delta(before: pd.DataFrame, after: pd.DataFrame, out: Path) -> None:
    def row_pct(cm: pd.DataFrame) -> np.ndarray:
        values = cm.to_numpy(float)
        den = values.sum(axis=1, keepdims=True)
        return np.divide(values, den, out=np.zeros_like(values), where=den > 0) * 100
    delta = row_pct(after) - row_pct(before)
    vmax = max(float(np.abs(delta).max()), 1.0)
    fig, ax = plt.subplots(figsize=(6.8, 5.5), layout="constrained")
    image = ax.imshow(delta, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    labels = list(before.index)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=10.0)
    ax.set_yticks(range(len(labels)), labels, fontsize=10.0)
    ax.set_xlabel("Predicted Class", fontsize=11.5, weight="medium")
    ax.set_ylabel("True Class", fontsize=11.5, weight="medium")

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            txt_color = "#FFFFFF" if abs(val) > 0.50 * vmax else "#000000"
            ax.text(
                j,
                i,
                f"{val:+.1f}",
                ha="center",
                va="center",
                fontsize=9.2,
                weight="bold",
                color=txt_color,
            )
    cbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Change (Percentage Points)", fontsize=10.5, color="#000000", weight="bold")
    cbar.outline.set_edgecolor("#000000")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(colors="#000000")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.0)
    save_figure(fig, out, "26_confusion_delta_after_rejection")


def generate_analysis_plots(
    data_dir: Path | None = None,
    figures_out: Path | None = None,
    tables_out: Path | None = None,
) -> None:
    """Generate non-scalability analysis figures 23-26 and associated summary CSVs."""
    configure_academic_style()
    d_dir = Path(data_dir or PROCESSED_DATA_DIR)
    f_out = Path(figures_out or FIGURES_OUTPUT_DIR)
    t_out = Path(tables_out or TABLES_OUTPUT_DIR)
    f_out.mkdir(parents=True, exist_ok=True)
    t_out.mkdir(parents=True, exist_ok=True)

    scores_path = resolve_data_path("exp2_scores.csv", d_dir)
    scores = pd.read_csv(scores_path)
    det_metrics = binary_detection_metrics(scores)
    curve = threshold_curve(scores)
    curve.to_csv(t_out / "open_set_threshold_operating_curve.csv", index=False)
    plot_threshold_operating_curve(curve, det_metrics["threshold"], f_out)

    before_path = resolve_data_path("exp2_confusion_before.csv", d_dir)
    after_path = resolve_data_path("exp2_confusion_after.csv", d_dir)
    before = load_confusion_matrix(before_path)
    after = load_confusion_matrix(after_path)
    per_class = pd.concat([class_metrics(before, "Before rejection"), class_metrics(after, "After rejection")], ignore_index=True)
    per_class.to_csv(t_out / "open_set_per_class_summary.csv", index=False)
    plot_classwise_tradeoff(per_class, f_out)

    agg = pd.DataFrame([aggregate_confusion_metrics(before, "Before rejection"), aggregate_confusion_metrics(after, "After rejection")])
    agg.to_csv(t_out / "open_set_aggregate_summary.csv", index=False)

    conv_mild_path = resolve_data_path("convergence_alpha1.csv", d_dir)
    conv_hard_path = resolve_data_path("convergence_alpha01.csv", d_dir)
    conv_mild = pd.read_csv(conv_mild_path)
    conv_hard = pd.read_csv(conv_hard_path)
    conv_summary = pd.concat([convergence_summary(conv_mild, 1.0), convergence_summary(conv_hard, 0.1)], ignore_index=True)
    conv_summary.to_csv(t_out / "non_scalability_convergence_summary.csv", index=False)
    plot_non_iid_retention(conv_summary, f_out)
    plot_confusion_delta(before, after, f_out)

    comm_mild_path = resolve_data_path("communication_alpha1.csv", d_dir)
    comm_mild = pd.read_csv(comm_mild_path)
    comm_summary = communication_summary(comm_mild)
    comm_summary.to_csv(t_out / "non_scalability_communication_summary.csv", index=False)
