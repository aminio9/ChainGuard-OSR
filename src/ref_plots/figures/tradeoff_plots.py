"""Closed/Open trade-off (21) and Pareto efficiency (22) figures for FedTROS publication."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ref_plots.config import FIGURES_OUTPUT_DIR, TABLES_OUTPUT_DIR
from src.ref_plots.data.loader import load_scalability_logs
from src.ref_plots.style import (
    CLIENT_STYLES,
    COLORS,
    apply_academic_spines,
    configure_academic_style,
    save_figure,
)


def compute_scalability_table_summary(logs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    common_round = min(int(df["round"].max()) for df in logs.values())
    for clients, df in logs.items():
        tail = df.tail(min(10, len(df)))
        common = df[df["round"] == common_round]
        row = {
            "Clients": clients,
            "Available rounds": len(df),
            "Total wall-clock (h)": df["round_time_sec"].sum() / 3600,
            "Common round": common_round,
            "Common-round open-set F1 (%)": common["round_openset_f1_macro"].mean() * 100 if not common.empty else np.nan,
            "Final-10 close-set acc (%)": tail["local_student_accuracy"].mean() * 100,
            "Final-10 close-set macro-F1 (%)": tail["local_student_f1_macro"].mean() * 100,
            "Final-10 mean client F1 (%)": tail["mean_client_macro_f1"].mean() * 100,
            "Final-10 worst client F1 (%)": tail["worst_client_macro_f1"].mean() * 100,
            "Final-10 open-set macro-F1 (%)": tail["round_openset_f1_macro"].mean() * 100,
            "Final-10 open-set overall acc (%)": tail["round_openset_overall_acc"].mean() * 100,
            "Final-10 known acc (%)": tail["round_openset_known_acc"].mean() * 100,
            "Final-10 unknown recall (%)": tail["round_openset_unknown_recall"].mean() * 100,
            "Final-10 AUROC (%)": tail["round_openset_auroc"].mean() * 100,
            "Final-10 FPR@95 (%)": tail["round_openset_fpr95"].mean() * 100,
            "Median round time (min)": tail["round_time_sec"].median() / 60,
            "Median client-fit time (min)": tail["client_fit_wall_time_sec"].median() / 60,
            "Median open-set eval time (min)": tail["open_set_round_eval_time_sec"].median() / 60,
            "Median aggregation time (s)": tail["server_aggregation_time_sec"].median(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Clients")


def plot_close_open_tradeoff(summary: pd.DataFrame, out: Path) -> None:
    clients = summary["Clients"].values
    fig, ax = plt.subplots(figsize=(7.8, 4.8), layout="constrained")

    series_specs = [
        ("Final-10 close-set acc (%)", "Closed-Set Accuracy", "#2E7D32", "o", "-", 8.0),
        ("Final-10 close-set macro-F1 (%)", "Closed-Set Macro-F1", "#1976D2", "s", "--", 7.5),
        ("Final-10 open-set overall acc (%)", "Open-Set Overall Acc.", "#00897B", "^", "-.", 8.0),
        ("Final-10 open-set macro-F1 (%)", "Open-Set Macro-F1", "#7B1FA2", "D", ":", 7.0),
        ("Final-10 known acc (%)", "Known-Class Acc.", "#546E7A", "v", "-", 7.5),
        ("Final-10 unknown recall (%)", "Unknown Recall (OSR)", "#C2185B", "P", "--", 8.0),
    ]

    for col, label, color, marker, ls, ms in series_specs:
        ax.plot(
            clients,
            summary[col].values,
            marker=marker,
            linestyle=ls,
            linewidth=2.3,
            markersize=ms,
            color=color,
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            label=label,
        )

    ax.set_xticks(clients, [f"{x} Clients" for x in clients], fontsize=10.5, weight="medium")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of Participating Clients")
    ax.set_ylabel("Performance (%)")
    ax.legend(framealpha=0.95, fontsize=9.5, ncol=2, loc="lower left")
    apply_academic_spines(ax)
    save_figure(fig, out, "21_scalability_close_open_tradeoff")


def plot_efficiency_robustness_pareto(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")

    for _, r in summary.iterrows():
        c_val = int(r["Clients"])
        st = CLIENT_STYLES.get(c_val, {"color": "#546E7A", "marker": "o", "label": f"{c_val} clients"})
        x = r["Median round time (min)"]
        y = r["Final-10 open-set macro-F1 (%)"]
        ur = r["Final-10 unknown recall (%)"]
        size = max(55.0, ur * 8.0)

        ax.scatter(
            x,
            y,
            s=size,
            alpha=0.90,
            color=st["color"],
            marker=st["marker"],
            edgecolors="#000000",
            linewidths=1.2,
            label=f"{c_val} Clients (Recall: {ur:.1f}%)",
            zorder=5,
        )
        ax.annotate(
            f"{c_val} Clients\nF1: {y:.1f}%\nRecall: {ur:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(10, -8 if c_val == 100 else 8),
            fontsize=9.0,
            weight="semibold",
            color="#000000",
        )

    ax.set_xlabel("Median Round Wall-Clock Time (Minutes)")
    ax.set_ylabel("Final-10-Round Open-Set Macro-F1 (%)")
    max_time = float(summary["Median round time (min)"].max())
    ax.set_xlim(0, max(max_time * 1.25, 0.01))
    y_values = pd.to_numeric(
        summary["Final-10 open-set macro-F1 (%)"], errors="coerce"
    ).dropna()
    if not y_values.empty:
        spread = float(y_values.max() - y_values.min())
        margin = max(3.0, 0.15 * max(spread, 1.0))
        ax.set_ylim(
            max(0.0, float(y_values.min()) - margin),
            min(105.0, float(y_values.max()) + margin),
        )
    ax.legend(loc="lower left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "22_scalability_efficiency_robustness_pareto")


def generate_tradeoff_plots(
    data_dir: Path | None = None,
    figures_out: Path | None = None,
    tables_out: Path | None = None,
) -> None:
    """Generate trade-off and Pareto figures (21, 22) and summary table."""
    configure_academic_style()
    f_out = Path(figures_out or FIGURES_OUTPUT_DIR)
    t_out = Path(tables_out or TABLES_OUTPUT_DIR)
    f_out.mkdir(parents=True, exist_ok=True)
    t_out.mkdir(parents=True, exist_ok=True)

    logs = load_scalability_logs(data_dir)
    summary = compute_scalability_table_summary(logs)
    summary.to_csv(t_out / "q1_scalability_table_summary.csv", index=False)

    plot_close_open_tradeoff(summary, f_out)
    plot_efficiency_robustness_pareto(summary, f_out)
