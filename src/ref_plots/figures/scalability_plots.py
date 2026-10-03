"""Scalability diagnostic figures 16–20 for FedTROS publication."""

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
    HATCHES,
    apply_academic_spines,
    configure_academic_style,
    save_figure,
)


def rolling(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


def set_percent_limits(ax: plt.Axes, values: list[float], *, floor: float = 0.0) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    spread = float(finite.max() - finite.min())
    margin = max(3.0, 0.15 * max(spread, 1.0))
    ax.set_ylim(
        max(floor, float(finite.min()) - margin),
        min(105.0, float(finite.max()) + margin),
    )


def plot_convergence_rounds(logs: dict[int, pd.DataFrame], out: Path) -> None:
    common_round = min(int(df["round"].max()) for df in logs.values())
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")

    plotted: list[float] = []
    for clients, df in logs.items():
        st = CLIENT_STYLES[clients]
        d = df[df["round"] <= common_round]
        y = 100 * rolling(d["round_openset_f1_macro"])
        plotted.extend(y.dropna().tolist())
        ax.plot(
            d["round"],
            y,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            marker=st["marker"],
            markevery=max(len(d) // 8, 1),
            markersize=st["markersize"],
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            label=f"{clients} clients",
        )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Open-Set Macro F1 (%)")
    ax.set_xlim(1, common_round)
    set_percent_limits(ax, plotted)
    ax.legend(loc="lower right", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "16_scalability_convergence_common_rounds")


def plot_convergence_wallclock(logs: dict[int, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")

    plotted: list[float] = []
    for clients, df in logs.items():
        st = CLIENT_STYLES[clients]
        y = 100 * rolling(df["round_openset_f1_macro"])
        plotted.extend(y.dropna().tolist())
        ax.plot(
            df["cumulative_hours"],
            y,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            marker=st["marker"],
            markevery=max(len(df) // 8, 1),
            markersize=st["markersize"],
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            label=f"{clients} clients",
        )

    ax.set_xlabel("Cumulative Wall-Clock Time (Hours)")
    ax.set_ylabel("Open-Set Macro F1 (%)")
    set_percent_limits(ax, plotted)
    ax.legend(loc="lower right", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "17_scalability_quality_vs_wallclock")


def plot_runtime_breakdown(logs: dict[int, pd.DataFrame], out: Path) -> None:
    rows = []
    for clients, df in logs.items():
        tail = df.tail(10)
        fit = float(tail["client_fit_wall_time_sec"].median())
        eval_t = float(tail["open_set_round_eval_time_sec"].median())
        agg = float(tail["server_aggregation_time_sec"].median())
        total = float(tail["round_time_sec"].median())
        other = max(total - fit - eval_t - agg, 0.0)
        rows.append((clients, fit, eval_t, agg, other, total))

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    bottoms = np.zeros(len(rows))
    
    components = [
        ("Client Fitting", [r[1] for r in rows], "#1976D2", "///"),
        ("Open-Set Evaluation", [r[2] for r in rows], "#7B1FA2", "\\\\\\"),
        ("Server Aggregation", [r[3] for r in rows], "#F57C00", "xxx"),
        ("Orchestration / Other", [r[4] for r in rows], "#546E7A", "..."),
    ]

    for label, values, fill_color, hatch in components:
        vals = np.asarray(values)
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.55,
            label=label,
            color=fill_color,
            edgecolor="#000000",
            linewidth=1.0,
            hatch=hatch,
        )
        bottoms += vals

    for i, r in enumerate(rows):
        ax.text(
            i,
            r[5] + max(bottoms) * 0.025,
            f"{r[5]/60:.1f} min",
            ha="center",
            fontsize=9.2,
            weight="bold",
            color="#000000",
        )

    ax.set_xticks(x, [f"{r[0]} Clients" for r in rows], fontsize=11.0, weight="medium")
    ax.set_ylabel("Time Per Round (Seconds)")
    ax.legend(ncol=2, loc="upper left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "18_scalability_runtime_breakdown")


def plot_client_fairness(logs: dict[int, pd.DataFrame], out: Path) -> None:
    clients = np.array(sorted(logs))
    mean = np.array([logs[n].tail(10)["mean_client_macro_f1"].mean() for n in clients]) * 100
    temporal_sd = np.nan_to_num(
        np.array(
            [logs[n].tail(10)["mean_client_macro_f1"].std(ddof=1) for n in clients]
        )
        * 100,
        nan=0.0,
    )
    worst = np.array([logs[n].tail(10)["worst_client_macro_f1"].mean() for n in clients]) * 100
    dispersion = np.array([logs[n].tail(10)["std_client_macro_f1"].mean() for n in clients]) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    ax.errorbar(
        clients,
        mean,
        yerr=temporal_sd,
        color=COLORS["Proposed"],
        marker="o",
        linestyle="-",
        linewidth=2.4,
        markersize=8.0,
        markeredgecolor="#000000",
        markeredgewidth=0.8,
        ecolor="#000000",
        elinewidth=1.4,
        capsize=5.0,
        capthick=1.4,
        label="Mean Client Macro F1 ± SD",
    )
    ax.plot(
        clients,
        worst,
        color=COLORS["Unknown"],
        marker="s",
        linestyle="--",
        linewidth=2.4,
        markersize=7.5,
        markeredgecolor="#000000",
        markeredgewidth=0.8,
        label="Worst-Client Macro F1",
    )

    for x, y, s in zip(clients, mean, dispersion):
        ax.annotate(
            f"SD: {s:.1f} pp",
            (x, y),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            weight="bold",
            color="#000000",
        )

    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("Client Macro F1 (%)")
    ax.set_xticks(clients)
    set_percent_limits(ax, [*mean, *worst])
    ax.legend(loc="lower left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "19_scalability_client_fairness")


def plot_detection_scaling(logs: dict[int, pd.DataFrame], out: Path) -> None:
    clients = np.array(sorted(logs))
    specs = [
        ("round_openset_auroc", "AUROC", COLORS["Proposed"], "o", "-", 8.0),
        ("round_openset_unknown_recall", "Unknown Recall", COLORS["Known"], "s", "--", 7.5),
        ("round_openset_fpr95", "FPR@95TPR", COLORS["Unknown"], "^", "-.", 8.0),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    
    for metric, label, color, marker, ls, ms in specs:
        means = np.array([logs[n].tail(10)[metric].mean() for n in clients]) * 100
        sds = np.nan_to_num(
            np.array([logs[n].tail(10)[metric].std(ddof=1) for n in clients])
            * 100,
            nan=0.0,
        )
        ax.errorbar(
            clients,
            means,
            yerr=sds,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.4,
            markersize=ms,
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            ecolor="#000000",
            elinewidth=1.4,
            capsize=5.0,
            capthick=1.4,
            label=label,
        )

    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("Metric (%)")
    ax.set_xticks(clients)
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, out, "20_scalability_open_set_detection")


def write_scalability_summary(logs: dict[int, pd.DataFrame], out_tables: Path) -> None:
    rows = []
    common_round = min(int(df["round"].max()) for df in logs.values())
    for clients, df in logs.items():
        tail = df.tail(10)
        at_common = df.loc[df["round"] == common_round].iloc[0]
        rows.append({
            "clients": clients,
            "available_rounds": int(df["round"].max()),
            "total_wallclock_hours": df["round_time_sec"].sum() / 3600,
            "common_round": common_round,
            "common_round_open_set_f1_pct": 100 * at_common["round_openset_f1_macro"],
            "final10_open_set_f1_mean_pct": 100 * tail["round_openset_f1_macro"].mean(),
            "final10_open_set_f1_temporal_sd_pp": 100 * (tail["round_openset_f1_macro"].std(ddof=1) if len(tail) > 1 else 0.0),
            "final10_auroc_mean_pct": 100 * tail["round_openset_auroc"].mean(),
            "final10_fpr95_mean_pct": 100 * tail["round_openset_fpr95"].mean(),
            "final10_unknown_recall_mean_pct": 100 * tail["round_openset_unknown_recall"].mean(),
            "final10_mean_client_f1_pct": 100 * tail["mean_client_macro_f1"].mean(),
            "final10_cross_client_sd_pp": 100 * tail["std_client_macro_f1"].mean(),
            "final10_worst_client_f1_pct": 100 * tail["worst_client_macro_f1"].mean(),
            "final10_round_time_median_sec": tail["round_time_sec"].median(),
        })
    out_tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_tables / "scalability_publication_summary.csv", index=False)


def generate_scalability_plots(
    data_dir: Path | None = None,
    figures_out: Path | None = None,
    tables_out: Path | None = None,
) -> None:
    """Generate all 5 scalability figures (16-20) and summary table."""
    configure_academic_style()
    f_out = Path(figures_out or FIGURES_OUTPUT_DIR)
    t_out = Path(tables_out or TABLES_OUTPUT_DIR)
    f_out.mkdir(parents=True, exist_ok=True)
    t_out.mkdir(parents=True, exist_ok=True)
    logs = load_scalability_logs(data_dir)

    plot_convergence_rounds(logs, f_out)
    plot_convergence_wallclock(logs, f_out)
    plot_runtime_breakdown(logs, f_out)
    plot_client_fairness(logs, f_out)
    plot_detection_scaling(logs, f_out)
    write_scalability_summary(logs, t_out)
