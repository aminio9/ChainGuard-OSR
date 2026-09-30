"""Publication-ready LaTeX tables generator and metric summary exports for FedTROS."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.ref_plots.config import TABLES_OUTPUT_DIR, DOCS_DIR
from src.ref_plots.data.loader import load_scalability_logs
from src.ref_plots.figures.tradeoff_plots import compute_scalability_table_summary


def fmt(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "--"
    return f"{v:.{digits}f}"


def generate_all_latex_tables(
    tables_out: Path | None = None,
    docs_out: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, str]:
    """Generate all three publication LaTeX tables for FedTROS and write to files."""
    t_out = Path(tables_out or TABLES_OUTPUT_DIR)
    d_out = Path(docs_out or DOCS_DIR)
    t_out.mkdir(parents=True, exist_ok=True)
    d_out.mkdir(parents=True, exist_ok=True)

    logs = load_scalability_logs(data_dir=data_dir)
    summary = compute_scalability_table_summary(logs)

    up_cols = {
        "Final-10 close-set acc (%)",
        "Final-10 close-set macro-F1 (%)",
        "Final-10 mean client F1 (%)",
        "Final-10 worst client F1 (%)",
        "Final-10 open-set macro-F1 (%)",
        "Final-10 open-set overall acc (%)",
        "Final-10 known acc (%)",
        "Final-10 unknown recall (%)",
        "Final-10 AUROC (%)",
    }
    down_cols = {
        "Final-10 FPR@95 (%)",
        "Median round time (min)",
        "Median client-fit time (min)",
        "Median open-set eval time (min)",
        "Median aggregation time (s)",
        "Total wall-clock (h)",
    }

    def bold_best(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
        best = {}
        for c in cols:
            if c in up_cols:
                best[c] = df[c].max()
            elif c in down_cols:
                best[c] = df[c].min()
        return best

    best = bold_best(summary, list(summary.columns))

    def cell(c: str, v: float) -> str:
        txt = fmt(v)
        if c in best and abs(v - best[c]) < 1e-9:
            return f"\\textbf{{{txt}}}"
        return txt

    # Table 1: Main scalability table
    lines1 = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main scalability results for the proposed framework across 10, 50, and 100 clients. Results are reported as final-10-round means unless otherwise stated. Best values are bold. $\\uparrow$ means higher is better and $\\downarrow$ means lower is better.}",
        "\\label{tab:main_scalability}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Clients & Rounds & Close-set Acc. $\\uparrow$ & Open-set Macro-F1 $\\uparrow$ & AUROC $\\uparrow$ & Unknown Recall $\\uparrow$ & FPR@95TPR $\\downarrow$ & Round Time (min) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        vals = [
            str(int(r["Clients"])),
            str(int(r["Available rounds"])),
            cell("Final-10 close-set acc (%)", r["Final-10 close-set acc (%)"]),
            cell("Final-10 open-set macro-F1 (%)", r["Final-10 open-set macro-F1 (%)"]),
            cell("Final-10 AUROC (%)", r["Final-10 AUROC (%)"]),
            cell("Final-10 unknown recall (%)", r["Final-10 unknown recall (%)"]),
            cell("Final-10 FPR@95 (%)", r["Final-10 FPR@95 (%)"]),
            cell("Median round time (min)", r["Median round time (min)"]),
        ]
        lines1.append(" & ".join(vals) + " \\\\")
    lines1.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    table_main = "\n".join(lines1)

    # Table 2: Trade-off table
    lines2 = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Detailed trade-off between closed-set utility, client-level fairness, and open-set robustness under increasing federation scale for the proposed framework. Results are final-10-round means. Best values are bold.}",
        "\\label{tab:tradeoff_scalability}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        "Clients & Close-F1 $\\uparrow$ & Mean Client F1 $\\uparrow$ & Worst Client F1 $\\uparrow$ & Open Overall Acc. $\\uparrow$ & Known Acc. $\\uparrow$ & Open Macro-F1 $\\uparrow$ & Unknown Recall $\\uparrow$ & AUROC $\\uparrow$ & FPR@95TPR $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        vals = [
            str(int(r["Clients"])),
            cell("Final-10 close-set macro-F1 (%)", r["Final-10 close-set macro-F1 (%)"]),
            cell("Final-10 mean client F1 (%)", r["Final-10 mean client F1 (%)"]),
            cell("Final-10 worst client F1 (%)", r["Final-10 worst client F1 (%)"]),
            cell("Final-10 open-set overall acc (%)", r["Final-10 open-set overall acc (%)"]),
            cell("Final-10 known acc (%)", r["Final-10 known acc (%)"]),
            cell("Final-10 open-set macro-F1 (%)", r["Final-10 open-set macro-F1 (%)"]),
            cell("Final-10 unknown recall (%)", r["Final-10 unknown recall (%)"]),
            cell("Final-10 AUROC (%)", r["Final-10 AUROC (%)"]),
            cell("Final-10 FPR@95 (%)", r["Final-10 FPR@95 (%)"]),
        ]
        lines2.append(" & ".join(vals) + " \\\\")
    lines2.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    table_tradeoff = "\n".join(lines2)

    # Table 3: Systems efficiency table
    lines3 = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{System-efficiency breakdown for the scalability study of the proposed framework. Median values are computed over the final 10 rounds. Best values are bold.}",
        "\\label{tab:systems_scalability}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Clients & Total Wall-clock (h) $\\downarrow$ & Client Fit (min) $\\downarrow$ & Open-set Eval. (min) $\\downarrow$ & Aggregation (s) $\\downarrow$ & Round Time (min) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        vals = [
            str(int(r["Clients"])),
            cell("Total wall-clock (h)", r["Total wall-clock (h)"]),
            cell("Median client-fit time (min)", r["Median client-fit time (min)"]),
            cell("Median open-set eval time (min)", r["Median open-set eval time (min)"]),
            cell("Median aggregation time (s)", r["Median aggregation time (s)"]),
            cell("Median round time (min)", r["Median round time (min)"]),
        ]
        lines3.append(" & ".join(vals) + " \\\\")
    lines3.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    table_systems = "\n".join(lines3)

    (t_out / "table_main_scalability.tex").write_text(table_main, encoding="utf-8")
    (t_out / "table_tradeoff_scalability.tex").write_text(table_tradeoff, encoding="utf-8")
    (t_out / "table_systems_scalability.tex").write_text(table_systems, encoding="utf-8")

    return {
        "main_scalability": table_main,
        "tradeoff_scalability": table_tradeoff,
        "systems_scalability": table_systems,
    }
