"""Automated LaTeX table generation from benchmark results."""

from __future__ import annotations

from typing import Dict, List


def generate_latex_benchmark_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "Comparison of Intrusion Detection Performance under Extreme Non-IID Setting",
    label: str = "tab:benchmark_results",
) -> str:
    """Generate publication-ready LaTeX table comparing methods across AUROC, F1, and False Alarm Rate."""
    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Method} & \\textbf{Closed-Set Acc (\\%)} & \\textbf{Open-Set AUROC (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{FAR (\\%)} \\\\",
        "\\hline",
    ]

    for method, metrics in results.items():
        acc = metrics.get("accuracy", 0.0) * 100.0
        auroc = metrics.get("auroc", 0.0) * 100.0
        f1 = metrics.get("open_set_macro_f1", metrics.get("macro_f1", 0.0)) * 100.0
        far = metrics.get("false_alarm_rate", 0.0) * 100.0

        latex.append(f"{method} & {acc:.2f} & {auroc:.2f} & {f1:.2f} & {far:.2f} \\\\")

    latex.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    return "\n".join(latex)
