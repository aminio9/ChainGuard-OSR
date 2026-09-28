"""Automated publication-ready LaTeX table generation from experimental benchmark results."""

from __future__ import annotations

from typing import Dict


def generate_latex_benchmark_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "Comparison of Intrusion Detection Performance under Extreme Non-IID Setting",
    label: str = "tab:benchmark_results",
) -> str:
    """Generate publication-ready LaTeX table matching paper Table 5 (Section 6.3)."""
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{AUROC (\%)} & \textbf{AUPR (\%)} & \textbf{Unk-Recall (\%)} & \textbf{Unk-F1 (\%)} & \textbf{OS Macro-F1 (\%)} & \textbf{KFR (\%)} \\",
        r"\midrule",
    ]

    for method, metrics in results.items():
        auroc = metrics.get("auroc", 0.0) * 100.0
        aupr = metrics.get("aupr", 0.0) * 100.0
        unk_rec = metrics.get("unknown_recall", 0.0) * 100.0
        unk_f1 = metrics.get("unknown_f1", 0.0) * 100.0
        os_f1 = metrics.get("open_set_macro_f1", metrics.get("macro_f1", 0.0)) * 100.0
        kfr = metrics.get("kfr", metrics.get("false_alarm_rate", 0.0)) * 100.0

        latex.append(
            f"{method} & {auroc:.2f} & {aupr:.2f} & {unk_rec:.2f} & {unk_f1:.2f} & {os_f1:.2f} & {kfr:.2f} \\\\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(latex)


def generate_latex_loao_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "Leave-One-Attack-Out Evaluation across Distinct Held-Out Attack Types",
    label: str = "tab:loao_results",
) -> str:
    """Generate publication-ready LaTeX table matching paper Table 6 (Section 6.5)."""
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Held-out} & \textbf{AUROC (\%)} & \textbf{AUPR (\%)} & \textbf{FPR95 (\%)} & \textbf{Unk-Recall (\%)} & \textbf{Unk-F1 (\%)} \\",
        r"\midrule",
    ]

    for attack_name, metrics in results.items():
        auroc = metrics.get("auroc", 0.0) * 100.0
        aupr = metrics.get("aupr", 0.0) * 100.0
        fpr95 = metrics.get("fpr95", 0.0) * 100.0
        unk_rec = metrics.get("unknown_recall", 0.0) * 100.0
        unk_f1 = metrics.get("unknown_f1", 0.0) * 100.0

        latex.append(
            f"{attack_name} & {auroc:.2f} & {aupr:.2f} & {fpr95:.2f} & {unk_rec:.2f} & {unk_f1:.2f} \\\\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)


def generate_latex_anchor_ablation_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "Anchor Regularization Ablation Study",
    label: str = "tab:anchor_ablation",
) -> str:
    """Generate publication-ready LaTeX table matching paper Table 9 (Section 6.6.2)."""
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Anchor Variant} & \textbf{AUROC (\%)} & \textbf{Unk-Recall (\%)} & \textbf{Unk-F1 (\%)} & \textbf{KFR (\%)} \\",
        r"\midrule",
    ]

    for variant, metrics in results.items():
        auroc = metrics.get("auroc", 0.0) * 100.0
        unk_rec = metrics.get("unknown_recall", 0.0) * 100.0
        unk_f1 = metrics.get("unknown_f1", 0.0) * 100.0
        kfr = metrics.get("kfr", metrics.get("false_alarm_rate", 0.0)) * 100.0

        latex.append(
            f"{variant} & {auroc:.2f} & {unk_rec:.2f} & {unk_f1:.2f} & {kfr:.2f} \\\\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)
