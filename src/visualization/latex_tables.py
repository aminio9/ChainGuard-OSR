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


def generate_latex_bla_ablation_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "Logit-Adjusted Balanced Risk Minimization Ablation Study",
    label: str = "tab:anchor_ablation",
) -> str:
    """Generate publication-ready LaTeX table matching paper Table 3 (Section 6.5.2)."""
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Loss Formulation} & \textbf{AUROC (\%)} & \textbf{Unk-Recall (\%)} & \textbf{Unk-F1 (\%)} & \textbf{KFR (\%)} \\",
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


# Backward compatibility alias
generate_latex_anchor_ablation_table = generate_latex_bla_ablation_table


def generate_latex_closed_set_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "Closed-Set Intrusion Detection Classification Performance",
    label: str = "tab:closed_set_results",
) -> str:
    """Generate publication-ready LaTeX table for closed-set classification (Section 6.2, Table 1)."""
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Accuracy (\%)} & \textbf{Balanced Acc. (\%)} & \textbf{Macro-F1 (\%)} \\",
        r"\midrule",
    ]

    for method, metrics in results.items():
        acc = metrics.get("accuracy", 0.0) * 100.0
        bal_acc = metrics.get("balanced_accuracy", 0.0) * 100.0
        macro_f1 = metrics.get("macro_f1", 0.0) * 100.0
        latex.append(
            f"{method} & {acc:.2f} & {bal_acc:.2f} & {macro_f1:.2f} \\\\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)


def generate_latex_multi_skew_closed_set_table(
    results_by_skew: Dict[float, Dict[str, Dict[str, float]]],
    caption: str = "Closed-set classification performance under varying Dirichlet label skew",
    label: str = "tab:noniid_closed_results",
) -> str:
    """Generate publication-ready LaTeX table for multi-skew closed-set classification (Section 6.2, Table 2)."""
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$\alpha_D$ & \textbf{Method} & \textbf{Accuracy (\%)} & \textbf{Balanced Acc. (\%)} & \textbf{Macro-F1 (\%)} \\",
        r"\midrule",
    ]

    skews = sorted(results_by_skew.keys(), reverse=True)
    for s_idx, alpha_d in enumerate(skews):
        methods_dict = results_by_skew[alpha_d]
        for idx, (method, metrics) in enumerate(methods_dict.items()):
            prefix = f"{alpha_d} & " if idx == 0 else " & "
            acc = metrics.get("accuracy", 0.0) * 100.0
            bal_acc = metrics.get("balanced_accuracy", 0.0) * 100.0
            macro_f1 = metrics.get("macro_f1", 0.0) * 100.0
            latex.append(
                f"{prefix}{method} & {acc:.2f} & {bal_acc:.2f} & {macro_f1:.2f} \\\\"
            )
        if s_idx < len(skews) - 1:
            latex.append(r"\midrule")

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)


def generate_latex_paper_table2(
    results_by_skew: Dict[float, Dict[str, float]],
    caption: str = "Closed-set B-NAT classification performance under varying Dirichlet label skew ($\\alpha_D \\in \\{1.0, 0.5, 0.1\\}$)",
    label: str = "tab:noniid_closed_results",
) -> str:
    """Generate exact publication-ready LaTeX table for paper Table 2 (Section 6.2)."""
    latex = [
        r"\begin{table}[pos=t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$\alpha_D$ & Accuracy (\%) & Balanced Acc. (\%) & Macro-F1 (\%) \\",
        r"\midrule",
    ]
    for alpha_d in sorted(results_by_skew.keys(), reverse=True):
        m = results_by_skew[alpha_d]
        acc = m.get("accuracy", 0.0) * 100.0
        bal_acc = m.get("balanced_accuracy", 0.0) * 100.0
        macro_f1 = m.get("macro_f1", 0.0) * 100.0
        latex.append(f"{alpha_d} & {acc:.3f} & {bal_acc:.3f} & {macro_f1:.3f} \\\\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(latex)


