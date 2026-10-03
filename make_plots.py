"""Consolidated real-data plotting entrypoint for FedTROS-MC.

Generates all 31 publication figures (34 figure stems across PNG + PDF) and 3 LaTeX tables
using the exact visual design, color palette, typography, hatches, layouts, and spines of
the reference suite (consolidated in `src.ref_plots`), driven 100% by real FedTROS-MC
experiment results from `D:\\PAPER\\Code`.

Usage:
    python make_plots.py --results_root results/runs_v2 --figures_dir results/runs_v2/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.replot_all import replot_all_figures
from src.ref_plots.config import ALL_FIGURE_STEMS
from src.ref_plots.contract_exporter_part3 import finalize_real_plot_contract
from src.ref_plots.figures.analysis_plots import generate_analysis_plots
from src.ref_plots.figures.base_plots import generate_base_plots
from src.ref_plots.figures.extended_plots import generate_extended_plots
from src.ref_plots.figures.scalability_plots import generate_scalability_plots
from src.ref_plots.figures.tradeoff_plots import generate_tradeoff_plots
from src.ref_plots.style import COLORS, apply_academic_spines, configure_academic_style, save_figure
from src.ref_plots.tables.latex_tables import generate_all_latex_tables


STEM_PROVENANCE_MAP: Dict[str, Dict[str, Any]] = {
    "01_scalability_clients": {"experiment": "E7_scalability", "contract_file": "result.json#plot_01_scalability"},
    "02_internal_non_iid_distribution": {"experiment": "E1_dirichlet_01", "contract_file": "client_distribution_alpha01.csv"},
    "02_internal_non_iid_distribution_2x2": {"experiment": "E1_iid..E1_dirichlet_01", "contract_file": "client_distribution_iid_alpha_sweep.csv"},
    "03_convergence_mild_non_iid": {"experiment": "E1_iid / E2_baselines (alpha=1.0)", "contract_file": "convergence_alpha1.csv"},
    "04_convergence_hard_non_iid": {"experiment": "E1_dirichlet_01 / E2_baselines (alpha=0.1)", "contract_file": "convergence_alpha01.csv"},
    "05_mahalanobis_nonconformity_distribution": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_scores.csv"},
    "06_openness_vs_auroc": {"experiment": "E2_E3_benchmarks (4 datasets)", "contract_file": "result.json#plot_06_openness"},
    "07_roc_unknown_detection": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_roc_curve.csv"},
    "08_pr_unknown_detection": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_pr_curve.csv"},
    "09_confusion_before_open_set": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before.csv"},
    "10_confusion_after_open_set": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_after.csv"},
    "09_10_confusion_comparison": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before/after.csv"},
    "09_10_confusion_comparison_percentage": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before/after.csv"},
    "11_non_iid_robustness_boxplot": {"experiment": "E1_iid..E1_dirichlet_01", "contract_file": "result.json#plot_11_robustness"},
    "12_adaptive_multicenter_geometry": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_latent_projection.csv"},
    "13_communication_efficiency": {"experiment": "E2_baselines / E5_comm", "contract_file": "communication_alpha1.csv"},
    "14_ablation_study": {"experiment": "A1_component_ablations + A5_blocks", "contract_file": "result.json#plot_14_ablation"},
    "15_method_comparison": {"experiment": "E2_baselines (alpha=1.0)", "contract_file": "result.json#plot_15_method_comparison"},
    "16_scalability_convergence_common_rounds": {"experiment": "E7_scalability (K=10,50,100)", "contract_file": "scalability_{10,50,100}_clients.csv"},
    "17_scalability_quality_vs_wallclock": {"experiment": "E7_scalability (K=10,50,100)", "contract_file": "scalability_{10,50,100}_clients.csv"},
    "18_scalability_runtime_breakdown": {"experiment": "E7_scalability (K=10,50,100)", "contract_file": "scalability_{10,50,100}_clients.csv"},
    "19_scalability_client_fairness": {"experiment": "E7_scalability (K=10,50,100)", "contract_file": "scalability_{10,50,100}_clients.csv"},
    "20_scalability_open_set_detection": {"experiment": "E7_scalability (K=10,50,100)", "contract_file": "scalability_{10,50,100}_clients.csv"},
    "21_scalability_close_open_tradeoff": {"experiment": "E7_scalability (K=10,50,100)", "contract_file": "scalability_{10,50,100}_clients.csv"},
    "22_scalability_efficiency_robustness_pareto": {"experiment": "E7_scalability (K=10,50,100)", "contract_file": "scalability_{10,50,100}_clients.csv"},
    "23_conformal_calibration_validation": {"experiment": "E2_E3_benchmarks + A6_conformal", "contract_file": "exp2_scores.csv"},
    "24_classwise_open_set_tradeoff": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before/after.csv"},
    "25_non_iid_accuracy_retention": {"experiment": "E1_iid..E1_dirichlet_01", "contract_file": "convergence_alpha1/01.csv"},
    "26_confusion_delta_after_rejection": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before/after.csv"},
    "27_multidataset_generalization": {"experiment": "E2_E3_benchmarks (4 datasets)", "contract_file": "multidataset_benchmark.csv"},
    "28_leave_one_attack_out_breakdown": {"experiment": "E6_loao", "contract_file": "loao_breakdown.csv"},
    "29_training_dynamics_gating_temperature": {"experiment": "E2_E3_benchmarks (Phase I losses)", "contract_file": "training_dynamics.csv"},
    "30_hyperparameter_sensitivity": {"experiment": "A2_vib_beta_*", "contract_file": "hyperparameter_sensitivity.csv"},
    "31_client_forgetting_mitigation": {"experiment": "E1_dirichlet_01 / E2_baselines", "contract_file": "client_forgetting_mitigation.csv"},
}


def _plot_and_tabulate_all_baselines(results_root: Path, figures_dir: Path, tables_dir: Path) -> None:
    """Export comprehensive plot and table of all 13 baselines + FedTROS-MC from E2_baselines."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    e2_summary_path = results_root / "E2_baselines" / "b_nat" / "summary.json"
    if not e2_summary_path.exists():
        return

    payload = json.loads(e2_summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if not summary:
        return

    rows = []
    for method_name, metrics in summary.items():
        display_name = "Proposed (FedTROS-MC)" if "FedTROS-MC" in method_name else method_name
        rows.append({
            "method": display_name,
            "auroc_pct": float(metrics["auroc"]["mean"] * 100.0),
            "aupr_pct": float(metrics["aupr"]["mean"] * 100.0),
            "fpr95_pct": float(metrics.get("fpr95", {"mean": 0.0})["mean"] * 100.0),
            "unknown_f1_pct": float(metrics["unknown_f1"]["mean"] * 100.0),
            "open_set_macro_f1_pct": float(metrics["open_set_macro_f1"]["mean"] * 100.0),
            "closed_accuracy_pct": float(metrics["accuracy"]["mean"] * 100.0),
            "kfr_pct": float(metrics["kfr"]["mean"] * 100.0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "all_baselines_summary.csv", index=False)

    # Generate LaTeX table for all 13 baselines + Proposed
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive comparison across all 13 federated and open-set baselines versus the proposed framework under Non-IID Dirichlet skew ($\alpha_D = 0.5$).}",
        r"\label{tab:all_13_baselines}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Closed Acc. (\%) $\uparrow$ & AUROC (\%) $\uparrow$ & AUPR (\%) $\uparrow$ & FPR@95 (\%) $\downarrow$ & Unknown-F1 (\%) $\uparrow$ & Open-Set Macro-F1 (\%) $\uparrow$ & KFR (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        m_str = f"\\textbf{{{r['method']}}}" if "Proposed" in str(r["method"]) else str(r["method"])
        tex_lines.append(
            f"{m_str} & {r['closed_accuracy_pct']:.2f} & {r['auroc_pct']:.2f} & {r['aupr_pct']:.2f} & "
            f"{r['fpr95_pct']:.2f} & {r['unknown_f1_pct']:.2f} & {r['open_set_macro_f1_pct']:.2f} & {r['kfr_pct']:.2f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    (tables_dir / "table_all_baselines.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # Generate Figure 32: All 13 Baselines + Proposed Comparison
    configure_academic_style()
    df_plot = df.sort_values("open_set_macro_f1_pct", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 6.2), layout="constrained")
    y = np.arange(len(df_plot))
    height = 0.38

    b1 = ax.barh(
        y - height / 2,
        df_plot["open_set_macro_f1_pct"],
        height=height,
        label="Open-Set Macro-F1 (%)",
        color=COLORS["Proposed"],
        edgecolor="#000000",
        linewidth=0.8,
        hatch="///",
    )
    b2 = ax.barh(
        y + height / 2,
        df_plot["auroc_pct"],
        height=height,
        label="Unknown Detection AUROC (%)",
        color=COLORS["FedAvg"],
        edgecolor="#000000",
        linewidth=0.8,
        hatch="\\\\\\",
    )
    for bar in b1:
        w = bar.get_width()
        ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=7.8, weight="bold")
    for bar in b2:
        w = bar.get_width()
        ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=7.8)

    ax.set_yticks(y, df_plot["method"], fontsize=9.2)
    ax.set_xlabel("Performance (%)")
    ax.set_xlim(0, 112)
    ax.set_title("Comprehensive Benchmark Across All 13 Baselines and Proposed Method (E2)", fontsize=11.8, weight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    apply_academic_spines(ax)
    save_figure(fig, figures_dir, "32_all_baselines_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all FedTROS-MC publication figures and LaTeX tables from real experiment outputs."
    )
    parser.add_argument("--results_root", type=str, default="results/runs_v2")
    parser.add_argument("--figures_dir", type=str, default="results/runs_v2/figures")
    parser.add_argument("--tables_dir", type=str, default="results/runs_v2/tables")
    parser.add_argument("--refresh_contract", action="store_true", help="Re-export contract CSVs/JSON from real PyTorch runs.")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    contract_dir = results_root / "plot_contract"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_contract or not (contract_dir / "result.json").exists():
        print(f"[INFO] Exporting real-data plot contract into {contract_dir} ...")
        finalize_real_plot_contract(
            results_root=results_root,
            contract_dir=contract_dir,
            rounds=5,
            spc=250,
            seed=42,
        )

    results_json = contract_dir / "result.json"
    print("[1/7] Generating Base Figures (01-15 + 02_2x2 + 09_10 comparisons) ...")
    generate_base_plots(results_json_path=results_json, output_dir=figures_dir, data_dir=contract_dir)

    print("[2/7] Generating Scalability Figures (16-20) ...")
    generate_scalability_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[3/7] Generating Trade-off & Pareto Figures (21-22) ...")
    generate_tradeoff_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[4/7] Generating Non-Scalability Analysis Figures (23-26) ...")
    generate_analysis_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[5/7] Generating Extended Q1 Figures (27-31) ...")
    generate_extended_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[6/7] Generating All-13-Baselines Comparison (32) and Deep Q1 Figures ...")
    _plot_and_tabulate_all_baselines(results_root=results_root, figures_dir=figures_dir, tables_dir=tables_dir)
    replot_all_figures(results_dir=results_root, figures_dir=figures_dir)

    print("[7/7] Generating LaTeX Summary Tables ...")
    generate_all_latex_tables(tables_out=tables_dir, docs_out=tables_dir, data_dir=contract_dir)

    manifest: Dict[str, Any] = {}
    missing = []
    for stem in ALL_FIGURE_STEMS:
        png_p = figures_dir / f"{stem}.png"
        pdf_p = figures_dir / f"{stem}.pdf"
        if not png_p.exists() or not pdf_p.exists():
            missing.append(stem)
        info = STEM_PROVENANCE_MAP.get(stem, {})
        manifest[stem] = {
            "source_experiment": info.get("experiment", "real_suite"),
            "contract_source": str(contract_dir / info.get("contract_file", "result.json")),
            "png": str(png_p),
            "pdf": str(pdf_p),
        }

    if (figures_dir / "32_all_baselines_comparison.png").exists():
        manifest["32_all_baselines_comparison"] = {
            "source_experiment": "E2_baselines (all 13 baselines + Proposed)",
            "contract_source": str(results_root / "E2_baselines" / "b_nat" / "summary.json"),
            "png": str(figures_dir / "32_all_baselines_comparison.png"),
            "pdf": str(figures_dir / "32_all_baselines_comparison.pdf"),
        }

    if missing:
        raise RuntimeError(f"Missing expected figure outputs: {missing}")

    manifest_path = figures_dir / "figure_provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SUCCESS] All {len(ALL_FIGURE_STEMS)} reference figure stems + extended baseline/ablation figures generated in {figures_dir}")
    print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()
