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

from src.ref_plots.config import ALL_FIGURE_STEMS
from src.ref_plots.contract_exporter_part3 import finalize_real_plot_contract
from src.ref_plots.figures.analysis_plots import generate_analysis_plots
from src.ref_plots.figures.base_plots import generate_base_plots
from src.ref_plots.figures.extended_plots import generate_extended_plots
from src.ref_plots.figures.scalability_plots import generate_scalability_plots
from src.ref_plots.figures.tradeoff_plots import generate_tradeoff_plots
from src.ref_plots.tables.latex_tables import generate_all_latex_tables


STEM_PROVENANCE_MAP: Dict[str, Dict[str, Any]] = {
    "01_scalability_clients": {"experiment": "E7_scalability", "contract_file": "result.json#plot_01_scalability"},
    "02_non_iid_client_distribution": {"experiment": "E1_dirichlet_01", "contract_file": "client_distribution_alpha01.csv"},
    "02_non_iid_partition_2x2_comparison": {"experiment": "E1_iid..E1_dirichlet_01", "contract_file": "client_distribution_iid_alpha_sweep.csv"},
    "03_convergence_mild_non_iid": {"experiment": "E1_iid / E2_baselines (alpha=1.0)", "contract_file": "convergence_alpha1.csv"},
    "04_convergence_hard_non_iid": {"experiment": "E1_dirichlet_01 / E2_baselines (alpha=0.1)", "contract_file": "convergence_alpha01.csv"},
    "05_mahalanobis_nonconformity_distribution": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_scores.csv"},
    "06_openness_vs_auroc": {"experiment": "E2_E3_benchmarks (4 datasets)", "contract_file": "result.json#plot_06_openness"},
    "07_roc_curve_unknown_detection": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_roc_curve.csv"},
    "08_precision_recall_unknown_detection": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_pr_curve.csv"},
    "09_confusion_matrix_before_rejection": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before.csv"},
    "10_confusion_matrix_after_rejection": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_after.csv"},
    "09_10_confusion_matrix_comparison": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before/after.csv"},
    "09_10_confusion_matrix_comparison_percent": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_confusion_before/after.csv"},
    "11_non_iid_robustness_across_alpha": {"experiment": "E1_iid..E1_dirichlet_01", "contract_file": "result.json#plot_11_robustness"},
    "12_latent_space_geometry": {"experiment": "E2_E3_benchmarks (B-NAT)", "contract_file": "exp2_latent_projection.csv"},
    "13_accuracy_vs_communication": {"experiment": "E2_baselines / E5_comm", "contract_file": "communication_alpha1.csv"},
    "14_ablation_study": {"experiment": "A1_component_ablations + A5_blocks", "contract_file": "result.json#plot_14_ablation"},
    "15_method_comparison_mild_non_iid": {"experiment": "E2_baselines (alpha=1.0)", "contract_file": "result.json#plot_15_method_comparison"},
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all 31 FedTROS-MC figures (34 stems) in the exact reference style from real experiment outputs."
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
    print("[1/6] Generating Base Figures (01-15 + 02_2x2 + 09_10 comparisons) ...")
    generate_base_plots(results_json_path=results_json, output_dir=figures_dir, data_dir=contract_dir)

    print("[2/6] Generating Scalability Figures (16-20) ...")
    generate_scalability_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[3/6] Generating Trade-off & Pareto Figures (21-22) ...")
    generate_tradeoff_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[4/6] Generating Non-Scalability Analysis Figures (23-26) ...")
    generate_analysis_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[5/6] Generating Extended Q1 Figures (27-31) ...")
    generate_extended_plots(data_dir=contract_dir, figures_out=figures_dir, tables_out=tables_dir)

    print("[6/6] Generating LaTeX Summary Tables ...")
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

    if missing:
        raise RuntimeError(f"Missing expected figure outputs: {missing}")

    manifest_path = figures_dir / "figure_provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SUCCESS] All {len(ALL_FIGURE_STEMS)} figure stems (PNG + PDF) generated in {figures_dir}")
    print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()
