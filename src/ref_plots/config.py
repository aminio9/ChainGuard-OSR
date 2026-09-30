"""Configuration paths and registries for the existing publication plot system."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_ROOT / "src"
DATA_DIR = PACKAGE_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
FIGURES_OUTPUT_DIR = OUTPUTS_DIR / "figures"
TABLES_OUTPUT_DIR = OUTPUTS_DIR / "tables"
DOCS_DIR = PACKAGE_ROOT / "docs"
ARCHIVE_DIR = PACKAGE_ROOT / "archive"

DEFAULT_RESULTS_JSON = METADATA_DIR / "result.json"

BASE_PLOT_STEMS = {
    "plot_01_scalability": ["01_scalability_clients"],
    "plot_02_non_iid_distribution": [
        "02_internal_non_iid_distribution",
        "02_internal_non_iid_distribution_2x2",
    ],
    "plot_03_convergence_mild": ["03_convergence_mild_non_iid"],
    "plot_04_convergence_hard": ["04_convergence_hard_non_iid"],
    "plot_05_score_distribution": ["05_mahalanobis_nonconformity_distribution"],
    "plot_06_openness": ["06_openness_vs_auroc"],
    "plot_07_roc": ["07_roc_unknown_detection"],
    "plot_08_pr": ["08_pr_unknown_detection"],
    "plot_09_10_confusion": [
        "09_confusion_before_open_set",
        "10_confusion_after_open_set",
        "09_10_confusion_comparison",
        "09_10_confusion_comparison_percentage",
    ],
    "plot_11_robustness": ["11_non_iid_robustness_boxplot"],
    "plot_12_latent_geometry": ["12_adaptive_multicenter_geometry"],
    "plot_13_communication": ["13_communication_efficiency"],
    "plot_14_ablation": ["14_ablation_study"],
    "plot_15_method_comparison": ["15_method_comparison"],
}

SCALABILITY_PLOT_STEMS = [
    "16_scalability_convergence_common_rounds",
    "17_scalability_quality_vs_wallclock",
    "18_scalability_runtime_breakdown",
    "19_scalability_client_fairness",
    "20_scalability_open_set_detection",
]

TRADEOFF_PLOT_STEMS = [
    "21_scalability_close_open_tradeoff",
    "22_scalability_efficiency_robustness_pareto",
]

ANALYSIS_PLOT_STEMS = [
    "23_conformal_calibration_validation",
    "24_classwise_open_set_tradeoff",
    "25_non_iid_accuracy_retention",
    "26_confusion_delta_after_rejection",
]

EXTENDED_PLOT_STEMS = [
    "27_multidataset_generalization",
    "28_leave_one_attack_out_breakdown",
    "29_training_dynamics_gating_temperature",
    "30_hyperparameter_sensitivity",
    "31_client_forgetting_mitigation",
]

ALL_FIGURE_STEMS = (
    [stem for stems in BASE_PLOT_STEMS.values() for stem in stems]
    + SCALABILITY_PLOT_STEMS
    + TRADEOFF_PLOT_STEMS
    + ANALYSIS_PLOT_STEMS
    + EXTENDED_PLOT_STEMS
)

# Kept for the optional bundle-preview renderer.  It is not the master plotting
# pipeline and does not replace the 29-figure registry above.
MAIN_FIGURE_STEMS = [
    "01_client_class_support_heatmap",
    "02_unknown_score_operating_diagnostic",
    "03_noniid_closed_macro_f1",
    "04_noniid_osr_operating_tradeoff",
    "05_leave_one_attack_out_heatmap",
    "06_component_ablation_deltas",
    "07_performance_vs_communication",
    "08_fixed_data_scalability",
]
