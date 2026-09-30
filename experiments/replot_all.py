"""Standalone Decoupled Publication Plot Generator (Zero Retraining Required).

Reads saved `.json` and `.npz` audit bundles from `results/` and regenerates all
IEEE Transactions vector `.pdf` and 300-DPI `.png` figures in seconds.

Usage:
    python experiments/replot_all.py --results_dir results --figures_dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, List
import numpy as np

from src.tracking.logger import setup_logger
from src.visualization.publication_plots import (
    plot_auroc_vs_openness,
    plot_conformal_calibration_diagonal,
    plot_ldp_privacy_utility_mia,
    plot_subblock_pareto_curve,
    plot_svd_singular_value_spectrum,
)


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def replot_all_figures(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "results/figures",
) -> List[str]:
    """Regenerate all IEEE TIFS/TDSC figures from decoupled .json and .npz files without training."""
    logger = setup_logger(name="replot_all")
    res_p = Path(results_dir)
    fig_p = Path(figures_dir)
    fig_p.mkdir(parents=True, exist_ok=True)
    generated_files: List[str] = []

    # 1. Openness vs AUROC Curve (E2)
    openness_json = _load_json_if_exists(res_p / "openness" / "openness_sweep.json")
    if openness_json and "openness_values" in openness_json and "method_results" in openness_json:
        open_vals = openness_json["openness_values"]
        method_res = openness_json["method_results"]
        ci_bounds = openness_json.get("ci_bounds", None)
        out_open = fig_p / "fig_e2_auroc_vs_openness.pdf"
        plot_auroc_vs_openness(open_vals, method_res, out_open, ci_bounds=ci_bounds)
        generated_files.append(str(out_open))
        logger.info(f"Generated {out_open} (.pdf & .png)")
    else:
        logger.warning("Skipping fig_e2_auroc_vs_openness.pdf: missing results/openness/openness_sweep.json")

    # 2. Deep Ablations (A1 SVD Spectrum, A5 Sub-Block Pareto, A6 Conformal Diagonal, LDP+MIA)
    deep_json = _load_json_if_exists(res_p / "ablations" / "q1_deep_ablations.json")
    deep_data = deep_json.get("results", deep_json)
    deep_npz_path = res_p / "ablations" / "q1_deep_ablations.npz"
    deep_npz = dict(np.load(deep_npz_path)) if deep_npz_path.exists() else {}

    # A1: SVD Singular Value Spectrum
    svd_spectra: Dict[str, List[float]] = {}
    for b_vib, label in [
        (0.0, r"$\lambda_{\mathrm{vib}} = 0$ (Collapse)"),
        (1e-3, r"$\lambda_{\mathrm{vib}} = 10^{-3}$"),
        (5e-3, r"$\lambda_{\mathrm{vib}} = 5\times 10^{-3}$ (Proposed)"),
        (1e-1, r"$\lambda_{\mathrm{vib}} = 10^{-1}$ (Over-Smoothed)"),
    ]:
        arr_key = f"svd_norm_{b_vib}"
        if arr_key in deep_npz:
            svd_spectra[label] = deep_npz[arr_key].tolist()
    if svd_spectra:
        out_a1 = fig_p / "fig_a1_svd_spectrum.pdf"
        plot_svd_singular_value_spectrum(svd_spectra, out_a1)
        generated_files.append(str(out_a1))
        logger.info(f"Generated {out_a1} (.pdf & .png)")
    else:
        logger.warning("Skipping fig_a1_svd_spectrum.pdf: missing results/ablations/q1_deep_ablations.npz")

    # A5: Sub-Block Size Pareto Front
    pareto_data = deep_data.get("A5_subblock_pareto", {})
    if pareto_data:
        out_a5 = fig_p / "fig_a5_subblock_pareto.pdf"
        plot_subblock_pareto_curve(pareto_data, out_a5)
        generated_files.append(str(out_a5))
        logger.info(f"Generated {out_a5} (.pdf & .png)")
    else:
        logger.warning("Skipping fig_a5_subblock_pareto.pdf: missing A5_subblock_pareto with real AUROC values")

    # A6: Conformal Calibration Diagonal
    a6_data = deep_data.get("A6_conformal_calibration", {})
    if a6_data and "alphas" in a6_data and "mondrian_kfr" in a6_data and "global_kfr" in a6_data:
        out_a6 = fig_p / "fig_a6_conformal_calibration.pdf"
        plot_conformal_calibration_diagonal(
            a6_data["alphas"], a6_data["mondrian_kfr"], a6_data["global_kfr"], out_a6
        )
        generated_files.append(str(out_a6))
        logger.info(f"Generated {out_a6} (.pdf & .png)")
    else:
        logger.warning("Skipping fig_a6_conformal_calibration.pdf: missing A6_conformal_calibration results")

    # LDP Privacy-Utility-MIA Curve
    ldp_data = deep_data.get("LDP_privacy_mia", {})
    eps_list = [1.0, 2.0, 5.0, 10.0]
    if ldp_data:
        aurocs_ldp = [float(ldp_data[f"eps_{e:.1f}"]["auroc"]) for e in eps_list if f"eps_{e:.1f}" in ldp_data]
        mias_ldp = [float(ldp_data[f"eps_{e:.1f}"]["mia_auc"]) for e in eps_list if f"eps_{e:.1f}" in ldp_data]
        if len(aurocs_ldp) == len(eps_list):
            out_ldp = fig_p / "fig_ldp_privacy_utility_mia.pdf"
            plot_ldp_privacy_utility_mia(eps_list, aurocs_ldp, mias_ldp, out_ldp)
            generated_files.append(str(out_ldp))
            logger.info(f"Generated {out_ldp} (.pdf & .png)")
    else:
        logger.warning("Skipping fig_ldp_privacy_utility_mia.pdf: missing LDP_privacy_mia results")

    logger.info(f"Successfully generated {len(generated_files)} dual-format (.pdf + .png) figures in {fig_p}")
    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Decoupled Re-Plotter (Zero Retraining)")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="results/figures")
    args = parser.parse_args()
    replot_all_figures(results_dir=args.results_dir, figures_dir=args.figures_dir)
