"""Comprehensive ablation study suite examining core algorithmic components.

Aligned with LaTeX Section 6.6:
  - Table 8: Local representation guidance vs. pure empirical risk
  - Table 9: Anchor Regularization as a failure-control mechanism (None vs Fixed vs Adaptive)
  - Component ablations across VIB, Conformal calibration, Class-Balanced weighting, and Byzantine MAD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict
import yaml

from src.config import load_config
from src.tracking.logger import setup_logger
from src.utils.io import save_json
from experiments.run_baselines import evaluate_method_configuration
from src.visualization.latex_tables import (
    generate_latex_benchmark_table,
    generate_latex_anchor_ablation_table,
)


def run_component_ablations(
    rounds: int = 10,
    beta_dirichlet: float = 0.1,
    output_dir: str = "results/ablations",
) -> Dict[str, Dict[str, float]]:
    """Run standard 6-component ablation suite."""
    logger = setup_logger(name="ablation_runner")
    logger.info(f"Executing 6-component ablation suite (rounds={rounds}, beta={beta_dirichlet})...")

    ablations = {
        "Full Method (Proposed)": "configs/default.yaml",
        "w/o VIB Latent Regularization": "configs/ablations/no_vib.yaml",
        "w/o Orthogonal Anchoring": "configs/ablations/no_anchor.yaml",
        "w/o Primal-Dual Consensus": "configs/ablations/no_primal_dual.yaml",
        "w/o Covariance Prototype Aggregation": "configs/ablations/no_cov_proto.yaml",
        "w/o Mondrian Conformal Calibration": "configs/ablations/no_conformal.yaml",
        "w/o Class-Balanced Focal Loss": "configs/ablations/no_cb_loss.yaml",
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, config_file in ablations.items():
        logger.info(f"--> Running Component Ablation: {name}")
        metrics = evaluate_method_configuration(
            name, config_file, rounds=rounds, beta_dirichlet=beta_dirichlet
        )
        results[name] = metrics
        logger.info(
            f"    AUROC: {metrics['auroc']*100:.2f}% | "
            f"Unk-Recall: {metrics['unknown_recall']*100:.2f}% | "
            f"Unk-F1: {metrics['unknown_f1']*100:.2f}% | "
            f"OS Macro-F1: {metrics['open_set_macro_f1']*100:.2f}% | "
            f"KFR: {metrics['kfr']*100:.2f}%"
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(results, out_p / f"component_ablations_beta_{beta_dirichlet}.json")

    latex_tbl = generate_latex_benchmark_table(
        results,
        caption=f"Component Ablation Study on Open-Set Detection and False Alarm Control ($\\alpha_D = {beta_dirichlet}$)",
        label=f"tab:component_ablations_{str(beta_dirichlet).replace('.', '_')}",
    )
    with open(out_p / f"table_component_ablations_beta_{beta_dirichlet}.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Component ablation suite completed. Table written to {out_p}")
    return results


def run_anchor_ablations(
    rounds: int = 10,
    beta_dirichlet: float = 0.1,
    output_dir: str = "results/ablations",
) -> Dict[str, Dict[str, float]]:
    """Run Anchor Regularization ablation matching LaTeX Table 9 (None vs Fixed vs Adaptive)."""
    logger = setup_logger(name="anchor_ablation")
    logger.info(f"Executing Anchor Regularization ablation (Table 9, beta={beta_dirichlet})...")

    anchor_variants = {
        "None (w/o Anchor)": {
            "loss": {"lambda_anchor": 0.0, "max_lambda_anchor": 0.0}
        },
        "Fixed Anchor": {
            "loss": {
                "lambda_anchor": 1.0,
                "max_lambda_anchor": 1.0,
                "anchor_kappa": 0.0,
                "rho_warm": 0.0,
            }
        },
        "Adaptive Dynamic Anchor": {
            "loss": {
                "lambda_anchor": 1.0,
                "max_lambda_anchor": 1.0,
                "anchor_kappa": 10.0,
                "rho_warm": 0.15,
            }
        },
    }

    results: Dict[str, Dict[str, float]] = {}
    temp_cfg_path = Path("configs/ablations/_temp_anchor.yaml")
    temp_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for name, overrides in anchor_variants.items():
            logger.info(f"--> Running Anchor Variant: {name}")
            base_cfg = load_config("configs/default.yaml", overrides=overrides)
            with open(temp_cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(base_cfg.model_dump(), f)

            metrics = evaluate_method_configuration(
                name, str(temp_cfg_path), rounds=rounds, beta_dirichlet=beta_dirichlet
            )
            results[name] = metrics
            logger.info(
                f"    AUROC: {metrics['auroc']*100:.2f}% | "
                f"Unk-Recall: {metrics['unknown_recall']*100:.2f}% | "
                f"Unk-F1: {metrics['unknown_f1']*100:.2f}% | "
                f"KFR: {metrics['kfr']*100:.2f}%"
            )
    finally:
        if temp_cfg_path.exists():
            temp_cfg_path.unlink()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(results, out_p / f"anchor_ablations_beta_{beta_dirichlet}.json")

    latex_tbl = generate_latex_anchor_ablation_table(
        results,
        caption=f"Anchor Regularization as Failure-Control Mechanism (LaTeX Table 9, $\\alpha_D = {beta_dirichlet}$)",
        label=f"tab:anchor_ablation_{str(beta_dirichlet).replace('.', '_')}",
    )
    with open(out_p / f"table_anchor_ablation_beta_{beta_dirichlet}.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Anchor ablation suite completed. Table written to {out_p}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Suites")
    parser.add_argument("--suite", type=str, default="components", choices=["components", "anchor", "all"])
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per trial")
    parser.add_argument("--beta", type=float, default=0.1, help="Dirichlet concentration alpha_D")
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    args = parser.parse_args()

    if args.suite in ["components", "all"]:
        run_component_ablations(rounds=args.rounds, beta_dirichlet=args.beta, output_dir=args.output_dir)
    if args.suite in ["anchor", "all"]:
        run_anchor_ablations(rounds=args.rounds, beta_dirichlet=args.beta, output_dir=args.output_dir)
