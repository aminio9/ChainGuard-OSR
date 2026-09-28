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
from src.visualization.latex_tables import generate_latex_benchmark_table


def run_ablation_suite(rounds: int = 10, output_dir: str = "results/ablations") -> None:
    logger = setup_logger(name="ablation_runner")
    logger.info("Executing comprehensive ablation suite across 6 core components...")

    ablations = {
        "Full Method": "configs/default.yaml",
        "w/o VIB Latent Regularization": "configs/ablations/no_vib.yaml",
        "w/o Orthogonal Anchoring": "configs/ablations/no_anchor.yaml",
        "w/o Primal-Dual Consensus": "configs/ablations/no_primal_dual.yaml",
        "w/o Covariance Prototype Aggregation": "configs/ablations/no_cov_proto.yaml",
        "w/o Mondrian Conformal Calibration": "configs/ablations/no_conformal.yaml",
        "w/o Class-Balanced Focal Loss": "configs/ablations/no_cb_loss.yaml",
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, config_file in ablations.items():
        logger.info(f"--> Running Ablation: {name}")

        # If ablation yaml specifies overrides, merge with default
        if config_file != "configs/default.yaml":
            with open(config_file, "r", encoding="utf-8") as f:
                raw_ablation = yaml.safe_load(f)
            overrides = raw_ablation.get("overrides", {})
        else:
            overrides = {}

        base_cfg = load_config("configs/default.yaml", overrides=overrides)
        # Temporarily save merged config
        temp_cfg_path = Path("configs/ablations/_active_ablation.yaml")
        temp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(base_cfg.model_dump(), f)

        metrics = evaluate_method_configuration(name, str(temp_cfg_path), rounds=rounds)
        results[name] = metrics

        logger.info(
            f"    AUROC: {metrics['auroc']*100:.2f}% | Open-Set F1: {metrics['open_set_macro_f1']*100:.2f}% | FAR: {metrics['false_alarm_rate']*100:.2f}%"
        )

        if temp_cfg_path.exists():
            temp_cfg_path.unlink()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(results, out_p / "ablation_results.json")

    latex_tbl = generate_latex_benchmark_table(
        results,
        caption="Component Ablation Study on Open-Set Detection and False Alarm Control",
        label="tab:ablation_results",
    )
    with open(out_p / "table_ablations.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Ablation studies completed successfully. Output in {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per ablation trial")
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    args = parser.parse_args()

    run_ablation_suite(rounds=args.rounds, output_dir=args.output_dir)
