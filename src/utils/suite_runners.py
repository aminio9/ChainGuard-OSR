"""Experiment suite runners for E2 baselines, A1/A3/A4/A6 ablations, E6 LOAO, and E7 scalability."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.utils.artifact_logger import aggregate_MultiSeed_Summary, save_seed_run_artifacts
from experiments.run_ablations import run_bla_ablations, run_deep_q1_ablations
from experiments.run_baselines import evaluate_method_configuration
from experiments.run_conformal_sweep import run_conformal_sweep_experiment
from experiments.run_covariance_distance_ablations import run_covariance_distance_ablations
from experiments.run_loao import run_loao_benchmark
from experiments.run_scalability import run_scalability_experiment


def run_suite_extensions(
    project_root: Path,
    output_root: Path,
    seeds: List[int],
    rounds: int,
    spc: Optional[int],
) -> None:
    """Run all 13 baselines (E2), component ablations (A1), A2 BLA, A3/A4, A6, deep Q1 ablations, E6 LOAO, and E7 scalability."""
    key_methods = {
        "FedAvg": "configs/baselines/fedavg_msp.yaml",
        "FedProx": "configs/baselines/fedprox_openmax.yaml",
        "SCAFFOLD": "configs/baselines/scaffold.yaml",
        "FedDyn": "configs/baselines/feddyn.yaml",
        "FedRoD": "configs/baselines/fedrod.yaml",
        "FedDecorr": "configs/baselines/feddecorr.yaml",
        "FedProto": "configs/baselines/fedproto_energy.yaml",
        "FedPD": "configs/baselines/fedpd.yaml",
        "FedPD++": "configs/baselines/fedpd_plus_plus.yaml",
        "FedOSS": "configs/baselines/fedoss.yaml",
        "OpenMax": "configs/baselines/fedprox_openmax.yaml",
        "Energy-OOD": "configs/baselines/fedproto_energy.yaml",
        "Vanilla-Mahalanobis": "configs/baselines/vanilla_mahalanobis.yaml",
        "FedTROS-MC (Ours)": "configs/default.yaml",
    }

    ablation_configs = {
        "Full FedTROS-MC": "configs/default.yaml",
        "w/o VIB Latent Regularization (A1)": "configs/ablations/no_vib.yaml",
        "w/o Logit-Adjusted Loss (A2)": "configs/ablations/no_bla_loss.yaml",
        "w/o Decoupled Aggregation (A3)": "configs/ablations/no_decoupled.yaml",
        "w/o Mondrian Conformal Calibration (A6)": "configs/ablations/no_conformal.yaml",
    }

    for seed in seeds:
        set_seed(seed)
        cfg = load_config(str(project_root / "configs" / "default.yaml"))
        cfg.federated.num_rounds = rounds
        cfg.experiment.seed = seed

        split_meta = {
            "dataset": "b_nat",
            "samples_per_class": spc,
            "beta_dirichlet": 0.5,
            "rounds": rounds,
        }

        # E2 Baselines (All 13 modular baselines + Proposed FedTROS-MC)
        baseline_metrics: Dict[str, Any] = {}
        for m_name, cfg_path in key_methods.items():
            t0 = time.perf_counter()
            m_res = evaluate_method_configuration(
                method_name=m_name,
                config_path=str(project_root / cfg_path),
                rounds=rounds,
                beta_dirichlet=0.5,
                samples_per_class=spc,
                seed=seed,
            )
            m_res["wall_time_sec"] = float(time.perf_counter() - t0)
            baseline_metrics[m_name] = m_res

        save_seed_run_artifacts(
            output_root=output_root,
            experiment_id="E2_baselines",
            dataset_name="b_nat",
            seed=seed,
            config=cfg,
            metrics=baseline_metrics,
            split_metadata=split_meta,
        )

        # A1 Component Ablations
        a1_metrics: Dict[str, Any] = {}
        for v_name, cfg_path in ablation_configs.items():
            a1_metrics[v_name] = evaluate_method_configuration(
                method_name=v_name,
                config_path=str(project_root / cfg_path),
                rounds=rounds,
                beta_dirichlet=0.5,
                samples_per_class=spc,
                seed=seed,
            )

        save_seed_run_artifacts(
            output_root=output_root,
            experiment_id="A1_component_ablations",
            dataset_name="b_nat",
            seed=seed,
            config=cfg,
            metrics=a1_metrics,
            split_metadata=split_meta,
        )

    aggregate_MultiSeed_Summary(output_root, "E2_baselines", "b_nat")
    aggregate_MultiSeed_Summary(output_root, "A1_component_ablations", "b_nat")

    # Run A2 BLA sweep, Deep Q1 Ablations (A1 SVD, A4 Frobenius, A5 Pareto, A6 Conformal, LDP+MIA),
    # A3/A4 Cov/Dist, A6 Conformal Sweep, E6 LOAO, E7 Scalability, and E5 Comm into output_root
    run_bla_ablations(
        rounds=rounds,
        beta_dirichlet=0.5,
        samples_per_class=spc,
        output_dir=str(output_root / "A2_bla_ablations"),
    )
    (output_root / "ablations").mkdir(parents=True, exist_ok=True)
    import shutil
    for f in (output_root / "A2_bla_ablations").glob("bla_ablations_*.json"):
        shutil.copy(f, output_root / "ablations" / f.name)

    run_deep_q1_ablations(
        dataset_name="b_nat",
        num_clients=10,
        rounds=min(rounds, 8),
        beta_dirichlet=0.5,
        samples_per_class=spc,
        output_dir=str(output_root / "ablations"),
        seed=seeds[0],
    )
    run_covariance_distance_ablations(
        dataset_name="b_nat",
        rounds=rounds,
        beta_dirichlet=0.5,
        max_samples=spc,
        output_dir=str(output_root / "A3_A4_cov_dist"),
        seed=seeds[0],
    )
    run_conformal_sweep_experiment(
        dataset_name="b_nat",
        rounds=rounds,
        beta_dirichlet=0.5,
        max_samples=spc,
        output_dir=str(output_root / "A6_conformal"),
        seed=seeds[0],
    )
    run_loao_benchmark(
        rounds=rounds,
        max_samples=spc,
        output_dir=str(output_root / "E6_loao"),
    )
    run_scalability_experiment(
        dataset_name="b_nat",
        client_counts=[10, 50, 100],
        rounds=rounds,
        beta_dirichlet=0.5,
        max_samples=spc,
        output_dir=str(output_root / "E7_scalability"),
        seed=seeds[0],
    )

    # E5 Communication Efficiency artifacts
    e5_dir = output_root / "E5_comm" / "b_nat"
    e5_dir.mkdir(parents=True, exist_ok=True)
    num_params = 512 * 34 + 256 * 512 + 128 * 256 + 64 * 128
    phase1_bytes = int(2 * num_params * 4 * rounds)
    phase2_bytes = int(4 * (64 + 4 * (16 ** 2) + 2) * 4)
    e5_metrics = {
        "rounds": float(rounds),
        "phase1_comm_bytes_per_client": float(phase1_bytes),
        "phase2_comm_bytes_per_client": float(phase2_bytes),
        "total_comm_bytes_per_client": float(phase1_bytes + phase2_bytes),
        "total_comm_mb_per_client": float((phase1_bytes + phase2_bytes) / (1024.0 * 1024.0)),
        "communication_reduction_vs_fedavg_x": 1.0,
    }
    (e5_dir / f"seed_{seeds[0]}.json").write_text(json.dumps({"metrics": e5_metrics}, indent=2), encoding="utf-8")
    (e5_dir / "summary.json").write_text(
        json.dumps({"summary": {k: {"mean": float(v), "std": 0.0} for k, v in e5_metrics.items()}}, indent=2),
        encoding="utf-8",
    )
