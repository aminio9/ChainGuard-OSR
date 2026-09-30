"""Master Experiment Orchestrator executing the complete empirical suite (E1 to E8, A1 to A6).

Allows execution of individual experiments or the entire benchmark suite in either:
  1. Full production mode (e.g., T=100 rounds, all samples).
  2. Quick dry-run mode (e.g., T=2 rounds, small sample subsets) to verify end-to-end mathematical correctness.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.tracking.logger import setup_logger
from experiments.run_baselines import (
    run_all_baselines,
    run_closed_set_baselines,
    run_closed_set_multi_skew_benchmark,
)
from experiments.run_real_benchmarks import run_all_four_benchmarks
from experiments.run_loao import run_loao_benchmark
from experiments.run_scalability import run_scalability_experiment
from experiments.run_wire_speed_profiling import run_wire_speed_experiment
from experiments.run_ablations import run_component_ablations, run_anchor_ablations
from experiments.run_covariance_distance_ablations import run_covariance_distance_ablations
from experiments.run_conformal_sweep import run_conformal_sweep_experiment


def run_master_suite(
    mode: str = "paper",
    dry_run: bool = False,
    output_root: str = "results",
) -> None:
    """Execute all 8 main experiments and 6 ablations."""
    logger = setup_logger(name="master_orchestrator")
    if dry_run:
        mode = "debug"

    mode_str = {
        "debug": "DRY-RUN (verification mode, fast subset)",
        "paper": "PAPER TABLE 1 BENCHMARK (exact Q1 publication protocol)",
        "full": "FULL MASSIVE BENCHMARK (100% raw dataset partitions)",
    }.get(mode, mode)

    logger.info(f"=================================================================")
    logger.info(f"Starting Master Evaluation Suite: {mode_str}")
    logger.info(f"=================================================================")

    start_all = time.perf_counter()
    if mode == "debug":
        rounds = 2
        max_samples = 300
    elif mode == "paper":
        rounds = 15
        max_samples = None
    else:  # full
        rounds = 25
        max_samples = None

    # E1 & E2: Closed-Set Classification Utility (Section 6.2, Tables 1 & 2)
    # Strictly closed-set evaluation - NO OSR rejection metrics or open-set tests
    logger.info("\n>>> [1/7] Running E1 & E2: Closed-Set Classification Utility (Tables 1 & 2)...")
    logger.info("--> Table 1: IID Closed-Set Classification Benchmark (FedAvg, FedProx, FedTROS-MC)...")
    run_closed_set_baselines(
        rounds=rounds,
        beta_dirichlet=None,
        samples_per_class=max_samples,
        output_dir=f"{output_root}/closed_set",
    )
    logger.info("--> Table 2: Non-IID Dirichlet Skew Benchmark on FedTROS-MC (alpha_D in {1.0, 0.5, 0.1})...")
    run_closed_set_multi_skew_benchmark(
        rounds=rounds,
        skews=[0.5] if dry_run else [1.0, 0.5, 0.1],
        samples_per_class=max_samples,
        output_dir=f"{output_root}/closed_set",
    )

    # E3 & E4: Canonical Open-Set Detection Baselines & Non-IID Skew (Section 6.3, Table 3)
    logger.info("\n>>> [2/7] Running E3 & E4: Canonical Open-Set Baselines & Non-IID Skew...")
    run_all_baselines(
        rounds=rounds,
        beta_dirichlet=0.5,
        samples_per_class=max_samples,
        output_dir=f"{output_root}/baselines",
    )

    # E5: Cross-Dataset on 4 Benchmarks (B-NAT, B-TAT, ToN-IoT, CIC-IDS2017)
    logger.info("\n>>> [3/7] Running E5: Cross-Dataset Generalization on 4 Real Benchmarks...")
    run_all_four_benchmarks(
        rounds=rounds,
        samples_per_class=max_samples,
        beta_dirichlet=0.5,
        output_dir=f"{output_root}/cross_dataset",
    )

    # E6: Leave-One-Attack-Out (LOAO) on B-NAT
    logger.info("\n>>> [4/7] Running E6: Leave-One-Attack-Out (LOAO) Semantic Benchmark...")
    run_loao_benchmark(rounds=rounds, max_samples=max_samples, output_dir=f"{output_root}/loao")

    # E7: Scalability across K in {10, 50, 100} clients
    logger.info("\n>>> [5/7] Running E7: Federation Scalability (K in {10, 50, 100})...")
    run_scalability_experiment(
        dataset_name="b_nat",
        client_counts=[10, 50] if dry_run else [10, 50, 100],
        rounds=rounds,
        beta_dirichlet=0.5,
        max_samples=max_samples,
        output_dir=f"{output_root}/scalability",
    )

    # E8: Wire-Speed Latency, Complexity, and FLOPs Profiling
    logger.info("\n>>> [6/7] Running E8: Wire-Speed Latency and FLOPs Profiling...")
    run_wire_speed_experiment(output_dir=f"{output_root}/wire_speed")

    # A1 to A6: Comprehensive Ablation Matrix
    logger.info("\n>>> [7/7] Running Ablations A1-A6 (Losses, Anchors, Covariance, Metrics, Conformal)...")
    run_component_ablations(
        rounds=rounds,
        beta_dirichlet=0.5,
        samples_per_class=max_samples,
        output_dir=f"{output_root}/ablations",
    )
    run_anchor_ablations(
        rounds=rounds,
        beta_dirichlet=0.5,
        samples_per_class=max_samples,
        output_dir=f"{output_root}/ablations",
    )
    run_covariance_distance_ablations(
        dataset_name="b_nat",
        rounds=rounds,
        beta_dirichlet=0.5,
        max_samples=max_samples,
        output_dir=f"{output_root}/ablations_cov_metric",
    )
    run_conformal_sweep_experiment(dataset_name="b_nat", rounds=rounds, beta_dirichlet=0.5, max_samples=max_samples, output_dir=f"{output_root}/conformal_sweep")

    total_time = time.perf_counter() - start_all
    logger.info(f"\n=================================================================")
    logger.info(f"Master Evaluation Suite Completed in {total_time:.2f} seconds ({total_time/60.0:.2f} minutes)!")
    logger.info(f"All artifacts, summary JSONs, and publication-ready LaTeX tables are stored in: {output_root}")
    logger.info(f"=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Experiment Suite (E1-E8, A1-A6)")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "full", "debug"], help="Benchmark execution mode")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry-run alias (sets mode to debug)")
    parser.add_argument("--output_root", type=str, default="results")
    args = parser.parse_args()

    run_master_suite(mode=args.mode, dry_run=args.dry_run, output_root=args.output_root)
