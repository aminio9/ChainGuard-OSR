"""Master Experiment Orchestrator executing the complete empirical suite (E1 to E9, A1 to A6).

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
from experiments.run_baselines import run_all_baselines
from experiments.run_real_benchmarks import run_all_four_benchmarks
from experiments.run_loao import run_loao_benchmark
from experiments.run_scalability import run_scalability_experiment
from experiments.run_wire_speed_profiling import run_wire_speed_experiment
from experiments.run_byzantine_robustness import run_byzantine_robustness_experiment
from experiments.run_ablations import run_component_ablations, run_anchor_ablations
from experiments.run_covariance_distance_ablations import run_covariance_distance_ablations
from experiments.run_conformal_sweep import run_conformal_sweep_experiment


def run_master_suite(
    dry_run: bool = False,
    output_root: str = "results",
) -> None:
    """Execute all 9 main experiments and 6 ablations."""
    logger = setup_logger(name="master_orchestrator")
    mode_str = "DRY-RUN (verification mode)" if dry_run else "FULL BENCHMARK (production mode)"
    logger.info(f"=================================================================")
    logger.info(f"Starting Master Evaluation Suite: {mode_str}")
    logger.info(f"=================================================================")

    start_all = time.perf_counter()
    rounds = 2 if dry_run else 10
    max_samples = 300 if dry_run else 200

    # E1, E2, E3, E4: Canonical Baselines & Non-IID Skew
    logger.info("\n>>> [1/7] Running E1-E4: Canonical Baselines & Non-IID Skew...")
    run_all_baselines(rounds=rounds, beta_dirichlet=0.5, output_dir=f"{output_root}/baselines")

    # E5: Cross-Dataset on 4 Benchmarks (B-NAT, B-TAT, ToN-IoT, CIC-IDS2017)
    logger.info("\n>>> [2/7] Running E5: Cross-Dataset Generalization on 4 Real Benchmarks...")
    run_all_four_benchmarks(
        rounds=rounds,
        samples_per_class=max_samples,
        beta_dirichlet=0.5,
        output_dir=f"{output_root}/cross_dataset",
    )

    # E6: Leave-One-Attack-Out (LOAO) on B-NAT
    logger.info("\n>>> [3/7] Running E6: Leave-One-Attack-Out (LOAO) Semantic Benchmark...")
    run_loao_benchmark(rounds=rounds, max_samples=max_samples, output_dir=f"{output_root}/loao")

    # E7: Scalability across K in {10, 50, 100} clients
    logger.info("\n>>> [4/7] Running E7: Federation Scalability (K in {10, 50, 100})...")
    run_scalability_experiment(
        dataset_name="b_nat",
        client_counts=[10, 50] if dry_run else [10, 50, 100],
        rounds=rounds,
        beta_dirichlet=0.5,
        max_samples=max_samples,
        output_dir=f"{output_root}/scalability",
    )

    # E8: Wire-Speed Latency, Complexity, and FLOPs Profiling
    logger.info("\n>>> [5/7] Running E8: Wire-Speed Latency and FLOPs Profiling...")
    run_wire_speed_experiment(output_dir=f"{output_root}/wire_speed")

    # E9: Byzantine Poisoning Robustness (f in {10%, 20%, 30%})
    logger.info("\n>>> [6/7] Running E9: Byzantine Poisoning Robustness (f in {0%, 10%, 20%, 30%})...")
    run_byzantine_robustness_experiment(
        dataset_name="b_nat",
        num_clients=10,
        rounds=rounds,
        poison_fractions=[0.0, 0.20] if dry_run else [0.0, 0.10, 0.20, 0.30],
        max_samples=max_samples,
        output_dir=f"{output_root}/byzantine",
    )

    # A1 to A6: Comprehensive Ablation Matrix
    logger.info("\n>>> [7/7] Running Ablations A1-A6 (Losses, Anchors, Covariance, Metrics, Conformal)...")
    run_component_ablations(rounds=rounds, beta_dirichlet=0.5, output_dir=f"{output_root}/ablations")
    run_anchor_ablations(rounds=rounds, beta_dirichlet=0.5, output_dir=f"{output_root}/ablations")
    run_covariance_distance_ablations(dataset_name="b_nat", rounds=rounds, beta_dirichlet=0.5, max_samples=max_samples, output_dir=f"{output_root}/ablations_cov_metric")
    run_conformal_sweep_experiment(dataset_name="b_nat", rounds=rounds, beta_dirichlet=0.5, max_samples=max_samples, output_dir=f"{output_root}/conformal_sweep")

    total_time = time.perf_counter() - start_all
    logger.info(f"\n=================================================================")
    logger.info(f"Master Evaluation Suite Completed in {total_time:.2f} seconds ({total_time/60.0:.2f} minutes)!")
    logger.info(f"All artifacts, summary JSONs, and publication-ready LaTeX tables are stored in: {output_root}")
    logger.info(f"=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Experiment Suite (E1-E9, A1-A6)")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry-run to verify mathematical integrity")
    parser.add_argument("--output_root", type=str, default="results")
    args = parser.parse_args()

    run_master_suite(dry_run=args.dry_run, output_root=args.output_root)
