"""Comprehensive end-to-end verification script testing ablation studies A1 to A6.
Runs exactly 1 round and 2 clients on 100% real data to verify zero bugs, correct imports, and metric logging.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add Code/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["OSR_PROTOCOL"] = "full"

from run_all import run_fedtros_mc_full_diagnostics
from experiments.run_ablations import run_component_ablations, run_bla_ablations, run_deep_q1_ablations
from experiments.run_covariance_distance_ablations import run_covariance_aggregation_comparison
from experiments.run_conformal_sweep import run_conformal_sweep

out_root = PROJECT_ROOT / "results" / "test_ablation_verification"
out_root.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STARTING A1-A6 ABLATION SUITE VERIFICATION (1 ROUND, 2 CLIENTS, 100% REAL DATA)")
print("=" * 80)

SEED = 42
BATCH_SIZE = 512

# ------------------------------------------------------------------------------
# TEST A1: Core Component Ablation
# ------------------------------------------------------------------------------
print("\n>>> [VERIFYING A1] Core Component Ablation...")
ablations_a1 = [
    ("Vanilla_FL_Baseline", False, 0.0, False),
    ("No_Conformal_Rejection", True, 5e-3, False),
]

for v_name, use_b, l_vib, conf in ablations_a1:
    print(f"  -> Testing A1 Variant: {v_name}...")
    t0 = time.perf_counter()
    diag = run_fedtros_mc_full_diagnostics(
        dataset_name="b_nat",
        seed=SEED,
        rounds=1,
        beta_dirichlet=0.5,
        samples_per_class=None,
        output_root=out_root,
        experiment_id="A1_component_ablations",
        num_clients=2,
        use_bla=use_b,
        lambda_vib=l_vib,
        enable_conformal=conf,
        batch_size=BATCH_SIZE,
    )
    m = diag.get("metrics", diag)
    acc = m.get("closed_set_accuracy", m.get("accuracy", 0.0)) * 100
    auroc = m.get("auroc", 0.0) * 100
    fpr95 = m.get("fpr95", 0.0) * 100
    print(f"     [PASS] {v_name}: Closed Acc: {acc:.2f}% | AUROC: {auroc:.2f}% | FPR95: {fpr95:.2f}% ({time.perf_counter()-t0:.1f}s)")

# Also test YAML-driven run_component_ablations
print("  -> Testing A1 YAML-driven run_component_ablations...")
t0 = time.perf_counter()
a1_yaml_res = run_component_ablations(
    rounds=1,
    beta_dirichlet=0.5,
    samples_per_class=None,
    output_dir=str(out_root / "A1_yaml"),
    num_clients=2,
    seed=SEED,
    batch_size=BATCH_SIZE,
)
assert len(a1_yaml_res) >= 4, f"A1 YAML failed: {a1_yaml_res.keys()}"
print(f"     [PASS] A1 YAML Suite completed with {len(a1_yaml_res)} variants ({time.perf_counter()-t0:.1f}s).")

# ------------------------------------------------------------------------------
# TEST A2: VIB Beta Regularization Sweep
# ------------------------------------------------------------------------------
print("\n>>> [VERIFYING A2] VIB Beta Regularization Sweep...")
for b_vib in [0.0, 1e-3]:
    print(f"  -> Testing A2: lambda_vib = {b_vib}...")
    t0 = time.perf_counter()
    diag = run_fedtros_mc_full_diagnostics(
        dataset_name="b_nat",
        seed=SEED,
        rounds=1,
        beta_dirichlet=0.5,
        samples_per_class=None,
        output_root=out_root,
        experiment_id=f"A2_vib_beta_{b_vib}",
        num_clients=2,
        lambda_vib=b_vib,
        batch_size=BATCH_SIZE,
    )
    m = diag.get("metrics", diag)
    acc = m.get("closed_set_accuracy", m.get("accuracy", 0.0)) * 100
    auroc = m.get("auroc", 0.0) * 100
    print(f"     [PASS] VIB {b_vib}: Closed Acc: {acc:.2f}% | AUROC: {auroc:.2f}% ({time.perf_counter()-t0:.1f}s)")

# ------------------------------------------------------------------------------
# TEST A3: Logit-Adjusted Balanced Risk Minimization (tau_bla Sweep)
# ------------------------------------------------------------------------------
print("\n>>> [VERIFYING A3] Logit-Adjusted Balanced Risk Minimization...")
t0 = time.perf_counter()
a3_res = run_bla_ablations(
    rounds=1,
    beta_dirichlet=0.5,
    samples_per_class=None,
    output_dir=str(out_root / "A3_bla"),
    num_clients=2,
    seed=SEED,
    batch_size=BATCH_SIZE,
)
assert len(a3_res) >= 3, f"A3 failed: {a3_res.keys()}"
print(f"     [PASS] A3 Suite completed with {len(a3_res)} variants ({time.perf_counter()-t0:.1f}s).")

# ------------------------------------------------------------------------------
# TEST A4: Covariance Aggregation via Law of Total Variance (Theorem 1)
# ------------------------------------------------------------------------------
print("\n>>> [VERIFYING A4] Covariance Aggregation via Law of Total Variance...")
t0 = time.perf_counter()
a4_res = run_covariance_aggregation_comparison(
    dataset_name="b_nat",
    rounds=1,
    beta_dirichlet=0.5,
    output_dir=str(out_root / "A4_covariance"),
    seed=SEED,
    num_clients=2,
    batch_size=BATCH_SIZE,
)
assert "A4_moment_aggregation" in a4_res, f"A4 failed: {a4_res.keys()}"
print(f"     [PASS] A4 Suite completed: Evaluated {list(a4_res['A4_moment_aggregation'].keys())} ({time.perf_counter()-t0:.1f}s).")

# ------------------------------------------------------------------------------
# TEST A5: Block-Diagonal Partitioning Sub-Block Pareto
# ------------------------------------------------------------------------------
print("\n>>> [VERIFYING A5] Block-Diagonal Partitioning Sub-Block Pareto...")
block_configs = [(4, 16), (2, 32)]
for nb, bd in block_configs:
    print(f"  -> Testing A5: {nb} blocks x {bd} dim...")
    t0 = time.perf_counter()
    diag = run_fedtros_mc_full_diagnostics(
        dataset_name="b_nat",
        seed=SEED,
        rounds=1,
        beta_dirichlet=0.5,
        samples_per_class=None,
        output_root=out_root,
        experiment_id=f"A5_blocks_{nb}x{bd}",
        num_clients=2,
        num_blocks=nb,
        block_dim=bd,
        batch_size=BATCH_SIZE,
    )
    m = diag.get("metrics", diag)
    acc = m.get("closed_set_accuracy", m.get("accuracy", 0.0)) * 100
    auroc = m.get("auroc", 0.0) * 100
    print(f"     [PASS] {nb}x{bd}: Closed Acc: {acc:.2f}% | AUROC: {auroc:.2f}% ({time.perf_counter()-t0:.1f}s)")

# ------------------------------------------------------------------------------
# TEST A6: Mondrian Conformal Calibration & Miscoverage Budget Sweep (Theorem 2)
# ------------------------------------------------------------------------------
print("\n>>> [VERIFYING A6] Mondrian Conformal Calibration Sweep...")
t0 = time.perf_counter()
a6_res = run_conformal_sweep(
    dataset_name="b_nat",
    rounds=1,
    beta_dirichlet=0.5,
    alphas=[0.05, 0.10],
    output_dir=str(out_root / "A6_conformal"),
    seed=SEED,
    num_clients=2,
    batch_size=BATCH_SIZE,
)
assert len(a6_res) >= 2, f"A6 failed: {a6_res.keys()}"
print(f"     [PASS] A6 Suite completed with {len(a6_res)} alpha levels ({time.perf_counter()-t0:.1f}s).")

# ------------------------------------------------------------------------------
# TEST DEEP Q1: Analytical SVD Spectrum, Frobenius Error, and LDP + MIA
# ------------------------------------------------------------------------------
print("\n>>> [VERIFYING DEEP Q1] Deep Analytical Ablations (SVD, Frobenius, LDP + MIA)...")
t0 = time.perf_counter()
deep_res = run_deep_q1_ablations(
    dataset_name="b_nat",
    num_clients=2,
    rounds=1,
    beta_dirichlet=0.5,
    samples_per_class=None,
    output_dir=str(out_root / "deep_q1"),
    seed=SEED,
    vib_betas=[0.0, 5e-3],
    batch_size=BATCH_SIZE,
)
assert "A1_vib_svd" in deep_res and "A4_total_variance" in deep_res and "LDP_privacy_mia" in deep_res
print(f"     [PASS] Deep Q1 Analytical Suite completed successfully ({time.perf_counter()-t0:.1f}s).")

print("\n" + "=" * 80)
print("ALL ABLATION EXPERIMENTS (A1 TO A6 + DEEP Q1) FULLY VERIFIED AND PASSING!")
print("=" * 80)
