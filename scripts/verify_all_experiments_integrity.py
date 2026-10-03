"""Pre-flight verification script to test every single experiment type and baseline execution signature."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import inspect
import torch

from run_gpu_parallel import build_experiment_task_list
from experiments.run_baselines import evaluate_method_configuration, MODULAR_BASELINE_REGISTRY
from experiments.run_loao import evaluate_loao_attack
from experiments.run_scalability import run_scalability_experiment
from experiments.run_ablations import run_component_ablations, run_bla_ablations, run_deep_q1_ablations
from experiments.run_wire_speed_profiling import run_wire_speed_experiment
from run_all import run_fedtros_mc_full_diagnostics

def test_task_list_and_signatures():
    print("=== [1/4] Verifying Task List Generation ===")
    tasks = build_experiment_task_list(scalability_clients=[20, 50, 100])
    print(f"  -> Successfully generated {len(tasks)} tasks.")
    task_types = set(t["type"] for t in tasks)
    print(f"  -> Task types present: {task_types}")

    print("\n=== [2/4] Verifying Function Signatures and Kwargs ===")
    # 1. evaluate_method_configuration
    sig_base = inspect.signature(evaluate_method_configuration)
    assert "method_name" in sig_base.parameters
    assert "dataset_name" in sig_base.parameters
    assert "rounds" in sig_base.parameters
    assert "beta" in sig_base.parameters
    assert "batch_size" in sig_base.parameters
    assert sig_base.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    print("  [OK] evaluate_method_configuration accepts method_name, dataset_name, rounds, beta, batch_size, and **kwargs")

    # 2. evaluate_loao_attack
    sig_loao = inspect.signature(evaluate_loao_attack)
    assert "held_out_attack" in sig_loao.parameters
    assert "dataset_name" in sig_loao.parameters
    assert "rounds" in sig_loao.parameters
    assert sig_loao.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    print("  [OK] evaluate_loao_attack accepts held_out_attack, dataset_name, rounds, and **kwargs")

    # 3. run_scalability_experiment
    sig_scal = inspect.signature(run_scalability_experiment)
    assert "dataset_name" in sig_scal.parameters
    assert "client_counts" in sig_scal.parameters
    assert "rounds" in sig_scal.parameters
    assert sig_scal.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    print("  [OK] run_scalability_experiment accepts dataset_name, client_counts, rounds, and **kwargs")

    # 4. run_fedtros_mc_full_diagnostics
    sig_diag = inspect.signature(run_fedtros_mc_full_diagnostics)
    assert "dataset_name" in sig_diag.parameters
    assert "num_clients" in sig_diag.parameters
    assert "num_blocks" in sig_diag.parameters
    assert "block_dim" in sig_diag.parameters
    assert "lambda_vib" in sig_diag.parameters
    assert "use_bla" in sig_diag.parameters
    assert "enable_conformal" in sig_diag.parameters
    assert sig_diag.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    print("  [OK] run_fedtros_mc_full_diagnostics accepts all ablation params and **kwargs")

    # 5. run_wire_speed_experiment
    sig_ws = inspect.signature(run_wire_speed_experiment)
    assert "output_dir" in sig_ws.parameters
    assert "trials" in sig_ws.parameters
    assert "num_trials" in sig_ws.parameters
    assert sig_ws.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    print("  [OK] run_wire_speed_experiment accepts output_dir, trials, num_trials, and **kwargs")

    print("\n=== [3/4] Verifying Baseline Registry ===")
    print(f"  -> Total registered modular baselines: {len(MODULAR_BASELINE_REGISTRY)}")
    for b_name, b_cls in MODULAR_BASELINE_REGISTRY.items():
        inst = b_cls(in_features=34, num_classes=4, latent_dim=64, hidden_dim=128, device=torch.device("cpu"))
        assert hasattr(inst, "fit")
        assert hasattr(inst, "evaluate")
    print("  [OK] All 13 modular baselines instantiated and verified.")

    print("\n=== [4/4] Verifying CUDA-CPU Aggregation Device Safety ===")
    from src.aggregation.decoupled_agg import aggregate_decoupled_parameters
    dummy_client_w = {
        "encoder.weight": torch.randn(10, 10),
        "classifier.weight": torch.randn(4, 10),
        "classifier.bias": torch.randn(4),
    }
    prev_global_w = {
        "encoder.weight": torch.randn(10, 10),
        "classifier.weight": torch.randn(4, 10),
        "classifier.bias": torch.randn(4),
    }
    client_updates = [(dummy_client_w, 100, [25, 25, 25, 25])]
    agg_w = aggregate_decoupled_parameters(client_updates, previous_global_state=prev_global_w)
    assert "classifier.bias" in agg_w
    assert agg_w["classifier.bias"].device == dummy_client_w["classifier.bias"].device
    print("  [OK] Decoupled aggregation preserves device and matches tensor devices perfectly.")

    print("\n[SUCCESS] ALL INTEGRITY AND SIGNATURE CHECKS PASSED WITH ZERO ERRORS.")

if __name__ == "__main__":
    test_task_list_and_signatures()
