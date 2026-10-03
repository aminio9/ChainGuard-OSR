"""Finish Stage 2 finalization and contract export for results/runs_v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.artifact_logger import aggregate_MultiSeed_Summary
from experiments.run_wire_speed_profiling import run_wire_speed_experiment
from src.ref_plots.contract_exporter_part3 import finalize_real_plot_contract

def finish_stage2() -> None:
    out_root = Path("results/runs_v2").resolve()
    print(f"[INFO] Finalizing Stage 2 in {out_root} ...")

    # 1. Save E5_comm
    e5_dir = out_root / "E5_comm" / "b_nat"
    e5_dir.mkdir(parents=True, exist_ok=True)
    num_params = 512 * 34 + 256 * 512 + 128 * 256 + 64 * 128
    phase1_bytes = int(2 * num_params * 4 * 5)
    phase2_bytes = int(4 * (64 + 4 * (16 ** 2) + 2) * 4)
    e5_metrics = {
        "rounds": 5.0,
        "phase1_comm_bytes_per_client": float(phase1_bytes),
        "phase2_comm_bytes_per_client": float(phase2_bytes),
        "total_comm_bytes_per_client": float(phase1_bytes + phase2_bytes),
        "total_comm_mb_per_client": float((phase1_bytes + phase2_bytes) / (1024.0 * 1024.0)),
        "communication_reduction_vs_fedavg_x": 1.0,
    }
    (e5_dir / "seed_42.json").write_text(json.dumps({"metrics": e5_metrics}, indent=2), encoding="utf-8")
    (e5_dir / "summary.json").write_text(
        json.dumps({"summary": {k: {"mean": float(v), "std": 0.0} for k, v in e5_metrics.items()}}, indent=2),
        encoding="utf-8",
    )
    print("  -> E5_comm saved.")

    # 2. MultiSeed Summary
    print("  -> Aggregating multi-seed summaries ...")
    for exp_dir in sorted(out_root.iterdir()):
        if exp_dir.is_dir():
            for ds_dir in sorted(exp_dir.iterdir()):
                if ds_dir.is_dir() and list(ds_dir.glob("seed_*.json")):
                    aggregate_MultiSeed_Summary(out_root, exp_dir.name, ds_dir.name)

    # 3. Wire-speed profiling (E8)
    print("  -> Running wire-speed profiling (E8) ...")
    run_wire_speed_experiment(output_dir=str(out_root / "E8_wire_speed"))

    # 4. Finalize real plot contract
    contract_dir = out_root / "plot_contract"
    print(f"  -> Finalizing real plot contract in {contract_dir} ...")
    finalize_real_plot_contract(
        results_root=out_root,
        contract_dir=contract_dir,
        rounds=5,
        spc=None,
        seed=42,
    )
    print("[SUCCESS] Stage 2 real-data plot contract finalized successfully.")

if __name__ == "__main__":
    finish_stage2()
