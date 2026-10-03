"""High-Performance Parallel GPU Experiment Suite Runner for FedTROS-MC.

Designed for Google Colab and GPU Server clusters (A100, T4, V100, RTX 3090/4090).
Executes all federated baselines, Dirichlet skews, LOAO, ablations, and scalability
in parallel across GPU workers using a queue-based multi-process dispatcher.

Usage:
    python run_gpu_parallel.py --protocol full --rounds 100 --workers 4
    python run_gpu_parallel.py --protocol full --rounds 200 --workers 8 --scalability_clients 20,50,100
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import importlib.util

def ensure_dependencies() -> None:
    """Pre-flight check: verify all runtime dependencies and install any missing packages."""
    required_packages = {
        "torch": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "numpy": "numpy>=1.26.0",
        "scipy": "scipy>=1.12.0",
        "pandas": "pandas>=2.2.0",
        "sklearn": "scikit-learn>=1.4.0",
        "matplotlib": "matplotlib>=3.8.0",
        "seaborn": "seaborn>=0.13.0",
        "yaml": "pyyaml>=6.0.1",
        "pydantic": "pydantic>=2.6.0",
        "rich": "rich>=13.7.0",
    }

    missing = []
    for mod, pkg in required_packages.items():
        if importlib.util.find_spec(mod) is None:
            missing.append((mod, pkg))

    if missing:
        print(f"[PRE-FLIGHT] Detected {len(missing)} missing required packages: {[m[0] for m in missing]}")

        # 1. Verify pip availability and bootstrap if missing
        if importlib.util.find_spec("pip") is None:
            print("[PRE-FLIGHT] 'pip' module not found. Attempting automatic bootstrap...")
            try:
                import ensurepip
                ensurepip.bootstrap(upgrade=True, default_pip=True)
                print("[PRE-FLIGHT] Successfully bootstrapped pip via ensurepip.")
            except Exception:
                try:
                    import urllib.request
                    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "/tmp/get-pip.py")
                    subprocess.run([sys.executable, "/tmp/get-pip.py", "--user", "--break-system-packages", "-q"], check=False)
                except Exception:
                    pass

        # 2. Check if pip is now usable
        pip_available = False
        try:
            res_pip = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True)
            pip_available = (res_pip.returncode == 0)
        except Exception:
            pip_available = False

        if not pip_available:
            print("\n" + "=" * 75)
            print("[FATAL ERROR] 'pip' is not installed in the active Python environment:")
            print(f"    Python Executable: {sys.executable}")
            print("\nPlease install pip or activate a virtual/conda environment with one of these:")
            print("  A) Install pip without sudo (User mode):")
            print("     curl -sS https://bootstrap.pypa.io/get-pip.py | python3 --user --break-system-packages")
            print("  B) If using Conda (recommended for GPU clusters):")
            print("     conda activate <your_env>  (or source /opt/conda/bin/activate)")
            print("  C) If you have sudo access on Ubuntu/Debian:")
            print("     sudo apt-get update && sudo apt-get install -y python3-pip python3-venv")
            print("=" * 75 + "\n")
            sys.exit(1)

        # 3. Detect if --break-system-packages flag is supported (Ubuntu 23.04+)
        pip_extra: List[str] = []
        try:
            h_out = subprocess.run([sys.executable, "-m", "pip", "install", "--help"], capture_output=True, text=True).stdout
            if "--break-system-packages" in h_out:
                pip_extra.append("--break-system-packages")
        except Exception:
            pass

        installed_count = 0
        for mod, pkg in missing:
            print(f"  -> Installing missing package: {pkg} ...")
            cmd = [sys.executable, "-m", "pip", "install", "-q"] + pip_extra + pkg.split()
            res = subprocess.run(cmd)
            if res.returncode != 0:
                # Retry with --user in case of permission error
                cmd_user = [sys.executable, "-m", "pip", "install", "-q", "--user"] + pip_extra + pkg.split()
                res = subprocess.run(cmd_user)
            if res.returncode != 0 and "torch" in mod:
                print("  -> Retrying standard PyTorch install...")
                cmd_torch = [sys.executable, "-m", "pip", "install", "-q", "--user"] + pip_extra + ["torch", "torchvision"]
                res = subprocess.run(cmd_torch)
            if res.returncode == 0:
                installed_count += 1
            else:
                print(f"  [ERROR] Failed to install {pkg}. (return code: {res.returncode})")

        if installed_count < len(missing):
            print(f"\n[FATAL ERROR] Failed to install all dependencies ({installed_count}/{len(missing)} installed).")
            print("Please run manually: pip install -r requirements.txt\n")
            sys.exit(1)

        print("[PRE-FLIGHT] All missing packages installed successfully.\n")


ensure_dependencies()

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_optimal_worker_count(requested: Optional[int] = None) -> int:
    """Determine optimal number of parallel GPU workers based on CUDA VRAM."""
    if requested is not None and requested > 0:
        return requested
    if not torch.cuda.is_available():
        cpu_cores = mp.cpu_count()
        return max(1, min(4, cpu_cores // 2))

    device_count = torch.cuda.device_count()
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # For lightweight MLPs, each worker requires ~1.0 - 1.5 GB VRAM
    if total_vram_gb >= 38.0:  # A100 (40GB/80GB)
        return min(8 * device_count, 12)
    elif total_vram_gb >= 15.0:  # T4, V100, RTX 3090/4090 (16-24GB)
        return min(4 * device_count, 6)
    else:  # Smaller GPUs (<12GB)
        return max(1, min(2 * device_count, 3))


def worker_process(
    worker_id: int,
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    protocol: str,
    output_root: Path,
    rounds: int,
    batch_size: int,
) -> None:
    """Worker loop executing assigned experiment tasks on a specific GPU."""
    # Set CUDA device for this worker process
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        actual_gpu_id = gpu_id % max(1, num_gpus)
        torch.cuda.set_device(actual_gpu_id)
        device = torch.device(f"cuda:{actual_gpu_id}")
    else:
        device = torch.device("cpu")
    os.environ["OSR_PROTOCOL"] = protocol

    from src.config import load_config
    from src.tracking.reproducibility import set_seed

    while True:
        try:
            task = task_queue.get(timeout=3.0)
        except Exception:
            break

        if task is None:
            break

        task_name = task["name"]
        exp_type = task["type"]
        params = task["params"]

        print(
            f"\n>>> [WORKER #{worker_id} (GPU:{gpu_id})] Starting Task: '{task_name}' "
            f"(Type: {exp_type}, Rounds: {rounds}, BatchSize: {batch_size})...",
            flush=True,
        )

        t_start = time.perf_counter()
        status = "SUCCESS"
        error_msg = ""
        metrics = {}

        try:
            if exp_type == "baseline":
                from experiments.run_baselines import evaluate_method_configuration
                m_name = params["method"]
                m_dir = output_root / "E2_baselines" / "b_nat"
                m_dir.mkdir(parents=True, exist_ok=True)
                metrics = evaluate_method_configuration(
                    method_name=m_name,
                    config_path="configs/default.yaml",
                    dataset_name="b_nat",
                    rounds=rounds,
                    beta_dirichlet=params.get("beta", 0.5),
                    eval_mode="open_set",
                    batch_size=batch_size,
                )
                # Save seed artifact
                res_payload = {"experiment_id": "E2_baselines", "method": m_name, "metrics": metrics}
                (m_dir / f"{m_name.replace(' ', '_')}_seed_42.json").write_text(
                    json.dumps(res_payload, indent=2), encoding="utf-8"
                )

            elif exp_type == "skew":
                from run_all import run_fedtros_mc_full_diagnostics
                exp_id = params["exp_id"]
                beta = params["beta"]
                diag = run_fedtros_mc_full_diagnostics(
                    dataset_name="b_nat",
                    seed=42,
                    rounds=rounds,
                    beta_dirichlet=beta,
                    samples_per_class=None,
                    output_root=output_root,
                    experiment_id=exp_id,
                    num_clients=10,
                    batch_size=batch_size,
                )
                metrics = diag.get("metrics", diag)

            elif exp_type == "loao":
                from experiments.run_loao import evaluate_loao_attack
                atk = params["attack"]
                out_loao = output_root / "E6_loao"
                out_loao.mkdir(parents=True, exist_ok=True)
                m_dict = evaluate_loao_attack(
                    dataset_name="b_nat",
                    held_out_attack=atk,
                    rounds=rounds,
                    beta_dirichlet=0.5,
                    max_samples=None,
                    seed=42,
                    batch_size=batch_size,
                )
                metrics = m_dict
                (out_loao / f"loao_{atk}.json").write_text(json.dumps(m_dict, indent=2), encoding="utf-8")

            elif exp_type == "scalability":
                from experiments.run_scalability import run_scalability_experiment
                k_val = params["clients"]
                out_scal = output_root / "E7_scalability"
                out_scal.mkdir(parents=True, exist_ok=True)
                s_res = run_scalability_experiment(
                    dataset_name="b_nat",
                    client_counts=[k_val],
                    rounds=max(rounds // 2, 5),
                    beta_dirichlet=0.5,
                    max_samples=None,
                    output_dir=str(out_scal),
                    seed=42,
                    batch_size=batch_size,
                )
                metrics = s_res.get(f"clients_{k_val}", {})

            elif exp_type == "vib_ablation":
                from run_all import run_fedtros_mc_full_diagnostics
                b_vib = params["beta_vib"]
                exp_id = f"A2_vib_beta_{b_vib}"
                diag = run_fedtros_mc_full_diagnostics(
                    dataset_name="b_nat",
                    seed=42,
                    rounds=rounds,
                    beta_dirichlet=0.5,
                    samples_per_class=None,
                    output_root=output_root,
                    experiment_id=exp_id,
                    num_clients=10,
                    lambda_vib=b_vib,
                    batch_size=batch_size,
                )
                metrics = diag.get("metrics", diag)

            elif exp_type == "block_ablation":
                from run_all import run_fedtros_mc_full_diagnostics
                nb = params["num_blocks"]
                bd = params["block_dim"]
                exp_id = f"A5_blocks_{nb}x{bd}"
                diag = run_fedtros_mc_full_diagnostics(
                    dataset_name="b_nat",
                    seed=42,
                    rounds=rounds,
                    beta_dirichlet=0.5,
                    samples_per_class=None,
                    output_root=output_root,
                    experiment_id=exp_id,
                    num_clients=10,
                    num_blocks=nb,
                    block_dim=bd,
                    batch_size=batch_size,
                )
                metrics = diag.get("metrics", diag)

            elif exp_type == "component_ablation":
                from run_all import run_fedtros_mc_full_diagnostics
                var_name = params["variant"]
                exp_id = "A1_component_ablations"
                diag = run_fedtros_mc_full_diagnostics(
                    dataset_name="b_nat",
                    seed=42,
                    rounds=rounds,
                    beta_dirichlet=0.5,
                    samples_per_class=None,
                    output_root=output_root,
                    experiment_id=exp_id,
                    num_clients=10,
                    use_bla=params.get("use_bla", True),
                    lambda_vib=params.get("lambda_vib", 5e-3),
                    enable_conformal=params.get("enable_conformal", True),
                    batch_size=batch_size,
                )
                metrics = diag.get("metrics", diag)

            elif exp_type == "wire_speed":
                from experiments.run_wire_speed_profiling import run_wire_speed_experiment
                out_ws = output_root / "E8_wire_speed"
                out_ws.mkdir(parents=True, exist_ok=True)
                ws_res = run_wire_speed_experiment(output_dir=str(out_ws), trials=10000)
                metrics = {"status": "wire_speed_complete"}

        except Exception as exc:
            import traceback
            status = "FAILED"
            error_msg = f"{exc}\n{traceback.format_exc()}"

        elapsed = time.perf_counter() - t_start
        print(
            f"<<< [WORKER #{worker_id}] Completed Task: '{task_name}' in {elapsed:.1f}s [{status}]\n",
            flush=True,
        )
        result_queue.put({
            "task_name": task_name,
            "worker_id": worker_id,
            "status": status,
            "elapsed_sec": elapsed,
            "error_msg": error_msg,
            "metrics": metrics,
        })


def build_experiment_task_list(
    scalability_clients: List[int],
    include_benchmarks: bool = True,
) -> List[Dict[str, Any]]:
    """Build the comprehensive task list for all Q1 paper experiments."""
    tasks = []

    # 1. E2 Canonical Baselines (14 methods)
    baseline_methods = [
        "FedAvg", "FedProx", "SCAFFOLD", "FedDyn", "FedRoD", "FedDecorr",
        "FedProto", "FedPD", "FedPD++", "FedOSS", "OpenMax", "Energy-OOD",
        "Vanilla-Mahalanobis", "FedTROS-MC (Proposed)"
    ]
    for b in baseline_methods:
        tasks.append({
            "name": f"Baseline: {b}",
            "type": "baseline",
            "params": {"method": b, "beta": 0.5},
        })

    # 2. E1 Dirichlet Non-IID Skews (4 settings)
    for s_name, beta in [
        ("E1_iid", 100.0),
        ("E1_dirichlet_05", 0.5),
        ("E1_dirichlet_01", 0.1),
        ("E1_pathological_c2", 0.05),
    ]:
        tasks.append({
            "name": f"Skew: {s_name}",
            "type": "skew",
            "params": {"exp_id": s_name, "beta": beta},
        })

    # 3. E6 LOAO Zero-Day Breakdown (4 attacks)
    for atk in ["BP", "DoS", "MitM", "FoT"]:
        tasks.append({
            "name": f"LOAO Attack: {atk}",
            "type": "loao",
            "params": {"attack": atk},
        })

    # 4. E7 Scalability Tiers
    for k_val in scalability_clients:
        tasks.append({
            "name": f"Scalability: K={k_val}",
            "type": "scalability",
            "params": {"clients": k_val},
        })

    # 5. A2 VIB Beta Sensitivity (6 variants)
    for b_vib in [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 1e-1]:
        tasks.append({
            "name": f"Ablation VIB: beta={b_vib}",
            "type": "vib_ablation",
            "params": {"beta_vib": b_vib},
        })

    # 6. A5 Covariance Block Partitioning (5 variants)
    for nb, bd in [(1, 64), (2, 32), (4, 16), (8, 8), (64, 1)]:
        tasks.append({
            "name": f"Ablation Block: {nb}x{bd}",
            "type": "block_ablation",
            "params": {"num_blocks": nb, "block_dim": bd},
        })

    # 7. A1 Component Ablations (3 variants)
    for v_name, use_b, l_vib, conf in [
        ("Vanilla_FL_Baseline", False, 0.0, False),
        ("No_BLA_Margin", False, 5e-3, True),
        ("No_VIB_Encoder", True, 0.0, True),
        ("No_Conformal_Rejection", True, 5e-3, False),
    ]:
        tasks.append({
            "name": f"Ablation Component: {v_name}",
            "type": "component_ablation",
            "params": {"variant": v_name, "use_bla": use_b, "lambda_vib": l_vib, "enable_conformal": conf},
        })

    # 8. E8 Wire-speed profiling
    tasks.append({
        "name": "E8: Wire-Speed Latency Profiling",
        "type": "wire_speed",
        "params": {},
    })

    return tasks


def run_parallel_gpu_suite(
    protocol: str = "full",
    rounds: int = 100,
    workers: Optional[int] = None,
    scalability_clients: Optional[List[int]] = None,
    batch_size: int = 128,
    output_root: str = "results/colab_runs",
    make_plots: bool = True,
    zip_results: bool = True,
) -> None:
    """Launch and manage parallel GPU experiment execution."""
    out_dir = Path(output_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if scalability_clients is None:
        scalability_clients = [20, 50, 100]

    num_workers = get_optimal_worker_count(workers)
    tasks = build_experiment_task_list(scalability_clients)
    total_tasks = len(tasks)

    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    num_gpus = torch.cuda.device_count() if cuda_avail else 0

    print("=" * 75)
    print("FEDTROS-MC HIGH-PERFORMANCE PARALLEL GPU SUITE")
    print(f"Device:           {device_name} (Total GPUs: {num_gpus})")
    print(f"Parallel Workers: {num_workers}")
    print(f"Protocol:         {protocol.upper()} (Zero Subsampling)")
    print(f"Federated Rounds: {rounds}")
    print(f"Batch Size:       {batch_size} (Tensor-Core Optimized)")
    print(f"Scalability K:    {scalability_clients}")
    print(f"Total Tasks:      {total_tasks}")
    print(f"Output Directory: {out_dir}")
    print("=" * 75)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for t in tasks:
        task_queue.put(t)

    # Launch workers
    procs = []
    for wid in range(num_workers):
        gpu_id = wid % max(1, num_gpus) if num_gpus > 0 else 0
        p = ctx.Process(
            target=worker_process,
            args=(wid, gpu_id, task_queue, result_queue, protocol, out_dir, rounds, batch_size),
        )
        p.start()
        procs.append(p)

    # Monitor and collect results
    completed = 0
    failed = 0
    t_start_all = time.perf_counter()

    while completed < total_tasks:
        try:
            res = result_queue.get(timeout=2.0)
            completed += 1
            t_name = res["task_name"]
            status = res["status"]
            dur = res["elapsed_sec"]
            wid = res["worker_id"]

            if status == "SUCCESS":
                tag = "\033[92m[PASS]\033[0m"
            else:
                tag = "\033[91m[FAIL]\033[0m"
                failed += 1

            pct = (completed / total_tasks) * 100.0
            print(f"[{completed:02d}/{total_tasks:02d}] ({pct:5.1f}%) {tag} W#{wid} | {t_name:<38} | {dur:6.1f}s")
            if res["error_msg"]:
                err_lines = [l.strip() for l in res["error_msg"].strip().splitlines() if l.strip()]
                print(f"    --> Error: {err_lines[-1] if err_lines else 'Unknown'}")
                if len(err_lines) > 1:
                    print(f"        Location: {err_lines[-2]}")

        except Exception:
            # Check if all processes are still alive
            all_dead = all(not p.is_alive() for p in procs)
            if all_dead and result_queue.empty():
                print("[WARNING] All worker processes terminated early.")
                break

    for p in procs:
        p.join()

    total_time = time.perf_counter() - t_start_all
    print("=" * 75)
    print(f"SUITE COMPLETED: {completed}/{total_tasks} Tasks in {total_time:.1f}s ({total_time/60.0:.2f} min).")
    print(f"Passed: {completed - failed} | Failed: {failed}")
    print("=" * 75)

    # Aggregate summaries
    print("\n[Step 2/4] Aggregating Multi-Seed and Method Summaries ...")
    from src.utils.artifact_logger import aggregate_MultiSeed_Summary
    for exp_p in out_dir.iterdir():
        if exp_p.is_dir():
            for ds_p in exp_p.iterdir():
                if ds_p.is_dir() and list(ds_p.glob("*.json")):
                    aggregate_MultiSeed_Summary(out_dir, exp_p.name, ds_p.name)

    # Finalize Plot Contract
    print("\n[Step 3/4] Exporting Plot Contract ...")
    from src.ref_plots.contract_exporter_part3 import finalize_real_plot_contract
    contract_p = out_dir / "plot_contract"
    finalize_real_plot_contract(out_dir, contract_p, rounds=rounds, spc=None, seed=42)

    # Render plots & tables
    if make_plots:
        print("\n[Step 4/4] Generating Publication Figures and LaTeX Tables ...")
        figs_dir = out_dir / "figures"
        tbls_dir = out_dir / "tables"
        subprocess.run([
            sys.executable, "make_plots.py",
            "--results_root", str(out_dir),
            "--figures_dir", str(figs_dir),
            "--tables_dir", str(tbls_dir),
        ], check=False)

    # Package into downloadable zip
    if zip_results:
        import shutil
        zip_path = out_dir.parent / f"{out_dir.name}_bundle.zip"
        print(f"\n[Packaging] Compressing results to {zip_path} ...")
        shutil.make_archive(str(out_dir.parent / f"{out_dir.name}_bundle"), "zip", root_dir=str(out_dir))
        print(f"[SUCCESS] Downloadable bundle ready: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Parallel GPU Experiment Suite for FedTROS-MC")
    parser.add_argument("--protocol", type=str, default="full", choices=["full"], help="Dataset split protocol (100% full dataset)")
    parser.add_argument("--rounds", type=int, default=100, help="Federated training rounds per experiment")
    parser.add_argument("--workers", type=int, default=None, help="Number of concurrent GPU worker processes (default: auto)")
    parser.add_argument("--scalability_clients", type=str, default="20,50,100", help="Comma-separated client counts for scalability")
    parser.add_argument("--batch_size", type=int, default=128, help="Tensor-core batch size")
    parser.add_argument("--output_root", type=str, default="results/colab_runs", help="Output directory")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot rendering")
    parser.add_argument("--no_zip", action="store_true", help="Skip zip bundle creation")
    args = parser.parse_args()

    scal_clients = [int(c.strip()) for c in args.scalability_clients.split(",") if c.strip()]

    run_parallel_gpu_suite(
        protocol=args.protocol,
        rounds=args.rounds,
        workers=args.workers,
        scalability_clients=scal_clients,
        batch_size=args.batch_size,
        output_root=args.output_root,
        make_plots=not args.no_plots,
        zip_results=not args.no_zip,
    )
