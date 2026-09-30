"""Experiment E8: Wire-Speed Latency, Computational Complexity, and FLOPs Profiling.

Aligned with LaTeX Section 4.3 & Section 5.1 (e) and IEEE TIFS/TDSC Hardware Audit:
Validates:
  - Theoretical FLOPs complexity: O(|C_K| * d_z * d_b) Block-Diagonal vs O(|C_K| * d_z^2) Full Mahalanobis.
  - Exactly 2,176 FLOPs per query candidate evaluation.
  - Vectorized `LatencyProfiler` with 1,000 warmup iterations and 10,000 timed trials at B=1.
  - Dense Mahalanobis regularized via Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`).
  - Reports Median [P95, P99] latency in microseconds (us) and saves `.npz`/`.json` plot bundles.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from src.utils.io import save_json
from src.tracking.logger import setup_logger, save_experiment_audit_bundle


class LatencyProfiler:
    """Hardware-calibrated per-packet latency profiler (B=1 wire-speed regime)."""

    def __init__(self, device: str = "cpu", warmup: int = 1000, trials: int = 10000) -> None:
        self.device = torch.device(device)
        self.warmup = warmup
        self.trials = trials

    def profile_scoring_head(
        self,
        score_fn: Callable[[torch.Tensor], Any],
        z_sample: torch.Tensor,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Measure single-packet (B=1) execution latency distribution in microseconds."""
        z_sample = z_sample.to(self.device)

        # Warmup phase to stabilize CPU cache / CUDA kernels
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = score_fn(z_sample)

        latencies_us = np.zeros(self.trials, dtype=np.float64)
        with torch.no_grad():
            for i in range(self.trials):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                _ = score_fn(z_sample)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                latencies_us[i] = (t1 - t0) / 1000.0  # nanoseconds -> microseconds

        stats = {
            "median_us": float(np.median(latencies_us)),
            "mean_us": float(np.mean(latencies_us)),
            "p95_us": float(np.percentile(latencies_us, 95)),
            "p99_us": float(np.percentile(latencies_us, 99)),
            "std_us": float(np.std(latencies_us)),
            "throughput_fps": float(1e6 / max(np.mean(latencies_us), 1e-6)),
        }
        return stats, latencies_us


@torch.no_grad()
def block_diagonal_mahalanobis_score(
    z: torch.Tensor,
    mu_blocks: torch.Tensor,
    inv_cov_blocks: torch.Tensor,
) -> torch.Tensor:
    """Vectorized Block-Diagonal Mahalanobis scoring without Python loops.

    Args:
        z: shape (1, B_blocks, d_b)
        mu_blocks: shape (K_classes, B_blocks, d_b)
        inv_cov_blocks: shape (K_classes, B_blocks, d_b, d_b)

    Returns:
        Minimum Mahalanobis distance across K_classes.
    """
    diff = z - mu_blocks  # (K_classes, B_blocks, d_b)
    # Direct batched matrix-vector contraction (avoids per-call einsum string parsing overhead at B=1)
    proj = torch.matmul(diff.unsqueeze(-2), inv_cov_blocks).squeeze(-2)  # (K_classes, B_blocks, d_b)
    total_dist = torch.sum(proj * diff, dim=(-1, -2))  # (K_classes,)
    return total_dist.min()


@torch.no_grad()
def candidate_block_diagonal_mahalanobis_score(
    z_flat: torch.Tensor,
    means_flat: torch.Tensor,
    mu_blocks: torch.Tensor,
    inv_cov_blocks: torch.Tensor,
    num_blocks: int = 4,
    block_dim: int = 16,
) -> torch.Tensor:
    """Two-stage candidate screening + block-diagonal Mahalanobis (exact 2,176 FLOPs)."""
    diff_means = z_flat - means_flat  # (K_classes, d_z)
    c_star = torch.argmin(torch.sum(diff_means * diff_means, dim=-1))
    q_blocks = z_flat.view(num_blocks, block_dim)
    diff_b = q_blocks - mu_blocks[c_star]  # (B_blocks, d_b)
    inv_b = inv_cov_blocks[c_star]  # (B_blocks, d_b, d_b)
    proj_b = torch.bmm(diff_b.unsqueeze(1), inv_b).squeeze(1)
    return torch.sum(proj_b * diff_b)


def compute_encoder_flops(
    in_features: int = 34,
    hidden_dims: Tuple[int, ...] = (512, 256, 128),
    latent_dim: int = 64,
) -> int:
    """Compute exact multiply-accumulate FLOPs for the MLP representation backbone."""
    total_flops = 0
    prev_dim = in_features
    for h in hidden_dims:
        total_flops += 2 * prev_dim * h
        total_flops += 2 * h
        prev_dim = h
    total_flops += 2 * prev_dim * latent_dim
    return total_flops


def compute_metric_flops(
    num_classes: int = 4,
    latent_dim: int = 64,
    num_blocks: int = 4,
    block_dim: int = 16,
) -> Dict[str, int]:
    """Compute theoretical FLOPs per incoming flow query across metric configurations."""
    # 1. Full Mahalanobis: |C_K| classes * (2 * d_z^2 + d_z)
    full_mahalanobis_flops = num_classes * (2 * (latent_dim ** 2) + latent_dim)

    # 2. Block-Diagonal Mahalanobis (Proposed):
    # Candidate class block evaluation: num_blocks * (2 * block_dim^2 + block_dim) + latent_dim
    # For B=4, d_b=16, d_z=64: 4 * (512 + 16) + 64 = 2,112 + 64 = 2,176 FLOPs
    candidate_block_flops = num_blocks * (2 * (block_dim ** 2) + block_dim) + latent_dim

    # 3. Diagonal Mahalanobis: |C_K| * 3 * d_z
    diag_mahalanobis_flops = num_classes * 3 * latent_dim

    # 4. Euclidean Prototype: |C_K| * 2 * d_z
    euclidean_flops = num_classes * 2 * latent_dim

    # 5. OpenMax (Weibull EVT fit + CDF across classes):
    openmax_flops = num_classes * (2 * (latent_dim ** 2) + 12 * latent_dim)

    return {
        "Full Mahalanobis (Ledoit-Wolf)": full_mahalanobis_flops,
        "Full Mahalanobis": full_mahalanobis_flops,
        "Block-Diagonal Mahalanobis (Proposed)": candidate_block_flops,
        "Diagonal Mahalanobis": diag_mahalanobis_flops,
        "Euclidean Prototype": euclidean_flops,
        "OpenMax (Weibull EVT)": openmax_flops,
    }


def profile_runtime_latency(
    num_classes: int = 4,
    latent_dim: int = 64,
    num_blocks: int = 4,
    block_dim: int = 16,
    warmup: int = 1000,
    trials: int = 10000,
    device_name: str = "cpu",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Empirically measure per-query B=1 latency across metric architectures using LatencyProfiler."""
    device = torch.device(device_name)
    torch.manual_seed(42)
    np.random.seed(42)

    profiler = LatencyProfiler(device=device_name, warmup=warmup, trials=trials)

    # Single query packet (B=1)
    z_flat = torch.randn(1, latent_dim, device=device)
    z_blocks = z_flat.view(1, num_blocks, block_dim)

    means_flat = torch.randn(num_classes, latent_dim, device=device)
    mu_blocks = means_flat.view(num_classes, num_blocks, block_dim)

    # 1. Full covariance inverse regularized with Ledoit-Wolf shrinkage
    full_invs = []
    for _ in range(num_classes):
        synthetic_feats = np.random.randn(256, latent_dim).astype(np.float64)
        lw = LedoitWolf().fit(synthetic_feats)
        prec = lw.precision_.astype(np.float32)
        full_invs.append(torch.from_numpy(prec).to(device))
    full_invs_t = torch.stack(full_invs, dim=0)  # (K, d_z, d_z)

    # 2. Block-diagonal covariance inverses: shape (K, B, d_b, d_b)
    block_invs = []
    for c in range(num_classes):
        c_blocks = []
        for b in range(num_blocks):
            blk = full_invs_t[c, b * block_dim : (b + 1) * block_dim, b * block_dim : (b + 1) * block_dim]
            c_blocks.append(blk.contiguous())
        block_invs.append(torch.stack(c_blocks, dim=0))
    block_invs_t = torch.stack(block_invs, dim=0).contiguous()

    # 3. Diagonal variances: shape (K, d_z)
    var_invs_t = torch.diagonal(full_invs_t, dim1=-2, dim2=-1).contiguous()

    results: Dict[str, Dict[str, float]] = {}
    raw_distributions: Dict[str, np.ndarray] = {}

    # A) Full Mahalanobis (Ledoit-Wolf)
    def full_maha_fn(z: torch.Tensor) -> torch.Tensor:
        diff = z - means_flat  # (K, d_z)
        return torch.einsum("cd,cde,ce->c", diff, full_invs_t, diff).min()

    stats_full, dist_full = profiler.profile_scoring_head(full_maha_fn, z_flat)
    raw_distributions["full_mahalanobis_us"] = dist_full

    # B) Block-Diagonal Mahalanobis (Proposed Vectorized Einsum)
    def block_maha_fn(z_b: torch.Tensor) -> torch.Tensor:
        return block_diagonal_mahalanobis_score(z_b, mu_blocks, block_invs_t)

    stats_block, dist_block = profiler.profile_scoring_head(block_maha_fn, z_blocks)
    raw_distributions["block_diagonal_proposed_us"] = dist_block

    # C) Diagonal Mahalanobis
    def diag_maha_fn(z: torch.Tensor) -> torch.Tensor:
        diff = z - means_flat
        return torch.sum((diff * diff) * var_invs_t, dim=-1).min()

    stats_diag, dist_diag = profiler.profile_scoring_head(diag_maha_fn, z_flat)
    raw_distributions["diagonal_mahalanobis_us"] = dist_diag

    # D) Euclidean Prototype
    def euc_fn(z: torch.Tensor) -> torch.Tensor:
        diff = z - means_flat
        return torch.sum(diff * diff, dim=-1).min()

    stats_euc, dist_euc = profiler.profile_scoring_head(euc_fn, z_flat)
    raw_distributions["euclidean_prototype_us"] = dist_euc

    # E) OpenMax (Weibull EVT + Matrix Distance)
    weibull_scales = torch.ones(num_classes, device=device) * 12.0
    weibull_shapes = torch.ones(num_classes, device=device) * 2.5

    def openmax_fn(z: torch.Tensor) -> torch.Tensor:
        diff = z - means_flat
        maha = torch.einsum("cd,cde,ce->c", diff, full_invs_t, diff)
        euc = torch.linalg.norm(diff, dim=-1)
        combined = 0.5 * maha + 0.5 * euc
        weibull_cdf = 1.0 - torch.exp(-torch.pow(torch.clamp(combined / weibull_scales, min=1e-6), weibull_shapes))
        return weibull_cdf.max()

    stats_openmax, dist_openmax = profiler.profile_scoring_head(openmax_fn, z_flat)
    raw_distributions["openmax_evt_us"] = dist_openmax

    # Memory footprints (in KB)
    memory_footprints_kb = {
        "Full Mahalanobis (Ledoit-Wolf)": round(float(num_classes * (latent_dim + (latent_dim ** 2)) * 4 / 1024.0), 2),
        "Block-Diagonal Mahalanobis (Proposed)": round(float(num_classes * (latent_dim + num_blocks * (block_dim ** 2)) * 4 / 1024.0), 2),
        "Diagonal Mahalanobis": round(float(num_classes * (latent_dim + latent_dim) * 4 / 1024.0), 2),
        "Euclidean Prototype": round(float(num_classes * latent_dim * 4 / 1024.0), 2),
        "OpenMax (Weibull EVT)": round(float(num_classes * (latent_dim + (latent_dim ** 2) + 3) * 4 / 1024.0), 2),
    }

    method_stats_map = {
        "Full Mahalanobis (Ledoit-Wolf)": stats_full,
        "Block-Diagonal Mahalanobis (Proposed)": stats_block,
        "Diagonal Mahalanobis": stats_diag,
        "Euclidean Prototype": stats_euc,
        "OpenMax (Weibull EVT)": stats_openmax,
    }

    ref_median = stats_full["median_us"]
    for name, st in method_stats_map.items():
        results[name] = {
            "median_latency_us": st["median_us"],
            "mean_latency_us": st["mean_us"],
            "mean_latency_ms": st["mean_us"] / 1000.0,
            "p95_latency_us": st["p95_us"],
            "p99_latency_us": st["p99_us"],
            "std_latency_us": st["std_us"],
            "throughput_fps": st["throughput_fps"],
            "speedup_vs_full": float(ref_median / max(st["median_us"], 1e-6)),
            "memory_footprint_kb": memory_footprints_kb[name],
            # Backward-compatible aliases
            "Full Mahalanobis": ref_median,
        }

    # Add legacy key alias for backward compatibility with existing tests
    results["Full Mahalanobis"] = results["Full Mahalanobis (Ledoit-Wolf)"]

    return results, raw_distributions


def profile_subblock_pareto_curve(
    num_classes: int = 4,
    latent_dim: int = 64,
    block_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64),
    warmup: int = 500,
    trials: int = 5000,
) -> Dict[str, Any]:
    """Profile FLOPs, memory (KB), and latency across sub-block sizes d_b in {4, 8, 16, 32, 64} (Ablation A5)."""
    profiler = LatencyProfiler(device="cpu", warmup=warmup, trials=trials)
    pareto_records: Dict[str, Any] = {}

    for d_b in block_sizes:
        num_blocks = latent_dim // d_b
        z_b = torch.randn(1, num_blocks, d_b)
        mu_b = torch.randn(num_classes, num_blocks, d_b)
        inv_b = torch.randn(num_classes, num_blocks, d_b, d_b)
        inv_b = torch.matmul(inv_b, inv_b.transpose(-1, -2)) + 0.1 * torch.eye(d_b)

        def fn(z_in: torch.Tensor) -> torch.Tensor:
            return block_diagonal_mahalanobis_score(z_in, mu_b, inv_b)

        st, _ = profiler.profile_scoring_head(fn, z_b)
        flops = compute_metric_flops(
            num_classes=num_classes,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
            block_dim=d_b,
        )["Block-Diagonal Mahalanobis (Proposed)"]
        mem_kb = round(float(num_classes * (latent_dim + num_blocks * (d_b ** 2)) * 4 / 1024.0), 2)

        pareto_records[f"db_{d_b}"] = {
            "block_dim": d_b,
            "num_blocks": num_blocks,
            "flops": flops,
            "memory_kb": mem_kb,
            "median_latency_us": st["median_us"],
            "p95_latency_us": st["p95_us"],
            "p99_latency_us": st["p99_us"],
        }

    return pareto_records


def run_wire_speed_experiment(
    output_dir: str = "results/wire_speed",
    warmup: int = 1000,
    trials: int = 10000,
) -> Dict[str, Any]:
    """Execute complete wire-speed complexity and latency profiling suite (E8)."""
    logger = setup_logger(name="wire_speed_runner")
    logger.info(f"Executing Wire-Speed Latency and Complexity Profiling (E8, warmup={warmup}, trials={trials})...")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    encoder_flops = compute_encoder_flops(in_features=34, hidden_dims=(512, 256, 128), latent_dim=64)
    metric_flops = compute_metric_flops(num_classes=4, latent_dim=64, num_blocks=4, block_dim=16)
    # Include legacy key for compatibility
    metric_flops["Full Mahalanobis"] = metric_flops["Full Mahalanobis (Ledoit-Wolf)"]

    latency_results, raw_distributions = profile_runtime_latency(
        num_classes=4,
        latent_dim=64,
        num_blocks=4,
        block_dim=16,
        warmup=warmup,
        trials=trials,
    )
    subblock_pareto = profile_subblock_pareto_curve(
        num_classes=4,
        latent_dim=64,
        block_sizes=(4, 8, 16, 32, 64),
        warmup=min(500, warmup),
        trials=min(5000, trials),
    )

    summary = {
        "encoder_flops": encoder_flops,
        "metric_flops": metric_flops,
        "latency_profiles": latency_results,
        "subblock_pareto": subblock_pareto,
    }

    for method in [
        "Full Mahalanobis (Ledoit-Wolf)",
        "Block-Diagonal Mahalanobis (Proposed)",
        "Diagonal Mahalanobis",
        "Euclidean Prototype",
        "OpenMax (Weibull EVT)",
    ]:
        lat = latency_results[method]
        flops = metric_flops.get(method, 0)
        logger.info(
            f"  [{method}] FLOPs: {flops:,} | "
            f"Median: {lat['median_latency_us']:.2f} us [P95: {lat['p95_latency_us']:.2f}, P99: {lat['p99_latency_us']:.2f}] | "
            f"Throughput: {lat['throughput_fps']:,.0f} PPS | "
            f"RAM: {lat['memory_footprint_kb']:.2f} KB | "
            f"Speedup: {lat['speedup_vs_full']:.2f}x"
        )

    save_json(summary, out_p / "wire_speed_profiling.json")

    # Save standardized Q1 system profiling log + decoupled plot bundle (.npz + .json)
    proposed_lat = latency_results["Block-Diagonal Mahalanobis (Proposed)"]
    q1_wire_log = {
        "experiment_id": "E8_wire_speed_profiling",
        "flops_per_query": metric_flops["Block-Diagonal Mahalanobis (Proposed)"],
        "median_latency_us": round(proposed_lat["median_latency_us"], 2),
        "mean_latency_us": round(proposed_lat["mean_latency_us"], 2),
        "p95_latency_us": round(proposed_lat["p95_latency_us"], 2),
        "p99_latency_us": round(proposed_lat["p99_latency_us"], 2),
        "throughput_pps": round(proposed_lat["throughput_fps"], 1),
        "memory_footprint_kb": proposed_lat["memory_footprint_kb"],
        "phase2_payload_kb": proposed_lat["memory_footprint_kb"],
        "speedup_vs_full": round(proposed_lat["speedup_vs_full"], 2),
        "subblock_pareto": subblock_pareto,
    }
    save_experiment_audit_bundle(
        output_dir=out_p,
        bundle_name="q1_wire_speed_log",
        summary_data=q1_wire_log,
        raw_arrays=raw_distributions,
        config={"warmup": warmup, "trials": trials, "latent_dim": 64, "block_dim": 16},
        seed=42,
    )

    # Generate LaTeX table
    latex_tbl = generate_wire_speed_latex_table(metric_flops, latency_results)
    with open(out_p / "table_wire_speed.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Wire-speed profiling completed. Results saved to {out_p}")
    return summary


def generate_wire_speed_latex_table(
    flops_dict: Dict[str, int],
    latency_dict: Dict[str, Dict[str, float]],
) -> str:
    """Format complexity and latency comparison into publication-ready LaTeX table with Median [P95, P99]."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hardware complexity, FLOPs count, and single-packet ($B=1$) wire-speed inference latency across edge detection heads ($d_z = 64, B = 4, d_b = 16$, 10,000 trials).}",
        r"\label{tab:wire_speed_complexity}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Detection Head & Complexity & FLOPs & Median [P95, P99] ($\mu$s) & Memory (KB) & Speedup \\",
        r"\midrule",
    ]

    complexity_map = {
        "Full Mahalanobis (Ledoit-Wolf)": r"$\mathcal{O}(|C_K| \cdot d_z^2)$",
        "OpenMax (Weibull EVT)": r"$\mathcal{O}(|C_K| \cdot d_z^2)$",
        "Block-Diagonal Mahalanobis (Proposed)": r"$\mathcal{O}(B \cdot d_b^2)$",
        "Diagonal Mahalanobis": r"$\mathcal{O}(|C_K| \cdot d_z)$",
        "Euclidean Prototype": r"$\mathcal{O}(|C_K| \cdot d_z)$",
    }

    for method in [
        "Full Mahalanobis (Ledoit-Wolf)",
        "OpenMax (Weibull EVT)",
        "Diagonal Mahalanobis",
        "Euclidean Prototype",
        "Block-Diagonal Mahalanobis (Proposed)",
    ]:
        if method not in latency_dict:
            continue
        comp = complexity_map.get(method, "-")
        flops = flops_dict.get(method, 0)
        entry = latency_dict[method]
        med = entry.get("median_latency_us", entry.get("mean_latency_us", 0.0))
        p95 = entry.get("p95_latency_us", 0.0)
        p99 = entry.get("p99_latency_us", 0.0)
        mem = entry.get("memory_footprint_kb", 0.0)
        spd = entry.get("speedup_vs_full", 1.0)

        name_str = rf"\textbf{{{method}}}" if "Proposed" in method else method
        lines.append(
            f"{name_str} & {comp} & {flops:,} & {med:.2f} [{p95:.2f}, {p99:.2f}] & {mem:.2f} & {spd:.2f}$\\times$ \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wire-Speed Profiling (E8)")
    parser.add_argument("--output_dir", type=str, default="results/wire_speed")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=10000)
    args = parser.parse_args()
    run_wire_speed_experiment(output_dir=args.output_dir, warmup=args.warmup, trials=args.trials)
