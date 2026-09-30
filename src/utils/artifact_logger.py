"""Structured experiment artifact logger for reproducible multi-seed runs and plotting."""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def get_git_commit_hash(repo_dir: str = ".") -> str:
    """Return current git commit hash or 'untracked' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "untracked"
    except Exception:
        return "untracked"


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return _to_serializable(obj.model_dump())
    if dataclasses.is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_seed_run_artifacts(
    output_root: str | Path,
    experiment_id: str,
    dataset_name: str,
    seed: int,
    config: Any,
    metrics: Dict[str, Any],
    split_metadata: Dict[str, Any],
    round_history: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[Dict[int, float]] = None,
    calibration_counts: Optional[Dict[int, int]] = None,
    per_class_kfr: Optional[Dict[int, float]] = None,
    per_class_coverage: Optional[Dict[int, float]] = None,
    scores_table: Optional[Dict[str, np.ndarray]] = None,
    confusion_before: Optional[np.ndarray] = None,
    confusion_after: Optional[np.ndarray] = None,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Save structured per-seed experiment outputs to results/{experiment_id}/{dataset}/seed_{seed}/."""
    seed_dir = Path(output_root) / experiment_id / dataset_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "seed": int(seed),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "split_metadata": _to_serializable(split_metadata),
    }

    with open(seed_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    cfg_dict = _to_serializable(config)
    with open(seed_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

    full_metrics = {
        "provenance": provenance,
        "metrics": _to_serializable(metrics),
    }
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    if thresholds is not None or calibration_counts is not None:
        thresh_payload = {
            "thresholds": _to_serializable(thresholds or {}),
            "calibration_counts": _to_serializable(calibration_counts or {}),
            "per_class_kfr": _to_serializable(per_class_kfr or {}),
            "per_class_coverage": _to_serializable(per_class_coverage or {}),
        }
        with open(seed_dir / "thresholds.json", "w", encoding="utf-8") as f:
            json.dump(thresh_payload, f, indent=2)

    if round_history:
        keys = sorted({k for row in round_history for k in row.keys()})
        with open(seed_dir / "round_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in round_history:
                writer.writerow({k: row.get(k, "") for k in keys})

    if scores_table:
        keys = list(scores_table.keys())
        n_rows = len(scores_table[keys[0]]) if keys else 0
        with open(seed_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for idx in range(n_rows):
                writer.writerow([scores_table[k][idx] for k in keys])

    if confusion_before is not None:
        np.savetxt(seed_dir / "confusion_before.csv", confusion_before, delimiter=",", fmt="%d")
    if confusion_after is not None:
        np.savetxt(seed_dir / "confusion_after.csv", confusion_after, delimiter=",", fmt="%d")
    if extra_arrays:
        np.savez_compressed(seed_dir / "diagnostics.npz", **extra_arrays)

    return seed_dir


def aggregate_MultiSeed_Summary(
    output_root: str | Path,
    experiment_id: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """Aggregate all seed_*/metrics.json files into summary.json with mean and std."""
    exp_dir = Path(output_root) / experiment_id / dataset_name
    seed_files = sorted(exp_dir.glob("seed_*/metrics.json"))
    if not seed_files:
        return {}

    seed_payloads = []
    for sf in seed_files:
        with open(sf, "r", encoding="utf-8") as f:
            seed_payloads.append(json.load(f))

    # Group scalar metrics across seeds
    sample_metrics = seed_payloads[0].get("metrics", {})
    summary_metrics: Dict[str, Any] = {}

    def _aggregate_dict(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        all_keys = sorted({k for d in dicts for k in d.keys()})
        for k in all_keys:
            vals = [d[k] for d in dicts if k in d and d[k] is not None]
            if not vals:
                continue
            if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
                arr = np.asarray(vals, dtype=np.float64)
                out[k] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "values": [float(v) for v in arr],
                }
            elif isinstance(vals[0], dict):
                out[k] = _aggregate_dict(vals)
            else:
                out[k] = vals[0]
        return out

    summary_metrics = _aggregate_dict([p.get("metrics", {}) for p in seed_payloads])
    summary = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "num_seeds": len(seed_payloads),
        "seeds": [p.get("provenance", {}).get("seed") for p in seed_payloads],
        "summary": summary_metrics,
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
