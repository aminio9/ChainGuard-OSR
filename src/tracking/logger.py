"""Structured logger with console and JSONL output for research experiments."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from src.utils.io import ensure_dir


class JsonlHandler(logging.Handler):
    """Custom logging handler to record structured metrics into a JSONL file."""

    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        ensure_dir(filepath.parent)

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "structured_data"):
            data = getattr(record, "structured_data")
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")


def setup_logger(
    name: str = "fl_osr",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup structured application logger with stream and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream Handler
    stream_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(stream_fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        ld = ensure_dir(log_dir)
        text_handler = logging.FileHandler(ld / "experiment.log", encoding="utf-8")
        text_handler.setFormatter(stream_fmt)
        logger.addHandler(text_handler)

        jsonl_handler = JsonlHandler(ld / "metrics.jsonl")
        logger.addHandler(jsonl_handler)

    return logger


def format_q1_experiment_log(
    dataset: str,
    alpha_dirichlet: Optional[float | str],
    round_num: int,
    closed_metrics: Dict[str, Any],
    osr_metrics: Optional[Dict[str, Any]] = None,
    system_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate standardized Q1 peer-review audit log record matching IEEE TIFS/TDSC requirements."""
    metrics_block: Dict[str, Any] = {
        "closed_set": {
            "accuracy": float(closed_metrics.get("accuracy", 0.0)),
            "balanced_acc": float(closed_metrics.get("balanced_accuracy", 0.0)),
            "macro_f1": float(closed_metrics.get("macro_f1", 0.0)),
        }
    }

    if "class_wise_f1" in closed_metrics and closed_metrics["class_wise_f1"]:
        metrics_block["closed_set"]["class_wise_f1"] = [round(float(x), 4) for x in closed_metrics["class_wise_f1"]]
    if "confusion_matrix" in closed_metrics and closed_metrics["confusion_matrix"]:
        metrics_block["closed_set"]["confusion_matrix"] = closed_metrics["confusion_matrix"]

    if osr_metrics is not None:
        metrics_block["open_set"] = {
            "auroc": float(osr_metrics.get("auroc", 0.0)),
            "aupr": float(osr_metrics.get("aupr", 0.0)),
            "fpr95": float(osr_metrics.get("fpr95", 0.0)),
            "unknown_recall": float(osr_metrics.get("unknown_recall", 0.0)),
            "unknown_f1": float(osr_metrics.get("unknown_f1", 0.0)),
            "nominal_kfr": float(osr_metrics.get("nominal_kfr", 0.05)),
            "realized_kfr": float(osr_metrics.get("kfr", 0.0)),
            "open_set_macro_f1": float(osr_metrics.get("open_set_macro_f1", 0.0)),
        }

    if system_metrics is not None:
        metrics_block["system_profiling"] = system_metrics

    alpha_val = alpha_dirichlet if alpha_dirichlet is not None else "IID"
    record = {
        "dataset": dataset,
        "alpha_dirichlet": alpha_val,
        "round": round_num,
        "metrics": metrics_block,
    }
    return record


def save_q1_log(record: Dict[str, Any], output_path: Path | str) -> None:
    """Save formatted Q1 log record to json file."""
    p = Path(output_path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def generate_audit_metadata(
    config: Dict[str, Any],
    seed: int,
    data_dir: Path | str = "data/raw",
) -> Dict[str, Any]:
    """Extract deterministic system, git commit, and dataset provenance checksum for Q1 auditability."""
    import hashlib
    import subprocess

    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit_hash = "untracked_workspace"

    data_p = Path(data_dir)
    hasher = hashlib.sha256()
    if data_p.exists():
        data_files = sorted(list(data_p.glob("*.csv")))
        for f in data_files[:5]:
            try:
                with open(f, "rb") as fp:
                    hasher.update(fp.read(4096))
            except Exception:
                pass
    dataset_hash = hasher.hexdigest()[:16]

    return {
        "git_commit": commit_hash,
        "dataset_provenance_hash": dataset_hash,
        "random_seed": int(seed),
        "hyperparameters": config,
    }


def save_experiment_audit_bundle(
    output_dir: Path | str,
    experiment_name: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    round_history: Optional[Dict[str, Any]] = None,
    summary_metrics: Optional[Dict[str, Any]] = None,
    final_eval_arrays: Optional[Dict[str, Any]] = None,
    *,
    bundle_name: Optional[str] = None,
    summary_data: Optional[Dict[str, Any]] = None,
    raw_arrays: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[Path, Optional[Path]]:
    """Save decoupled lightweight JSON audit log and compressed .npz array bundle for standalone replotting."""
    import numpy as np

    out_p = ensure_dir(output_dir)
    exp_name = bundle_name or experiment_name or "experiment"
    eff_meta = meta if meta is not None else generate_audit_metadata(config or {}, seed=seed)
    eff_summary = summary_data if summary_data is not None else (summary_metrics or {})
    eff_arrays = raw_arrays if raw_arrays is not None else final_eval_arrays

    payload = {
        "meta": eff_meta,
        "round_history": round_history or {},
        "summary_metrics": eff_summary,
        "results": eff_summary,
        **(eff_summary if isinstance(eff_summary, dict) else {}),
    }

    json_path = out_p / f"{exp_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    audit_json_path = out_p / f"{exp_name}_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    npz_path: Optional[Path] = None
    if eff_arrays:
        npz_path = out_p / f"{exp_name}.npz"
        clean_arrays = {
            k: np.asarray(v) for k, v in eff_arrays.items() if v is not None
        }
        np.savez_compressed(npz_path, **clean_arrays)
        np.savez_compressed(out_p / f"{exp_name}_arrays.npz", **clean_arrays)

    return json_path, npz_path



