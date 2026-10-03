"""Data loading and verification functions for experiment metrics."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.ref_plots.config import PROCESSED_DATA_DIR, METADATA_DIR, DEFAULT_RESULTS_JSON


_ACTIVE_DATA_DIR: ContextVar[Path | None] = ContextVar("plot_data_dir", default=None)


@contextmanager
def using_data_dir(data_dir: str | Path | None) -> Iterator[None]:
    """Temporarily make ``data_dir`` authoritative for all legacy plot loaders."""
    resolved = Path(data_dir).expanduser().resolve() if data_dir is not None else None
    token = _ACTIVE_DATA_DIR.set(resolved)
    try:
        yield
    finally:
        _ACTIVE_DATA_DIR.reset(token)


def resolve_data_path(file_ref: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve an input without silently preferring stale repository data.

    An explicitly supplied ``base_dir`` (or the active directory installed by
    :func:`using_data_dir`) takes precedence over bundled sample data.  Both the
    complete relative reference and its basename are supported because historical
    result files used ``real_plot_data/<name>`` while adapters emit flat files.
    """
    path = Path(file_ref).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()

    roots: list[Path] = []
    if base_dir is not None:
        roots.append(Path(base_dir).expanduser().resolve())
    active = _ACTIVE_DATA_DIR.get()
    if active is not None and active not in roots:
        roots.append(active)
    if PROCESSED_DATA_DIR not in roots:
        roots.append(PROCESSED_DATA_DIR)

    candidates = [candidate for root in roots for candidate in (root / path, root / path.name)]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"Could not locate data file {file_ref!s}; searched: {searched}")


def load_json_data(path: Path | None = None) -> dict[str, Any]:
    """Load and validate result.json from metadata directory."""
    p = Path(path).expanduser().resolve() if path is not None else DEFAULT_RESULTS_JSON
    active = _ACTIVE_DATA_DIR.get()
    if path is None and active is not None and (active / "result.json").exists():
        p = active / "result.json"
    if not p.exists():
        p = METADATA_DIR / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"result.json not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_scalability_logs(data_dir: Path | None = None) -> dict[int, pd.DataFrame]:
    """Load round-level scalability metrics for 10, 50, and 100 clients."""
    d = Path(data_dir).expanduser().resolve() if data_dir is not None else (
        _ACTIVE_DATA_DIR.get() or PROCESSED_DATA_DIR
    )
    files = {
        10: d / "scalability_10_clients.csv",
        50: d / "scalability_50_clients.csv",
        100: d / "scalability_100_clients.csv",
    }
    
    logs: dict[int, pd.DataFrame] = {}
    required = {
        "round", "num_clients", "mean_client_macro_f1", "std_client_macro_f1",
        "worst_client_macro_f1", "client_fit_wall_time_sec", "round_time_sec",
        "server_aggregation_time_sec", "open_set_round_eval_time_sec",
        "round_openset_f1_macro", "round_openset_auroc",
        "round_openset_fpr95", "round_openset_unknown_recall",
    }
    metric_columns = required - {
        "round",
        "num_clients",
        "client_fit_wall_time_sec",
        "round_time_sec",
        "server_aggregation_time_sec",
        "open_set_round_eval_time_sec",
    }
    
    for clients, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing scalability CSV for {clients} clients: {p}")
        df = pd.read_csv(p).sort_values("round").reset_index(drop=True)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p.name}: missing columns {sorted(missing)}")
        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        empty = sorted(column for column in metric_columns if not df[column].notna().any())
        if empty:
            raise ValueError(f"{p.name}: all-null required metrics {empty}")
        df["cumulative_hours"] = df["round_time_sec"].cumsum() / 3600.0
        logs[clients] = df
    return logs


def load_confusion_matrix(path: Path) -> pd.DataFrame:
    """Load a confusion matrix CSV with numeric indices and columns."""
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.apply(pd.to_numeric, errors="raise")


def load_score_arrays(scores_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load known/unknown canonical Mahalanobis nonconformity scores."""
    frame = pd.read_csv(scores_path)
    forbidden = {"prototype_rank_score", "rank_score", "prototype_score", "selected_threshold_used"}
    bad = forbidden.intersection(frame.columns)
    if bad:
        raise ValueError(f"Legacy score fields are forbidden in canonical v2: {sorted(bad)}")
    if "nonconformity_score" not in frame.columns or "known_or_unknown" not in frame.columns:
        raise ValueError("scores CSV must contain nonconformity_score and known_or_unknown")
    unknown_mask = frame["known_or_unknown"].astype(str).str.lower().eq("unknown")
    scores = pd.to_numeric(frame["nonconformity_score"], errors="coerce")
    return scores[~unknown_mask].dropna().to_numpy(), scores[unknown_mask].dropna().to_numpy()
