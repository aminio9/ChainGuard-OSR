"""Tabular loader and preprocessor for real intrusion detection benchmarks.

Supports all 4 paper benchmarks (Section 5.1 & Table 2):
  1. B-NAT (Blockchain Network Attack Traffic)
  2. B-TAT (Blockchain Transaction and Attack Traffic)
  3. ToN-IoT (Industrial IoT Telemetry and Network Flow)
  4. CIC-IDS2017 (Enterprise Network Intrusion Benchmark)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from src.data.base import FlowDataset

BENCHMARK_METADATA: Dict[str, Dict[str, Any]] = {
    "b_nat": {
        "filename": "b_nat.csv",
        "target_col": "label",
        "known_classes": ["Normal", "BP", "DoS", "MitM"],
        "unknown_classes": ["FoT"],
    },
    "b_tat": {
        "filename": "b_tat.csv",
        "target_col": "label",
        "known_classes": ["Normal", "DoS", "OaU", "FoT", "FDV"],
        "unknown_classes": ["Re", "DeC"],
    },
    "ton_iot": {
        "filename": "ton_iot.csv",
        "target_col": "type",
        "known_classes": ["normal", "backdoor", "ddos", "dos", "injection", "password", "scanning"],
        "unknown_classes": ["mitm", "ransomware", "xss"],
    },
    "cic_ids2017": {
        "filename": "cic_ids2017.csv",
        "target_col": "Label",
        "known_classes": [
            "BENIGN",
            "DDoS",
            "DoS Hulk",
            "PortScan",
            "DoS GoldenEye",
            "FTP-Patator",
            "SSH-Patator",
            "DoS Slowhttptest",
        ],
        "unknown_classes": ["Bot", "Web Attack-Brute Force", "Web Attack-XSS"],
    },
}


def load_p2p_tabular_dataset(
    csv_path: str | Path,
    target_column: str = "label",
    normalize: bool = False,
    max_samples: Optional[int] = None,
) -> FlowDataset:
    """Load tabular CSV network flow dataset, clean infs/nans, normalize features, and return FlowDataset."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found at: {path}")

    # Case-insensitive resolution of target column
    preview = pd.read_csv(path, nrows=5)
    candidates = [c for c in preview.columns if c.strip().lower() == target_column.lower()]
    if candidates:
        target_column = candidates[0]
    elif "label" in [c.lower() for c in preview.columns]:
        target_column = [c for c in preview.columns if c.lower() == "label"][0]
    elif "type" in [c.lower() for c in preview.columns]:
        target_column = [c for c in preview.columns if c.lower() == "type"][0]

    df = pd.read_csv(path, nrows=max_samples, low_memory=False)

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in CSV columns: {df.columns.tolist()[:10]}")

    y_raw = df[target_column].astype(str).str.strip()
    unique_labels = sorted(y_raw.unique())
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_arr = y_raw.map(label_to_id).to_numpy(dtype=np.int64)

    X_df = df.drop(columns=[target_column])
    for col_name in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col_name]):
            X_df[col_name] = pd.factorize(X_df[col_name].astype(str))[0]
    X_numeric = X_df.select_dtypes(include=[np.number])
    X_arr = X_numeric.to_numpy(dtype=np.float32)

    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)

    if normalize:
        mean = np.mean(X_arr, axis=0, keepdims=True)
        std = np.std(X_arr, axis=0, keepdims=True) + 1e-7
        X_arr = (X_arr - mean) / std

    return FlowDataset(torch.from_numpy(X_arr.copy()), torch.from_numpy(y_arr.copy()))


def load_benchmark_dataset(
    dataset_name: str,
    data_dir: str = "data/raw",
    samples_per_class: Optional[int] = None,
    cache_dir: str = "data/processed",
) -> Tuple[FlowDataset, Dict[str, Any]]:
    """Load, encode, and return one of the 4 real benchmark datasets on 100% of data.

    Args:
        dataset_name: One of 'b_nat', 'b_tat', 'ton_iot', 'cic_ids2017'.
        data_dir: Directory containing the raw CSV files.
        samples_per_class: Optional subsampling limit (default None = 100% of all data).
        cache_dir: Directory to save/load preprocessed tensor caches.

    Returns:
        (FlowDataset, metadata_dict)
    """
    key = dataset_name.lower().replace("-", "_")
    if key not in BENCHMARK_METADATA:
        raise ValueError(f"Unknown benchmark dataset '{dataset_name}'. Choose from: {list(BENCHMARK_METADATA.keys())}")

    meta = BENCHMARK_METADATA[key]
    csv_path = Path(data_dir) / meta["filename"]
    if not csv_path.is_file():
        fallback_root = Path(__file__).resolve().parent.parent.parent
        alt_path = fallback_root / data_dir / meta["filename"]
        if alt_path.is_file():
            csv_path = alt_path
        else:
            alt_root = fallback_root.parent / data_dir / meta["filename"]
            if alt_root.is_file():
                csv_path = alt_root
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV for benchmark '{dataset_name}' not found at: {csv_path}")

    # Safety safeguard: For CIC-IDS2017 (902 MB, ~2.8M rows), loading without subsampling
    # exceeds 14 GB RAM and causes fatal OOM kills in memory-constrained environments (Colab 12.7 GB limit).
    # If samples_per_class is None, default to a statistically sound 2000 samples/class (~22k flows).
    # Pass samples_per_class=-1 to explicitly force loading the entire 2.8M rows on large servers.
    if key == "cic_ids2017":
        if samples_per_class is None:
            print("  [INFO] CIC-IDS2017 contains 2.8M rows (902 MB). Capping at stratified samples_per_class=2000 to prevent system OOM (Colab 12.7 GB RAM limit). (Pass samples_per_class=-1 to force loading entire 2.8M rows).")
            samples_per_class = 2000
        elif samples_per_class == -1:
            samples_per_class = None

    c_dir = Path(cache_dir)
    if not c_dir.is_dir():
        c_alt = Path(__file__).resolve().parent.parent.parent / cache_dir
        if c_alt.exists():
            c_dir = c_alt
    cache_path = c_dir / f"{key}_spc_{samples_per_class}.pt"
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, weights_only=False)
            return cached["dataset"], cached["metadata"]
        except (EOFError, RuntimeError, Exception) as cache_err:
            print(f"  [WARN] Corrupt/incomplete cache detected at {cache_path} ({cache_err}). Rebuilding from raw CSV...")
            try:
                cache_path.unlink()
            except Exception:
                pass

    known_list = meta["known_classes"]
    unknown_list = meta["unknown_classes"]
    total_needed = set(known_list + unknown_list)

    # Deterministic label mapping: Known: 0..len(known)-1, Unknown: len(known)..
    label_to_id: Dict[str, int] = {}
    for i, name in enumerate(known_list):
        label_to_id[name] = i
    for j, name in enumerate(unknown_list):
        label_to_id[name] = len(known_list) + j

    target_col_name = meta["target_col"]

    # Sample rows across the dataset
    if samples_per_class is not None:
        counts = {c: 0 for c in total_needed}
        collected_chunks = {c: [] for c in total_needed}

        chunk_size = 50000 if key != "cic_ids2017" else 100000
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            col_candidates = [c for c in chunk.columns if c.strip().lower() == target_col_name.lower()]
            col = col_candidates[0] if col_candidates else target_col_name
            chunk[col] = chunk[col].astype(str).str.strip()

            for cname in total_needed:
                if counts[cname] < samples_per_class:
                    matches = chunk[chunk[col] == cname]
                    if len(matches) > 0:
                        take = matches.head(samples_per_class - counts[cname])
                        collected_chunks[cname].append(take)
                        counts[cname] += len(take)

            if all(counts[c] >= samples_per_class for c in total_needed):
                break

        all_dfs = [pd.concat(dfs, ignore_index=True) for dfs in collected_chunks.values() if len(dfs) > 0]
        df = pd.concat(all_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        df = pd.read_csv(csv_path, low_memory=False)
        col_candidates = [c for c in df.columns if c.strip().lower() == target_col_name.lower()]
        col = col_candidates[0] if col_candidates else target_col_name
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col].isin(total_needed)].reset_index(drop=True)

    target_col = [c for c in df.columns if c.strip().lower() == target_col_name.lower()][0]
    y_raw = df[target_col].astype(str).str.strip()
    y_arr = y_raw.map(label_to_id).to_numpy(dtype=np.int64)

    X_df = df.drop(columns=[target_col])

    from src.data.traffic_preprocessor import DATASET_RECIPES, ZeroLeakageTrafficPreprocessor

    raw_df_copy = X_df.copy()

    # Pre-clean known special columns like value in B-TAT
    if key == "b_tat" and "value" in X_df.columns:
        X_df["value"] = pd.to_numeric(X_df["value"], errors="coerce").fillna(0.0)

    if key in DATASET_RECIPES:
        recipe = DATASET_RECIPES[key]
        preprocessor = ZeroLeakageTrafficPreprocessor(recipe)
        preprocessor.fit(X_df)
        X_arr = preprocessor.transform(X_df)
    else:
        # Fallback for datasets without recipes
        for col_name in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col_name]):
                X_df[col_name] = pd.factorize(X_df[col_name].astype(str))[0]
        X_numeric = X_df.select_dtypes(include=[np.number])
        X_arr = X_numeric.to_numpy(dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)

    dataset = FlowDataset(
        torch.from_numpy(X_arr.copy()),
        torch.from_numpy(y_arr.copy()),
        raw_df=raw_df_copy,
        recipe_name=key,
    )

    metadata = {
        "dataset_name": key,
        "in_features": int(X_arr.shape[1]),
        "total_samples": int(len(dataset)),
        "num_known_classes": len(known_list),
        "num_unknown_classes": len(unknown_list),
        "known_classes": list(range(len(known_list))),
        "unknown_classes": list(range(len(known_list), len(known_list) + len(unknown_list))),
        "known_class_names": known_list,
        "unknown_class_names": unknown_list,
        "label_to_id": label_to_id,
        "id_to_label": {v: k for k, v in label_to_id.items()},
    }

    # Save to cache
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": dataset, "metadata": metadata}, cache_path)

    return dataset, metadata
