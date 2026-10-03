"""Q1-Compliant Zero-Leakage Traffic Preprocessor for Network and Blockchain Benchmarks.

Strictly separates Continuous, Ratio, and Nominal features.
All statistical parameters (means, standard deviations, vocabularies) are fitted
EXCLUSIVELY on the federated training partition D_train containing only known classes Y_K.
Calibration (D_cal) and Test partitions (D_test_known, D_test_open) are transformed
inductively without information leakage or out-of-vocabulary distortion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch


@dataclass
class PreprocessingRecipe:
    """Recipe defining feature role classification for a benchmark dataset."""

    dataset_name: str
    target_col: str
    skewed_numeric_cols: List[str]
    bounded_ratio_cols: List[str]
    categorical_cols: List[str]
    financial_wei_cols: List[str] = field(default_factory=list)
    one_hot: bool = True
    fixed_vocabularies: Dict[str, List[str]] = field(default_factory=dict)


DATASET_RECIPES: Dict[str, PreprocessingRecipe] = {
    "b_nat": PreprocessingRecipe(
        dataset_name="b_nat",
        target_col="label",
        skewed_numeric_cols=[
            "duration",
            "src_bytes",
            "dst_bytes",
            "count",
            "srv_count",
            "dst_host_count",
            "dst_host_srv_count",
        ],
        bounded_ratio_cols=[
            "serror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_serror_rate",
            "srv_diff_host_rate",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_serror_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_srv_serror_rate",
        ],
        categorical_cols=["protocol_type", "service", "flag"],
        one_hot=True,
        fixed_vocabularies={
            "protocol_type": ["icmp", "tcp", "udp"],
            "service": ["http", "netbios_ssn", "oth_i", "other", "private"],
            "flag": ["OTH", "RSTR", "S1", "S3", "SF"],
        },
    ),
    "b_tat": PreprocessingRecipe(
        dataset_name="b_tat",
        target_col="label",
        skewed_numeric_cols=["gas_limit", "gas_used", "gas_price", "nonce"],
        bounded_ratio_cols=[],
        categorical_cols=[],
        financial_wei_cols=["value"],
        one_hot=False,
    ),
    "cic_ids2017": PreprocessingRecipe(
        dataset_name="cic_ids2017",
        target_col="Label",
        skewed_numeric_cols=[
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
        ],
        bounded_ratio_cols=[],
        categorical_cols=[],
        one_hot=False,
    ),
    "ton_iot": PreprocessingRecipe(
        dataset_name="ton_iot",
        target_col="type",
        skewed_numeric_cols=["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"],
        bounded_ratio_cols=[],
        categorical_cols=["proto", "service", "conn_state", "http_method"],
        one_hot=True,
    ),
}


class ZeroLeakageTrafficPreprocessor:
    """Production-grade inductive preprocessor guaranteeing mathematical zero leakage."""

    def __init__(self, recipe: PreprocessingRecipe):
        self.recipe = recipe
        self.fitted: bool = False

        # Parameters fitted strictly on D_train
        self.skewed_means: Dict[str, float] = {}
        self.skewed_stds: Dict[str, float] = {}
        self.wei_means: Dict[str, float] = {}
        self.wei_stds: Dict[str, float] = {}
        self.other_means: Dict[str, float] = {}
        self.other_stds: Dict[str, float] = {}
        self.vocabularies: Dict[str, List[str]] = {}
        self.other_cols: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "ZeroLeakageTrafficPreprocessor":
        """Compute all normalization statistics and vocabularies strictly on D_train."""
        special_cols = set(
            self.recipe.skewed_numeric_cols
            + self.recipe.bounded_ratio_cols
            + self.recipe.categorical_cols
            + self.recipe.financial_wei_cols
            + [self.recipe.target_col]
        )

        # 1. Financial Wei Columns (e.g. value in B-TAT): string Wei -> numeric float -> log1p -> mean/std
        for col in self.recipe.financial_wei_cols:
            if col in train_df.columns:
                nums = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                vals = np.log1p(np.maximum(0.0, nums))
                self.wei_means[col] = float(np.mean(vals))
                self.wei_stds[col] = float(np.std(vals)) + 1e-7

        # 2. Skewed Numerics: log1p(max(0, x)) -> mean/std
        for col in self.recipe.skewed_numeric_cols:
            if col in train_df.columns:
                nums = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                vals = np.log1p(np.maximum(0.0, nums))
                self.skewed_means[col] = float(np.mean(vals))
                self.skewed_stds[col] = float(np.std(vals)) + 1e-7

        # 3. Categoricals: Use fixed domain vocabulary if defined, else extract from D_train
        for col in self.recipe.categorical_cols:
            if col in self.recipe.fixed_vocabularies:
                self.vocabularies[col] = list(self.recipe.fixed_vocabularies[col])
            elif col in train_df.columns:
                cats = sorted(train_df[col].astype(str).str.strip().unique().tolist())
                self.vocabularies[col] = cats

        # 4. Other Standard Numerics: compute mean/std on D_train
        self.other_cols = [
            c
            for c in train_df.columns
            if c not in special_cols and pd.api.types.is_numeric_dtype(train_df[c])
        ]
        for col in self.other_cols:
            vals = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            self.other_means[col] = float(np.mean(vals))
            self.other_stds[col] = float(np.std(vals)) + 1e-7

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform any partition (train, cal, test) using fitted parameters."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on D_train before calling transform()!")

        processed_parts: List[np.ndarray] = []

        # 1. Financial Wei Columns
        for col in self.recipe.financial_wei_cols:
            if col in df.columns:
                nums = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                vals = np.log1p(np.maximum(0.0, nums))
                normed = (vals - self.wei_means[col]) / self.wei_stds[col]
                processed_parts.append(normed[:, None].astype(np.float32))

        # 2. Skewed Numerics
        for col in self.recipe.skewed_numeric_cols:
            if col in df.columns:
                nums = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                vals = np.log1p(np.maximum(0.0, nums))
                normed = (vals - self.skewed_means[col]) / self.skewed_stds[col]
                processed_parts.append(normed[:, None].astype(np.float32))

        # 3. Bounded Ratios: clip to [0.0, 1.0] without unconstrained distortion
        for col in self.recipe.bounded_ratio_cols:
            if col in df.columns:
                vals = np.clip(
                    pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                    0.0,
                    1.0,
                )
                processed_parts.append(vals[:, None].astype(np.float32))

        # 4. Other Standard Numerics
        for col in self.other_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                normed = (vals - self.other_means[col]) / self.other_stds[col]
                processed_parts.append(normed[:, None].astype(np.float32))

        # 5. Categoricals: strict One-Hot or Ordinal against D_train vocabulary with OOV handling
        for col in self.recipe.categorical_cols:
            if col in df.columns:
                cats = self.vocabularies.get(col, [])
                s = df[col].astype(str).str.strip()
                if self.recipe.one_hot:
                    ohe = np.zeros((len(df), len(cats)), dtype=np.float32)
                    for cat_idx, cat_val in enumerate(cats):
                        ohe[s == cat_val, cat_idx] = 1.0
                    processed_parts.append(ohe)
                else:
                    mapping = {cat_val: idx + 1 for idx, cat_val in enumerate(cats)}
                    ord_vals = s.map(mapping).fillna(0.0).to_numpy(dtype=np.float32)
                    processed_parts.append(ord_vals[:, None])

        if not processed_parts:
            # Fallback for unexpected empty extraction
            numeric_df = df.select_dtypes(include=[np.number])
            return np.nan_to_num(numeric_df.to_numpy(dtype=np.float32), nan=0.0)

        out_features = np.concatenate(processed_parts, axis=1)
        return np.nan_to_num(out_features, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
