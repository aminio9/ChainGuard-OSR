"""Client partition sweeps and non-IID Dirichlet distribution generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_dirichlet_partition(
    client_count: int = 10,
    groups: list[str] | None = None,
    alpha: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate client class distributions under a Dirichlet distribution."""
    groups = groups or ["DoS", "DDoS", "PortScan", "Botnet", "Benign"]
    rng = np.random.default_rng(seed)
    rows = []
    for client in range(1, client_count + 1):
        values = rng.dirichlet(np.full(len(groups), alpha))
        for group, val in zip(groups, values):
            rows.append(
                {
                    "client": f"Client {client}",
                    "client_id": client,
                    "traffic_group": group,
                    "proportion": float(val),
                }
            )
    return pd.DataFrame(rows)


def build_partition_sweep_dataset(
    output_path: Path,
    client_count: int = 10,
    groups: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build comparison dataset across IID and Dirichlet alphas."""
    groups = groups or ["Benign", "DDoS", "PortScan", "Botnet", "WebAttack"]
    settings = [
        ("iid", 1, "IID"),
        ("alpha_1_0", 2, "Dirichlet \u03b1 = 1.0"),
        ("alpha_0_5", 3, "Dirichlet \u03b1 = 0.5"),
        ("alpha_0_1", 4, "Dirichlet \u03b1 = 0.1"),
    ]
    rows = []
    for setting, order, title in settings:
        if setting == "iid":
            for c in range(1, client_count + 1):
                for g in groups:
                    rows.append({
                        "setting": setting,
                        "setting_order": order,
                        "panel_title": title,
                        "client_id": c,
                        "traffic_group": g,
                        "proportion": 1.0 / len(groups),
                    })
        else:
            alpha = float(setting.replace("alpha_", "").replace("_", "."))
            df = generate_dirichlet_partition(client_count, groups, alpha, seed)
            for _, r in df.iterrows():
                rows.append({
                    "setting": setting,
                    "setting_order": order,
                    "panel_title": title,
                    "client_id": r["client_id"],
                    "traffic_group": r["traffic_group"],
                    "proportion": r["proportion"],
                })
    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return out_df
