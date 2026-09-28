"""
Configuration definitions and loaders for the research framework.
Validates experiment, dataset, model, federated, loss, and OSR parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional
import yaml
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    name: str = "federated_osr_experiment"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results/runs"
    checkpoint_dir: str = "checkpoints"
    save_checkpoints: bool = True
    save_frequency: int = 10


class DataConfig(BaseModel):
    dataset_name: str = "synthetic_p2p"
    data_dir: str = "data"
    in_features: int = 43
    num_classes_total: int = 10
    known_classes: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    unknown_classes: List[int] = Field(default_factory=lambda: [7, 8, 9])
    openness: float = 0.1835
    test_split_ratio: float = 0.2
    cal_split_ratio: float = 0.1
    num_samples: Optional[int] = 5000
    target_column: Optional[str] = "label"
    normalize: bool = True
    normalization_type: str = "zscore"
    noise_level: float = 0.05


class PartitionConfig(BaseModel):
    strategy: str = "dirichlet"
    beta_dirichlet: float = 0.1


class FederatedConfig(BaseModel):
    algorithm: str = "primal_dual_consensus"
    num_clients: int = 10
    client_fraction: float = 0.5
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 128
    primal_dual_rho: float = 0.1
    covariance_prototype_aggregation: bool = True
    share_classifier_head: bool = False
    fedprox_mu: float = 0.0
    partition: PartitionConfig = Field(default_factory=PartitionConfig)


class ModelConfig(BaseModel):
    architecture: str = "vib_dual_branch"
    in_features: int = 43
    latent_dim: int = 64
    hidden_dim: int = 128
    density_dim: int = 16
    dropout: float = 0.1
    use_batchnorm: bool = True
    activation: str = "relu"
    stochastic_encoder: bool = True


class OptimizationConfig(BaseModel):
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    lr_decay_step: int = 40
    lr_decay_gamma: float = 0.5


class LossConfig(BaseModel):
    beta_cb: float = 0.999
    focal_gamma: float = 2.0
    lambda_vib: float = 0.01
    lambda_anchor: float = 0.05
    primal_dual_rho: float = 0.1


class OSRConfig(BaseModel):
    method: str = "mondrian_conformal"
    weibull_tail_size: int = 20
    energy_temperature: float = 1.0
    conformal_alpha: float = 0.05
    enable_weibull: bool = True
    enable_energy: bool = True
    enable_conformal: bool = True
    msp_threshold: float = 0.5


class AppConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    osr: OSRConfig = Field(default_factory=OSRConfig)


def _deep_update(base_dict: dict[str, Any], update_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(config_path: str | Path, overrides: Optional[dict[str, Any]] = None) -> AppConfig:
    """Load configuration from a YAML file, merging with default configurations."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    if overrides:
        raw_cfg = _deep_update(raw_cfg, overrides)

    return AppConfig.model_validate(raw_cfg)
