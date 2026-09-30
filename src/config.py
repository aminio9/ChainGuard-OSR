"""
Configuration definitions and loaders for the research framework.
Validates experiment, dataset, model, federated, loss, and OSR parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Union
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
    algorithm: str = "fedavg"
    num_clients: int = 10
    client_fraction: float = 1.0
    num_rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 64
    primal_dual_rho: float = 0.0
    covariance_prototype_aggregation: bool = False
    share_classifier_head: bool = True
    fedprox_mu: float = 0.0
    num_blocks: int = 4
    block_dim: int = 16
    min_cov_samples: int = 3
    tikhonov_eps: float = 1e-4
    aggregation_strategy: str = "decoupled"
    partition: PartitionConfig = Field(default_factory=PartitionConfig)


class ModelConfig(BaseModel):
    architecture: str = "vib_network"
    in_features: int = 34
    latent_dim: int = 64
    hidden_dim: Union[int, List[int]] = Field(default_factory=lambda: [512, 256, 128])
    density_dim: int = 0
    dropout: float = 0.1
    use_batchnorm: bool = False
    use_layernorm: bool = True
    activation: str = "gelu"
    stochastic_encoder: bool = True


class OptimizationConfig(BaseModel):
    optimizer: str = "adam"
    learning_rate: float = 0.001
    eta_min: float = 1e-6
    weight_decay: float = 0.0001
    clip_grad_norm: float = 1.0
    lr_decay_step: int = 40
    lr_decay_gamma: float = 0.5


class LossConfig(BaseModel):
    tau_bla: float = 1.0
    gamma_bla: Optional[float] = None
    laplace_eps: float = 0.1
    lambda_vib: float = 0.01
    beta_cb: float = 0.999
    focal_gamma: float = 2.0
    lambda_anchor: float = 0.0
    primal_dual_rho: float = 0.0
    fedprox_mu: float = 0.0
    tau_g: float = 2.0
    max_lambda_anchor: float = 1.0
    anchor_kappa: float = 10.0
    rho_warm: float = 0.15
    anchor_t0: float = 0.15


class OSRConfig(BaseModel):
    method: str = "mondrian_conformal"
    weibull_tail_size: int = 20
    energy_temperature: float = 1.0
    conformal_alpha: float = 0.05
    enable_weibull: bool = True
    enable_energy: bool = True
    enable_conformal: bool = True
    enable_block_mahalanobis: bool = True
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

    # Determine project root and default config path
    root_dir = Path(__file__).resolve().parent.parent
    default_cfg_path = root_dir / "configs" / "default.yaml"

    base_dict: dict[str, Any] = {}
    if default_cfg_path.is_file() and path.resolve() != default_cfg_path.resolve():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            base_dict = yaml.safe_load(f) or {}

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    # If the YAML file encapsulates its modifications under an 'overrides' block (common in ablation configs)
    if "overrides" in raw_cfg and isinstance(raw_cfg["overrides"], dict):
        file_overrides = raw_cfg.pop("overrides")
        raw_cfg = _deep_update(raw_cfg, file_overrides)

    # Resolve modular config pointers if present
    sub_config_keys = {
        "federated_config": "federated",
        "model_config": "model",
        "data_config": "data",
        "dataset_config": "data",
        "loss_config": "loss",
        "osr_config": "osr",
    }
    for pointer_key, target_section in sub_config_keys.items():
        if pointer_key in raw_cfg:
            sub_path = Path(raw_cfg[pointer_key])
            if not sub_path.is_absolute():
                sub_path = root_dir / sub_path
            if sub_path.is_file():
                with open(sub_path, "r", encoding="utf-8") as sf:
                    sub_dict = yaml.safe_load(sf) or {}
                raw_cfg.setdefault(target_section, {})
                raw_cfg[target_section] = _deep_update(sub_dict, raw_cfg.get(target_section, {}))

    # Synchronize fedprox_mu if algorithm is fedprox
    if raw_cfg.get("federated", {}).get("algorithm") == "fedprox":
        mu = raw_cfg["federated"].get("fedprox_mu", 0.01)
        raw_cfg.setdefault("loss", {})
        raw_cfg["loss"]["fedprox_mu"] = mu

    # Merge into base_dict
    final_dict = _deep_update(base_dict, raw_cfg)

    if overrides:
        final_dict = _deep_update(final_dict, overrides)

    return AppConfig.model_validate(final_dict)
