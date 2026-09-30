"""Shared fixtures for unit and integration tests."""

import pytest
import torch
from src.config import AppConfig
from src.data.base import FlowDataset
from src.data.synthetic import generate_synthetic_p2p_data
from src.models.vib_network import VIBClassifierNetwork


@pytest.fixture
def sample_config() -> AppConfig:
    """Fixture providing a lightweight AppConfig for rapid testing."""
    cfg = AppConfig()
    cfg.federated.num_clients = 3
    cfg.federated.client_fraction = 1.0
    cfg.federated.num_rounds = 2
    cfg.federated.local_epochs = 1
    cfg.federated.batch_size = 16
    cfg.data.num_samples = 300
    cfg.data.known_classes = [0, 1, 2]
    cfg.data.unknown_classes = [3, 4]
    cfg.data.in_features = 12
    cfg.model.in_features = 12
    cfg.model.latent_dim = 16
    cfg.model.hidden_dim = 32
    cfg.experiment.seed = 42
    return cfg


@pytest.fixture
def synthetic_dataset(sample_config) -> FlowDataset:
    """Fixture providing synthetic flow dataset."""
    return generate_synthetic_p2p_data(
        num_samples=sample_config.data.num_samples,
        in_features=sample_config.data.in_features,
        num_classes=5,
        seed=42,
    )


@pytest.fixture
def decoupled_model(sample_config) -> VIBClassifierNetwork:
    """Fixture providing a miniature canonical VIB network."""
    return VIBClassifierNetwork(
        in_features=sample_config.model.in_features,
        latent_dim=sample_config.model.latent_dim,
        num_classes=len(sample_config.data.known_classes),
        hidden_dim=sample_config.model.hidden_dim,
    )
