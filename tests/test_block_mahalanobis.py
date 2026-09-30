"""Unit tests for Linear-Time Block-Diagonal Mahalanobis Edge Detector (Algorithm 2)."""

import pytest
import torch
import torch.nn as nn

from src.aggregation.total_variance import GlobalClassProfile
from src.osr.block_mahalanobis import (
    WireSpeedEdgeDetector,
    compute_block_mahalanobis_distances,
)


class DummyEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        # Linear projection
        self.linear = nn.Linear(10, latent_dim, bias=False)

    def forward(self, x):
        z = self.linear(x)
        # Return tuple matching VIBClassifierNetwork: (logits, z, mu, logvar)
        return (None, z, z, None)


def test_block_mahalanobis_computation():
    d_z = 64
    num_blocks = 4
    d_b = 16

    # Setup dummy profiles for 2 classes
    mu_0 = torch.zeros(d_z)
    mu_1 = torch.ones(d_z) * 5.0

    covs_0 = [torch.eye(d_b) for _ in range(num_blocks)]
    covs_1 = [torch.eye(d_b) for _ in range(num_blocks)]

    profiles = {
        0: GlobalClassProfile(0, 100, mu_0, covs_0, covs_0, conformal_threshold=50.0),
        1: GlobalClassProfile(1, 100, mu_1, covs_1, covs_1, conformal_threshold=50.0),
    }

    # Query point at mu_0
    z_at_0 = torch.zeros(d_z)
    dists = compute_block_mahalanobis_distances(z_at_0, profiles, num_blocks=num_blocks, block_dim=d_b)

    assert dists[0].item() == pytest.approx(0.0, abs=1e-5)
    # Distance to mu_1 should be 64 * (5^2) = 1600.0
    assert dists[1].item() == pytest.approx(64.0 * 25.0, abs=1e-3)


def test_wire_speed_edge_detector_decision():
    d_z = 64
    encoder = DummyEncoder(latent_dim=d_z)

    mu_0 = torch.zeros(d_z)
    covs_0 = [torch.eye(16) for _ in range(4)]
    profiles = {
        0: GlobalClassProfile(0, 100, mu_0, covs_0, covs_0, conformal_threshold=20.0),
    }

    detector = WireSpeedEdgeDetector(
        frozen_encoder=encoder,
        global_profiles=profiles,
        unknown_label=-1,
        num_blocks=4,
        block_dim=16,
    )

    # 1. Close input: should classify as class 0
    x_close = torch.zeros(1, 10)
    res_close = detector.predict_sample(x_close)
    assert res_close.predicted_label == 0
    assert not res_close.is_unknown

    # 2. Distant input with large norm: should trigger Unknown (Zero-Day Exploit Intercepted)
    x_far = torch.ones(1, 10) * 100.0
    res_far = detector.predict_sample(x_far)
    assert res_far.predicted_label == -1
    assert res_far.is_unknown
