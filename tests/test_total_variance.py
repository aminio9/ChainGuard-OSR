"""Unit tests for Sample-Weighted MAD Byzantine Filter and Law of Total Variance (Theorem 1, Equations 14-18 & LaTeX Eq. 151-167)."""

import pytest
import torch

from src.aggregation.byzantine_mad import ClientClassMoment, filter_byzantine_moments
from src.aggregation.total_variance import aggregate_total_variance_moments


def test_byzantine_mad_rejection():
    # 5 honest clients with close means, 1 malicious client with an extreme poison attack
    d_z = 64
    d_b = 16
    honest_mean = torch.zeros(d_z)
    dummy_covs = [torch.eye(d_b) for _ in range(4)]

    submissions = [
        ClientClassMoment(client_id=0, sample_count=100, mu=honest_mean + 0.05 * torch.randn(d_z), block_covs=dummy_covs),
        ClientClassMoment(client_id=1, sample_count=80, mu=honest_mean + 0.04 * torch.randn(d_z), block_covs=dummy_covs),
        ClientClassMoment(client_id=2, sample_count=120, mu=honest_mean + 0.06 * torch.randn(d_z), block_covs=dummy_covs),
        ClientClassMoment(client_id=3, sample_count=90, mu=honest_mean + 0.03 * torch.randn(d_z), block_covs=dummy_covs),
        ClientClassMoment(client_id=4, sample_count=110, mu=honest_mean + 0.05 * torch.randn(d_z), block_covs=dummy_covs),
        # Poisoned attacker injecting massive shift: ||mu|| -> 100
        ClientClassMoment(client_id=99, sample_count=100, mu=honest_mean + 100.0, block_covs=dummy_covs),
    ]

    clean, rejected = filter_byzantine_moments(submissions, gamma_0=3.0, zeta=1.5)

    assert 99 in rejected
    assert len(clean) == 5
    for c in clean:
        assert c.client_id != 99


def test_byzantine_variance_explosion_rejection():
    """Verify Theorem 1 Part 2 & Eq. 153: Arbitrary covariance trace explosion triggers Byzantine rejection."""
    d_z = 64
    d_b = 16
    honest_mean = torch.zeros(d_z)
    dummy_covs = [torch.eye(d_b) for _ in range(4)]
    # Exploded covariance attacker: Tr(Sigma) -> 100000
    exploded_covs = [torch.eye(d_b) * 10000.0 for _ in range(4)]

    submissions = [
        ClientClassMoment(client_id=0, sample_count=100, mu=honest_mean + 0.01 * torch.randn(d_z), block_covs=dummy_covs),
        ClientClassMoment(client_id=1, sample_count=80, mu=honest_mean + 0.01 * torch.randn(d_z), block_covs=dummy_covs),
        ClientClassMoment(client_id=2, sample_count=120, mu=honest_mean + 0.01 * torch.randn(d_z), block_covs=dummy_covs),
        ClientClassMoment(client_id=3, sample_count=90, mu=honest_mean + 0.01 * torch.randn(d_z), block_covs=dummy_covs),
        # Attacker transmits normal centroid but exploded variance to blind OSR
        ClientClassMoment(client_id=88, sample_count=100, mu=honest_mean, block_covs=exploded_covs),
    ]

    clean, rejected = filter_byzantine_moments(submissions, gamma_0=3.0, zeta=1.5)

    assert 88 in rejected
    assert len(clean) == 4
    for c in clean:
        assert c.client_id != 88


def test_law_of_total_variance_exactness():
    """Verify Theorem 1: Decentralized pooled block covariance algebraically matches centralized covariance."""
    d_z = 64
    num_blocks = 4
    d_b = 16

    # Generate partitioned data across 3 clients
    torch.manual_seed(42)
    x1 = torch.randn(100, d_z) + 1.0
    x2 = torch.randn(150, d_z) + 2.0
    x3 = torch.randn(200, d_z) + 3.0

    all_x = torch.cat([x1, x2, x3], dim=0)
    total_n = len(all_x)

    # 1. Centralized calculation for sub-blocks
    centralized_mu = torch.mean(all_x, dim=0)
    centralized_block_covs = []
    for b in range(num_blocks):
        start = b * d_b
        end = start + d_b
        x_b = all_x[:, start:end]
        diff_b = x_b - centralized_mu[start:end]
        # Formula uses 1/N divisor (MLE covariance):
        cov_b = torch.matmul(diff_b.t(), diff_b) / float(total_n)
        centralized_block_covs.append(cov_b)

    # 2. Local client moments
    submissions = []
    for cid, x_data in enumerate([x1, x2, x3]):
        n_k = len(x_data)
        mu_k = torch.mean(x_data, dim=0)
        covs_k = []
        for b in range(num_blocks):
            start = b * d_b
            end = start + d_b
            x_b = x_data[:, start:end]
            diff_b = x_b - mu_k[start:end]
            cov_k_b = torch.matmul(diff_b.t(), diff_b) / float(n_k)
            covs_k.append(cov_k_b)

        submissions.append(
            ClientClassMoment(client_id=cid, sample_count=n_k, mu=mu_k, block_covs=covs_k)
        )

    # 3. Decentralized aggregation via Law of Total Variance
    profiles, _ = aggregate_total_variance_moments(
        class_submissions={0: submissions},
        num_blocks=num_blocks,
        block_dim=d_b,
        tikhonov_eps=0.0,  # Zero Tikhonov to test exact algebraic equality
        apply_mad_filter=False,
    )

    prof = profiles[0]

    # Verify centroid exactness: mu_decentralized == mu_centralized
    assert torch.allclose(prof.mu, centralized_mu, atol=1e-5)

    # Verify Theorem 1 exactness: Sigma_decentralized == Sigma_centralized for all blocks
    for b in range(num_blocks):
        assert torch.allclose(prof.block_covs[b], centralized_block_covs[b], atol=1e-5)
