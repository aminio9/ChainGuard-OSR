"""Unit tests for Law of Total Variance Exact Moment Aggregation (Theorem 1)."""

import torch

from src.aggregation.total_variance import ClientClassMoment, aggregate_total_variance_moments


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
    profiles = aggregate_total_variance_moments(
        class_submissions={0: submissions},
        num_blocks=num_blocks,
        block_dim=d_b,
        tikhonov_eps=0.0,  # Zero Tikhonov to test exact algebraic equality
    )

    prof = profiles[0]

    # Verify centroid exactness: mu_decentralized == mu_centralized
    assert torch.allclose(prof.mu, centralized_mu, atol=1e-5)

    # Verify Theorem 1 exactness: Sigma_decentralized == Sigma_centralized for all blocks
    for b in range(num_blocks):
        assert torch.allclose(prof.block_covs[b], centralized_block_covs[b], atol=1e-5)

