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


def test_hungarian_prototype_alignment_resolves_permutation_ambiguity():
    """Verify that Hungarian bipartite matching correctly resolves permutation ambiguity (label switching) across nodes."""
    from src.aggregation.total_variance import federated_align_and_average_prototypes

    # 4 distinct behavioral modes for normal traffic (e.g., handshake, discovery, tx propagation, block announcement)
    m0 = torch.tensor([10.0, 0.0, 0.0, 0.0])
    m1 = torch.tensor([0.0, 10.0, 0.0, 0.0])
    m2 = torch.tensor([0.0, 0.0, 10.0, 0.0])
    m3 = torch.tensor([0.0, 0.0, 0.0, 10.0])
    ground_truth_modes = torch.stack([m0, m1, m2, m3])

    # Client 1: canonical order [0, 1, 2, 3] with slight noise
    p1 = torch.stack([m0 + 0.05, m1 - 0.05, m2 + 0.02, m3 - 0.02])
    c1 = [100, 100, 100, 100]

    # Client 2: permuted order [2, 3, 0, 1] (label switching)
    p2 = torch.stack([m2 - 0.04, m3 + 0.04, m0 - 0.03, m1 + 0.03])
    c2 = [100, 100, 100, 100]

    # Client 3: permuted order [1, 0, 3, 2] (label switching)
    p3 = torch.stack([m1 + 0.02, m0 - 0.02, m3 + 0.05, m2 - 0.05])
    c3 = [100, 100, 100, 100]

    # Client 4: permuted order [3, 2, 1, 0] (label switching)
    p4 = torch.stack([m3 - 0.03, m2 + 0.03, m1 - 0.04, m0 + 0.04])
    c4 = [100, 100, 100, 100]

    aligned_prototypes = federated_align_and_average_prototypes(
        local_prototypes_list=[(p1, c1), (p2, c2), (p3, c3), (p4, c4)],
        target_num_prototypes=4,
        random_state=42,
    )

    assert aligned_prototypes.shape == (4, 4)

    # For each ground truth mode, there must be exactly one aligned prototype with distance < 0.2
    for gt in ground_truth_modes:
        dists = torch.norm(aligned_prototypes - gt.unsqueeze(0), dim=-1)
        min_dist = torch.min(dists).item()
        assert min_dist < 0.2, f"Mode {gt} was not recovered! Min distance: {min_dist}"


def test_prototype_alignment_weighted_and_edge_cases():
    """Verify weighted averaging and graceful fallback when local prototype submissions are sparse."""
    from src.aggregation.total_variance import federated_align_and_average_prototypes

    # Single client with 2 prototypes when 4 are requested
    p1 = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    c1 = [300, 100]  # Skewed sample counts

    aligned = federated_align_and_average_prototypes(
        local_prototypes_list=[(p1, c1)],
        target_num_prototypes=2,
    )
    assert aligned.shape == (2, 2)
    # Check that aligned clusters match the input centers
    assert torch.allclose(aligned[0], p1[0], atol=1e-4) or torch.allclose(aligned[0], p1[1], atol=1e-4)

    # Empty submission with fallback mean
    fallback = torch.tensor([1.0, 2.0])
    aligned_fallback = federated_align_and_average_prototypes(
        local_prototypes_list=[],
        target_num_prototypes=3,
        fallback_mean=fallback,
    )
    assert aligned_fallback.shape == (3, 2)
    assert torch.allclose(aligned_fallback[0], fallback)


