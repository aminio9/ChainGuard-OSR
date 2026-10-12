import math
import torch
import torch.nn as nn
from src.aggregation.total_variance import GlobalClassProfile, compute_server_side_oas_shrinkage
from src.osr.block_mahalanobis import (
    WireSpeedEdgeDetector,
    compute_block_mahalanobis_distances,
)


class DummyEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(10, latent_dim, bias=False)

    def forward(self, x):
        z = self.linear(x)
        return (None, z, z, None)


def test_multi_prototype_backward_compatibility():
    """Verify that when prototypes is None, distance matches single-centroid exactly."""
    d_z = 64
    num_blocks = 4
    d_b = 16

    mu_0 = torch.zeros(d_z)
    mu_1 = torch.ones(d_z) * 2.0
    covs_0 = [torch.eye(d_b) for _ in range(num_blocks)]
    covs_1 = [torch.eye(d_b) for _ in range(num_blocks)]

    # Single-centroid profile
    prof_single = {
        0: GlobalClassProfile(0, 100, mu_0, covs_0, covs_0, conformal_threshold=10.0),
        1: GlobalClassProfile(1, 100, mu_1, covs_1, covs_1, conformal_threshold=10.0),
    }

    z_test = torch.randn(8, d_z)
    dists_single = compute_block_mahalanobis_distances(
        z_test, prof_single, num_blocks=num_blocks, block_dim=d_b, normalize_features=False
    )

    # Multi-prototype profile with 1 prototype equal to mu
    prof_mp1 = {
        0: GlobalClassProfile(0, 100, mu_0, covs_0, covs_0, conformal_threshold=10.0, prototypes=mu_0.unsqueeze(0)),
        1: GlobalClassProfile(1, 100, mu_1, covs_1, covs_1, conformal_threshold=10.0, prototypes=mu_1.unsqueeze(0)),
    }
    dists_mp1 = compute_block_mahalanobis_distances(
        z_test, prof_mp1, num_blocks=num_blocks, block_dim=d_b, normalize_features=False
    )

    torch.testing.assert_close(dists_single[0], dists_mp1[0])
    torch.testing.assert_close(dists_single[1], dists_mp1[1])


def test_multi_prototype_minimum_distance():
    """Verify that multi-prototype computes exact minimum distance to closest prototype."""
    d_z = 16
    num_blocks = 1
    d_b = 16

    # Create 3 prototypes for class 0: at 0, at +5, at -5
    p0 = torch.zeros(d_z)
    p1 = torch.ones(d_z) * 5.0
    p2 = torch.ones(d_z) * -5.0
    prototypes_0 = torch.stack([p0, p1, p2], dim=0) # (3, 16)

    covs = [torch.eye(d_b)]
    prof = {
        0: GlobalClassProfile(0, 300, p0, covs, covs, conformal_threshold=10.0, prototypes=prototypes_0)
    }

    # Query point at p1
    z_at_p1 = torch.ones(d_z) * 5.0
    dists = compute_block_mahalanobis_distances(
        z_at_p1, prof, num_blocks=num_blocks, block_dim=d_b, normalize_features=False
    )
    # Minimum distance should be 0.0 because it matches p1 exactly
    assert math.isclose(dists[0].item(), 0.0, abs_tol=1e-5)

    # Query point midway between p0 and p1: at 2.5
    z_mid = torch.ones(d_z) * 2.5
    dists_mid = compute_block_mahalanobis_distances(
        z_mid, prof, num_blocks=num_blocks, block_dim=d_b, normalize_features=False
    )
    # Distance to p0: 16 * (2.5^2) = 100.0, distance to p1: 16 * (2.5^2) = 100.0
    assert math.isclose(dists_mid[0].item(), 100.0, abs_tol=1e-3)


def test_wire_speed_edge_detector_with_multi_prototypes():
    """Verify WireSpeedEdgeDetector routes known vs unknown with multi-prototype profiles."""
    d_z = 64
    encoder = DummyEncoder(latent_dim=d_z)

    p0_a = torch.zeros(d_z)
    p0_b = torch.zeros(d_z)
    p0_b[:10] = 3.0
    prototypes_0 = torch.stack([p0_a, p0_b], dim=0)

    p1 = torch.ones(d_z) * 10.0
    covs = [torch.eye(16) for _ in range(4)]

    profiles = {
        0: GlobalClassProfile(0, 200, p0_a, covs, covs, conformal_threshold=5.0, prototypes=prototypes_0),
        1: GlobalClassProfile(1, 100, p1, covs, covs, conformal_threshold=5.0),
    }

    detector = WireSpeedEdgeDetector(
        frozen_encoder=encoder,
        global_profiles=profiles,
        unknown_label=-1,
        num_blocks=4,
        block_dim=16,
        normalize_features=False,
    )

    # Test sample exactly at p0_b -> should be accepted as class 0
    x_known = torch.ones(10) * 3.0
    with torch.no_grad():
        # Inject linear weights as identity-like
        encoder.linear.weight.zero_()
        for i in range(10):
            encoder.linear.weight[i, i] = 1.0

    res = detector.predict_sample(x_known)
    assert res.candidate_class == 0
    assert not res.is_unknown


if __name__ == "__main__":
    test_multi_prototype_backward_compatibility()
    test_multi_prototype_minimum_distance()
    test_wire_speed_edge_detector_with_multi_prototypes()
    print("ALL MULTI-PROTOTYPE TESTS PASSED SUCCESSFULLY!")
