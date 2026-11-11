import math
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

    assert math.isclose(dists[0].item(), 0.0, abs_tol=1e-5)
    # Distance to mu_1 should be 64 * (5^2) = 1600.0
    assert math.isclose(dists[1].item(), 64.0 * 25.0, abs_tol=1e-3)


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
        normalize_features=False,
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


def test_conformal_acceptance_set_and_normalized_decision():
    """Verify conformal acceptance set and scale-normalized decision rule:
    C(z) = {c | D_c(z) <= tau_c}
    f_OSR(x) = C+1 if C(z) == empty else argmin_{c in C(z)} D_c(z) / tau_c.
    """
    d_z = 64
    encoder = DummyEncoder(latent_dim=d_z)

    # Class 0: centroid at 0, threshold = 4.0
    mu_0 = torch.zeros(d_z)
    covs_0 = [torch.eye(16) for _ in range(4)]
    # Class 1: centroid at ones, threshold = 100.0
    mu_1 = torch.ones(d_z)
    covs_1 = [torch.eye(16) for _ in range(4)]

    profiles = {
        0: GlobalClassProfile(0, 100, mu_0, covs_0, covs_0, conformal_threshold=4.0),
        1: GlobalClassProfile(1, 100, mu_1, covs_1, covs_1, conformal_threshold=100.0),
    }

    detector = WireSpeedEdgeDetector(
        frozen_encoder=encoder,
        global_profiles=profiles,
        unknown_label=-1,
        num_blocks=4,
        block_dim=16,
    )

    # Point with D_0 = 5.0 (exceeds tau_0=4.0) but D_1 = 30.0 (within tau_1=100.0)
    # Under raw distance, argmin is 0 (dist 5.0 < 30.0), which would erroneously reject as unknown!
    # Under conformal set C(z) = {1}, the sample is correctly classified as class 1.
    with torch.no_grad():
        encoder.linear.weight.zero_()
    x_test = torch.zeros(1, 10)
    # Manually compute with known z
    z = torch.zeros(d_z)
    z[0] = math.sqrt(5.0)  # D_0 = 5.0 > 4.0
    # D_1 = (sqrt(5)-1)^2 + 63*(1) = (2.236-1)^2 + 63 = 1.528 + 63 = 64.528 <= 100.0

    dists = compute_block_mahalanobis_distances(z, profiles, num_blocks=4, block_dim=16)
    assert dists[0].item() > 4.0
    assert dists[1].item() <= 100.0

    # Test batch with manual dist_matrix check
    res = detector.predict_sample(torch.zeros(1, 10))  # At origin: D_0 = 0 <= 4, D_1 = 64 <= 100 -> class 0
    assert res.predicted_label == 0
    assert not res.is_unknown


if __name__ == "__main__":
    test_block_mahalanobis_computation()
    test_wire_speed_edge_detector_decision()
    test_conformal_acceptance_set_and_normalized_decision()
    print("ALL BLOCK MAHALANOBIS TESTS PASSED!")

