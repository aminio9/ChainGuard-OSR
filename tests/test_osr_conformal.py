"""Tests for Open-Set Recognition and Mondrian Conformal Prediction (LaTeX Section 4.4 & Algorithm 2)."""

import math
import numpy as np
import torch
from src.osr.weibull_evt import WeibullEVTFitter, compute_evt_probabilities
from src.osr.energy_ood import compute_free_energy_score
from src.osr.conformal import MondrianConformalCalibrator
from src.osr.block_mahalanobis import WireSpeedEdgeDetector
from src.aggregation.total_variance import GlobalClassProfile


def test_free_energy_score():
    # Strong logit -> deep energy basin (negative value)
    logits_known = torch.tensor([[10.0, 1.0, 0.0]])
    energy_known = compute_free_energy_score(logits_known, temperature=1.0)

    # Low uncertain logits -> higher energy
    logits_ood = torch.tensor([[0.1, 0.1, 0.1]])
    energy_ood = compute_free_energy_score(logits_ood, temperature=1.0)

    assert energy_known.item() < energy_ood.item()


def test_mondrian_conformal_guarantee():
    calibrator = MondrianConformalCalibrator(alpha=0.1)  # 90% guarantee
    # 100 well-calibrated samples
    probs = np.zeros((100, 2))
    probs[:50, 0] = 0.95
    probs[:50, 1] = 0.05
    probs[50:, 0] = 0.05
    probs[50:, 1] = 0.95
    labels = np.array([0] * 50 + [1] * 50)

    quantiles = calibrator.calibrate(probs, labels, num_classes=2)
    assert 0 in quantiles and 1 in quantiles

    # Predict on test
    test_probs = np.array([[0.95, 0.05], [0.05, 0.95], [0.01, 0.01]])
    sets = calibrator.predict_conformal_sets(test_probs)

    assert 0 in sets[0]
    assert 1 in sets[1]
    # Far-away OOD point produces empty set
    assert len(sets[2]) == 0


def test_wire_speed_edge_detector_zero_day(decoupled_model):
    latent_dim = decoupled_model.latent_dim
    num_blocks = 4
    block_dim = latent_dim // num_blocks

    profiles = {
        0: GlobalClassProfile(
            class_id=0,
            total_samples=100,
            mu=torch.zeros(latent_dim),
            block_covs=[torch.eye(block_dim) for _ in range(num_blocks)],
            inv_block_covs=[torch.eye(block_dim) for _ in range(num_blocks)],
            conformal_threshold=10.0,
        )
    }

    detector = WireSpeedEdgeDetector(
        frozen_encoder=decoupled_model.encoder,
        global_profiles=profiles,
        unknown_label=-1,
        num_blocks=num_blocks,
        block_dim=block_dim,
    )

    # In-distribution sample
    x_in = torch.randn(decoupled_model.in_features) * 0.1
    res_in = detector.predict_sample(x_in)
    assert res_in.candidate_class == 0

    # Batch prediction
    x_batch = torch.randn(5, decoupled_model.in_features)
    preds, is_unk, cands, scores = detector.predict_batch(x_batch)
    assert len(preds) == 5
    assert len(is_unk) == 5


def test_distributed_conformal_overflow_bin_and_conservative_quantile():
    """Verify LaTeX Eqs. 4.7-4.8:
    Histogram binning with overflow bin (B_c, +inf) and finite-sample quantile aggregation.
    """
    from src.osr.conformal import compute_right_closed_histogram, aggregate_distributed_cdfs
    import numpy as np

    b_c = 50.0
    num_bins = 50
    bin_edges = np.linspace(0.0, b_c, num_bins + 1)

    # Client 1: 95 normal scores in [0, 40] and 5 extreme outliers > 50 (in overflow bin)
    scores_k1 = np.concatenate([np.linspace(1.0, 40.0, 95), np.array([55.0, 60.0, 75.0, 80.0, 100.0])])
    counts_k1, edges_k1 = compute_right_closed_histogram(scores_k1, bin_edges=bin_edges, include_overflow=True)
    assert len(counts_k1) == num_bins + 1
    assert counts_k1[-1] == 5
    assert np.isinf(edges_k1[-1])

    # Client 2: 100 normal scores in [5, 45] and 0 outliers
    scores_k2 = np.linspace(5.0, 45.0, 100)
    counts_k2, edges_k2 = compute_right_closed_histogram(scores_k2, bin_edges=bin_edges, include_overflow=True)
    assert counts_k2[-1] == 0

    # Aggregate histograms across clients
    alpha = 0.05
    tau_agg = aggregate_distributed_cdfs([(counts_k1, edges_k1), (counts_k2, edges_k2)], alpha=alpha)
    assert math.isfinite(tau_agg)
    assert 40.0 <= tau_agg <= 50.0

    # Test edge case: insufficient samples where r_c > N_c returns +inf
    tau_inf = aggregate_distributed_cdfs([(np.array([1, 0]), np.array([0.0, 10.0, np.inf]))], alpha=0.01)
    assert np.isinf(tau_inf)
