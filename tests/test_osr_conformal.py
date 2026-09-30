"""Tests for Open-Set Recognition and Mondrian Conformal Prediction (LaTeX Section 4.4 & Algorithm 2)."""

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
