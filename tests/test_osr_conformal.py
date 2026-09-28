"""Tests for Open-Set Recognition and Mondrian Conformal Prediction."""

import numpy as np
import torch
from src.osr.weibull_evt import WeibullEVTFitter, compute_evt_probabilities
from src.osr.energy_ood import compute_free_energy_score
from src.osr.conformal import MondrianConformalCalibrator
from src.osr.edge_inference import EdgeInferenceEngine


def test_free_energy_score():
    # Strong logit -> deep energy basin (negative value)
    logits_known = torch.tensor([[10.0, 1.0, 0.0]])
    energy_known = compute_free_energy_score(logits_known, temperature=1.0)

    # Low uncertain logits -> higher energy
    logits_ood = torch.tensor([[0.1, 0.1, 0.1]])
    energy_ood = compute_free_energy_score(logits_ood, temperature=1.0)

    assert energy_known.item() < energy_ood.item()


def test_weibull_fitter_and_evt_discount():
    fitter = WeibullEVTFitter(tail_size=10)
    features = torch.randn(50, 4)
    labels = torch.zeros(50, dtype=torch.long)
    proto = torch.zeros(4)

    fitter.fit(features, labels, {0: proto})
    assert 0 in fitter.weibull_models

    z = torch.randn(2, 4)
    logits = torch.tensor([[5.0], [5.0]])
    cal_probs = compute_evt_probabilities(z, logits, {0: proto}, fitter.weibull_models)
    assert cal_probs.shape == (2, 2)  # Index 0 unknown, Index 1 known
    assert torch.allclose(cal_probs.sum(dim=-1), torch.ones(2), atol=1e-5)


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


def test_edge_inference_engine_zero_day(decoupled_model):
    calibrator = MondrianConformalCalibrator(alpha=0.05)
    # Calibrate with dummy quantiles: requires probability >= 0.8 to be accepted
    calibrator.quantiles = {0: 0.2, 1: 0.2, 2: 0.2}

    prototypes = {i: torch.randn(decoupled_model.latent_dim) for i in range(3)}
    weibull_models = {i: (1.5, 2.0, 1.0) for i in range(3)}

    engine = EdgeInferenceEngine(
        model=decoupled_model,
        prototypes=prototypes,
        weibull_models=weibull_models,
        conformal_calibrator=calibrator,
        unknown_class_id=-1,
    )

    # Test single flow
    x = torch.randn(decoupled_model.encoder.network[0].in_features)
    pred_cls, c_set, energy, u_prob = engine.predict_flow(x)

    assert isinstance(pred_cls, (int, np.integer))
    assert isinstance(c_set, list)
    assert isinstance(energy, float)
    assert isinstance(u_prob, float)
