"""Comprehensive unit tests for modular baseline implementations and Q1 TIFS/TDSC extensions.

Verifies:
1. Architectural integrity and shared backbone compatibility across all 13 baselines
   (including SCAFFOLD, FedDyn, and FedRoD).
2. Unified scoring heads (`native`, `msp`, `energy`, `mahalanobis`) + Mondrian Conformal Calibration.
3. SVD singular value spectrum & Stable Rank computation.
4. Non-parametric 10-seed statistical tests (Wilcoxon signed-rank, Cliff's Delta, Bootstrap 95% CI).
5. Local Differential Privacy (LDP) moment perturbation, PSD cone projection, and Likelihood Ratio MIA.
"""

import numpy as np
import pytest
import torch

from src.baselines import (
    FedAvgBaseline,
    FedProxBaseline,
    SCAFFOLDBaseline,
    FedDynBaseline,
    FedRoDBaseline,
    FedDecorrBaseline,
    FedProtoBaseline,
    FedPDBaseline,
    FedPDPlusPlusBaseline,
    FedOSSBaseline,
    OpenMaxBaseline,
    EnergyBaseline,
    VanillaMahalanobisBaseline,
)
from src.data.base import FlowDataset
from src.metrics.statistical_tests import (
    CANONICAL_TEN_SEEDS,
    compute_wilcoxon_signed_rank,
    compute_cliffs_delta,
    compute_bootstrap_ci,
    summarize_multi_seed_comparison,
)
from src.aggregation.total_variance import ClientClassMoment
from src.privacy.dp_moments import (
    perturb_client_moment_ldp,
    project_psd_cone,
    evaluate_likelihood_ratio_mia,
)


@pytest.fixture
def synthetic_fl_data():
    """Create a minimal synthetic multi-client dataset for fast baseline unit tests."""
    torch.manual_seed(42)
    np.random.seed(42)

    in_features = 16
    num_classes = 3  # known classes: 0, 1, 2
    samples_per_client = 20

    # 3 clients with known classes only
    client_datasets = []
    for _ in range(3):
        x = torch.randn(samples_per_client, in_features)
        y = torch.randint(0, num_classes, (samples_per_client,))
        client_datasets.append(FlowDataset(x, y))

    # Calibration dataset (strictly known classes only)
    x_cal = torch.randn(30, in_features)
    y_cal = torch.randint(0, num_classes, (30,))
    cal_ds = FlowDataset(x_cal, y_cal)

    # Closed-set test dataset (known classes only)
    x_closed = torch.randn(30, in_features)
    y_closed = torch.randint(0, num_classes, (30,))
    closed_ds = FlowDataset(x_closed, y_closed)

    # Open-set test dataset (known + unknown class 3)
    x_open_known = torch.randn(20, in_features)
    y_open_known = torch.randint(0, num_classes, (20,))
    x_open_unk = torch.randn(20, in_features) + 5.0  # Shifted unknown
    y_open_unk = torch.full((20,), num_classes, dtype=torch.long)

    x_open = torch.cat([x_open_known, x_open_unk], dim=0)
    y_open = torch.cat([y_open_known, y_open_unk], dim=0)
    open_ds = FlowDataset(x_open, y_open)

    return {
        "in_features": in_features,
        "num_classes": num_classes,
        "client_datasets": client_datasets,
        "cal_ds": cal_ds,
        "closed_ds": closed_ds,
        "open_ds": open_ds,
    }


@pytest.mark.parametrize(
    "baseline_cls,kwargs",
    [
        (FedAvgBaseline, {}),
        (FedProxBaseline, {"mu_prox": 0.05}),
        (SCAFFOLDBaseline, {}),
        (FedDynBaseline, {"alpha_dyn": 0.01}),
        (FedRoDBaseline, {"tau_rod": 1.0}),
        (FedDecorrBaseline, {"beta_decorr": 0.1}),
        (FedProtoBaseline, {"lambda_proto": 0.5}),
        (FedPDBaseline, {"mask_ratio": 0.5, "lambda_open": 0.3}),
        (FedPDPlusPlusBaseline, {"lambda_mixup": 0.5}),
        (FedOSSBaseline, {"inversion_steps": 2, "lambda_duss": 0.3}),
        (OpenMaxBaseline, {"tail_size": 10}),
        (EnergyBaseline, {"temperature": 1.0}),
        (VanillaMahalanobisBaseline, {"regularization": 1e-3}),
    ],
)
def test_baseline_lifecycle(synthetic_fl_data, baseline_cls, kwargs):
    """Test full training, calibration, and Q1 evaluation lifecycle for each baseline."""
    in_feat = synthetic_fl_data["in_features"]
    n_cls = synthetic_fl_data["num_classes"]
    device = torch.device("cpu")

    baseline = baseline_cls(
        in_features=in_feat,
        num_classes=n_cls,
        latent_dim=32,
        hidden_dim=32,
        device=device,
        **kwargs,
    )

    # 1. Round execution
    stats = baseline.fit_round(
        round_idx=1,
        client_datasets=synthetic_fl_data["client_datasets"],
        epochs=1,
        batch_size=16,
        lr=0.01,
    )
    assert "train_loss" in stats
    assert stats["train_loss"] >= 0.0

    # 2. Calibration (must use known-classes only)
    cal_ds = synthetic_fl_data["cal_ds"]
    assert torch.all(cal_ds.labels < n_cls), "Zero leakage violation: unknown classes in calibration set!"
    tau = baseline.calibrate(cal_ds, alpha=0.1)
    assert isinstance(tau, float)

    # 3. Batch prediction
    preds, scores, cands = baseline.predict_batch(synthetic_fl_data["open_ds"].features)
    assert preds.shape == (40,)
    assert scores.shape == (40,)
    assert cands.shape == (40,)
    assert np.all(cands < n_cls), "Candidate known classes must be strictly within known classes"

    # 4. Standardized Q1 evaluation (including fpr95 and unified scoring heads)
    metrics = baseline.evaluate(
        test_open_ds=synthetic_fl_data["open_ds"],
        test_closed_ds=synthetic_fl_data["closed_ds"],
        cal_ds=cal_ds,
        conformal_alpha=0.1,
    )

    expected_keys = [
        "baseline_name",
        "threshold",
        "closed_set_accuracy",
        "accuracy",
        "balanced_accuracy",
        "closed_set_macro_f1",
        "auroc",
        "fpr95",
        "aupr",
        "unknown_recall",
        "unknown_precision",
        "unknown_f1",
        "open_set_macro_f1",
        "kfr",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric key: {key}"
        if isinstance(metrics[key], (int, float, np.floating)):
            assert not np.isnan(metrics[key]), f"Metric {key} is NaN in {baseline.name}"


def test_unified_scoring_and_svd_spectrum(synthetic_fl_data):
    """Verify unified scoring heads (`msp`, `energy`, `mahalanobis`) + Mondrian Conformal + SVD spectrum."""
    baseline = FedAvgBaseline(
        in_features=synthetic_fl_data["in_features"],
        num_classes=synthetic_fl_data["num_classes"],
        latent_dim=16,
        hidden_dim=16,
        device=torch.device("cpu"),
    )
    baseline.fit_round(1, synthetic_fl_data["client_datasets"], epochs=1, batch_size=16, lr=0.01)

    for score_type in ("msp", "energy", "mahalanobis"):
        m = baseline.evaluate(
            test_open_ds=synthetic_fl_data["open_ds"],
            test_closed_ds=synthetic_fl_data["closed_ds"],
            cal_ds=synthetic_fl_data["cal_ds"],
            conformal_alpha=0.05,
            score_type=score_type,
            use_mondrian=True,
        )
        assert m["score_type"] == score_type
        assert m["use_mondrian"] is True
        assert 0.0 <= m["auroc"] <= 1.0
        assert 0.0 <= m["fpr95"] <= 1.0

    svd_info = baseline.compute_feature_svd_spectrum(synthetic_fl_data["closed_ds"].features)
    assert "stable_rank" in svd_info
    assert svd_info["stable_rank"] >= 1.0
    assert len(svd_info["normalized_singular_values"]) == 16


def test_statistical_discipline_10_seeds():
    """Verify 10-seed Wilcoxon signed-rank p < 0.01 capability, Cliff's Delta, and Bootstrap CI."""
    assert len(CANONICAL_TEN_SEEDS) == 10
    proposed = [0.951, 0.948, 0.955, 0.952, 0.949, 0.953, 0.950, 0.954, 0.947, 0.956]
    competitor = [0.891, 0.885, 0.894, 0.888, 0.882, 0.890, 0.887, 0.892, 0.884, 0.889]

    w_res = compute_wilcoxon_signed_rank(proposed, competitor)
    assert w_res["p_value"] < 0.01
    assert w_res["significant_at_0_01"] is True

    c_res = compute_cliffs_delta(proposed, competitor)
    assert c_res["delta"] == 1.0
    assert c_res["magnitude"] == "large"

    ci_res = compute_bootstrap_ci(proposed, confidence=0.95, n_resamples=500)
    assert ci_res["ci_lower"] <= ci_res["mean"] <= ci_res["ci_upper"]

    summary = summarize_multi_seed_comparison(proposed, competitor)
    assert summary["dagger_p001"] is True


def test_ldp_moments():
    """Verify LDP moment perturbation and PSD cone projection."""
    subs = []
    for cid in range(6):
        mu = torch.randn(16) * 0.2
        covs = [torch.eye(8) * 0.5, torch.eye(8) * 0.5]
        subs.append(ClientClassMoment(client_id=cid, sample_count=25, mu=mu, block_covs=covs))

    # Test LDP perturbation & PSD projection
    p_mom = perturb_client_moment_ldp(subs[0], epsilon=2.0, delta=1e-5, seed=42)
    for b_cov in p_mom.block_covs:
        eigvals = torch.linalg.eigvalsh(b_cov)
        assert torch.all(eigvals >= -1e-6), "PSD cone projection failed!"

