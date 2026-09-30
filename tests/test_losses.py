"""Tests for canonical multi-objective loss functions (LaTeX Equations 23-87)."""

import torch
from src.losses.class_balanced import class_balanced_focal_loss, compute_class_balanced_weights
from src.losses.vib_loss import vib_kl_divergence_loss
from src.losses.anchor_loss import DynamicGlobalAnchorLoss
from src.losses.composite_loss import CompositeLossCalculator
from src.config import LossConfig


def test_class_balanced_focal_loss():
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))
    samples_per_cls = [1000, 50, 10]  # Imbalanced

    loss = class_balanced_focal_loss(logits, targets, samples_per_class=samples_per_cls, beta=0.999)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    # Rare class should have higher weight
    weights = compute_class_balanced_weights(samples_per_cls, beta=0.999)
    assert weights[2] > weights[0]


def test_vib_kl_divergence_loss():
    # If mu=0 and logvar=0, KL divergence must be exactly 0
    mu_zero = torch.zeros(4, 8)
    logvar_zero = torch.zeros(4, 8)
    kl_zero = vib_kl_divergence_loss(mu_zero, logvar_zero)
    assert abs(kl_zero.item()) < 1e-6

    # Non-zero mu or variance should yield positive KL
    mu_pos = torch.randn(4, 8)
    logvar_pos = torch.randn(4, 8)
    kl_pos = vib_kl_divergence_loss(mu_pos, logvar_pos)
    assert kl_pos.item() > 0.0


def test_dynamic_global_anchor_loss():
    anchor_loss_fn = DynamicGlobalAnchorLoss(tau_g=2.0, max_lambda=1.0)
    student_logits = torch.randn(8, 4)
    teacher_logits = torch.randn(8, 4)

    loss = anchor_loss_fn(student_logits, teacher_logits, t_ratio=0.5)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


from src.losses.logit_adjusted import (
    LogitAdjustedBalancedLoss,
    logit_adjusted_balanced_loss,
    compute_laplace_class_priors,
)


def test_logit_adjusted_balanced_loss():
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))
    samples_per_cls = [1000, 50, 0]  # Severely Imbalanced with missing class

    priors = compute_laplace_class_priors(samples_per_cls, num_classes=3, epsilon=0.1)
    assert priors.size(0) == 3
    assert abs(priors.sum().item() - 1.0) < 1e-5
    assert priors[0] > priors[1] > priors[2]

    loss = logit_adjusted_balanced_loss(logits, targets, class_priors=priors, tau=1.0)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_composite_loss_calculator(decoupled_model):
    cfg = LossConfig(tau_bla=1.0, lambda_vib=0.01)
    calc = CompositeLossCalculator(cfg)

    x = torch.randn(4, decoupled_model.in_features)
    y = torch.tensor([0, 1, 2, 0])
    logits, z, mu, logvar = decoupled_model(x)

    total_loss, breakdown = calc(
        logits=logits,
        targets=y,
        mu=mu,
        logvar=logvar,
        samples_per_class=[100, 20, 5],
    )

    assert total_loss.item() > 0.0
    assert "loss_total" in breakdown
    assert "loss_bla" in breakdown
    assert "loss_vib" in breakdown

