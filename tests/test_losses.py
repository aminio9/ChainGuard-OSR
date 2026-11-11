"""Unit tests for local representation learning loss functions."""

import torch
from src.losses.vib_loss import vib_kl_divergence_loss
from src.losses.logit_adjusted import (
    LogitAdjustedBalancedLoss,
    logit_adjusted_balanced_loss,
    compute_laplace_class_priors,
    compute_local_class_frequencies,
)
from src.losses.composite_loss import CompositeLossCalculator
from src.config import LossConfig


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


def test_compute_local_class_frequencies():
    counts = [100, 50, 0, 0]
    freqs = compute_local_class_frequencies(counts, num_classes=4)
    assert freqs.size(0) == 4
    assert abs(freqs[:2].sum().item() - 1.0) < 1e-6
    assert abs(freqs[0].item() - 100.0 / 150.0) < 1e-6
    assert abs(freqs[1].item() - 50.0 / 150.0) < 1e-6
    assert freqs[2].item() == 0.0
    assert freqs[3].item() == 0.0


def test_bla_loss_absent_class_masking():
    """Verify absent classes are masked from denominator sum and receive zero gradient."""
    logits = torch.randn(6, 4, requires_grad=True)
    # Target only in present classes {0, 1}
    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    # Class 2 and 3 are absent locally
    counts = [100, 50, 0, 0]
    freqs = compute_local_class_frequencies(counts, num_classes=4)

    loss_fn = LogitAdjustedBalancedLoss(tau=1.0)
    loss = loss_fn(logits, targets, freqs)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    # Gradients for absent classes 2 and 3 must be strictly zero (no gradient corruption)
    assert torch.all(logits.grad[:, 2] == 0.0)
    assert torch.all(logits.grad[:, 3] == 0.0)
    # Gradients for present classes 0 and 1 must be non-zero
    assert torch.any(logits.grad[:, 0] != 0.0)
    assert torch.any(logits.grad[:, 1] != 0.0)


def test_logit_adjusted_balanced_loss():
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))
    samples_per_cls = [1000, 50, 0]  # Severely imbalanced with missing class

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
