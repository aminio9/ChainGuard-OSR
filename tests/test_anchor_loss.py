"""Unit tests for Dynamic Global Anchor Regularization with Shannon Entropy Gating (Equations 5-8)."""

import math
import pytest
import torch
import torch.nn.functional as F

from src.losses.anchor_loss import (
    DynamicGlobalAnchorLoss,
    compute_anchor_weight,
    compute_shannon_certainty,
)


def test_shannon_certainty_extremes():
    num_classes = 5
    # 1. Perfectly confident one-hot distribution: certainty should approach 1.0
    confident_logits = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])
    u_confident = compute_shannon_certainty(confident_logits, temperature=2.0)
    assert u_confident.item() == pytest.approx(1.0, abs=1e-3)

    # 2. Completely uncertain uniform distribution: certainty should approach 0.0
    uniform_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    u_uniform = compute_shannon_certainty(uniform_logits, temperature=2.0)
    assert u_uniform.item() == pytest.approx(0.0, abs=1e-3)


def test_anchor_weight_progression():
    u_anc = torch.tensor([1.0, 0.5])
    # At t_ratio = 0.0 (early warm-up, well before t0 = 0.25): weight should be very low
    w_early = compute_anchor_weight(u_anc, t_ratio=0.0, max_lambda=1.0, kappa=10.0, t0=0.25)
    assert (w_early < 0.15).all()

    # At t_ratio = 1.0 (late rounds, well after t0 = 0.25): weight should approach max_lambda * u_anc
    w_late = compute_anchor_weight(u_anc, t_ratio=1.0, max_lambda=1.0, kappa=10.0, t0=0.25)
    assert w_late[0].item() == pytest.approx(1.0, abs=0.01)
    assert w_late[1].item() == pytest.approx(0.5, abs=0.01)


def test_dynamic_anchor_loss_gradients():
    loss_fn = DynamicGlobalAnchorLoss(tau_g=2.0, max_lambda=1.0, kappa=10.0, t0=0.25)

    student_logits = torch.randn(4, 5, requires_grad=True)
    teacher_logits = torch.tensor([
        [5.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5.0, 0.0],
    ])

    loss = loss_fn(student_logits, teacher_logits, t_ratio=0.5)
    assert loss.item() > 0.0

    loss.backward()
    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()
    # Gradients should be non-zero for student logits (Equation 8)
    assert torch.norm(student_logits.grad) > 0.0
