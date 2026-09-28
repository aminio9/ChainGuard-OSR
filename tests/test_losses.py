"""Tests for multi-objective loss functions."""

import torch
import torch.nn as nn
from src.losses.class_balanced import class_balanced_focal_loss, compute_class_balanced_weights
from src.losses.vib_loss import vib_kl_divergence_loss
from src.losses.orthogonal_loss import orthogonal_anchor_loss
from src.losses.primal_dual_loss import primal_dual_consensus_loss
from src.losses.composite_loss import CompositeLossCalculator
from src.config import LossConfig


def test_class_balanced_focal_loss():
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))
    samples_per_cls = [1000, 50, 10]  # Imbalanced

    loss = class_balanced_focal_loss(logits, targets, samples_per_cls=samples_per_cls, beta=0.999)
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


def test_orthogonal_anchor_loss():
    # Construct strictly orthogonal weight matrix
    # e.g. 2 orthogonal rows in 4-dim space
    W_ortho = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    loss_ortho = orthogonal_anchor_loss(W_ortho, r_max=2.0)
    assert abs(loss_ortho.item()) < 1e-6

    # Collinear rows should produce high loss
    W_collinear = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    loss_coll = orthogonal_anchor_loss(W_collinear, r_max=2.0)
    assert loss_coll.item() > 0.9


def test_primal_dual_consensus_loss():
    local_mod = nn.Linear(4, 2, bias=False)
    global_params = {k: v.clone() for k, v in local_mod.state_dict().items()}

    # If local == global and lambda == 0, loss is 0
    dual_zero = torch.zeros(8)
    loss_eq = primal_dual_consensus_loss(local_mod, global_params, dual_zero, rho=0.1)
    assert abs(loss_eq.item()) < 1e-6

    # If local diverges from global, proximal penalty increases quadratically
    with torch.no_grad():
        local_mod.weight.add_(1.0)
    loss_diff = primal_dual_consensus_loss(local_mod, global_params, dual_zero, rho=0.1)
    assert loss_diff.item() > 0.0


def test_composite_loss_calculator(decoupled_model):
    cfg = LossConfig(beta_cb=0.999, lambda_vib=0.01, lambda_anchor=0.05, primal_dual_rho=0.1)
    calc = CompositeLossCalculator(cfg)

    x = torch.randn(4, 12)
    y = torch.tensor([0, 1, 2, 0])
    logits, _, z, mu, logvar = decoupled_model(x)

    global_params = decoupled_model.get_encoder_parameters()
    dual_lambda = torch.zeros(100)

    total_loss, breakdown = calc(
        logits=logits,
        targets=y,
        mu=mu,
        logvar=logvar,
        classifier_weights=decoupled_model.classifier.weight_matrix,
        local_encoder=decoupled_model.encoder,
        global_encoder_params=global_params,
        dual_lambda=dual_lambda,
    )

    assert total_loss.item() > 0.0
    assert "loss_total" in breakdown
    assert "loss_cb" in breakdown
    assert "loss_vib" in breakdown
    assert "loss_anchor" in breakdown
