"""Composite Phase I Local Loss Calculator (LaTeX Section 4.2, Eq. eq:total_client_loss).

Formulation:
    L_local(theta_k, psi_k) = L_BLA(theta_k, psi_k) + lambda_vib * L_VIB(theta_k)

Features:
    1. Logit-Adjusted Balanced Risk Minimization (L_BLA): Enlarges rare-class margins among
       present classes (c in Y_k) and suppresses absent-class gradients (c in Y_K \\ Y_k)
       on the shared encoder f_{theta_k}.
    2. Variational Information Bottleneck (L_VIB): Pulls posterior means mu(x; theta_k) toward
       the origin and injects Gaussian noise during training to reduce packet-noise sensitivity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.losses.logit_adjusted import (
    LogitAdjustedBalancedLoss,
    compute_laplace_class_priors,
)
from src.losses.vib_loss import vib_kl_divergence_loss
from src.config import LossConfig


class CompositeLossCalculator:
    """Computes total composite training loss across Phase I objectives: L_local = L_BLA + lambda_vib * L_VIB."""

    def __init__(self, config: LossConfig):
        self.config = config
        tau_cfg = getattr(config, "tau_bla", 1.0)
        gamma_cfg = getattr(config, "gamma_bla", None)
        self.tau_bla = float(tau_cfg)
        self.gamma_bla = float(gamma_cfg) if gamma_cfg is not None else self.tau_bla
        self.laplace_eps = getattr(config, "laplace_eps", 0.1)
        self.lambda_vib = getattr(config, "lambda_vib", 0.01)
        self.bla_loss_fn = LogitAdjustedBalancedLoss(tau=self.tau_bla, gamma=self.gamma_bla)

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        class_priors: Optional[torch.Tensor] = None,
        samples_per_class: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        return_tensors: bool = False,
        **kwargs: Any,  # Gracefully accept legacy unused arguments (teacher_logits, t_ratio, etc.)
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute Phase I loss components.

        Args:
            logits: Output classification logits ell_k(x) in R^{|C_K|}.
            targets: Ground-truth class labels y in C_K.
            mu: Latent representation mean from VIB mu(x; theta).
            logvar: Latent representation log-variance from VIB log sigma^2(x; theta).
            class_priors: Precomputed Laplace class priors pi^{(k)}.
            samples_per_class: Class-frequency distribution if priors are not precomputed.
            return_tensors: If True, returns detached tensors in breakdown to avoid GPU-CPU sync stalls.

        Returns:
            (total_loss, breakdown_metrics)
        """
        # 1. Obtain or compute class priors
        if class_priors is None:
            if samples_per_class is not None:
                class_priors = compute_laplace_class_priors(
                    samples_per_class,
                    num_classes=logits.size(1),
                    epsilon=self.laplace_eps,
                    device=logits.device,
                )
            else:
                # Uniform fallback
                class_priors = torch.ones(logits.size(1), device=logits.device) / float(logits.size(1))
        else:
            class_priors = class_priors.to(logits.device)

        # 2. Logit-Adjusted Balanced Task Loss (L_BLA)
        l_bla = self.bla_loss_fn(logits, targets, class_priors)

        # 3. Variational Information Bottleneck KL Regularization (L_VIB)
        if self.lambda_vib > 0.0:
            l_vib = vib_kl_divergence_loss(mu, logvar, normalize_dim=True)
        else:
            l_vib = torch.tensor(0.0, device=logits.device)

        # 4. Total Loss
        total_loss = l_bla + self.lambda_vib * l_vib

        if return_tensors:
            breakdown = {
                "loss_total": total_loss.detach(),
                "loss_bla": l_bla.detach(),
                "loss_vib": l_vib.detach(),
            }
        else:
            breakdown = {
                "loss_total": float(total_loss.item()),
                "loss_bla": float(l_bla.item()),
                "loss_vib": float(l_vib.item()),
            }

        return total_loss, breakdown
