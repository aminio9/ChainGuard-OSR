"""Composite multi-objective local loss calculator (Equations 9-11 & Algorithm 1 Line 8)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.losses.class_balanced import class_balanced_focal_loss
from src.losses.vib_loss import vib_kl_divergence_loss
from src.losses.orthogonal_loss import orthogonal_anchor_loss
from src.losses.anchor_loss import DynamicGlobalAnchorLoss
from src.losses.primal_dual_loss import primal_dual_consensus_loss
from src.config import LossConfig


class CompositeLossCalculator:
    """Computes total composite training loss across Phase I objectives (Equations 9-11).

    Mathematical Formulation (Equations 9 & 11):
        min_{w_k} L_local = L_CB + lambda_vib * L_VIB + lambda_anc(x, t) * L_anchor [+ L_PD]
    """

    def __init__(self, config: LossConfig):
        self.config = config
        self.dynamic_anchor = DynamicGlobalAnchorLoss(
            tau_g=getattr(config, "tau_g", 2.0),
            max_lambda=getattr(config, "max_lambda_anchor", config.lambda_anchor if config.lambda_anchor > 0 else 1.0),
            rho_warm=getattr(config, "rho_warm", getattr(config, "anchor_t0", 0.15)),
        )

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        classifier_weights: torch.Tensor,
        local_encoder: Optional[nn.Module] = None,
        global_encoder_params: Optional[Dict[str, torch.Tensor]] = None,
        dual_lambda: Optional[torch.Tensor] = None,
        samples_per_class: Optional[List[int] | np.ndarray] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        t_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all loss components and combine into total loss scalar.

        Args:
            logits: Local student prediction logits.
            targets: True class labels.
            mu: Latent representation mean from VIB.
            logvar: Latent representation log-variance from VIB.
            classifier_weights: Linear classification head weight matrix.
            local_encoder: Local feature encoder module (for primal-dual).
            global_encoder_params: Frozen global parameters (for primal-dual).
            dual_lambda: Lagrange multiplier vector (for primal-dual).
            samples_per_class: Class-frequency distribution for effective number weighting.
            teacher_logits: Logits from frozen global anchor model (for entropy-gated KD).
            t_ratio: Current federation round progression ratio t / T in [0, 1].

        Returns:
            (total_loss, metrics_dict)
        """
        # 1. Class-Balanced Focal Cross-Entropy (Equations 1-2)
        l_cb = class_balanced_focal_loss(
            logits,
            targets,
            samples_per_class=samples_per_class,
            beta=self.config.beta_cb,
            gamma=self.config.focal_gamma,
        )

        # 2. Variational Information Bottleneck KL Regularization (Equations 3-4)
        if self.config.lambda_vib > 0.0:
            l_vib = vib_kl_divergence_loss(mu, logvar)
        else:
            l_vib = torch.tensor(0.0, device=logits.device)

        # 3. Dynamic Global Anchor (Equations 5-8) OR Orthogonal Anchor Regularizer
        if teacher_logits is not None:
            # Full Research-Grade Dynamic Anchor with Shannon Entropy Gating (Equations 5-8)
            l_anchor = self.dynamic_anchor(logits, teacher_logits, t_ratio=t_ratio)
            anchor_loss_weighted = l_anchor  # lambda_anc is already integrated inside dynamic_anchor
        elif self.config.lambda_anchor > 0.0:
            # Baseline Orthogonal Anchor Regularizer (Ablation)
            l_anchor = orthogonal_anchor_loss(classifier_weights)
            anchor_loss_weighted = self.config.lambda_anchor * l_anchor
        else:
            l_anchor = torch.tensor(0.0, device=logits.device)
            anchor_loss_weighted = torch.tensor(0.0, device=logits.device)

        # 4. Primal-Dual Augmented Lagrangian Consensus OR FedProx Proximal Penalty (LaTeX Section 5.3)
        eff_rho = max(self.config.primal_dual_rho, getattr(self.config, "fedprox_mu", 0.0))
        if eff_rho > 0.0 and local_encoder is not None and global_encoder_params is not None:
            effective_lambda = dual_lambda if dual_lambda is not None else torch.zeros(1, device=logits.device)
            l_pd = primal_dual_consensus_loss(
                local_encoder,
                global_encoder_params,
                effective_lambda,
                rho=eff_rho,
            )
        else:
            l_pd = torch.tensor(0.0, device=logits.device)

        # Composite sum (Equation 9)
        total_loss = (
            l_cb
            + self.config.lambda_vib * l_vib
            + anchor_loss_weighted
            + l_pd
        )

        breakdown = {
            "loss_total": float(total_loss.item()),
            "loss_cb": float(l_cb.item()),
            "loss_vib": float(l_vib.item()),
            "loss_anchor": float(l_anchor.item()),
            "loss_pd": float(l_pd.item()),
        }

        return total_loss, breakdown
