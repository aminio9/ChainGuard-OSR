"""Composite multi-objective local loss calculator (Formula 9 & Algorithm 1 Line 8)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.losses.class_balanced import class_balanced_focal_loss
from src.losses.vib_loss import vib_kl_divergence_loss
from src.losses.orthogonal_loss import orthogonal_anchor_loss
from src.losses.primal_dual_loss import primal_dual_consensus_loss
from src.config import LossConfig


class CompositeLossCalculator:
    """Computes total composite training loss across all objectives (Formula 9).

    Mathematical Formulation:
        L_local = L_CB + lambda_vib * L_VIB + lambda_anchor * L_anchor + L_PD
    """

    def __init__(self, config: LossConfig):
        self.config = config

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        classifier_weights: torch.Tensor,
        local_encoder: nn.Module,
        global_encoder_params: Optional[Dict[str, torch.Tensor]],
        dual_lambda: Optional[torch.Tensor],
        samples_per_class: Optional[List[int] | np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all loss components and combine into total loss scalar.

        Returns:
            (total_loss, metrics_dict)
        """
        # 1. Class-Balanced Focal Cross-Entropy
        l_cb = class_balanced_focal_loss(
            logits,
            targets,
            samples_per_class=samples_per_class,
            beta=self.config.beta_cb,
            gamma=self.config.focal_gamma,
        )

        # 2. Variational Information Bottleneck KL Regularization
        if self.config.lambda_vib > 0.0:
            l_vib = vib_kl_divergence_loss(mu, logvar)
        else:
            l_vib = torch.tensor(0.0, device=logits.device)

        # 3. Fixed / Penalized Orthogonal Anchor Regularization
        if self.config.lambda_anchor > 0.0:
            l_anchor = orthogonal_anchor_loss(classifier_weights)
        else:
            l_anchor = torch.tensor(0.0, device=logits.device)

        # 4. Primal-Dual Augmented Lagrangian Consensus
        if (
            self.config.primal_dual_rho > 0.0
            and global_encoder_params is not None
            and dual_lambda is not None
        ):
            l_pd = primal_dual_consensus_loss(
                local_encoder,
                global_encoder_params,
                dual_lambda,
                rho=self.config.primal_dual_rho,
            )
        else:
            l_pd = torch.tensor(0.0, device=logits.device)

        # Composite sum
        total_loss = (
            l_cb
            + self.config.lambda_vib * l_vib
            + self.config.lambda_anchor * l_anchor
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
