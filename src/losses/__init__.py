"""Canonical Loss Functions for FedTROS-MC (LaTeX Section 4.1, Equations 23-87).

1. Class-Balanced Loss (L_CB, Eq. 23-30)
2. Variational Information Bottleneck KL Regularization (L_VIB, Eq. 46)
3. Dynamic Global Anchor Knowledge Distillation (L_anchor, Eq. 59-69)
4. Composite Objective Calculator (L_local, Eq. 87)
"""

from src.losses.class_balanced import class_balanced_focal_loss
from src.losses.vib_loss import vib_kl_divergence_loss
from src.losses.anchor_loss import (
    DynamicGlobalAnchorLoss,
    compute_shannon_certainty,
    compute_anchor_weight,
)
from src.losses.composite_loss import CompositeLossCalculator

__all__ = [
    "class_balanced_focal_loss",
    "vib_kl_divergence_loss",
    "DynamicGlobalAnchorLoss",
    "compute_shannon_certainty",
    "compute_anchor_weight",
    "CompositeLossCalculator",
]
