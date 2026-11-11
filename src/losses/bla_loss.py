"""Balanced Logit Adjustment Loss (BLA).

Corresponds to the balanced logit adjustment loss formulation:
    L_BLA adjusts classification logits by class priors to counter severe Non-IID
    label skew across distributed nodes, while restricting the softmax denominator
    strictly to locally present classes.
"""

from __future__ import annotations

from src.losses.logit_adjusted import (
    LogitAdjustedBalancedLoss,
    logit_adjusted_balanced_loss,
    compute_local_class_frequencies,
    compute_laplace_class_priors,
)

__all__ = [
    "LogitAdjustedBalancedLoss",
    "logit_adjusted_balanced_loss",
    "compute_local_class_frequencies",
    "compute_laplace_class_priors",
]
