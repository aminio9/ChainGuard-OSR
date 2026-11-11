"""Loss Functions for Federated Representation Learning.

Includes:
    - Balanced Logit Adjustment Loss (LogitAdjustedBalancedLoss, BLALoss)
    - Variational Information Bottleneck KL Regularizer (vib_kl_divergence_loss)
    - Local Composite Training Loss Calculator (CompositeLossCalculator)
"""

from src.losses.logit_adjusted import (
    LogitAdjustedBalancedLoss,
    logit_adjusted_balanced_loss,
    compute_local_class_frequencies,
    compute_laplace_class_priors,
)
from src.losses.bla_loss import (
    LogitAdjustedBalancedLoss as BLALoss,
    logit_adjusted_balanced_loss as bla_loss,
)
from src.losses.vib_loss import vib_kl_divergence_loss
from src.losses.composite_loss import CompositeLossCalculator

__all__ = [
    "LogitAdjustedBalancedLoss",
    "logit_adjusted_balanced_loss",
    "BLALoss",
    "bla_loss",
    "compute_local_class_frequencies",
    "compute_laplace_class_priors",
    "vib_kl_divergence_loss",
    "CompositeLossCalculator",
]
