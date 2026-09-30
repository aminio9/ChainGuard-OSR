"""Logit-Adjusted Balanced Risk Minimization (LaTeX Section 4.2, Equation eq:bla_loss).

Based on Menon et al. (ICLR 2021) and Chen & Chao (FedRoD, ICLR 2022).
Addresses non-IID label skew and locally absent classes:
1. Under standard cross-entropy (L_CE), a locally absent class c in Y_K \\ Y_k (N_{k,c} = 0)
   has no positive samples on node k, yet its logit receives a positive gradient
   dL_CE / d ell_{k,c} = p_{k,c} > 0 from every sample of a present class y_{k,i} in Y_k.
   Local updates continually push down absent-class logits, and backpropagating these gradients
   through psi_{k,c} can bias the shared encoder f_{theta_k} and may distort learned features.
2. In L_BLA, for any non-target class c != y_{k,i}, dL_BLA / d ell_{k,c} = p_{k,c}^{adj}(x).
   First, among locally present classes c in Y_k, the factor pi_{k,c}^gamma is larger for
   majority classes, so a non-target majority class (c != y_{k,i}) is pushed down more strongly,
   enlarging the decision margin of rare local classes. Second, for any locally absent class
   c in Y_K \\ Y_k (N_{k,c} = 0), choosing eps_pi << n_k ensures pi_{k,c} = eps_pi / (n_k + C * eps_pi) << 1,
   which makes gamma * log pi_{k,c} strongly negative and reduces p_{k,c}^{adj}(x) proportional to
   pi_{k,c}^gamma to nearly zero (recovering a restricted softmax over Y_k as eps_pi -> 0).

Mathematical Formulation:
    L_BLA(theta_k, psi_k) = - (1 / |B_k|) sum_{(x, y) in B_k} log [
        exp(ell_{k, y}(x) + gamma * log pi_{k, y}) /
        sum_{c in Y_K} pi_{k, c}^gamma * exp(ell_{k, c}(x))
    ]
where:
    pi_{k, c} = (N_{k, c} + eps_pi) / (n_k + C * eps_pi),   with 0 < eps_pi << n_k.
"""

from __future__ import annotations

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_laplace_class_priors(
    samples_per_class: Union[List[int], np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
    epsilon: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute Laplace-smoothed class prior distribution pi_k for client k (Eq. eq:smoothed_prior).

    Args:
        samples_per_class: Array or tensor of sample counts N_{k,c} for each class.
        num_classes: Total number of known classes C = |Y_K|.
        epsilon: Laplace smoothing parameter eps_pi > 0 (default: 0.1).
        device: PyTorch device for returned tensor.

    Returns:
        Tensor of class priors pi_k of shape (C,) summing to 1.0.
    """
    if isinstance(samples_per_class, (list, tuple)):
        counts = torch.tensor(samples_per_class, dtype=torch.float32)
    elif isinstance(samples_per_class, np.ndarray):
        counts = torch.from_numpy(samples_per_class).float()
    else:
        counts = samples_per_class.float()

    if num_classes is not None and counts.size(0) < num_classes:
        padded = torch.zeros(num_classes, dtype=torch.float32)
        padded[: counts.size(0)] = counts
        counts = padded

    c_k = float(counts.size(0))
    n_k = float(counts.sum().item())

    # Laplace smoothing: (N_{k,c} + eps_pi) / (n_k + C * eps_pi)
    priors = (counts + epsilon) / (n_k + c_k * epsilon)

    if device is not None:
        priors = priors.to(device)

    return priors


class LogitAdjustedBalancedLoss(nn.Module):
    """Logit-Adjusted Balanced Risk Minimization Module (L_BLA, Eq. eq:bla_loss)."""

    def __init__(self, tau: float = 1.0, gamma: Optional[float] = None):
        super().__init__()
        scale = float(gamma if gamma is not None else tau)
        self.gamma = scale
        self.tau = scale  # Backward-compatible alias

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_priors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L_BLA over mini-batch B_k.

        Args:
            logits: Output classification logits ell_k(x) of shape (|B_k|, C).
            targets: Class labels y in Y_K of shape (|B_k|,).
            class_priors: Local class priors pi_k of shape (C,) on same device.

        Returns:
            Scalar balanced cross-entropy loss.
        """
        priors = class_priors.to(logits.device).clamp(min=1e-12)
        log_priors = torch.log(priors)
        adjusted_logits = logits + self.gamma * log_priors.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets)


def logit_adjusted_balanced_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_priors: torch.Tensor,
    tau: float = 1.0,
    gamma: Optional[float] = None,
) -> torch.Tensor:
    """Functional interface for Logit-Adjusted Balanced Loss (L_BLA)."""
    scale = float(gamma if gamma is not None else tau)
    priors = class_priors.to(logits.device).clamp(min=1e-12)
    log_priors = torch.log(priors)
    adjusted_logits = logits + scale * log_priors.unsqueeze(0)
    return F.cross_entropy(adjusted_logits, targets)
