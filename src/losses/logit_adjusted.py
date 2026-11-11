"""Balanced Logit Adjustment Loss (BLA).

Adjusts the classifier's output logits by adding class frequency priors:
    ell_c(x) + tau * log(pi_c)
This expands decision margins for rare classes to counter severe Non-IID class imbalance.
Crucially, the softmax denominator is restricted strictly to classes present at the local node,
masking absent classes to prevent zero-frequency log errors and spurious gradient updates.
"""

from __future__ import annotations

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_local_class_frequencies(
    samples_per_class: Union[List[int], np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute normalized local class frequencies pi_c = N_c / N for locally observed classes.

    For absent classes (N_c == 0), the frequency is set to 0.0 so their logits are masked
    out from the softmax denominator.

    Args:
        samples_per_class: Array or tensor of local sample counts for each class.
        num_classes: Total number of known classes.
        device: PyTorch device for returned tensor.

    Returns:
        Tensor of class frequencies summing to 1.0 over present classes.
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

    n_k = float(counts.sum().item())
    if n_k > 0:
        freqs = counts / n_k
    else:
        freqs = torch.ones_like(counts) / float(counts.size(0))

    if device is not None:
        freqs = freqs.to(device)

    return freqs


def compute_laplace_class_priors(
    samples_per_class: Union[List[int], np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
    epsilon: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute Laplace-smoothed class prior distribution for client nodes.

    Args:
        samples_per_class: Array or tensor of sample counts for each class.
        num_classes: Total number of known classes.
        epsilon: Laplace smoothing constant (default: 0.1).
        device: PyTorch device for returned tensor.

    Returns:
        Tensor of smoothed class priors summing to 1.0.
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
    """Balanced Logit Adjustment loss module.

    Restricts denominator strictly to locally present classes, eliminating absent-class
    gradient corruption and log(0) errors.
    """

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
        """Compute balanced cross-entropy loss over a mini-batch.

        Args:
            logits: Output classification logits of shape (batch_size, num_classes).
            targets: Class labels of shape (batch_size,).
            class_priors: Local class frequencies of shape (num_classes,).
                          Entries with class_priors[c] <= 0 (absent classes) are masked.

        Returns:
            Scalar balanced cross-entropy loss.
        """
        priors = class_priors.to(logits.device)
        present_mask = priors > 0

        # If all classes present and gamma == 0, standard cross-entropy
        if self.gamma == 0.0 and torch.all(present_mask):
            return F.cross_entropy(logits, targets)

        adjusted_logits = logits.clone()

        if self.gamma != 0.0:
            log_priors = torch.zeros_like(priors)
            log_priors[present_mask] = torch.log(priors[present_mask])
            adjusted_logits = adjusted_logits + self.gamma * log_priors.unsqueeze(0)

        # Restrict denominator strictly to locally present classes:
        # Setting absent class logits to -1e9 forces exp(logit) to zero
        if not torch.all(present_mask):
            adjusted_logits = adjusted_logits.masked_fill(~present_mask.unsqueeze(0), -1e9)

        return F.cross_entropy(adjusted_logits, targets)


def logit_adjusted_balanced_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_priors: torch.Tensor,
    tau: float = 1.0,
    gamma: Optional[float] = None,
) -> torch.Tensor:
    """Functional interface for Balanced Logit Adjustment loss."""
    scale = float(gamma if gamma is not None else tau)
    priors = class_priors.to(logits.device)
    present_mask = priors > 0

    if scale == 0.0 and torch.all(present_mask):
        return F.cross_entropy(logits, targets)

    adjusted_logits = logits.clone()
    if scale != 0.0:
        log_priors = torch.zeros_like(priors)
        log_priors[present_mask] = torch.log(priors[present_mask])
        adjusted_logits = adjusted_logits + scale * log_priors.unsqueeze(0)

    if not torch.all(present_mask):
        adjusted_logits = adjusted_logits.masked_fill(~present_mask.unsqueeze(0), -1e9)

    return F.cross_entropy(adjusted_logits, targets)
