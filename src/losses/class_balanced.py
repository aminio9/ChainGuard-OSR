"""Class-Balanced Focal Cross-Entropy Loss based on Effective Number of Samples (Formula 4)."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F


def compute_class_balanced_weights(
    samples_per_class: List[int] | np.ndarray,
    beta: float = 0.999,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute normalized class-balanced weights based on effective number of samples:

        E_n = (1 - beta^n) / (1 - beta)
        W_c = (1 - beta) / (1 - beta^{N_c})
    """
    counts = np.array(samples_per_class, dtype=np.float64)
    # Avoid zero division
    counts = np.maximum(counts, 1.0)

    if beta <= 0.0 or beta >= 1.0:
        # Uniform weighting
        weights = np.ones_like(counts, dtype=np.float32)
    else:
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        # Normalize weights so that mean is 1.0
        weights = weights / np.sum(weights) * len(counts)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def class_balanced_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    samples_per_class: Optional[List[int] | np.ndarray] = None,
    samples_per_cls: Optional[List[int] | np.ndarray] = None,
    beta: float = 0.999,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-Balanced Focal Loss (Formula 4).

    Args:
        logits: Raw predictions of shape (batch, num_classes).
        targets: Ground truth class indices of shape (batch,).
        samples_per_class: Per-class sample counts in the client's local dataset.
        samples_per_cls: Alias for samples_per_class.
        beta: Effective number hyperparameter (default 0.999).
        gamma: Focal modulation parameter (default 2.0).

    Returns:
        Scalar loss tensor.
    """
    if samples_per_class is None:
        samples_per_class = samples_per_cls
    num_classes = logits.size(-1)
    if samples_per_class is None:
        samples_per_class = np.ones(num_classes, dtype=int)

    weights = compute_class_balanced_weights(samples_per_class, beta=beta, device=logits.device)

    # Standard unreduced cross entropy
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    p_t = torch.exp(-ce_loss)  # probability of true class
    focal_weight = torch.pow(1.0 - p_t, gamma)

    class_weight = weights[targets]
    loss = class_weight * focal_weight * ce_loss
    return loss.mean()
