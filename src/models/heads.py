"""Known-Class Classification Head C_phi (LaTeX Section 4.1.1, Eq. 32)."""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.base import BaseNetwork


class ClassificationHead(BaseNetwork):
    """Linear classification head mapping latent space to known class logits: C_phi(z) -> R^{|C_K|} (Eq. 32)."""

    def __init__(self, latent_dim: int = 64, num_classes: int = 4, bias: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(latent_dim, num_classes, bias=bias)

    @property
    def weight_matrix(self) -> torch.Tensor:
        """Returns the classifier weight matrix of shape (num_classes, latent_dim)."""
        return self.linear.weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized classification logits."""
        return self.linear(z)


LinearClassifierHead = ClassificationHead

__all__ = ["ClassificationHead", "LinearClassifierHead"]
