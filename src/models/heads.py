"""Decoupled output heads: classification head C_phi and density/uncertainty head H_psi (Formula 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.base import BaseNetwork


class ClassificationHead(BaseNetwork):
    """Linear classification head mapping latent space to known class logits (C_phi).

    Uses bias=False to align with Stiefel manifold orthogonal anchoring constraints.
    """

    def __init__(self, latent_dim: int = 64, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(latent_dim, num_classes, bias=False)

    @property
    def weight_matrix(self) -> torch.Tensor:
        """Returns the classifier weight matrix of shape (num_classes, latent_dim)."""
        return self.linear.weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute raw classification logits."""
        return self.linear(z)


class DensityHead(BaseNetwork):
    """Auxiliary uncertainty and density projection head (H_psi).

    Lightweight projection head used for density estimation and energy-based OSR scoring.
    """

    def __init__(self, latent_dim: int = 64, density_dim: int = 16):
        super().__init__()
        self.density_dim = density_dim
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, density_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute compact uncertainty/density representations."""
        return self.projection(z)
