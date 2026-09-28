"""Encoder architectures: Variational Information Bottleneck (VIB) and deterministic baselines."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
from src.models.base import BaseNetwork


class VIBEncoder(BaseNetwork):
    """Variational Information Bottleneck (VIB) Stochastic Feature Encoder (Formulas 5 & 6).

    Maps raw flow vector x to latent Gaussian distribution parameters (mu, logvar).
    Reparameterization Trick:
        z = mu + epsilon * exp(0.5 * logvar),  where epsilon ~ N(0, I) during training.
        z = mu deterministically during inference (eval mode).
    """

    def __init__(
        self,
        in_features: int = 43,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        layers = [nn.Linear(in_features, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, latent_dim * 2))  # mu and logvar
        self.network = nn.Sequential(*layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick during training; return mu during evaluation."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (z, mu, logvar) where z is the latent vector of shape (batch, latent_dim).
        """
        params = self.network(x)
        mu, logvar = torch.split(params, self.latent_dim, dim=-1)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class DeterministicEncoder(BaseNetwork):
    """Deterministic feature encoder for standard baselines and ablation studies."""

    def __init__(
        self,
        in_features: int = 43,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        layers = [nn.Linear(in_features, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (z, z, zeros_like(z)) for interface compatibility."""
        z = self.network(x)
        zeros = torch.zeros_like(z)
        return z, z, zeros
