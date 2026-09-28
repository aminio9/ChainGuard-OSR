"""Encoder architectures: Variational Information Bottleneck (VIB) and deterministic baselines."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union
import torch
import torch.nn as nn
from src.models.base import BaseNetwork


def get_activation(act_name: str) -> nn.Module:
    """Instantiate activation function by name."""
    act = act_name.lower().strip()
    if act == "gelu":
        return nn.GELU()
    if act == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if act == "elu":
        return nn.ELU()
    return nn.ReLU()


class VIBEncoder(BaseNetwork):
    """Variational Information Bottleneck (VIB) Feature Encoder (Equations 3-4 & Section 5.3).

    Supports MLP backbone (e.g. 512 -> 256 -> 128 with GELU/ReLU) mapping to Gaussian parameters (mu, logvar).
    Reparameterization Trick:
        z = mu + epsilon * exp(0.5 * logvar),  where epsilon ~ N(0, I) during training.
        z = mu deterministically during inference (eval mode).
    """

    def __init__(
        self,
        in_features: int = 43,
        hidden_dim: Union[int, Sequence[int]] = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_layernorm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.latent_dim = latent_dim

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        layers: List[nn.Module] = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            elif use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection to Gaussian parameters: mu and logvar
        layers.append(nn.Linear(prev_dim, latent_dim * 2))
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
        hidden_dim: Union[int, Sequence[int]] = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_layernorm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.latent_dim = latent_dim

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        layers: List[nn.Module] = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            elif use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (z, z, zeros_like(z)) for interface compatibility."""
        z = self.network(x)
        zeros = torch.zeros_like(z)
        return z, z, zeros
