"""Variational Information Bottleneck (VIB) Analytical KL Divergence Loss (Formula 5)."""

from __future__ import annotations

import torch


def vib_kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute analytical KL Divergence between N(mu, diag(exp(logvar))) and standard N(0, I).

    Mathematical Formulation (Formula 5):
        L_VIB = 0.5 * sum_j [ mu_j^2 + sigma_j^2 - log(sigma_j^2) - 1 ]
              = -0.5 * sum_j [ 1 + logvar_j - mu_j^2 - exp(logvar_j) ]

    Args:
        mu: Mean tensor of shape (batch, latent_dim).
        logvar: Log-variance tensor of shape (batch, latent_dim).

    Returns:
        Scalar KL divergence loss averaged over the batch.
    """
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
    return kl.mean()
