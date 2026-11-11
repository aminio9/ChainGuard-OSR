"""Variational Information Bottleneck (VIB) Analytical KL Divergence.

Regularizes the encoder's diagonal Gaussian posterior q(z|x) toward a standard normal prior N(0, I):
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
Averaged across the batch to constrain information leakage into the latent space.
"""

from __future__ import annotations

import torch


def vib_kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor, normalize_dim: bool = True) -> torch.Tensor:
    """Compute analytical KL divergence regularizer for the VIB encoder.

    Args:
        mu: Encoder posterior mean tensor of shape (batch_size, latent_dim).
        logvar: Encoder posterior log-variance tensor of shape (batch_size, latent_dim).
        normalize_dim: If True, scales by 1 / latent_dim to keep loss magnitude invariant to dimension.

    Returns:
        Scalar KL divergence loss averaged over the mini-batch.
    """
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
    if normalize_dim and mu.size(-1) > 0:
        kl = kl / float(mu.size(-1))
    return kl.mean()
