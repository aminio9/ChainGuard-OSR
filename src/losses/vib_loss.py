"""Variational Information Bottleneck (VIB) Analytical KL Divergence Loss (LaTeX Section 4.2, Eq. eq:vib_kl).

Regularizes the diagonal Gaussian encoder posterior:
    q_{theta_k}(z | x_{k,i}) = N(z; m_{theta_k}(x_{k,i}), diag(sigma_{theta_k}^2(x_{k,i})))
toward the standard Gaussian prior p(z) = N(0, I_{d_z}). During local training, reparameterized
sampling z_{k,i} = m_{theta_k}(x_{k,i}) + eps * sigma_{theta_k}(x_{k,i}) injects Gaussian noise
with the learned scale sigma_{theta_k}(x_{k,i}), while the KL loss pulls the mean vectors
m_{theta_k}(x_{k,i}) toward the origin and regularizes the variances sigma_{theta_k}^2(x_{k,i})
toward one. Because the KL penalty can shrink unused coordinates of m_theta(x) toward zero, a
shrinkage term eps_v * I_{d_b} is added when inverting block covariances in Section 4.3.
"""

from __future__ import annotations

import torch


def vib_kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute analytical KL Divergence between N(m_{theta_k}(x), diag(sigma_{theta_k}^2(x))) and N(0, I_{d_z}).

    Mathematical Formulation (Eq. eq:vib_kl):
        L_VIB(theta_k) = (1 / (2 * |B_k|)) * sum_{(x, y) in B_k} sum_{j=1}^{d_z} [
            m_{theta_k, j}^2(x) + sigma_{theta_k, j}^2(x) - log(sigma_{theta_k, j}^2(x)) - 1
        ]

    Args:
        mu: Encoder posterior mean tensor m_{theta_k}(x) of shape (|B_k|, d_z).
        logvar: Encoder posterior log-variance tensor log(sigma_{theta_k}^2(x)) of shape (|B_k|, d_z).

    Returns:
        Scalar KL divergence loss averaged over the mini-batch B_k.
    """
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
    return kl.mean()
