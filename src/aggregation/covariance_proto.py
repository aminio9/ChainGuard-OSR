"""Bayesian Covariance-Weighted Global Prototype Synthesis (Formula 10 & Algorithm 1 Line 14)."""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch


def aggregate_prototypes_covariance(
    client_prototypes: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]],
    latent_dim: int = 64,
    ridge_epsilon: float = 1e-4,
) -> Dict[int, torch.Tensor]:
    """Synthesize global class prototypes weighted by inverse local covariance matrices.

    Mathematical Formulation (Formula 10):
        mu_{global, c} = [ sum_{k in K_c} (Sigma_{k,c} + eps * I)^{-1} ]^{-1} *
                         [ sum_{k in K_c} (Sigma_{k,c} + eps * I)^{-1} * mu_{k,c} ]

    Args:
        client_prototypes: Dict mapping class_id -> list of tuples (mu_k_c, cov_k_c, sample_count).
        latent_dim: Dimensionality of latent feature representations (d_z = 64).
        ridge_epsilon: Ridge diagonal regularization parameter to prevent matrix singularity.

    Returns:
        Dict mapping class_id -> aggregated global prototype tensor of shape (latent_dim,).
    """
    global_prototypes: Dict[int, torch.Tensor] = {}

    for c, submissions in client_prototypes.items():
        if not submissions:
            continue

        if len(submissions) == 1:
            # Single client has this class: global prototype equals the local prototype
            global_prototypes[c] = submissions[0][0].clone()
            continue

        device = submissions[0][0].device
        inv_cov_sum = torch.zeros((latent_dim, latent_dim), dtype=torch.float32, device=device)
        weighted_mu_sum = torch.zeros(latent_dim, dtype=torch.float32, device=device)

        ridge_eye = ridge_epsilon * torch.eye(latent_dim, dtype=torch.float32, device=device)

        for mu, cov, _ in submissions:
            mu_f = mu.float()
            cov_f = cov.float()
            # Ensure 2D covariance matrix
            if cov_f.ndim == 1:
                cov_mat = torch.diag(cov_f)
            else:
                cov_mat = cov_f

            # Invert with ridge stabilizer
            reg_cov = cov_mat + ridge_eye
            try:
                inv_cov = torch.linalg.inv(reg_cov)
            except RuntimeError:
                # SVD pseudo-inverse fallback in case of severe collinearity
                inv_cov = torch.linalg.pinv(reg_cov)

            inv_cov_sum += inv_cov
            weighted_mu_sum += torch.mv(inv_cov, mu_f)

        try:
            global_cov = torch.linalg.inv(inv_cov_sum + ridge_eye)
        except RuntimeError:
            global_cov = torch.linalg.pinv(inv_cov_sum + ridge_eye)

        global_mu = torch.mv(global_cov, weighted_mu_sum)
        global_prototypes[c] = global_mu

    return global_prototypes
