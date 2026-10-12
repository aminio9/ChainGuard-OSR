"""Differential Privacy (LDP) Moment Perturbation with Sensitivity Clipping, Symmetric Gaussian Noise, and PSD Cone Projection.

Implements the exact mathematical formulation for Phase II privacy protection:
1. L2 and Frobenius Sensitivity Clipping:
   mu_bar = mu / max(1, ||mu||_2 / C_mu)
   Sigma_bar_b = Sigma_b / max(1, ||Sigma_b||_F / C_Sigma)
2. Symmetric Gaussian Mechanism ((epsilon, delta)-LDP):
   sigma = (Delta * sqrt(2 * ln(1.25 / delta))) / epsilon
   W_b = (G + G^T) / sqrt(2), G ~ N(0, sigma_Sigma^2)^{d_b x d_b}
3. Positive Semi-Definite (PSD) Cone Projection (Pi_{S_+}):
   Sigma_tilde_b = sum_{i=1}^{d_b} max(0, lambda_i) v_i v_i^T
4. Likelihood Ratio Membership Inference Attack (MIA) Auditing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np
import torch

from src.aggregation.total_variance import ClientClassMoment
from src.metrics.osr_metrics import compute_binary_auroc


def clip_mean_l2(mu: torch.Tensor, clip_bound: float = 5.0) -> torch.Tensor:
    """Clip mean vector L2 norm to sensitivity bound C_mu."""
    norm_val = float(torch.linalg.norm(mu, ord=2).item())
    scale = max(1.0, norm_val / max(clip_bound, 1e-8))
    return mu / scale


def clip_covariance_frobenius(cov_block: torch.Tensor, clip_bound: float = 10.0) -> torch.Tensor:
    """Clip covariance sub-block Frobenius norm to sensitivity bound C_Sigma."""
    frob_val = float(torch.linalg.norm(cov_block, ord="fro").item())
    scale = max(1.0, frob_val / max(clip_bound, 1e-8))
    return cov_block / scale


def project_psd_cone(sym_matrix: torch.Tensor, min_eig: float = 0.0) -> torch.Tensor:
    """Project symmetric noisy matrix onto the Positive Semi-Definite (PSD) cone S_+ via eigenvalue clipping.

    Formula:
        Pi_{S_+}(Sigma_hat) = sum_{i=1}^{d_b} max(min_eig, lambda_i) v_i v_i^T
    """
    orig_dtype = sym_matrix.dtype
    sym = 0.5 * (sym_matrix.to(torch.float64) + sym_matrix.to(torch.float64).t())
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eff_min = max(float(min_eig), 1e-5)
    clipped_eigvals = torch.clamp(eigvals, min=eff_min)
    projected = torch.matmul(eigvecs * clipped_eigvals.unsqueeze(0), eigvecs.t())
    projected = 0.5 * (projected + projected.t())
    return projected.to(orig_dtype)


def compute_gaussian_noise_std(
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
) -> float:
    """Compute analytical Gaussian mechanism standard deviation for (epsilon, delta)-DP."""
    if epsilon <= 0 or math.isinf(epsilon):
        return 0.0
    c_factor = math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12)))
    return float((sensitivity * c_factor) / float(epsilon))


def perturb_client_moment_ldp(
    moment: ClientClassMoment,
    epsilon: float,
    delta: float = 1e-5,
    c_mu: float = 5.0,
    c_sigma: float = 10.0,
    clip_norm_mu: float | None = None,
    clip_norm_cov: float | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> ClientClassMoment:
    """Apply (epsilon, delta)-LDP perturbation to a client's sufficient statistics (mu, block_covs)."""
    eff_c_mu = float(clip_norm_mu) if clip_norm_mu is not None else c_mu
    eff_c_sigma = float(clip_norm_cov) if clip_norm_cov is not None else c_sigma
    if generator is None and seed is not None:
        generator = torch.Generator(device=moment.mu.device)
        generator.manual_seed(int(seed))

    n_k = max(1, moment.sample_count)
    mu_clipped = clip_mean_l2(moment.mu, clip_bound=eff_c_mu)

    # Sensitivity of sample mean is 2 * C_mu / n_k; split privacy budget 50/50 between mean and covariance
    eps_mu = 0.5 * epsilon
    eps_cov = 0.5 * epsilon

    sens_mu = (2.0 * eff_c_mu) / float(n_k)
    sigma_mu = compute_gaussian_noise_std(sens_mu, eps_mu, delta=delta)

    if sigma_mu > 0.0:
        noise_mu = torch.randn(
            mu_clipped.shape,
            dtype=mu_clipped.dtype,
            device=mu_clipped.device,
            generator=generator,
        ) * sigma_mu
        mu_tilde = mu_clipped + noise_mu
    else:
        mu_tilde = mu_clipped.clone()

    sens_cov = (2.0 * eff_c_sigma) / float(n_k)
    sigma_cov = compute_gaussian_noise_std(sens_cov, eps_cov, delta=delta)

    noisy_blocks: List[torch.Tensor] = []
    for b_cov in moment.block_covs:
        b_clipped = clip_covariance_frobenius(b_cov, clip_bound=eff_c_sigma)
        if sigma_cov > 0.0:
            g_mat = torch.randn(
                b_clipped.shape,
                dtype=b_clipped.dtype,
                device=b_clipped.device,
                generator=generator,
            ) * sigma_cov
            # Symmetric Gaussian noise: W_b = (G + G^T) / sqrt(2)
            w_sym = (g_mat + g_mat.t()) / math.sqrt(2.0)
            b_hat = b_clipped + w_sym
            b_tilde = project_psd_cone(b_hat, min_eig=0.0)
        else:
            b_tilde = project_psd_cone(b_clipped, min_eig=0.0)
        noisy_blocks.append(b_tilde)

    return ClientClassMoment(
        client_id=moment.client_id,
        sample_count=moment.sample_count,
        mu=mu_tilde,
        block_covs=noisy_blocks,
        cal_quantile=moment.cal_quantile,
    )


def evaluate_likelihood_ratio_mia(
    member_z: torch.Tensor | np.ndarray | None = None,
    non_member_z: torch.Tensor | np.ndarray | None = None,
    target_mu: torch.Tensor | np.ndarray | None = None,
    target_block_covs: List[torch.Tensor] | torch.Tensor | np.ndarray | None = None,
    population_mu: torch.Tensor | np.ndarray | None = None,
    population_block_covs: List[torch.Tensor] | torch.Tensor | np.ndarray | None = None,
    tikhonov_eps: float = 1e-4,
    *,
    member_features: torch.Tensor | np.ndarray | None = None,
    non_member_features: torch.Tensor | np.ndarray | None = None,
    published_mu: torch.Tensor | np.ndarray | None = None,
    published_cov: torch.Tensor | np.ndarray | None = None,
    prior_mu: torch.Tensor | np.ndarray | None = None,
    prior_cov: torch.Tensor | np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate Likelihood Ratio Membership Inference Attack (MIA) AUC against published moments."""
    mem_t = torch.as_tensor(member_features if member_features is not None else member_z, dtype=torch.float32)
    non_t = torch.as_tensor(non_member_features if non_member_features is not None else non_member_z, dtype=torch.float32)
    t_mu = torch.as_tensor(published_mu if published_mu is not None else target_mu, dtype=torch.float32)
    p_mu = torch.as_tensor(prior_mu if prior_mu is not None else population_mu, dtype=torch.float32)

    raw_t_cov = published_cov if published_cov is not None else target_block_covs
    raw_p_cov = prior_cov if prior_cov is not None else population_block_covs
    t_blocks = [torch.as_tensor(b, dtype=torch.float32) for b in raw_t_cov] if isinstance(raw_t_cov, list) else [torch.as_tensor(raw_t_cov, dtype=torch.float32)]
    p_blocks = [torch.as_tensor(b, dtype=torch.float32) for b in raw_p_cov] if isinstance(raw_p_cov, list) else [torch.as_tensor(raw_p_cov, dtype=torch.float32)]

    z_all = torch.cat([mem_t, non_t], dim=0)
    y_mia = np.concatenate([
        np.ones(mem_t.size(0), dtype=int),
        np.zeros(non_t.size(0), dtype=int),
    ])

    def _block_maha_sq(z: torch.Tensor, mu: torch.Tensor, blocks: List[torch.Tensor]) -> torch.Tensor:
        diff = z - mu.unsqueeze(0)
        block_dim = blocks[0].size(0)
        total_sq = torch.zeros(z.size(0), device=z.device)
        for b_idx, cov_b in enumerate(blocks):
            start = b_idx * block_dim
            end = start + block_dim
            d_b = diff[:, start:end]
            reg_cov = cov_b + tikhonov_eps * torch.eye(block_dim).to(z.device)
            prec_b = torch.linalg.pinv(reg_cov)
            total_sq += torch.sum(torch.matmul(d_b, prec_b) * d_b, dim=-1)
        return total_sq

    with torch.no_grad():
        d_target_sq = _block_maha_sq(z_all, t_mu, t_blocks)
        d_pop_sq = _block_maha_sq(z_all, p_mu, p_blocks)
        llr_scores = (0.5 * (d_pop_sq - d_target_sq)).cpu().numpy()

    auc = float(compute_binary_auroc(y_mia, llr_scores))
    return {"mia_auc": auc, "auc": auc}

