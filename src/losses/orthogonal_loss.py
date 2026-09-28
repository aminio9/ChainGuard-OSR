"""Fixed/Penalized Orthogonal Anchor Regularizer (Formula 7)."""

from __future__ import annotations

import torch


def orthogonal_anchor_loss(
    weight_matrix: torch.Tensor,
    r_max: float = 2.0,
    lambda_r: float = 0.01,
) -> torch.Tensor:
    """Enforce pairwise orthogonality on classifier weight vectors and prevent norm explosion.

    Mathematical Formulation (Formula 7):
        L_anchor = (1 / (C * (C - 1))) * sum_{i != j} | cos(w_i, w_j) | + lambda_r * sum_c max(0, ||w_c||_2 - r_max)

    Args:
        weight_matrix: Classifier weight tensor of shape (num_classes, latent_dim).
        r_max: Maximum allowed norm radius for each class vector.
        lambda_r: Penalty coefficient for radius violation.

    Returns:
        Scalar orthogonality penalty tensor.
    """
    c = weight_matrix.size(0)
    if c <= 1:
        return torch.tensor(0.0, device=weight_matrix.device)

    # 1. Normalize row vectors to unit sphere
    norms = torch.norm(weight_matrix, p=2, dim=1, keepdim=True) + 1e-7
    normalized_w = weight_matrix / norms

    # 2. Pairwise Cosine Similarity: Shape (C, C)
    cosine_sim = torch.matmul(normalized_w, normalized_w.t())

    # 3. Off-diagonal absolute values (exclude diagonal where cos(w_i, w_i) = 1)
    identity = torch.eye(c, device=weight_matrix.device)
    off_diag_abs = (cosine_sim - identity).abs()
    ortho_term = off_diag_abs.sum() / (c * (c - 1))

    # 4. Radius constraint penalty
    radius_violation = torch.clamp(norms.squeeze(1) - r_max, min=0.0)
    norm_penalty = lambda_r * radius_violation.sum()

    return ortho_term + norm_penalty
