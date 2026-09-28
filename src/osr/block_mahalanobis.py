"""Linear-Time Block-Diagonal Mahalanobis Nonconformity Metric and Decision Engine (Equations 19-21 & 27 / Algorithm 2).

Achieves wire-speed (<0.25 ms) edge intrusion detection and zero-day threat interception
by evaluating block-diagonal elliptic distances across B independent sub-blocks.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.aggregation.total_variance import GlobalClassProfile


class MahalanobisInferenceResult(NamedTuple):
    """Structured output for block-diagonal Mahalanobis edge decision."""

    predicted_label: int  # Known class index or unknown_label (-1)
    is_unknown: bool  # True if flagged as zero-day exploit
    candidate_class: int  # c_hat = argmin D_block(z, c)
    nonconformity_score: float  # S(x) = D_block(z, c_hat)
    conformal_threshold: float  # tau_{alpha, c_hat}
    class_distances: Dict[int, float]  # All D_block(z, c)


def compute_single_block_distance(
    z_block: torch.Tensor,
    mu_block: torch.Tensor,
    inv_cov_block: torch.Tensor,
) -> torch.Tensor:
    """Compute quadratic Mahalanobis form for a single sub-block G_b: (z - mu)^T Sigma^{-1} (z - mu).

    Args:
        z_block: Latent feature sub-vector of shape (batch_size, d_b) or (d_b,).
        mu_block: Centroid sub-vector of shape (d_b,).
        inv_cov_block: Inverted block covariance matrix of shape (d_b, d_b).

    Returns:
        Tensor of shape (batch_size,) or scalar containing the block distance.
    """
    diff = z_block - mu_block  # Shape: (..., d_b)
    # Quadratic form: diff @ inv_cov_block @ diff^T
    temp = torch.matmul(diff, inv_cov_block)  # Shape: (..., d_b)
    dist = torch.sum(temp * diff, dim=-1)  # Shape: (...)
    return dist


def compute_block_mahalanobis_distances(
    z: torch.Tensor,
    profiles: Dict[int, GlobalClassProfile],
    num_blocks: int = 4,
    block_dim: int = 16,
) -> Dict[int, torch.Tensor]:
    """Compute Block-Diagonal Mahalanobis distance D_block(z, c) for all classes (Equation 19).

    Mathematical Formulation (Equation 19):
        D_block(z, c) = sum_{b=1}^B (z[G_b] - mu_c[G_b])^T (Sigma_{c, G_b})^{-1} (z[G_b] - mu_c[G_b])

    Args:
        z: Latent query embedding of shape (batch_size, d_z) or (d_z,).
        profiles: Dict mapping class_id -> GlobalClassProfile.
        num_blocks: Number of diagonal sub-blocks B (default: 4).
        block_dim: Dimensionality of each sub-block d_b (default: 16).

    Returns:
        Dict mapping class_id -> distance tensor of shape (batch_size,) or scalar.
    """
    device = z.device
    dtype = torch.float32
    z_f = z.to(dtype=dtype, device=device)

    distances: Dict[int, torch.Tensor] = {}

    for c, profile in profiles.items():
        mu = profile.mu.to(dtype=dtype, device=device)
        total_dist = torch.zeros(z_f.shape[:-1], dtype=dtype, device=device)

        for b in range(num_blocks):
            start = b * block_dim
            end = start + block_dim
            z_b = z_f[..., start:end]
            mu_b = mu[start:end]
            inv_cov_b = profile.inv_block_covs[b].to(dtype=dtype, device=device)

            dist_b = compute_single_block_distance(z_b, mu_b, inv_cov_b)
            total_dist = total_dist + dist_b

        distances[c] = total_dist

    return distances


class WireSpeedEdgeDetector:
    """Real-Time Edge Detector implementing Algorithm 2 with finite-sample conformal guarantees.

    Complexity: O(|C_K| * d_z * d_b), runtime < 0.25 ms.
    """

    def __init__(
        self,
        frozen_encoder: nn.Module,
        global_profiles: Dict[int, GlobalClassProfile],
        unknown_label: int = -1,
        num_blocks: int = 4,
        block_dim: int = 16,
    ) -> None:
        self.encoder = frozen_encoder
        self.encoder.eval()
        self.profiles = global_profiles
        self.unknown_label = unknown_label
        self.num_blocks = num_blocks
        self.block_dim = block_dim

    @torch.no_grad()
    def predict_sample(self, x: torch.Tensor) -> MahalanobisInferenceResult:
        """Execute Algorithm 2 for a single input flow vector x.

        Lines 1-10:
            1. z = mu(x; theta_g^*)
            2. Compute D_block(z, c) for all c in C_K
            3. c_hat = argmin D_block(z, c)
            4. S(x) = D_block(z, c_hat)
            5. If S(x) > tau_{alpha, c_hat}: return Unknown
            6. Else: return c_hat
        """
        device = next(self.encoder.parameters()).device
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(device)

        # Line 1: Deterministic forward pass z = mu(x)
        # Handle dual branch or single encoder
        output = self.encoder(x)
        if isinstance(output, tuple):
            # dual branch returns (logits, latent_z, z_sample, mu, logvar) or similar
            if len(output) >= 4:
                mu = output[3]  # mu(x)
            else:
                mu = output[0]
        else:
            mu = output

        z_test = mu.squeeze(0)  # Shape: (d_z,)

        # Lines 2-4: Block-Diagonal Mahalanobis Distances
        dists = compute_block_mahalanobis_distances(
            z_test,
            self.profiles,
            num_blocks=self.num_blocks,
            block_dim=self.block_dim,
        )

        dist_dict: Dict[int, float] = {c: float(v.item()) for c, v in dists.items()}

        # Line 5: c_hat = argmin D_c
        best_class = min(dist_dict.keys(), key=lambda c: dist_dict[c])
        s_score = dist_dict[best_class]

        threshold = self.profiles[best_class].conformal_threshold

        # Line 7-10: Decision Rule (Equation 27)
        is_unknown = s_score > threshold
        pred_label = self.unknown_label if is_unknown else best_class

        return MahalanobisInferenceResult(
            predicted_label=pred_label,
            is_unknown=is_unknown,
            candidate_class=best_class,
            nonconformity_score=s_score,
            conformal_threshold=threshold,
            class_distances=dist_dict,
        )

    @torch.no_grad()
    def predict_batch(
        self, x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute Algorithm 2 on a batch of test flow vectors.

        Returns:
            (predictions, is_unknown_flags, candidate_classes, nonconformity_scores)
        """
        device = next(self.encoder.parameters()).device
        x = x.to(device)

        output = self.encoder(x)
        if isinstance(output, tuple):
            if len(output) >= 4:
                z_batch = output[3]
            else:
                z_batch = output[0]
        else:
            z_batch = output

        dists = compute_block_mahalanobis_distances(
            z_batch,
            self.profiles,
            num_blocks=self.num_blocks,
            block_dim=self.block_dim,
        )

        class_keys = list(self.profiles.keys())
        # Stack distances into (N, |C_K|)
        dist_matrix = torch.stack([dists[c] for c in class_keys], dim=1)  # (N, C)
        best_indices = torch.argmin(dist_matrix, dim=1)  # (N,)

        thresholds = torch.tensor(
            [self.profiles[c].conformal_threshold for c in class_keys],
            dtype=torch.float32,
            device=device,
        )

        preds = []
        is_unknown_list = []
        cand_list = []
        scores_list = []

        for i in range(len(x)):
            best_idx = int(best_indices[i].item())
            cand_c = class_keys[best_idx]
            score = float(dist_matrix[i, best_idx].item())
            thresh = float(thresholds[best_idx].item())

            is_unk = score > thresh
            pred = self.unknown_label if is_unk else cand_c

            preds.append(pred)
            is_unknown_list.append(is_unk)
            cand_list.append(cand_c)
            scores_list.append(score)

        return (
            np.array(preds, dtype=np.int64),
            np.array(is_unknown_list, dtype=bool),
            np.array(cand_list, dtype=np.int64),
            np.array(scores_list, dtype=np.float32),
        )
