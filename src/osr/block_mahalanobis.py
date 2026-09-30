"""Linear-Time Block-Diagonal Mahalanobis Nonconformity Metric and Decision Engine (Equations 19-21 & 27 / Algorithm 2).

Achieves wire-speed (<0.25 ms) edge intrusion detection and zero-day threat interception
by evaluating block-diagonal elliptic distances across B independent sub-blocks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.aggregation.total_variance import GlobalClassProfile


class MahalanobisInferenceResult(NamedTuple):
    """Structured output for block-diagonal Mahalanobis edge decision."""

    predicted_label: int  # Known class index or unknown_label (-1)
    is_unknown: bool  # True if flagged as zero-day exploit
    candidate_class: int  # c_hat = argmin D_block(z, c) / tau_{alpha, c}
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
    return torch.clamp(dist, min=0.0)


def compute_block_mahalanobis_distances(
    z: torch.Tensor,
    profiles: Dict[int, GlobalClassProfile],
    num_blocks: int = 4,
    block_dim: int = 16,
) -> Dict[int, torch.Tensor]:
    """Compute Block-Diagonal Mahalanobis distance D_block(z, c) for all classes (Section 4.3, Eq. eq:block_mahalanobis).

    Mathematical Formulation (Eq. eq:block_mahalanobis):
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
    """Real-Time Edge Detector implementing Section 4.4 with finite-sample conformal guarantees.

    Complexity: O(C * d_z * d_b), runtime < 0.25 ms.
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
        """Execute Section 4.4 decision rule for a single input flow vector x.

        Steps (Section 4.4, Eq. eq:osr_decision_rule):
            1. z_test = f_{theta^*}(x_test) = m_{theta^*}(x_test)
            2. Compute D_block(z_test, c) for all c in Y_K
            3. hat{y}(x_test) = argmin_{c in Y_K} D_block(z_test, c) / tau_{alpha, c}
            4. S(x_test) = min_{c in Y_K} D_block(z_test, c) / tau_{alpha, c}
                         = D_block(z_test, hat{y}(x_test)) / tau_{alpha, hat{y}(x_test)}
            5. Reject as Unknown iff S(x_test) > 1 (equivalently D_block(z_test, hat{y}(x_test)) > tau_{alpha, hat{y}(x_test)},
               i.e., C_{1-alpha}(x_test) == empty since 0 < tau_{alpha, c} < inf for all c in Y_K)
            6. Else: return hat{y}(x_test)
        """
        device = next(self.encoder.parameters()).device
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(device)

        # Line 1: Deterministic forward pass z = mu(x)
        if hasattr(self.encoder, "extract_features"):
            mu = self.encoder.extract_features(x)
        else:
            output = self.encoder(x)
            if isinstance(output, tuple):
                # VIBClassifierNetwork returns (logits, z, mu, logvar)
                if len(output) == 4:
                    mu = output[2] if output[2] is not None else output[1]
                elif len(output) >= 5:
                    mu = output[3]
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

        # Threshold-normalized candidate selection: c_hat = argmin_c D_block(z, c) / tau_{alpha, c}
        # Ensures D_block(z, c_hat) > tau_{alpha, c_hat} iff D_block(z, c) > tau_{alpha, c} for all c in Y_K
        best_class = min(
            dist_dict.keys(),
            key=lambda c: (
                dist_dict[c] / self.profiles[c].conformal_threshold
                if (self.profiles[c].conformal_threshold > 0.0 and math.isfinite(self.profiles[c].conformal_threshold))
                else dist_dict[c]
            ),
        )
        s_score = dist_dict[best_class]

        threshold = self.profiles[best_class].conformal_threshold

        # Decision Rule (Eq. eq:osr_decision_rule): Reject iff D(z, c_hat) > tau_{alpha, c_hat} (i.e. C_{1-alpha}(x) == empty)
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
        """Execute Section 4.4 decision rule on a batch of test flow vectors.

        Returns:
            (predictions, is_unknown_flags, candidate_classes, nonconformity_scores)
        """
        device = next(self.encoder.parameters()).device
        x = x.to(device)

        if hasattr(self.encoder, "extract_features"):
            z_batch = self.encoder.extract_features(x)
        else:
            output = self.encoder(x)
            if isinstance(output, tuple):
                if len(output) == 4:
                    z_batch = output[2] if output[2] is not None else output[1]
                elif len(output) >= 5:
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
        # Stack distances into (N, C)
        dist_matrix = torch.stack([dists[c] for c in class_keys], dim=1)  # (N, C)

        thresholds = torch.tensor(
            [self.profiles[c].conformal_threshold for c in class_keys],
            dtype=torch.float32,
            device=device,
        )

        # Threshold-normalized distance ratio D(z, c) / tau_{alpha, c}
        valid_tau = (thresholds > 0.0) & torch.isfinite(thresholds)
        safe_thresholds = torch.where(valid_tau, thresholds, torch.ones_like(thresholds))
        norm_dist_matrix = dist_matrix / safe_thresholds.unsqueeze(0)
        best_indices = torch.argmin(norm_dist_matrix, dim=1)  # (N,)

        # Set-based acceptance matrix: accepted_mask[i, j] is True iff D_block(z_i, c_j) <= tau_{alpha, c_j}
        accepted_mask = dist_matrix <= thresholds.unsqueeze(0)

        preds = []
        is_unknown_list = []
        cand_list = []
        scores_list = []

        for i in range(len(x)):
            best_idx = int(best_indices[i].item())
            cand_c = class_keys[best_idx]
            norm_score = float(norm_dist_matrix[i, best_idx].item())
            raw_score = float(dist_matrix[i, best_idx].item())
            tau_val = float(thresholds[best_idx].item())

            # Set-based rejection rule: reject iff no known class is accepted (C_{1-alpha}(x_i) == empty)
            has_accepted_class = bool(torch.any(accepted_mask[i]).item())
            is_unk = not has_accepted_class
            pred = self.unknown_label if is_unk else cand_c

            preds.append(pred)
            is_unknown_list.append(is_unk)
            cand_list.append(cand_c)
            scores_list.append(norm_score if (tau_val > 0.0 and math.isfinite(tau_val)) else raw_score)

        return (
            np.array(preds, dtype=np.int64),
            np.array(is_unknown_list, dtype=bool),
            np.array(cand_list, dtype=np.int64),
            np.array(scores_list, dtype=np.float32),
        )

    @torch.no_grad()
    def predict_batch_detailed(self, x: torch.Tensor) -> Dict[str, Any]:
        """Execute Section 4.4 decision rule and return full diagnostic arrays including conformal prediction sets."""
        device = next(self.encoder.parameters()).device
        x = x.to(device)
        if hasattr(self.encoder, "extract_features"):
            z_batch = self.encoder.extract_features(x)
        else:
            output = self.encoder(x)
            z_batch = output[2] if isinstance(output, tuple) and len(output) == 4 else (output[0] if isinstance(output, tuple) else output)

        dists = compute_block_mahalanobis_distances(
            z_batch, self.profiles, num_blocks=self.num_blocks, block_dim=self.block_dim
        )
        class_keys = list(self.profiles.keys())
        dist_matrix = torch.stack([dists[c] for c in class_keys], dim=1)
        thresholds = torch.tensor(
            [self.profiles[c].conformal_threshold for c in class_keys],
            dtype=torch.float32,
            device=device,
        )
        valid_tau = (thresholds > 0.0) & torch.isfinite(thresholds)
        safe_thresholds = torch.where(valid_tau, thresholds, torch.ones_like(thresholds))
        norm_dist_matrix = dist_matrix / safe_thresholds.unsqueeze(0)
        best_indices = torch.argmin(norm_dist_matrix, dim=1)
        accepted_mask = dist_matrix <= thresholds.unsqueeze(0)

        preds, is_unk_list, cand_list, norm_scores, raw_scores, tau_used, pred_sets = [], [], [], [], [], [], []
        for i in range(len(x)):
            best_idx = int(best_indices[i].item())
            cand_c = class_keys[best_idx]
            pset = [class_keys[j] for j in range(len(class_keys)) if bool(accepted_mask[i, j].item())]
            is_unk = len(pset) == 0
            preds.append(self.unknown_label if is_unk else cand_c)
            is_unk_list.append(is_unk)
            cand_list.append(cand_c)
            norm_scores.append(float(norm_dist_matrix[i, best_idx].item()))
            raw_scores.append(float(dist_matrix[i, best_idx].item()))
            tau_used.append(float(thresholds[best_idx].item()))
            pred_sets.append(pset)

        return {
            "predictions": np.array(preds, dtype=np.int64),
            "is_unknown": np.array(is_unk_list, dtype=bool),
            "candidate_classes": np.array(cand_list, dtype=np.int64),
            "normalized_scores": np.array(norm_scores, dtype=np.float32),
            "raw_scores": np.array(raw_scores, dtype=np.float32),
            "thresholds_used": np.array(tau_used, dtype=np.float32),
            "prediction_sets": pred_sets,
            "latent_z": z_batch.detach().cpu().numpy(),
        }

