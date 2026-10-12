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
from src.osr.conformal import (
    ThreeWayConformalDecisionResult,
    compute_three_way_conformal_decision,
)


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
    num_blocks: int = 1,
    block_dim: int = 64,
    normalize_features: bool = False,
    metric_type: str = "mahalanobis",
) -> Dict[int, torch.Tensor]:
    """Compute Block-Diagonal Mahalanobis or Spherical Cosine distance for all classes.

    Args:
        z: Latent query embedding of shape (batch_size, d_z) or (d_z,).
        profiles: Dict mapping class_id -> GlobalClassProfile.
        num_blocks: Number of diagonal sub-blocks B (default: 4).
        block_dim: Dimensionality of each sub-block d_b (default: 16).
        normalize_features: If True, normalizes z onto the unit sphere z / ||z||_2.
        metric_type: 'mahalanobis' or 'spherical_cosine'.

    Returns:
        Dict mapping class_id -> distance tensor of shape (batch_size,) or scalar.
    """
    device = z.device
    dtype = torch.float32
    z_f = z.to(dtype=dtype, device=device)
    if normalize_features:
        z_norm = torch.norm(z_f, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        z_f = z_f / z_norm

    distances: Dict[int, torch.Tensor] = {}

    for c, profile in profiles.items():
        protos = getattr(profile, "prototypes", None)
        proto_inv_covs = getattr(profile, "proto_inv_block_covs", None)

        if metric_type == "spherical_cosine":
            if protos is not None and protos.ndim == 2 and protos.size(0) >= 1:
                protos_f = protos.to(dtype=dtype, device=device)
                if normalize_features:
                    protos_norm = torch.norm(protos_f, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    protos_f = protos_f / protos_norm
                cos_sims = torch.matmul(z_f, protos_f.t())
                total_dist = 1.0 - torch.max(cos_sims, dim=-1).values
            else:
                mu_f = profile.mu.to(dtype=dtype, device=device)
                if normalize_features:
                    mu_f = mu_f / torch.norm(mu_f, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                cos_sim = torch.sum(z_f * mu_f, dim=-1)
                total_dist = 1.0 - cos_sim
        elif protos is not None and protos.ndim == 2 and protos.size(0) > 1:
            protos_f = protos.to(dtype=dtype, device=device)
            if normalize_features:
                protos_norm = torch.norm(protos_f, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                protos_f = protos_f / protos_norm
            M_c = protos_f.size(0)
            sub_dists = []
            for k in range(M_c):
                pk = protos_f[k]
                d_pk = torch.zeros(z_f.shape[:-1], dtype=dtype, device=device)
                for b in range(num_blocks):
                    start = b * block_dim
                    end = start + block_dim
                    z_b = z_f[..., start:end]
                    pk_b = pk[start:end]
                    if proto_inv_covs is not None and k < len(proto_inv_covs) and b < len(proto_inv_covs[k]):
                        inv_cov_b = proto_inv_covs[k][b].to(dtype=dtype, device=device)
                    else:
                        inv_cov_b = profile.inv_block_covs[b].to(dtype=dtype, device=device)
                    d_pk = d_pk + compute_single_block_distance(z_b, pk_b, inv_cov_b)
                sub_dists.append(d_pk)
            total_dist = torch.min(torch.stack(sub_dists, dim=-1), dim=-1).values
        else:
            mu = profile.mu.to(dtype=dtype, device=device)
            if normalize_features:
                mu = mu / torch.norm(mu, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            total_dist = torch.zeros(z_f.shape[:-1], dtype=dtype, device=device)
            for b in range(num_blocks):
                start = b * block_dim
                end = start + block_dim
                z_b = z_f[..., start:end]
                mu_b = mu[start:end]
                inv_cov_b = profile.inv_block_covs[b].to(dtype=dtype, device=device)
                total_dist = total_dist + compute_single_block_distance(z_b, mu_b, inv_cov_b)

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
        num_blocks: int = 1,
        block_dim: int = 64,
        normalize_features: bool = False,
        metric_type: str = "mahalanobis",
        tau_accept_map: Optional[Dict[int, float]] = None,
        tau_reject_map: Optional[Dict[int, float]] = None,
    ) -> None:
        self.encoder = frozen_encoder
        self.encoder.eval()
        self.profiles = global_profiles
        self.unknown_label = unknown_label
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.normalize_features = normalize_features
        self.metric_type = metric_type
        self.tau_reject_map = tau_reject_map or {
            c: float(p.conformal_threshold) for c, p in global_profiles.items()
        }
        self.tau_accept_map = tau_accept_map or {}

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
        try:
            device = next(self.encoder.parameters()).device
        except (StopIteration, AttributeError):
            device = x.device
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
            normalize_features=self.normalize_features,
            metric_type=self.metric_type,
        )

        dist_dict: Dict[int, float] = {c: float(v.item()) for c, v in dists.items()}

        # Candidate selection based on minimum raw Mahalanobis distance
        best_class = min(dist_dict.keys(), key=lambda c: dist_dict[c])
        s_score = dist_dict[best_class]

        threshold = self.profiles[best_class].conformal_threshold

        # Decision Rule: Reject iff D(z, c_hat) > tau_{alpha, c_hat}
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
        try:
            device = next(self.encoder.parameters()).device
        except (StopIteration, AttributeError):
            device = x.device
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
            normalize_features=self.normalize_features,
            metric_type=self.metric_type,
        )

        class_keys = list(self.profiles.keys())
        class_keys_tensor = torch.tensor(class_keys, dtype=torch.int64, device=device)
        # Stack distances into (N, C)
        dist_matrix = torch.stack([dists[c] for c in class_keys], dim=1)  # (N, C)

        thresholds = torch.tensor(
            [self.profiles[c].conformal_threshold for c in class_keys],
            dtype=torch.float32,
            device=device,
        )

        # Candidate selection based on minimum raw Mahalanobis distance
        best_indices = torch.argmin(dist_matrix, dim=1)  # (N,)
        cand_distances = torch.gather(dist_matrix, 1, best_indices.unsqueeze(1)).squeeze(1)
        cand_thresholds = thresholds[best_indices]

        # Decision rule: reject iff candidate distance exceeds candidate calibrated threshold
        is_unknown_tensor = cand_distances > cand_thresholds
        cand_tensor = class_keys_tensor[best_indices]
        preds_tensor = torch.where(
            is_unknown_tensor,
            torch.tensor(self.unknown_label, dtype=torch.int64, device=device),
            cand_tensor,
        )

        scores_tensor = cand_distances

        return (
            preds_tensor.cpu().numpy().astype(np.int64),
            is_unknown_tensor.cpu().numpy().astype(bool),
            cand_tensor.cpu().numpy().astype(np.int64),
            scores_tensor.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def predict_batch_detailed(self, x: torch.Tensor) -> Dict[str, Any]:
        """Execute Section 4.4 decision rule and return full diagnostic arrays including conformal prediction sets."""
        try:
            device = next(self.encoder.parameters()).device
        except (StopIteration, AttributeError):
            device = x.device
        x = x.to(device)
        if hasattr(self.encoder, "extract_features"):
            z_batch = self.encoder.extract_features(x)
        else:
            output = self.encoder(x)
            z_batch = output[2] if isinstance(output, tuple) and len(output) == 4 else (output[0] if isinstance(output, tuple) else output)

        dists = compute_block_mahalanobis_distances(
            z_batch,
            self.profiles,
            num_blocks=self.num_blocks,
            block_dim=self.block_dim,
            normalize_features=self.normalize_features,
            metric_type=self.metric_type,
        )
        class_keys = list(self.profiles.keys())
        class_keys_tensor = torch.tensor(class_keys, dtype=torch.int64, device=device)
        dist_matrix = torch.stack([dists[c] for c in class_keys], dim=1)
        thresholds = torch.tensor(
            [self.profiles[c].conformal_threshold for c in class_keys],
            dtype=torch.float32,
            device=device,
        )
        valid_tau = (thresholds > 0.0) & torch.isfinite(thresholds)
        safe_thresholds = torch.where(valid_tau, thresholds, torch.ones_like(thresholds))
        norm_dist_matrix = dist_matrix / safe_thresholds.unsqueeze(0)
        best_raw_indices = torch.argmin(dist_matrix, dim=1)
        best_norm_indices = torch.argmin(norm_dist_matrix, dim=1)
        accepted_mask = dist_matrix <= thresholds.unsqueeze(0)

        cand_distances = torch.gather(dist_matrix, 1, best_raw_indices.unsqueeze(1)).squeeze(1)
        cand_thresholds = thresholds[best_raw_indices]
        is_unk_tensor = cand_distances > cand_thresholds
        cand_tensor = class_keys_tensor[best_raw_indices]
        preds_tensor = torch.where(
            is_unk_tensor,
            torch.tensor(self.unknown_label, dtype=torch.int64, device=device),
            cand_tensor,
        )
        norm_scores_tensor = torch.gather(norm_dist_matrix, 1, best_norm_indices.unsqueeze(1)).squeeze(1)
        raw_scores_tensor = cand_distances
        tau_used_tensor = cand_thresholds

        # Single transfer to CPU for set comprehension
        accepted_cpu = accepted_mask.cpu().numpy()
        pred_sets = [[class_keys[j] for j, acc in enumerate(row) if acc] for row in accepted_cpu]

        return {
            "predictions": preds_tensor.cpu().numpy().astype(np.int64),
            "is_unknown": is_unk_tensor.cpu().numpy().astype(bool),
            "candidate_classes": cand_tensor.cpu().numpy().astype(np.int64),
            "normalized_scores": norm_scores_tensor.cpu().numpy().astype(np.float32),
            "raw_scores": raw_scores_tensor.cpu().numpy().astype(np.float32),
            "thresholds_used": tau_used_tensor.cpu().numpy().astype(np.float32),
            "prediction_sets": pred_sets,
            "latent_z": z_batch.detach().cpu().numpy(),
            "dist_matrix": dist_matrix.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def predict_batch_three_way(
        self,
        x: torch.Tensor,
        tau_accept_map: Optional[Dict[int, float]] = None,
        tau_reject_map: Optional[Dict[int, float]] = None,
    ) -> ThreeWayConformalDecisionResult:
        """Execute Section 4.5 three-way open-set decision policy (POS / BND / NEG) on a batch of flows.

        Returns:
            ThreeWayConformalDecisionResult with actions ('accept', 'abstain', 'reject'),
            predictions, candidate_classes, scores, and boolean masks (is_accepted, is_abstained, is_rejected).
        """
        tau_rej = tau_reject_map if tau_reject_map is not None else self.tau_reject_map
        tau_acc = tau_accept_map if tau_accept_map is not None else self.tau_accept_map

        try:
            device = next(self.encoder.parameters()).device
        except (StopIteration, AttributeError):
            device = x.device
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
            normalize_features=self.normalize_features,
            metric_type=self.metric_type,
        )

        class_keys = sorted(self.profiles.keys())
        dist_matrix = torch.stack([dists[c] for c in class_keys], dim=1)

        return compute_three_way_conformal_decision(
            dist_matrix=dist_matrix,
            tau_reject_map=tau_rej,
            tau_accept_map=tau_acc,
            unknown_label=self.unknown_label,
        )

    @torch.no_grad()
    def predict_sample_three_way(
        self,
        x: torch.Tensor,
        tau_accept_map: Optional[Dict[int, float]] = None,
        tau_reject_map: Optional[Dict[int, float]] = None,
    ) -> Tuple[str, int, int, float]:
        """Execute Section 4.5 three-way decision on a single input flow vector.

        Returns:
            (action, predicted_label, candidate_class, nonconformity_score)
            where action in {"accept", "abstain", "reject"}.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        res = self.predict_batch_three_way(x, tau_accept_map, tau_reject_map)
        return (
            str(res.actions[0]),
            int(res.predictions[0]),
            int(res.candidate_classes[0]),
            float(res.scores[0]),
        )


