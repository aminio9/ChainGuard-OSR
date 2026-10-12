"""Vanilla Mahalanobis Distance OSR baseline (Lee et al., NeurIPS 2018) adapted to FL.

Reference:
    Lee et al., 'A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks',
    Advances in Neural Information Processing Systems (NeurIPS), 2018.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.baselines.base import BaseBaseline
from src.baselines.fedavg.fedavg import FedAvgBaseline
from src.data.base import FlowDataset


class VanillaMahalanobisBaseline(BaseBaseline):
    """Vanilla Mahalanobis Distance detector with shared pooled covariance."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        regularization: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="Vanilla-Mahalanobis",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.regularization = regularization
        self.fedavg = FedAvgBaseline(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.class_means: Dict[int, torch.Tensor] = {}
        self.inv_cov: Optional[torch.Tensor] = None

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one round of federated training using FedAvg backbone."""
        return self.fedavg.fit_round(
            round_idx=round_idx,
            client_datasets=client_datasets,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            **kwargs,
        )

    def calibrate(self, cal_dataset: FlowDataset, alpha: float = 0.05) -> float:
        """Estimate class-conditional Gaussian means and pooled covariance matrix on calibration set."""
        self.fedavg.model.eval()
        x_cal = cal_dataset.features.to(self.device)
        y_cal = cal_dataset.labels.to(self.device)

        with torch.no_grad():
            _, z_cal = self.fedavg.model(x_cal)

        n_total = z_cal.size(0)
        pooled_cov = torch.zeros((self.latent_dim, self.latent_dim), device=self.device)

        # 1. Compute empirical class means
        for c in range(self.num_classes):
            idx_c = torch.where(y_cal == c)[0]
            if len(idx_c) > 0:
                mean_c = z_cal[idx_c].mean(dim=0)
                self.class_means[c] = mean_c
                # Accumulate centered outer products
                diff = z_cal[idx_c] - mean_c
                pooled_cov += torch.matmul(diff.t(), diff)
            else:
                self.class_means[c] = torch.zeros(self.latent_dim, device=self.device)

        # Normalize and regularize pooled covariance
        pooled_cov = pooled_cov / max(1.0, float(n_total))
        pooled_cov += self.regularization * torch.eye(self.latent_dim, device="cpu").to(self.device)
        self.inv_cov = torch.linalg.pinv(pooled_cov)

        # 2. Compute minimum Mahalanobis distances on calibration set
        cal_dists = self._compute_mahalanobis_distances(z_cal)
        min_cal, _ = torch.min(cal_dists, dim=-1)
        cal_scores = min_cal.cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(cal_scores, q_val))
        self.threshold = tau
        return tau

    def _compute_mahalanobis_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance from each sample to each class mean."""
        assert self.inv_cov is not None, "Baseline must be calibrated before distance evaluation"
        batch_size = z.size(0)
        dists = torch.zeros((batch_size, self.num_classes), device=self.device)

        for c in range(self.num_classes):
            diff = z - self.class_means[c].unsqueeze(0)  # (B, d)
            # D_M^2 = diff * inv_cov * diff^T
            left = torch.matmul(diff, self.inv_cov)
            d_sq = torch.sum(left * diff, dim=-1)
            dists[:, c] = torch.sqrt(torch.clamp(d_sq, min=0.0))

        return dists

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and Mahalanobis anomaly scores."""
        self.fedavg.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, z = self.fedavg.model(x)
            dists = self._compute_mahalanobis_distances(z)
            min_dist, cand_class = torch.min(dists, dim=-1)
            anomaly_scores = min_dist.cpu().numpy()
            cand_classes = cand_class.cpu().numpy()

        preds = np.where(anomaly_scores > self.threshold, self.num_classes, cand_classes)
        return preds, anomaly_scores, cand_classes
