"""Free Helmholtz Energy-based OSR baseline adapted to Federated Learning.

Reference:
    Liu et al., 'Energy-based Out-of-distribution Detection', NeurIPS 2020.
    Souza et al., 'Open-Set Energy-based Flow Classifier for Network Intrusion Detection', Computers & Security 2025.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.baselines.base import BaseBaseline
from src.baselines.fedavg.fedavg import FedAvgBaseline
from src.data.base import FlowDataset
from src.osr.energy_ood import compute_free_energy_score


class EnergyBaseline(BaseBaseline):
    """Energy-based OSR baseline using Free Helmholtz Energy score."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="Energy-OOD",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.temperature = temperature
        self.fedavg = FedAvgBaseline(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

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
        """Calibrate rejection threshold using Free Helmholtz Energy."""
        self.fedavg.model.eval()
        x_cal = cal_dataset.features.to(self.device)
        with torch.no_grad():
            logits, _ = self.fedavg.model(x_cal)
            energy_scores = compute_free_energy_score(logits, temperature=self.temperature).cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(energy_scores, q_val))
        self.threshold = tau
        return tau

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and energy anomaly scores."""
        self.fedavg.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, _ = self.fedavg.model(x)
            energy_scores = compute_free_energy_score(logits, temperature=self.temperature).cpu().numpy()
            cand_classes = torch.argmax(logits, dim=-1).cpu().numpy()

        preds = np.where(energy_scores > self.threshold, self.num_classes, cand_classes)
        return preds, energy_scores, cand_classes
