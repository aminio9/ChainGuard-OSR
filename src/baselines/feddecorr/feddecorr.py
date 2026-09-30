"""FedDecorr (Shi et al., ICLR 2023) baseline for Non-IID representation decorrelation.

Reference:
    Shi et al., 'FedDecorr: An Extreme and Agnostic Decoupled Representation Learning Method for Non-IID Federated Learning',
    International Conference on Learning Representations (ICLR), 2023.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baselines.base import BaseBaseline, create_baseline_optimizer
from src.baselines.fedavg.fedavg import FedAvgModel
from src.data.base import FlowDataset


def compute_decorrelation_loss(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute representation correlation penalty to prevent dimensional collapse.

    Mathematical formulation:
        C = (1 / B) * Z_tilde^T * Z_tilde
        L_decorr = (1 / (d * (d - 1))) * sum_{i != j} C_{ij}^2
    """
    b, d = z.size()
    if b <= 1 or d <= 1:
        return torch.tensor(0.0, device=z.device)

    # Center and normalize features
    z_mean = z.mean(dim=0, keepdim=True)
    z_std = z.std(dim=0, keepdim=True) + eps
    z_norm = (z - z_mean) / z_std

    # Cross-correlation matrix
    corr = torch.matmul(z_norm.t(), z_norm) / float(b)

    # Zero out diagonal
    diag = torch.diag(torch.diag(corr))
    off_diag = corr - diag

    decorr_loss = torch.sum(off_diag ** 2) / float(d * (d - 1))
    return decorr_loss


class FedDecorrBaseline(BaseBaseline):
    """FedDecorr baseline regularizing representation cross-correlation."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        beta_decorr: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedDecorr",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.beta_decorr = beta_decorr
        self.model = FedAvgModel(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one communication round with decorrelation loss."""
        self.model.eval()
        global_state = copy.deepcopy(self.model.state_dict())

        client_weights: List[Dict[str, torch.Tensor]] = []
        client_sample_sizes: List[int] = []
        total_loss = 0.0
        active_clients = 0

        criterion = nn.CrossEntropyLoss()

        for c_idx, c_data in enumerate(client_datasets):
            if len(c_data) < 2:
                continue

            local_model = copy.deepcopy(self.model)
            local_model.train()
            optimizer = create_baseline_optimizer(local_model, lr=lr, optimizer_name=kwargs.get("optimizer_name", "adam"))
            loader = DataLoader(c_data, batch_size=batch_size, shuffle=True, drop_last=(len(c_data) > batch_size))

            client_loss = 0.0
            steps = 0
            for _ in range(epochs):
                for x_b, y_b in loader:
                    if x_b.size(0) <= 1:
                        continue
                    x_b = x_b.to(self.device)
                    y_b = y_b.to(self.device)

                    optimizer.zero_grad()
                    logits, z = local_model(x_b)
                    loss_ce = criterion(logits, y_b)
                    loss_decorr = compute_decorrelation_loss(z)
                    loss = loss_ce + self.beta_decorr * loss_decorr

                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_loss += loss.item()
                    steps += 1

            client_weights.append(local_model.state_dict())
            client_sample_sizes.append(len(c_data))
            if steps > 0:
                total_loss += client_loss / steps
                active_clients += 1

        # Coordinate-wise weighted averaging
        total_samples = sum(client_sample_sizes)
        if total_samples > 0 and len(client_weights) > 0:
            new_state = {}
            for key in global_state.keys():
                weighted_sum = sum(
                    cw[key].float() * (cs / total_samples)
                    for cw, cs in zip(client_weights, client_sample_sizes)
                )
                new_state[key] = weighted_sum.to(global_state[key].dtype)
            self.model.load_state_dict(new_state)

        avg_loss = total_loss / max(1, active_clients)
        return {"round": round_idx, "train_loss": avg_loss}

    def calibrate(self, cal_dataset: FlowDataset, alpha: float = 0.05) -> float:
        """Calibrate rejection threshold using Maximum Softmax Probability (MSP)."""
        self.model.eval()
        x_cal = cal_dataset.features.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x_cal)
            probs = torch.softmax(logits, dim=-1)
            max_p, _ = torch.max(probs, dim=-1)
            cal_scores = (1.0 - max_p).cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(cal_scores, q_val))
        self.threshold = tau
        return tau

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and anomaly scores."""
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            max_p, cand_class = torch.max(probs, dim=-1)
            anomaly_scores = (1.0 - max_p).cpu().numpy()
            cand_classes = cand_class.cpu().numpy()

        preds = np.where(anomaly_scores > self.threshold, self.num_classes, cand_classes)
        return preds, anomaly_scores, cand_classes
