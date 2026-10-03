"""FedAvg (McMahan et al., 2017) baseline with Calibrated Maximum Softmax Probability (MSP).

Reference:
    McMahan et al., 'Communication-Efficient Learning of Deep Networks from Decentralized Data', AISTATS 2017.
    Hendrycks & Gimpel, 'A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks', ICLR 2017.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baselines.base import BaseBaseline, create_baseline_optimizer
from src.data.base import FlowDataset
from src.models.encoder import DeterministicEncoder
from src.models.heads import LinearClassifierHead


class FedAvgModel(nn.Module):
    """End-to-end model comprising a deterministic MLP encoder and linear classification head."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
    ):
        super().__init__()
        self.encoder = DeterministicEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            use_batchnorm=True,
            activation="relu",
        )
        self.classifier = LinearClassifierHead(latent_dim=latent_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, _, _ = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


class FedAvgBaseline(BaseBaseline):
    """FedAvg baseline with empirical cross-entropy loss and MSP open-set calibration."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedAvg",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
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
        """Execute one communication round of FedAvg."""
        self.model.eval()
        global_state = self.model.state_dict()

        client_weights: List[Dict[str, torch.Tensor]] = []
        client_sample_sizes: List[int] = []
        total_loss = 0.0
        active_clients = 0

        criterion = nn.CrossEntropyLoss()

        if not hasattr(self, "_local_model"):
            self._local_model = FedAvgModel(
                in_features=self.in_features,
                num_classes=self.num_classes,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

        local_model = self._local_model

        for c_idx, c_data in enumerate(client_datasets):
            if len(c_data) < 2:
                continue

            local_model.load_state_dict(global_state)
            local_model.train()
            optimizer = create_baseline_optimizer(local_model, lr=lr, optimizer_name=kwargs.get("optimizer_name", "adam"))
            loader = DataLoader(c_data, batch_size=batch_size, shuffle=True, drop_last=(len(c_data) > batch_size))

            client_running_loss = torch.tensor(0.0, device=self.device)
            steps = 0
            for _ in range(epochs):
                for x_b, y_b in loader:
                    if x_b.size(0) <= 1:
                        continue
                    x_b = x_b.to(self.device)
                    y_b = y_b.to(self.device)

                    optimizer.zero_grad()
                    logits, _ = local_model(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_running_loss += loss.detach()
                    steps += 1

            client_weights.append({k: v.clone() for k, v in local_model.state_dict().items()})
            client_sample_sizes.append(len(c_data))
            if steps > 0:
                total_loss += float((client_running_loss / float(steps)).item())
                active_clients += 1

        # FedAvg Aggregation: coordinate-wise weighted average
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
            # Anomaly score is 1 - max_softmax_prob
            cal_scores = (1.0 - max_p).cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(cal_scores, q_val))
        self.threshold = tau
        return tau

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict class labels and MSP anomaly scores."""
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
