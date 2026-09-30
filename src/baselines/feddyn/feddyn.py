"""FedDyn (Acar et al., ICLR 2021) baseline with Dynamic Regularization.

Reference:
    Acar et al., 'Federated Learning Based on Dynamic Regularization',
    International Conference on Learning Representations (ICLR), 2021.
    Official GitHub: https://github.com/alpemreacar/FedDyn

Mathematical Formulation:
    1. Local Objective: L_i(theta) - <h_i^{t-1}, theta> + (alpha_dyn / 2) * ||theta - theta^{t-1}||^2
    2. Local Dual Update: h_i^t = h_i^{t-1} - alpha_dyn * (theta_i^t - theta^{t-1})
    3. Server State Update: h^t = h^{t-1} - (alpha_dyn / N) * sum_{i in S_t} (theta_i^t - theta^{t-1})
    4. Server Aggregation: theta^t = (1 / |S_t|) * sum_{i in S_t} theta_i^t - (1 / alpha_dyn) * h^t
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
from src.osr.conformal import compute_finite_sample_quantile


class FedDynBaseline(BaseBaseline):
    """FedDyn baseline aligning stationary points across Non-IID clients via exact dynamic regularization."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        alpha_dyn: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedDyn",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.alpha_dyn = alpha_dyn
        self.model = FedAvgModel(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Server dual state h^t
        self.server_h: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data, device=self.device)
            for name, param in self.model.named_parameters()
        }
        # Client dual states h_i^t
        self.client_h: Dict[int, Dict[str, torch.Tensor]] = {}

    def _get_client_h(self, client_idx: int) -> Dict[str, torch.Tensor]:
        if client_idx not in self.client_h:
            self.client_h[client_idx] = {
                name: torch.zeros_like(param.data, device=self.device)
                for name, param in self.model.named_parameters()
            }
        return self.client_h[client_idx]

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one communication round of FedDyn."""
        self.model.eval()
        global_state = copy.deepcopy(self.model.state_dict())
        global_params = {name: param.data.clone() for name, param in self.model.named_parameters()}

        client_weights: List[Dict[str, torch.Tensor]] = []
        client_sample_sizes: List[int] = []
        delta_theta_sum: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(val, device=self.device) for name, val in self.server_h.items()
        }

        total_loss = 0.0
        active_clients = 0
        num_total_clients = max(1, len(client_datasets))
        criterion = nn.CrossEntropyLoss()

        for c_idx, c_data in enumerate(client_datasets):
            if len(c_data) < 2:
                continue

            local_model = copy.deepcopy(self.model)
            local_model.train()
            h_i = self._get_client_h(c_idx)
            optimizer = create_baseline_optimizer(
                local_model,
                lr=lr,
                optimizer_name=kwargs.get("optimizer_name", "adam"),
            )
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
                    logits, _ = local_model(x_b)
                    loss_ce = criterion(logits, y_b)

                    # Dynamic regularizer: - <h_i, theta> + (alpha_dyn / 2) * ||theta - theta_global||^2
                    linear_penalty = 0.0
                    quad_penalty = 0.0
                    for name, param in local_model.named_parameters():
                        linear_penalty += torch.sum(param * h_i[name])
                        quad_penalty += torch.sum((param - global_params[name]) ** 2)

                    loss = loss_ce - linear_penalty + 0.5 * self.alpha_dyn * quad_penalty
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_loss += loss.item()
                    steps += 1

            if steps > 0:
                with torch.no_grad():
                    for name, param in local_model.named_parameters():
                        diff = param.data - global_params[name]
                        h_i[name] = h_i[name] - self.alpha_dyn * diff
                        delta_theta_sum[name] += diff

                client_weights.append(local_model.state_dict())
                client_sample_sizes.append(len(c_data))
                total_loss += client_loss / steps
                active_clients += 1

        if active_clients > 0 and len(client_weights) > 0:
            # Update server state h^t
            for name in self.server_h:
                self.server_h[name] = self.server_h[name] - (self.alpha_dyn / float(num_total_clients)) * delta_theta_sum[name]

            new_state = {}
            total_samples = sum(client_sample_sizes)
            for key in global_state.keys():
                if key in self.server_h:
                    avg_param = sum(cw[key].float() for cw in client_weights) / float(len(client_weights))
                    corrected = avg_param - (1.0 / self.alpha_dyn) * self.server_h[key]
                    new_state[key] = corrected.to(global_state[key].dtype)
                else:
                    # Non-trainable buffers (e.g. BatchNorm running stats)
                    weighted_sum = sum(
                        cw[key].float() * (cs / total_samples)
                        for cw, cs in zip(client_weights, client_sample_sizes)
                    )
                    new_state[key] = weighted_sum.to(global_state[key].dtype)

            self.model.load_state_dict(new_state)

        avg_loss = total_loss / max(1, active_clients)
        return {"round": round_idx, "train_loss": avg_loss}

    def calibrate(self, cal_dataset: FlowDataset, alpha: float = 0.05) -> float:
        """Calibrate rejection threshold using finite-sample conformal quantile on Energy score."""
        self.model.eval()
        x_cal = cal_dataset.features.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x_cal)
            cal_scores = (-torch.logsumexp(logits, dim=-1)).cpu().numpy()

        tau = compute_finite_sample_quantile(cal_scores, alpha=alpha)
        self.threshold = tau
        return tau

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and Energy anomaly scores."""
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            anomaly_scores = (-torch.logsumexp(logits, dim=-1)).cpu().numpy()
            cand_classes = torch.argmax(logits, dim=-1).cpu().numpy()

        preds = np.where(anomaly_scores > self.threshold, self.num_classes, cand_classes)
        return preds, anomaly_scores, cand_classes
