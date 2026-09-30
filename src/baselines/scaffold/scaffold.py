"""SCAFFOLD (Karimireddy et al., ICML 2020) baseline with Stochastic Controlled Averaging.

Reference:
    Karimireddy et al., 'SCAFFOLD: Stochastic Controlled Averaging for Federated Learning',
    International Conference on Machine Learning (ICML), 2020.

Algorithmic Formulation (Option II):
    1. Local Gradient Correction: g_i <- g_i - c_i + c
    2. Local Control Variate Update: c_i^+ <- c_i - c + (1 / (K * eta)) * (w_global - w_i)
    3. Server Control Variate Update: c <- c + (1 / N) * sum_{i in S} (c_i^+ - c_i)
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


class SCAFFOLDBaseline(BaseBaseline):
    """SCAFFOLD baseline correcting client-drift via server and client control variates."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="SCAFFOLD",
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

        # Server control variate c
        self.server_c: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data, device=self.device)
            for name, param in self.model.named_parameters()
        }
        # Client control variates c_i
        self.client_c: Dict[int, Dict[str, torch.Tensor]] = {}

    def _get_client_c(self, client_idx: int) -> Dict[str, torch.Tensor]:
        if client_idx not in self.client_c:
            self.client_c[client_idx] = {
                name: torch.zeros_like(param.data, device=self.device)
                for name, param in self.model.named_parameters()
            }
        return self.client_c[client_idx]

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one communication round of SCAFFOLD with control variate variance reduction."""
        self.model.eval()
        global_state = copy.deepcopy(self.model.state_dict())
        global_params = {name: param.data.clone() for name, param in self.model.named_parameters()}

        client_weights: List[Dict[str, torch.Tensor]] = []
        client_sample_sizes: List[int] = []
        delta_c_accum: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(val, device=self.device) for name, val in self.server_c.items()
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
            c_i = self._get_client_c(c_idx)
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
                    loss = criterion(logits, y_b)
                    loss.backward()

                    # SCAFFOLD gradient correction: g_i <- g_i - c_i + c
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            if param.grad is not None and name in self.server_c:
                                param.grad.add_(self.server_c[name] - c_i[name])

                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_loss += loss.item()
                    steps += 1

            if steps > 0:
                # Option II control variate update: c_i^+ = c_i - c + (w_global - w_local) / (steps * lr)
                eff_denom = max(1e-6, float(steps) * max(lr, 1e-4))
                with torch.no_grad():
                    for name, param in local_model.named_parameters():
                        c_plus = c_i[name] - self.server_c[name] + (global_params[name] - param.data) / eff_denom
                        # Clamp control variate magnitude for numerical stability under Adam
                        c_plus = torch.clamp(c_plus, min=-10.0, max=10.0)
                        delta_c_accum[name] += (c_plus - c_i[name])
                        c_i[name] = c_plus

                client_weights.append(local_model.state_dict())
                client_sample_sizes.append(len(c_data))
                total_loss += client_loss / steps
                active_clients += 1

        # Aggregate model parameters
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

            # Update server control variate c <- c + (1 / N) * sum (c_i^+ - c_i)
            for name in self.server_c:
                self.server_c[name] = torch.clamp(
                    self.server_c[name] + delta_c_accum[name] / float(num_total_clients),
                    min=-10.0,
                    max=10.0,
                )

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
