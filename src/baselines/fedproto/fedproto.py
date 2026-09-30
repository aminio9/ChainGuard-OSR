"""FedProto (Tan et al., AAAI 2022) baseline with Prototype-guided Representation Learning.

Reference:
    Tan et al., 'FedProto: Federated Prototype Learning across Heterogeneous Clients', AAAI 2022.
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


class FedProtoBaseline(BaseBaseline):
    """FedProto baseline communicating and regularizing representation prototypes."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        lambda_proto: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedProto",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.lambda_proto = lambda_proto
        self.model = FedAvgModel(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Global class prototypes in latent space
        self.global_prototypes: Dict[int, torch.Tensor] = {
            c: torch.zeros(latent_dim, device=self.device) for c in range(num_classes)
        }

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one communication round of FedProto."""
        self.model.eval()
        global_state = copy.deepcopy(self.model.state_dict())

        client_weights: List[Dict[str, torch.Tensor]] = []
        client_sample_sizes: List[int] = []
        client_prototypes_list: List[Dict[int, Tuple[torch.Tensor, int]]] = []
        total_loss = 0.0
        active_clients = 0

        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()

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
                    loss_ce = criterion_ce(logits, y_b)

                    # Prototype pull loss towards current global prototypes
                    proto_loss = torch.tensor(0.0, device=self.device)
                    if round_idx > 1:
                        target_protos = torch.stack([self.global_prototypes[y.item()] for y in y_b])
                        proto_loss = criterion_mse(z, target_protos)

                    loss = loss_ce + self.lambda_proto * proto_loss

                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_loss += loss.item()
                    steps += 1

            client_weights.append(local_model.state_dict())
            client_sample_sizes.append(len(c_data))

            # Extract local prototypes for this client
            local_model.eval()
            local_protos: Dict[int, Tuple[torch.Tensor, int]] = {}
            with torch.no_grad():
                for c in range(self.num_classes):
                    idx_c = torch.where(c_data.labels == c)[0]
                    if len(idx_c) > 0:
                        x_c = c_data.features[idx_c].to(self.device)
                        _, z_c = local_model(x_c)
                        local_protos[c] = (z_c.mean(dim=0), len(idx_c))
            client_prototypes_list.append(local_protos)

            if steps > 0:
                total_loss += client_loss / steps
                active_clients += 1

        # Model parameter aggregation
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

        # Prototype aggregation across clients
        for c in range(self.num_classes):
            proto_sum = torch.zeros(self.latent_dim, device=self.device)
            count_sum = 0
            for cp in client_prototypes_list:
                if c in cp:
                    vec, cnt = cp[c]
                    proto_sum += vec * cnt
                    count_sum += cnt
            if count_sum > 0:
                self.global_prototypes[c] = proto_sum / float(count_sum)

        avg_loss = total_loss / max(1, active_clients)
        return {"round": round_idx, "train_loss": avg_loss}

    def calibrate(self, cal_dataset: FlowDataset, alpha: float = 0.05) -> float:
        """Calibrate rejection threshold using minimum distance to prototypes."""
        self.model.eval()
        x_cal = cal_dataset.features.to(self.device)
        with torch.no_grad():
            _, z_cal = self.model(x_cal)
            for c in range(self.num_classes):
                if torch.all(self.global_prototypes[c] == 0):
                    idx_c = torch.where(cal_dataset.labels == c)[0]
                    if len(idx_c) > 0:
                        self.global_prototypes[c] = z_cal[idx_c].mean(dim=0)
                    elif hasattr(self.model.classifier, "weight_matrix"):
                        self.global_prototypes[c] = self.model.classifier.weight_matrix[c].detach()
            proto_matrix = torch.stack([self.global_prototypes[c] for c in range(self.num_classes)])  # (C, d)
            # Pairwise euclidean distances
            dists = torch.cdist(z_cal, proto_matrix)  # (N, C)
            min_dists, _ = torch.min(dists, dim=-1)
            cal_scores = min_dists.cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(cal_scores, q_val))
        self.threshold = tau
        return tau

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and prototype distance anomaly scores."""
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, z = self.model(x)
            proto_matrix = torch.stack([self.global_prototypes[c] for c in range(self.num_classes)])
            dists = torch.cdist(z, proto_matrix)
            min_dists, cand_class = torch.min(dists, dim=-1)
            anomaly_scores = min_dists.cpu().numpy()
            cand_classes = cand_class.cpu().numpy()

        preds = np.where(anomaly_scores > self.threshold, self.num_classes, cand_classes)
        return preds, anomaly_scores, cand_classes
