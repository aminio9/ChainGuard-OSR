"""FedRoD (Chen & Chao, ICLR 2022) baseline with Balanced Risk Minimization and Dual Heads.

Reference:
    Chen & Chao, 'On Bridging Generic and Personalized Federated Learning for Image Classification',
    International Conference on Learning Representations (ICLR), 2022.
    Official GitHub: https://github.com/hongyouc/FedRoD

Mathematical Formulation:
    1. Generic Global Head (G_phi): Trained with Balanced Risk Minimization (BRM) / Balanced Softmax:
       L_BRM(g_phi(z), y; N_k) = -log( (N_{k,y}^gamma * exp(g_y(z))) / sum_c N_{k,c}^gamma * exp(g_c(z)) )
    2. Personalized Head (P_psi): Trained with Empirical Risk Minimization (ERM) on detached generic logits:
       L_ERM(g_phi(z).detach() + p_psi(z.detach()), y)
    3. Server Aggregation: Aggregates shared encoder E_theta and generic head G_phi.
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
from src.models.heads import LinearClassifierHead
from src.osr.conformal import compute_finite_sample_quantile


class FedRoDBaseline(BaseBaseline):
    """FedRoD baseline decoupling Balanced Risk Minimization (generic) and Empirical Risk Minimization (personalized)."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        gamma_brm: float = 1.0,
        tau_rod: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedRoD",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.gamma_brm = float(tau_rod) if tau_rod is not None else gamma_brm
        self.model = FedAvgModel(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Client-specific personalized heads P_psi
        self.personalized_heads: Dict[int, LinearClassifierHead] = {}

    def _get_personalized_head(self, client_idx: int) -> LinearClassifierHead:
        if client_idx not in self.personalized_heads:
            self.personalized_heads[client_idx] = LinearClassifierHead(
                latent_dim=self.latent_dim,
                num_classes=self.num_classes,
            ).to(self.device)
        return self.personalized_heads[client_idx]

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one communication round of FedRoD."""
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

            # Compute client class counts N_{k,c} for Balanced Softmax (BRM)
            labels_np = c_data.labels.cpu().numpy()
            class_counts = np.zeros(self.num_classes, dtype=np.float32)
            for c in range(self.num_classes):
                class_counts[c] = float(np.sum(labels_np == c))
            # Laplace smoothing to avoid log(0) on missing classes
            prior_margin = torch.tensor(
                self.gamma_brm * np.log(class_counts + 1.0),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            local_model = copy.deepcopy(self.model)
            local_model.train()
            p_head = self._get_personalized_head(c_idx)
            p_head.train()

            optimizer_g = create_baseline_optimizer(
                local_model,
                lr=lr,
                optimizer_name=kwargs.get("optimizer_name", "adam"),
            )
            optimizer_p = create_baseline_optimizer(
                p_head,
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

                    # 1. Generic Balanced Risk Minimization (BRM) step
                    optimizer_g.zero_grad()
                    logits_g, z = local_model(x_b)
                    balanced_logits = logits_g + prior_margin
                    loss_brm = criterion(balanced_logits, y_b)
                    loss_brm.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer_g.step()

                    # 2. Personalized Empirical Risk Minimization (ERM) step
                    optimizer_p.zero_grad()
                    logits_p = p_head(z.detach()) + logits_g.detach()
                    loss_erm = criterion(logits_p, y_b)
                    loss_erm.backward()
                    optimizer_p.step()

                    client_loss += loss_brm.item()
                    steps += 1

            client_weights.append(local_model.state_dict())
            client_sample_sizes.append(len(c_data))
            if steps > 0:
                total_loss += client_loss / steps
                active_clients += 1

        # Aggregate generic model (encoder + generic balanced classifier head)
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
        """Predict labels and Energy anomaly scores from generic balanced model."""
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            anomaly_scores = (-torch.logsumexp(logits, dim=-1)).cpu().numpy()
            cand_classes = torch.argmax(logits, dim=-1).cpu().numpy()

        preds = np.where(anomaly_scores > self.threshold, self.num_classes, cand_classes)
        return preds, anomaly_scores, cand_classes
