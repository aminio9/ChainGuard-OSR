"""FedPD (Yang et al., ICCV 2023) baseline with Parameter Disentanglement for Federated OSR.

Reference:
    Yang et al., 'Federated Open-Set Recognition with Parameter Disentanglement',
    IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
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


def compute_taylor_importance(
    model: nn.Module,
    loss: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute first-order Taylor expansion parameter importance (Eq. 3): Omega = |grad(w) * w|."""
    model.zero_grad()
    loss.backward(retain_graph=True)
    importance: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            importance[name] = torch.abs(param.grad.detach() * param.detach())
        else:
            importance[name] = torch.zeros_like(param.detach())
    return importance


def generate_binary_mask(importance: Dict[str, torch.Tensor], top_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
    """Generate binary parameter mask selecting top-k important parameters (Eq. 6-7)."""
    masks: Dict[str, torch.Tensor] = {}
    for name, imp in importance.items():
        flat = imp.view(-1)
        if len(flat) == 0:
            masks[name] = torch.ones_like(imp)
            continue
        k = max(1, int(len(flat) * top_ratio))
        topk_val, _ = torch.topk(flat, k)
        threshold = topk_val[-1]
        masks[name] = (imp >= threshold).float()
    return masks


class FedPDBaseline(BaseBaseline):
    """FedPD baseline with Local Parameter Disentanglement and Divide-and-Conquer Aggregation."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        mask_ratio: float = 0.5,
        lambda_open: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedPD",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.mask_ratio = mask_ratio
        self.lambda_open = lambda_open
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
        """Execute one round of FedPD with parameter disentanglement and GDCA."""
        self.model.eval()
        global_state = copy.deepcopy(self.model.state_dict())

        client_models: List[Dict[str, torch.Tensor]] = []
        client_close_masks: List[Dict[str, torch.Tensor]] = []
        client_open_masks: List[Dict[str, torch.Tensor]] = []
        client_sample_sizes: List[int] = []

        total_loss = 0.0
        active_clients = 0
        criterion_ce = nn.CrossEntropyLoss()

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
                    loss_close = criterion_ce(logits, y_b)

                    # Boundary entropy constraint for open space (Eq. 1)
                    probs = torch.softmax(logits, dim=-1)
                    loss_open = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))

                    loss = loss_close + self.lambda_open * loss_open
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_loss += loss.item()
                    steps += 1

            # Extract Taylor importance for close and open objectives
            local_model.eval()
            x_sample = c_data.features[: min(128, len(c_data))].to(self.device)
            y_sample = c_data.labels[: min(128, len(c_data))].to(self.device)
            logits_s, _ = local_model(x_sample)
            l_c = criterion_ce(logits_s, y_sample)
            imp_close = compute_taylor_importance(local_model, l_c)

            p_s = torch.softmax(logits_s, dim=-1)
            l_o = -torch.mean(torch.sum(p_s * torch.log(p_s + 1e-8), dim=-1))
            imp_open = compute_taylor_importance(local_model, l_o)

            m_close = generate_binary_mask(imp_close, top_ratio=self.mask_ratio)
            m_open = generate_binary_mask(imp_open, top_ratio=self.mask_ratio)

            client_models.append(local_model.state_dict())
            client_close_masks.append(m_close)
            client_open_masks.append(m_open)
            client_sample_sizes.append(len(c_data))

            if steps > 0:
                total_loss += client_loss / steps
                active_clients += 1

        # Global Divide-and-Conquer Aggregation (GDCA - Eq. 8-10, 22)
        total_samples = sum(client_sample_sizes)
        if total_samples > 0 and len(client_models) > 0:
            new_state = {}
            for key in global_state.keys():
                param_shape = global_state[key].shape
                sum_close = torch.zeros(param_shape, device=self.device)
                sum_open = torch.zeros(param_shape, device=self.device)
                sum_share = torch.zeros(param_shape, device=self.device)
                weight_close = torch.zeros(param_shape, device=self.device)
                weight_open = torch.zeros(param_shape, device=self.device)
                weight_share = torch.zeros(param_shape, device=self.device)

                for cm, mc, mo, cs in zip(
                    client_models, client_close_masks, client_open_masks, client_sample_sizes
                ):
                    w = cm[key].to(self.device)
                    m_c = mc.get(key, torch.ones_like(w))
                    m_o = mo.get(key, torch.ones_like(w))

                    # Disentangle: P_share = M_open * M_close
                    p_share = m_open_share = m_c * m_o
                    p_close = m_c * (1.0 - m_o)
                    p_open = m_o * (1.0 - m_c)

                    sum_close += (w * p_close) * cs
                    weight_close += p_close * cs

                    sum_open += (w * p_open) * cs
                    weight_open += p_open * cs

                    sum_share += (w * p_share) * cs
                    weight_share += p_share * cs

                # Global Divide-and-Conquer Aggregation (GDCA - Eq. 8-10, 22)
                sum_total = sum_close + sum_open + sum_share
                weight_total = weight_close + weight_open + weight_share
                fused = torch.where(weight_total > 0, sum_total / weight_total, global_state[key].to(self.device))
                new_state[key] = fused.to(global_state[key].dtype)

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
