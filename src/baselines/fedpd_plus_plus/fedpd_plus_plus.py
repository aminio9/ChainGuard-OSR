"""FedPD++ (Yang et al., IJCV 2026) baseline with Adaptive Soft Masking and Manifold Mixup.

Reference:
    Yang et al., 'FedPD++: Enhanced Federated Open-Set Recognition with Parameter Disentanglement',
    International Journal of Computer Vision (IJCV), 2026.
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


def compute_soft_mask(importance: Dict[str, torch.Tensor], temperature: float = 0.1) -> Dict[str, torch.Tensor]:
    """Compute Adaptive Soft Mask (ASM) mapping continuous Taylor scores to [0, 1] via sigmoid."""
    masks: Dict[str, torch.Tensor] = {}
    for name, imp in importance.items():
        if imp.numel() == 0:
            masks[name] = torch.ones_like(imp)
            continue
        mean_val = imp.mean()
        std_val = imp.std() + 1e-6
        normalized = (imp - mean_val) / std_val
        masks[name] = torch.sigmoid(normalized / temperature)
    return masks


class FedPDPlusPlusBaseline(BaseBaseline):
    """FedPD++ baseline with PROSER-style manifold mixup and Adaptive Soft Masking."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        lambda_mixup: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedPD++",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.lambda_mixup = lambda_mixup
        # Output num_classes + 1 logits (class K is the unknown placeholder)
        self.model = FedAvgModel(
            in_features=in_features,
            num_classes=num_classes + 1,
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
        """Execute one communication round of FedPD++ with manifold mixup and ASM."""
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
            if len(c_data) < 4:
                continue

            local_model = copy.deepcopy(self.model)
            local_model.train()
            optimizer = create_baseline_optimizer(local_model, lr=lr, optimizer_name=kwargs.get("optimizer_name", "adam"))
            loader = DataLoader(c_data, batch_size=batch_size, shuffle=True, drop_last=(len(c_data) > batch_size))

            client_loss = 0.0
            steps = 0
            for _ in range(epochs):
                for x_b, y_b in loader:
                    if x_b.size(0) <= 2:
                        continue
                    x_b = x_b.to(self.device)
                    y_b = y_b.to(self.device)

                    optimizer.zero_grad()
                    logits_known, _ = local_model(x_b)
                    loss_close = criterion_ce(logits_known[:, : self.num_classes], y_b)

                    # PROSER Manifold Mixup (Eq. 4): create pseudo-unknowns between differing classes
                    perm = torch.randperm(x_b.size(0))
                    diff_mask = y_b != y_b[perm]
                    if diff_mask.sum() > 0:
                        x_i = x_b[diff_mask]
                        x_j = x_b[perm][diff_mask]
                        lam = np.random.beta(1.0, 1.0)
                        x_mix = lam * x_i + (1.0 - lam) * x_j

                        logits_mix, _ = local_model(x_mix)
                        # Train mixup samples to activate placeholder class K
                        y_placeholder = torch.full((x_mix.size(0),), self.num_classes, dtype=torch.long, device=self.device)
                        loss_open = criterion_ce(logits_mix, y_placeholder)
                    else:
                        loss_open = torch.tensor(0.0, device=self.device)

                    loss = loss_close + self.lambda_mixup * loss_open
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_loss += loss.item()
                    steps += 1

            # Compute Taylor importance for close and open objectives
            local_model.eval()
            x_sample = c_data.features[: min(128, len(c_data))].to(self.device)
            y_sample = c_data.labels[: min(128, len(c_data))].to(self.device)

            logits_s, _ = local_model(x_sample)
            l_c = criterion_ce(logits_s[:, : self.num_classes], y_sample)

            local_model.zero_grad()
            l_c.backward(retain_graph=True)
            imp_c = {n: torch.abs(p.grad.detach() * p.detach()) for n, p in local_model.named_parameters() if p.grad is not None}

            # Open-set importance via placeholder activation
            y_ph = torch.full((x_sample.size(0),), self.num_classes, dtype=torch.long, device=self.device)
            l_o = criterion_ce(logits_s, y_ph)
            local_model.zero_grad()
            l_o.backward(retain_graph=True)
            imp_o = {n: torch.abs(p.grad.detach() * p.detach()) for n, p in local_model.named_parameters() if p.grad is not None}

            m_close = compute_soft_mask(imp_c)
            m_open = compute_soft_mask(imp_o)

            client_models.append(local_model.state_dict())
            client_close_masks.append(m_close)
            client_open_masks.append(m_open)
            client_sample_sizes.append(len(c_data))

            if steps > 0:
                total_loss += client_loss / steps
                active_clients += 1

        # Global Divide-and-Conquer Aggregation with Adaptive Soft Masking (Eq. 14-22)
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
                    m_c = mc.get(key, torch.full_like(w, 0.5))
                    m_o = mo.get(key, torch.full_like(w, 0.5))

                    p_share = m_c * m_o
                    p_close = torch.clamp(m_c - p_share, min=0.0)
                    p_open = torch.clamp(m_o - p_share, min=0.0)

                    sum_close += (w * p_close) * cs
                    weight_close += p_close * cs

                    sum_open += (w * p_open) * cs
                    weight_open += p_open * cs

                    sum_share += (w * p_share) * cs
                    weight_share += p_share * cs

                # Global Divide-and-Conquer Aggregation with Adaptive Soft Masking (Eq. 14-22)
                sum_total = sum_close + sum_open + sum_share
                weight_total = weight_close + weight_open + weight_share
                fused = torch.where(weight_total > 1e-5, sum_total / weight_total, global_state[key].to(self.device))
                new_state[key] = fused.to(global_state[key].dtype)

            self.model.load_state_dict(new_state)

        avg_loss = total_loss / max(1, active_clients)
        return {"round": round_idx, "train_loss": avg_loss}

    def calibrate(self, cal_dataset: FlowDataset, alpha: float = 0.05) -> float:
        """Calibrate rejection threshold using placeholder class probability."""
        self.model.eval()
        x_cal = cal_dataset.features.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x_cal)
            probs = torch.softmax(logits, dim=-1)
            # Placeholder class K represents unknown probability
            p_unk = probs[:, self.num_classes].cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(p_unk, q_val))
        self.threshold = max(0.1, tau)
        return self.threshold

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and placeholder anomaly scores."""
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            p_unk = probs[:, self.num_classes].cpu().numpy()
            cand_classes = torch.argmax(probs[:, : self.num_classes], dim=-1).cpu().numpy()

        preds = np.where(p_unk > self.threshold, self.num_classes, cand_classes)
        return preds, p_unk, cand_classes
