"""FedOSS (Zhu et al., IEEE TMI 2024) baseline via Inter-Client Discrepancy and DUSS.

Reference:
    Zhu et al., 'FedOSS: Federated Open Set Recognition via Inter-Client Knowledge Discrepancy',
    IEEE Transactions on Medical Imaging (TMI), vol. 43, no. 1, pp. 190-202, 2024.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.base import BaseBaseline, create_baseline_optimizer
from src.data.base import FlowDataset
from src.models.encoder import DeterministicEncoder
from src.models.heads import LinearClassifierHead


class FedOSSModel(nn.Module):
    """FedOSS model with shared feature extractor, main (K+1) classifier, and personalized K classifier."""

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
        # Main classifier has K+1 outputs (index K is the unknown class)
        self.main_classifier = LinearClassifierHead(latent_dim=latent_dim, num_classes=num_classes + 1)
        # Personalized classifier has K outputs for known class boundary discrepancy
        self.personalized_classifier = LinearClassifierHead(latent_dim=latent_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, _, _ = self.encoder(x)
        logits_main = self.main_classifier(z)
        logits_pers = self.personalized_classifier(z)
        return logits_main, logits_pers, z


class FedOSSBaseline(BaseBaseline):
    """FedOSS baseline with Discrete Unknown Sample Synthesis (DUSS)."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        inversion_steps: int = 3,
        inversion_lr: float = 0.05,
        lambda_duss: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="FedOSS",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.inversion_steps = inversion_steps
        self.inversion_lr = inversion_lr
        self.lambda_duss = lambda_duss

        self.global_model = FedOSSModel(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Bank of personalized classifiers Omega = {W_pc^c}
        self.personalized_bank: List[Dict[str, torch.Tensor]] = []

    def _synthesize_duss_samples(
        self,
        model: FedOSSModel,
        dataset: FlowDataset,
        batch_size: int = 64,
    ) -> Optional[TensorDataset]:
        """Perform DUSS boundary recognition and inversion updating (Eq. 1-2)."""
        if len(self.personalized_bank) < 2 or len(dataset) < 4:
            return None

        model.eval()
        x_all = dataset.features.to(self.device)
        y_all = dataset.labels.to(self.device)

        # 1. Compute discrepancy score s(x_i) via personalized bank (Eq. 1)
        with torch.no_grad():
            z_all, _, _ = model.encoder(x_all)
            votes = torch.zeros(x_all.size(0), device=self.device)

            temp_pers = LinearClassifierHead(latent_dim=self.latent_dim, num_classes=self.num_classes).to(self.device)
            for pers_state in self.personalized_bank:
                temp_pers.load_state_dict(pers_state)
                p_logits = temp_pers(z_all)
                p_preds = torch.argmax(p_logits, dim=-1)
                votes += (p_preds == y_all).float()

            s_scores = votes / float(len(self.personalized_bank))

        # Boundary samples: 0 < s < 1
        boundary_idx = torch.where((s_scores > 0.0) & (s_scores < 1.0))[0]
        if len(boundary_idx) == 0:
            # Fallback: samples with lowest confidence
            boundary_idx = torch.topk(s_scores, k=min(16, len(s_scores)), largest=False)[1]

        x_boundary = x_all[boundary_idx].clone().detach().requires_grad_(True)
        y_boundary = y_all[boundary_idx]

        # 2. Inversion updating via gradient ascent on known CE risk (Eq. 2)
        ce_loss = nn.CrossEntropyLoss()
        for _ in range(self.inversion_steps):
            z_b, _, _ = model.encoder(x_boundary)
            logits_m = model.main_classifier(z_b)[:, : self.num_classes]
            loss_inv = ce_loss(logits_m, y_boundary)
            grad_x = torch.autograd.grad(loss_inv, x_boundary, retain_graph=False)[0]
            # Push across decision boundary: x = x + lr * grad
            with torch.no_grad():
                x_boundary += self.inversion_lr * grad_x.sign()

        # Virtual unknown target label is K
        y_synth = torch.full((x_boundary.size(0),), self.num_classes, dtype=torch.long, device=self.device)
        return TensorDataset(x_boundary.detach(), y_synth)

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one communication round of FedOSS."""
        self.global_model.eval()

        client_shared_weights: List[Dict[str, torch.Tensor]] = []
        new_personalized_bank: List[Dict[str, torch.Tensor]] = []
        client_sample_sizes: List[int] = []

        total_loss = 0.0
        active_clients = 0
        criterion_ce = nn.CrossEntropyLoss()

        for c_idx, c_data in enumerate(client_datasets):
            if len(c_data) < 4:
                continue

            local_model = copy.deepcopy(self.global_model)
            local_model.train()
            optimizer = create_baseline_optimizer(local_model, lr=lr, optimizer_name=kwargs.get("optimizer_name", "adam"))

            # Generate DUSS virtual unknown samples from boundary
            synth_dataset = self._synthesize_duss_samples(local_model, c_data, batch_size=batch_size)

            loader_known = DataLoader(c_data, batch_size=batch_size, shuffle=True, drop_last=(len(c_data) > batch_size))
            client_loss = 0.0
            steps = 0

            for _ in range(epochs):
                for x_b, y_b in loader_known:
                    if x_b.size(0) <= 1:
                        continue
                    x_b = x_b.to(self.device)
                    y_b = y_b.to(self.device)

                    optimizer.zero_grad()
                    logits_main, logits_pers, _ = local_model(x_b)

                    # Loss 1: Known data on personalized classifier and main classifier
                    l_pers = criterion_ce(logits_pers, y_b)
                    l_main_known = criterion_ce(logits_main[:, : self.num_classes], y_b)

                    # Loss 2: DUSS virtual unknown samples on class K
                    l_synth = torch.tensor(0.0, device=self.device)
                    if synth_dataset is not None and len(synth_dataset) > 0:
                        x_s, y_s = synth_dataset[:]
                        logits_s, _, _ = local_model(x_s)
                        l_synth = criterion_ce(logits_s, y_s)

                    loss = l_main_known + l_pers + self.lambda_duss * l_synth
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    client_loss += loss.item()
                    steps += 1

            # Extract shared parameters (encoder + main_classifier)
            shared_state = {
                k: v.clone()
                for k, v in local_model.state_dict().items()
                if k.startswith("encoder.") or k.startswith("main_classifier.")
            }
            client_shared_weights.append(shared_state)

            # Retain personalized classifier in bank
            pers_state = {
                k.replace("personalized_classifier.", ""): v.clone()
                for k, v in local_model.state_dict().items()
                if k.startswith("personalized_classifier.")
            }
            new_personalized_bank.append(pers_state)
            client_sample_sizes.append(len(c_data))

            if steps > 0:
                total_loss += client_loss / steps
                active_clients += 1

        # Server updates personalized bank
        self.personalized_bank = new_personalized_bank

        # Server aggregates shared parameters
        total_samples = sum(client_sample_sizes)
        if total_samples > 0 and len(client_shared_weights) > 0:
            current_state = self.global_model.state_dict()
            for key in client_shared_weights[0].keys():
                weighted_sum = sum(
                    cw[key].float() * (cs / total_samples)
                    for cw, cs in zip(client_shared_weights, client_sample_sizes)
                )
                current_state[key] = weighted_sum.to(current_state[key].dtype)
            self.global_model.load_state_dict(current_state)

        avg_loss = total_loss / max(1, active_clients)
        return {"round": round_idx, "train_loss": avg_loss}

    def calibrate(self, cal_dataset: FlowDataset, alpha: float = 0.05) -> float:
        """Calibrate rejection threshold using main classifier unknown probability (class K)."""
        self.global_model.eval()
        x_cal = cal_dataset.features.to(self.device)
        with torch.no_grad():
            logits_main, _, _ = self.global_model(x_cal)
            probs = torch.softmax(logits_main, dim=-1)
            p_unk = probs[:, self.num_classes].cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(p_unk, q_val))
        self.threshold = max(0.05, tau)
        return self.threshold

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and FedOSS unknown anomaly scores."""
        self.global_model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits_main, _, _ = self.global_model(x)
            probs = torch.softmax(logits_main, dim=-1)
            p_unk = probs[:, self.num_classes].cpu().numpy()
            cand_classes = torch.argmax(probs[:, : self.num_classes], dim=-1).cpu().numpy()

        preds = np.where(p_unk > self.threshold, self.num_classes, cand_classes)
        return preds, p_unk, cand_classes
