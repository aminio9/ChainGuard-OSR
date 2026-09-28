"""Federated client implementation executing local training epochs (Algorithm 1 Lines 6-11)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.base import FlowDataset
from src.models.dual_branch import DecoupledDualBranchNetwork
from src.losses.composite_loss import CompositeLossCalculator
from src.config import AppConfig


class FederatedClient:
    """Represents a blockchain edge node participating in federated learning.

    Executes local epochs (Lines 6-11 of Algorithm 1):
        - Forward pass through VIB encoder & decoupled heads
        - Multi-objective composite loss evaluation (CB + VIB + Anchor + PD)
        - Parameter gradient update
        - Empirical prototype & covariance extraction
    """

    def __init__(
        self,
        client_id: int,
        dataset: FlowDataset,
        config: AppConfig,
        device: torch.device,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.device = device

        # Initialize local decoupled model
        self.model = DecoupledDualBranchNetwork(
            in_features=config.model.in_features,
            latent_dim=config.model.latent_dim,
            num_classes=len(config.data.known_classes),
            hidden_dim=config.model.hidden_dim,
            density_dim=config.model.density_dim,
            dropout=config.model.dropout,
            use_batchnorm=config.model.use_batchnorm,
            stochastic_encoder=config.model.stochastic_encoder,
        ).to(self.device)

        # Local optimizer
        if config.optimization.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.optimization.learning_rate,
                weight_decay=config.optimization.weight_decay,
                momentum=0.9,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.optimization.learning_rate,
                weight_decay=config.optimization.weight_decay,
            )

        self.loss_calculator = CompositeLossCalculator(config.loss)

        # Count samples per class in local data for Class-Balanced Loss
        labels_np = self.dataset.labels.numpy()
        self.local_classes = np.unique(labels_np)
        self.samples_per_class = np.zeros(len(config.data.known_classes), dtype=int)
        for c in self.local_classes:
            if c < len(self.samples_per_class):
                self.samples_per_class[c] = int(np.sum(labels_np == c))

        # References to global state received from server
        self.global_encoder_params: Optional[Dict[str, torch.Tensor]] = None
        self.dual_lambda: Optional[torch.Tensor] = None

    def synchronize(
        self,
        global_encoder_params: Dict[str, torch.Tensor],
        dual_lambda: Optional[torch.Tensor] = None,
    ) -> None:
        """Receive global encoder parameters and dual variable from orchestrator (Algorithm 1 Line 5)."""
        self.model.set_encoder_parameters(global_encoder_params)
        self.global_encoder_params = {
            k: v.clone().to(self.device) for k, v in global_encoder_params.items()
        }
        if dual_lambda is not None:
            self.dual_lambda = dual_lambda.clone().to(self.device)

    def train_local(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[int, Tuple[torch.Tensor, torch.Tensor, int]],
        Dict[str, float],
    ]:
        """Execute local training epochs and return updated parameters and local prototypes.

        Returns:
            - updated_encoder_params
            - prototypes_dict: mapping class_id -> (mu_k_c, cov_k_c, sample_count)
            - average_loss_metrics
        """
        self.model.train()
        epochs = epochs or self.config.federated.local_epochs
        batch_size = batch_size or self.config.federated.batch_size

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        epoch_losses: List[Dict[str, float]] = []

        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_b = x_batch.to(self.device)
                y_b = y_batch.to(self.device)

                self.optimizer.zero_grad()

                logits, _, z, mu, logvar = self.model(x_b)

                total_loss, breakdown = self.loss_calculator(
                    logits=logits,
                    targets=y_b,
                    mu=mu,
                    logvar=logvar,
                    classifier_weights=self.model.classifier.weight_matrix,
                    local_encoder=self.model.encoder,
                    global_encoder_params=self.global_encoder_params,
                    dual_lambda=self.dual_lambda,
                    samples_per_class=self.samples_per_class,
                )

                total_loss.backward()
                # Gradient clipping for numerical stability
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                epoch_losses.append(breakdown)

        # Line 10: Compute empirical class prototypes mu_{k,c} and covariance matrices Sigma_{k,c}
        prototypes_data = self._compute_local_prototypes()

        # Compute average metrics across all batches
        avg_metrics: Dict[str, float] = {}
        if epoch_losses:
            for k in epoch_losses[0].keys():
                avg_metrics[k] = float(np.mean([entry[k] for entry in epoch_losses]))

        return self.model.get_encoder_parameters(), prototypes_data, avg_metrics

    @torch.no_grad()
    def _compute_local_prototypes(
        self,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]:
        """Compute empirical mean and covariance for each class present in client's local dataset."""
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=256, shuffle=False)
        all_z = []
        all_y = []

        for x_batch, y_batch in loader:
            x_b = x_batch.to(self.device)
            # Deterministic latent features for prototype construction
            _, _, z, mu, _ = self.model(x_b)
            all_z.append(mu.cpu())
            all_y.append(y_batch)

        z_all = torch.cat(all_z, dim=0)
        y_all = torch.cat(all_y, dim=0)

        prototypes: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        latent_dim = z_all.size(1)

        for c in self.local_classes:
            c = int(c)
            idx = (y_all == c)
            count = int(idx.sum().item())
            if count == 0:
                continue

            z_c = z_all[idx]  # shape: (count, latent_dim)
            mu_c = torch.mean(z_c, dim=0)  # shape: (latent_dim,)

            if count > 1:
                # Unbiased sample covariance matrix
                centered = z_c - mu_c
                cov_c = torch.matmul(centered.t(), centered) / float(count - 1)
            else:
                cov_c = torch.eye(latent_dim)

            prototypes[c] = (mu_c, cov_c, count)

        return prototypes
