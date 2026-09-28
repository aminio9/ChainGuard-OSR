"""Federated client implementation executing local training and moment extraction (Algorithm 1)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.base import FlowDataset
from src.models.dual_branch import DecoupledDualBranchNetwork
from src.losses.composite_loss import CompositeLossCalculator
from src.aggregation.byzantine_mad import ClientClassMoment
from src.osr.conformal import compute_finite_sample_quantile
from src.config import AppConfig


class FederatedClient:
    """Represents a blockchain edge node participating in federated learning.

    Executes Phase I local training (Equations 1-11) and Phase II moment extraction (Equations 12-13 & 22).
    """

    def __init__(
        self,
        client_id: int,
        dataset: FlowDataset,
        config: AppConfig,
        device: torch.device,
        cal_dataset: Optional[FlowDataset] = None,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.cal_dataset = cal_dataset
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
            use_layernorm=getattr(config.model, "use_layernorm", False),
            activation=getattr(config.model, "activation", "relu"),
            stochastic_encoder=config.model.stochastic_encoder,
        ).to(self.device)

        # Local optimizer: Cosine-annealed SGD with momentum 0.9 or Adam
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

        # Count samples per class in local data for Class-Balanced Loss (Equation 1)
        labels_np = self.dataset.labels.numpy()
        self.local_classes = np.unique(labels_np)
        self.samples_per_class = np.zeros(len(config.data.known_classes), dtype=int)
        for c in self.local_classes:
            if c < len(self.samples_per_class):
                self.samples_per_class[c] = int(np.sum(labels_np == c))

        # References to global state
        self.global_encoder_params: Optional[Dict[str, torch.Tensor]] = None
        self.dual_lambda: Optional[torch.Tensor] = None

    def set_learning_rate(self, lr: float) -> None:
        """Update optimizer learning rate according to Cosine Annealing (Equation 10)."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def synchronize(
        self,
        global_encoder_params: Dict[str, torch.Tensor],
        dual_lambda: Optional[torch.Tensor] = None,
        global_classifier_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Receive global encoder parameters, dual variable, and optional classifier head from orchestrator (Algorithm 1 Line 5)."""
        self.model.set_encoder_parameters(global_encoder_params)
        self.global_encoder_params = {
            k: v.clone().to(self.device) for k, v in global_encoder_params.items()
        }
        if dual_lambda is not None:
            self.dual_lambda = dual_lambda.clone().to(self.device)
        if global_classifier_params is not None and (
            getattr(self.config.federated, "share_classifier_head", False)
            or self.config.federated.algorithm in ["fedavg", "fedprox"]
        ):
            self.model.classifier.load_state_dict(global_classifier_params)

    def get_classifier_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract classifier head state dictionary for federated parameter exchange."""
        return {k: v.clone() for k, v in self.model.classifier.state_dict().items()}

    def train_local(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        teacher_model: Optional[nn.Module] = None,
        t_ratio: float = 0.0,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[int, Tuple[torch.Tensor, torch.Tensor, int]],
        Dict[str, float],
    ]:
        """Execute Phase I local training epochs (Algorithm 1 Lines 8-16).

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

                # Teacher anchor logits for entropy-gated distillation (Equations 5-8)
                teacher_logits: Optional[torch.Tensor] = None
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_out = teacher_model(x_b)
                        if isinstance(teacher_out, tuple):
                            teacher_logits = teacher_out[0]
                        else:
                            teacher_logits = teacher_out

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
                    teacher_logits=teacher_logits,
                    t_ratio=t_ratio,
                )

                total_loss.backward()
                # Gradient clipping for numerical stability (Section 5.3: max_norm=1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_losses.append(breakdown)

        # Baseline prototype calculation
        prototypes_data = self._compute_local_prototypes()

        # Compute average metrics across all batches
        avg_metrics: Dict[str, float] = {}
        if epoch_losses:
            for k in epoch_losses[0].keys():
                avg_metrics[k] = float(np.mean([entry[k] for entry in epoch_losses]))

        return self.model.get_encoder_parameters(), prototypes_data, avg_metrics

    @torch.no_grad()
    def extract_phase_two_moments(
        self,
        frozen_encoder: nn.Module,
        num_blocks: int = 4,
        block_dim: int = 16,
        alpha: float = 0.05,
    ) -> List[ClientClassMoment]:
        """Extract sub-block statistical moments and calibration quantiles (Phase II, Equations 12-13 & 22).

        Args:
            frozen_encoder: Permanently frozen global encoder theta_g^*.
            num_blocks: Number of sub-blocks B (default: 4).
            block_dim: Dimensionality of each sub-block d_b = d_z / B (default: 16).
            alpha: Mondrian conformal significance level.

        Returns:
            List of ClientClassMoment submissions for all local classes.
        """
        frozen_encoder.eval()
        loader = DataLoader(self.dataset, batch_size=256, shuffle=False)
        all_z = []
        all_y = []

        for x_b, y_b in loader:
            x_dev = x_b.to(self.device)
            out = frozen_encoder(x_dev)
            if isinstance(out, tuple):
                if len(out) >= 4:
                    mu_z = out[3]  # mu(x)
                else:
                    mu_z = out[0]
            else:
                mu_z = out

            all_z.append(mu_z.cpu())
            all_y.append(y_b.cpu())

        z_all = torch.cat(all_z, dim=0)
        y_all = torch.cat(all_y, dim=0)

        actual_latent = z_all.size(1)
        if num_blocks * block_dim != actual_latent:
            num_blocks = min(num_blocks, actual_latent)
            block_dim = actual_latent // num_blocks

        moments_list: List[ClientClassMoment] = []

        for c in self.local_classes:
            c = int(c)
            idx = (y_all == c)
            count = int(idx.sum().item())
            if count == 0:
                continue

            z_c = z_all[idx]  # Shape: (count, d_z)

            # Equation 12: mu_c^{(k)}
            mu_k_c = torch.mean(z_c, dim=0)  # Shape: (d_z,)

            # Equation 13: Sub-block covariances Sigma_{c, G_b}^{(k)}
            block_covs: List[torch.Tensor] = []
            for b in range(num_blocks):
                start = b * block_dim
                end = start + block_dim
                z_c_b = z_c[:, start:end]  # (count, d_b)
                mu_c_b = mu_k_c[start:end]  # (d_b,)

                diff_b = z_c_b - mu_c_b.unsqueeze(0)  # (count, d_b)
                if count > 1:
                    cov_b = torch.matmul(diff_b.t(), diff_b) / float(count)
                else:
                    cov_b = torch.eye(block_dim)

                block_covs.append(cov_b)

            # Compute local calibration quantile if calibration set is available
            cal_quantile: Optional[float] = None
            cal_ds = self.cal_dataset or self.dataset
            cal_loader = DataLoader(cal_ds, batch_size=256, shuffle=False)
            cal_z_list = []
            cal_y_list = []
            for cx, cy in cal_loader:
                cx_dev = cx.to(self.device)
                cout = frozen_encoder(cx_dev)
                c_mu = cout[3] if isinstance(cout, tuple) and len(cout) >= 4 else (cout[0] if isinstance(cout, tuple) else cout)
                cal_z_list.append(c_mu.cpu())
                cal_y_list.append(cy.cpu())

            cal_z_all = torch.cat(cal_z_list, dim=0)
            cal_y_all = torch.cat(cal_y_list, dim=0)
            c_cal_idx = (cal_y_all == c)

            if int(c_cal_idx.sum().item()) > 0:
                z_cal_c = cal_z_all[c_cal_idx]
                # Compute block Mahalanobis scores against local moments
                scores = []
                for zi in z_cal_c:
                    d_i = 0.0
                    for b in range(num_blocks):
                        start = b * block_dim
                        end = start + block_dim
                        z_i_b = zi[start:end]
                        mu_b = mu_k_c[start:end]
                        inv_c_b = torch.linalg.pinv(block_covs[b] + 1e-4 * torch.eye(block_dim))
                        diff_i_b = z_i_b - mu_b
                        d_i += float(torch.dot(diff_i_b, torch.mv(inv_c_b, diff_i_b)).item())
                    scores.append(d_i)
                cal_quantile = compute_finite_sample_quantile(scores, alpha=alpha)

            moments_list.append(
                ClientClassMoment(
                    client_id=self.client_id,
                    sample_count=count,
                    mu=mu_k_c,
                    block_covs=block_covs,
                    cal_quantile=cal_quantile,
                )
            )

        return moments_list

    @torch.no_grad()
    def _compute_local_prototypes(
        self,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]:
        """Compute empirical mean and covariance for baseline comparisons."""
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=256, shuffle=False)
        all_z = []
        all_y = []

        for x_batch, y_batch in loader:
            x_b = x_batch.to(self.device)
            out = self.model(x_b)
            mu = out[3] if isinstance(out, tuple) and len(out) >= 4 else (out[0] if isinstance(out, tuple) else out)
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

            z_c = z_all[idx]
            mu_c = torch.mean(z_c, dim=0)

            if count > 1:
                centered = z_c - mu_c
                cov_c = torch.matmul(centered.t(), centered) / float(count - 1)
            else:
                cov_c = torch.eye(latent_dim)

            prototypes[c] = (mu_c, cov_c, count)

        return prototypes
