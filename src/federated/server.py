"""Federated server orchestrator coordinating the 3-Phase protocol (Algorithm 1)."""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.models.dual_branch import DecoupledDualBranchNetwork
from src.federated.client import FederatedClient
from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.covariance_proto import aggregate_prototypes_covariance
from src.aggregation.primal_dual_agg import update_dual_multipliers, flatten_state_dict
from src.aggregation.byzantine_mad import ClientClassMoment
from src.aggregation.total_variance import (
    GlobalClassProfile,
    aggregate_total_variance_moments,
)
from src.osr.block_mahalanobis import WireSpeedEdgeDetector
from src.config import AppConfig


class FederatedServer:
    """Central orchestrator coordinating Phase I representation learning, Phase II moment aggregation, and Phase III edge deployment (Algorithm 1)."""

    def __init__(
        self,
        config: AppConfig,
        clients: List[FederatedClient],
        device: torch.device,
    ):
        self.config = config
        self.clients = clients
        self.device = device
        self.num_clients = len(clients)

        # Initialize global reference model
        self.global_model = DecoupledDualBranchNetwork(
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

        # Global encoder and classifier parameters
        self.global_encoder_params = self.global_model.get_encoder_parameters()
        self.global_classifier_params = {k: v.clone() for k, v in self.global_model.classifier.state_dict().items()}

        # Frozen anchor model from previous round for Phase I entropy-gated distillation
        self.anchor_model: Optional[nn.Module] = None

        # Dual variables lambda_k for each client k, initialized to zeros
        flat_size = flatten_state_dict(self.global_encoder_params).numel()
        self.dual_multipliers: Dict[int, torch.Tensor] = {
            c.client_id: torch.zeros(flat_size, device=self.device) for c in clients
        }

        # Baseline prototypes
        self.global_prototypes: Dict[int, torch.Tensor] = {}

        # Phase II Aggregated Profiles via Law of Total Variance (Equations 17-18)
        self.global_profiles: Dict[int, GlobalClassProfile] = {}
        self.byzantine_rejections: Dict[int, List[int]] = {}

        # Frozen encoder flag for Phase I -> Phase II transition (Line 7)
        self.encoder_frozen: bool = False

    def compute_cosine_learning_rate(
        self,
        round_idx: int,
        total_rounds: int,
        eta_0: float = 0.001,
        eta_min: float = 1e-6,
    ) -> float:
        """Compute scheduled learning rate via Cosine Annealing (LaTeX Equation 100-103).

        Mathematical Formulation (LaTeX Equation 101):
            eta_t = eta_min + 0.5 * (eta_0 - eta_min) * (1 + cos(pi * t / T))
        """
        ratio = min(max(float(round_idx) / max(total_rounds, 1), 0.0), 1.0)
        eta_t = eta_min + 0.5 * (eta_0 - eta_min) * (1.0 + math.cos(math.pi * ratio))
        return eta_t

    def select_active_clients(
        self, fraction: float, seed: Optional[int] = None
    ) -> List[FederatedClient]:
        """Select a random subset S_t of active blockchain clients for the round (Line 3)."""
        rng = np.random.default_rng(seed)
        m = max(1, int(fraction * self.num_clients))
        selected_indices = rng.choice(self.num_clients, size=m, replace=False)
        return [self.clients[i] for i in selected_indices]

    def run_round(self, round_idx: int) -> Dict[str, float]:
        """Execute one communication round of Phase I (Algorithm 1 Lines 2-6)."""
        total_rounds = self.config.federated.num_rounds
        t_ratio = float(round_idx) / max(total_rounds, 1)

        # Line 2 & 3: Compute scheduled learning rate eta_t (LaTeX Equation 101)
        lr_t = self.compute_cosine_learning_rate(
            round_idx=round_idx,
            total_rounds=total_rounds,
            eta_0=self.config.optimization.learning_rate,
            eta_min=getattr(self.config.optimization, "eta_min", 1e-6),
        )

        active_clients = self.select_active_clients(
            self.config.federated.client_fraction,
            seed=self.config.experiment.seed + round_idx,
        )

        client_updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
        client_submitted_params: Dict[int, Dict[str, torch.Tensor]] = {}
        all_prototypes_submissions: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = {
            c: [] for c in range(len(self.config.data.known_classes))
        }
        round_losses: List[Dict[str, float]] = []

        # Teacher anchor model: frozen snapshot from start of round (Equation 5)
        if self.anchor_model is None:
            self.anchor_model = copy.deepcopy(self.global_model)
        else:
            self.anchor_model.load_state_dict(self.global_model.state_dict())
        self.anchor_model.eval()

        # Lines 4-5: Parallel client local training
        for client in active_clients:
            # Set cosine annealed learning rate
            client.set_learning_rate(lr_t)

            # Synchronize global encoder parameters, dual multipliers, and classifier head
            client.synchronize(
                self.global_encoder_params,
                self.dual_multipliers[client.client_id],
                global_classifier_params=self.global_classifier_params,
            )

            # Train local epochs with dynamic anchor distillation
            encoder_params, local_protos, loss_metrics = client.train_local(
                teacher_model=self.anchor_model,
                t_ratio=t_ratio,
            )

            client_updates.append((encoder_params, len(client.dataset)))
            client_submitted_params[client.client_id] = encoder_params
            round_losses.append(loss_metrics)

            for c, proto_data in local_protos.items():
                if c in all_prototypes_submissions:
                    all_prototypes_submissions[c].append(proto_data)

        # Line 6: Aggregate encoder parameters via FedAvg
        new_global_params = aggregate_fedavg_parameters(client_updates)

        # Aggregate classifier head if shared or running FedAvg/FedProx baseline
        if (
            getattr(self.config.federated, "share_classifier_head", False)
            or self.config.federated.algorithm in ["fedavg", "fedprox"]
        ):
            classifier_updates = [
                (client.get_classifier_parameters(), len(client.dataset))
                for client in active_clients
            ]
            new_classifier_params = aggregate_fedavg_parameters(classifier_updates)
            self.global_model.classifier.load_state_dict(new_classifier_params)
            self.global_classifier_params = new_classifier_params

        # Line 13: Optional Primal-Dual multipliers update
        if self.config.loss.primal_dual_rho > 0.0:
            self.dual_multipliers = update_dual_multipliers(
                current_lambdas=self.dual_multipliers,
                client_params=client_submitted_params,
                global_params=new_global_params,
                rho=self.config.loss.primal_dual_rho,
            )

        # Baseline prototype aggregation
        if self.config.federated.covariance_prototype_aggregation:
            self.global_prototypes = aggregate_prototypes_covariance(
                all_prototypes_submissions,
                latent_dim=self.config.model.latent_dim,
            )

        # Update global reference model
        self.global_encoder_params = new_global_params
        self.global_model.set_encoder_parameters(new_global_params)

        avg_round_loss: Dict[str, float] = {}
        if round_losses:
            for k in round_losses[0].keys():
                avg_round_loss[k] = float(np.mean([entry[k] for entry in round_losses]))
        avg_round_loss["lr_scheduled"] = lr_t

        return avg_round_loss

    def freeze_global_encoder(self) -> None:
        """Line 7: Freeze global encoder parameters permanently: theta_g^* = theta_g^T."""
        self.global_model.eval()
        for param in self.global_model.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True

    def run_phase_two_profiling(
        self,
        num_blocks: int = 4,
        block_dim: int = 16,
        alpha: float = 0.05,
        apply_mad_filter: bool = True,
        mad_gamma_0: float = 3.0,
        mad_zeta: float = 1.5,
    ) -> Dict[int, GlobalClassProfile]:
        """Execute Phase II: Byzantine-Robust Profiling via Law of Total Variance (Algorithm 1 Lines 17-27).

        Returns:
            Dict mapping class_id -> GlobalClassProfile with exact pooled covariances and conformal thresholds.
        """
        if not self.encoder_frozen:
            self.freeze_global_encoder()

        latent_dim = self.config.model.latent_dim
        if num_blocks * block_dim != latent_dim:
            num_blocks = min(num_blocks, latent_dim)
            block_dim = latent_dim // num_blocks

        # Lines 17-18: Parallel extraction of moments from all clients
        class_moments_map: Dict[int, List[ClientClassMoment]] = {
            c: [] for c in range(len(self.config.data.known_classes))
        }

        for client in self.clients:
            m_list = client.extract_phase_two_moments(
                frozen_encoder=self.global_model.encoder,
                num_blocks=num_blocks,
                block_dim=block_dim,
                alpha=alpha,
            )
            for idx, c in enumerate(client.local_classes):
                c = int(c)
                if idx < len(m_list) and c in class_moments_map:
                    class_moments_map[c].append(m_list[idx])

        # Lines 19-27: Byzantine MAD filter + Law of Total Variance aggregation
        self.global_profiles, self.byzantine_rejections = aggregate_total_variance_moments(
            class_submissions=class_moments_map,
            num_blocks=num_blocks,
            block_dim=block_dim,
            tikhonov_eps=1e-4,
            apply_mad_filter=apply_mad_filter,
            mad_gamma_0=mad_gamma_0,
            mad_zeta=mad_zeta,
        )

        return self.global_profiles

    def get_edge_detector(
        self,
        unknown_label: int = -1,
        num_blocks: int = 4,
        block_dim: int = 16,
    ) -> WireSpeedEdgeDetector:
        """Instantiate the Phase III Wire-Speed Edge Detector (Algorithm 2)."""
        if not self.global_profiles:
            raise RuntimeError("Phase II profiling must be executed before creating Edge Detector.")

        latent_dim = self.config.model.latent_dim
        if num_blocks * block_dim != latent_dim:
            num_blocks = min(num_blocks, latent_dim)
            block_dim = latent_dim // num_blocks

        return WireSpeedEdgeDetector(
            frozen_encoder=self.global_model.encoder,
            global_profiles=self.global_profiles,
            unknown_label=unknown_label,
            num_blocks=num_blocks,
            block_dim=block_dim,
        )
