"""Federated server orchestrator coordinating communication rounds (Algorithm 1)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.models.dual_branch import DecoupledDualBranchNetwork
from src.federated.client import FederatedClient
from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.covariance_proto import aggregate_prototypes_covariance
from src.aggregation.primal_dual_agg import update_dual_multipliers, flatten_state_dict
from src.config import AppConfig


class FederatedServer:
    """Central server orchestrating federated learning, primal-dual updates, and prototype synthesis (Algorithm 1)."""

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
            stochastic_encoder=config.model.stochastic_encoder,
        ).to(self.device)

        # Global encoder parameters
        self.global_encoder_params = self.global_model.get_encoder_parameters()

        # Dual variables lambda_k for each client k, initialized to zeros (Line 1)
        flat_size = flatten_state_dict(self.global_encoder_params).numel()
        self.dual_multipliers: Dict[int, torch.Tensor] = {
            c.client_id: torch.zeros(flat_size, device=self.device) for c in clients
        }

        # Global Bayesian prototypes
        self.global_prototypes: Dict[int, torch.Tensor] = {}

    def select_active_clients(self, fraction: float, seed: Optional[int] = None) -> List[FederatedClient]:
        """Select a random subset S_t of active blockchain clients for the round (Line 3)."""
        rng = np.random.default_rng(seed)
        m = max(1, int(fraction * self.num_clients))
        selected_indices = rng.choice(self.num_clients, size=m, replace=False)
        return [self.clients[i] for i in selected_indices]

    def run_round(self, round_idx: int) -> Dict[str, float]:
        """Execute one complete federated communication round (Lines 2-14)."""
        # Line 3: Select active clients
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

        # Lines 4-11: Parallel client local training
        for client in active_clients:
            # Line 5: Synchronize global model and dual multipliers
            client.synchronize(
                self.global_encoder_params,
                self.dual_multipliers[client.client_id],
            )

            # Lines 6-11: Train local epochs & extract prototypes
            encoder_params, local_protos, loss_metrics = client.train_local()

            client_updates.append((encoder_params, len(client.dataset)))
            client_submitted_params[client.client_id] = encoder_params
            round_losses.append(loss_metrics)

            for c, proto_data in local_protos.items():
                if c in all_prototypes_submissions:
                    all_prototypes_submissions[c].append(proto_data)

        # Line 12: Aggregate encoder parameters via FedAvg
        new_global_params = aggregate_fedavg_parameters(client_updates)

        # Line 13: Update server-side dual multipliers lambda_k
        self.dual_multipliers = update_dual_multipliers(
            current_lambdas=self.dual_multipliers,
            client_params=client_submitted_params,
            global_params=new_global_params,
            rho=self.config.loss.primal_dual_rho,
        )

        # Line 14: Aggregate global prototypes via Bayesian inverse covariance
        if self.config.federated.covariance_prototype_aggregation:
            self.global_prototypes = aggregate_prototypes_covariance(
                all_prototypes_submissions,
                latent_dim=self.config.model.latent_dim,
            )

        # Update global reference model
        self.global_encoder_params = new_global_params
        self.global_model.set_encoder_parameters(new_global_params)

        # Compute summary metrics for tracking
        avg_round_loss: Dict[str, float] = {}
        if round_losses:
            for k in round_losses[0].keys():
                avg_round_loss[k] = float(np.mean([entry[k] for entry in round_losses]))

        return avg_round_loss
