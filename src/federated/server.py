"""Federated server orchestrator coordinating the 3-Phase protocol (LaTeX Algorithm 1)."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.models.vib_network import VIBClassifierNetwork
from src.federated.client import FederatedClient, extract_latent_mu
from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.decoupled_agg import aggregate_decoupled_parameters
from src.aggregation.total_variance import (
    ClientClassMoment,
    GlobalClassProfile,
    aggregate_total_variance_moments,
)
from src.osr.block_mahalanobis import WireSpeedEdgeDetector
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics
from src.config import AppConfig


class FederatedServer:
    """Central orchestrator coordinating Phase I representation learning, Phase II moment aggregation, and Phase III edge deployment (LaTeX Algorithm 1)."""

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
        self.aggregation_strategy = getattr(config.federated, "aggregation_strategy", "decoupled").lower()

        # Initialize global model: VIB Encoder E_theta + Classification Head C_phi (LaTeX Section 4.1)
        self.global_model = VIBClassifierNetwork(
            in_features=config.model.in_features,
            latent_dim=config.model.latent_dim,
            num_classes=len(config.data.known_classes),
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            use_batchnorm=config.model.use_batchnorm,
            use_layernorm=getattr(config.model, "use_layernorm", False),
            activation=getattr(config.model, "activation", "gelu"),
            stochastic_encoder=config.model.stochastic_encoder,
        ).to(self.device)

        # Global model parameters w^0 = (theta_g^0, phi_g^0)
        self.global_params = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        self.global_encoder_params = self.global_model.get_encoder_parameters()

        # Phase II Aggregated Profiles via Law of Total Variance (LaTeX Section 4.2)
        self.global_profiles: Dict[int, GlobalClassProfile] = {}
        self.calibration_counts: Dict[int, int] = {}
        self.calibration_thresholds: Dict[int, float] = {}
        self.round_history: List[Dict[str, Any]] = []

        # Frozen encoder flag for Phase I -> Phase II transition (LaTeX Algorithm 1)
        self.encoder_frozen: bool = False

    def compute_cosine_learning_rate(
        self,
        round_idx: int,
        total_rounds: int,
        eta_0: float = 0.001,
        eta_min: float = 1e-6,
    ) -> float:
        """Compute scheduled learning rate via Cosine Annealing (LaTeX Equation 93).

        eta_t = eta_min + 0.5 * (eta_0 - eta_min) * (1 + cos(pi * t / T))
        """
        ratio = min(max(float(round_idx) / max(total_rounds, 1), 0.0), 1.0)
        eta_t = eta_min + 0.5 * (eta_0 - eta_min) * (1.0 + math.cos(math.pi * ratio))
        return eta_t

    def select_active_clients(
        self, fraction: float, seed: Optional[int] = None
    ) -> List[FederatedClient]:
        """Select a random subset S_t of active blockchain clients for the round (LaTeX Algorithm 1 Line 292)."""
        rng = np.random.default_rng(seed)
        m = max(1, int(fraction * self.num_clients))
        selected_indices = rng.choice(self.num_clients, size=m, replace=False)
        return [self.clients[i] for i in selected_indices]

    def run_round(self, round_idx: int) -> Dict[str, float]:
        """Execute one communication round of Phase I (LaTeX Section 4.2 & 4.3)."""
        total_rounds = self.config.federated.num_rounds
        # Normalize 1-based caller round_idx (1..T) to 0-based t in {0, 1, ..., T-1} (Section 4.2, Eq. eq:cosine_lr)
        t_zero_based = (
            len(self.round_history)
            if (round_idx >= 1 and len(self.round_history) == round_idx - 1)
            else round_idx
        )

        # Compute scheduled learning rate eta_t via Cosine Annealing (Fixed across E local epochs of round t)
        lr_t = self.compute_cosine_learning_rate(
            round_idx=t_zero_based,
            total_rounds=total_rounds,
            eta_0=self.config.optimization.learning_rate,
            eta_min=getattr(self.config.optimization, "eta_min", 1e-6),
        )

        active_clients = self.select_active_clients(
            self.config.federated.client_fraction,
            seed=self.config.experiment.seed + t_zero_based,
        )

        client_updates: List[Tuple[Dict[str, torch.Tensor], int, Any]] = []
        round_losses: List[Dict[str, float]] = []

        # Parallel client local training
        current_global_w = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        for client in active_clients:
            client.set_learning_rate(lr_t)
            client.synchronize(current_global_w)

            # Train local epochs with L_BLA + lambda_vib * L_VIB (LaTeX Section 4.2)
            updated_params, _, loss_metrics = client.train_local()

            client_updates.append((updated_params, len(client.dataset), client.samples_per_class))
            round_losses.append(loss_metrics)

        # Parameter Aggregation (LaTeX Section 4.3)
        if self.aggregation_strategy == "fedavg":
            fedavg_updates = [(params, n) for params, n, _ in client_updates]
            new_global_w = aggregate_fedavg_parameters(fedavg_updates)
        else:
            # Decoupled Server Aggregation: average encoder theta over all K nodes,
            # and average each class head psi_c over active nodes K_c = {k in M : N_{k,c} > 0}
            new_global_w = aggregate_decoupled_parameters(
                client_updates,
                previous_global_state=current_global_w,
            )
        self.global_model.load_state_dict(new_global_w)
        self.global_params = new_global_w
        self.global_encoder_params = self.global_model.get_encoder_parameters()

        avg_round_loss: Dict[str, float] = {}
        if round_losses:
            for k in round_losses[0].keys():
                avg_round_loss[k] = float(np.mean([entry[k] for entry in round_losses]))
        avg_round_loss["lr_scheduled"] = lr_t
        avg_round_loss["round"] = int(t_zero_based + 1)
        self.round_history.append(dict(avg_round_loss))

        return avg_round_loss

    def freeze_global_encoder(self) -> None:
        """LaTeX Section 4.1 & 4.3: Freeze global encoder parameters permanently (theta^* = theta^T)."""
        self.global_model.eval()
        for param in self.global_model.parameters():
            param.requires_grad = False
        self.encoder_frozen = True

    def run_phase_two_profiling(
        self,
        num_blocks: int = 4,
        block_dim: int = 16,
        alpha: float = 0.05,
        cal_dataset: Optional[Any] = None,
    ) -> Dict[int, GlobalClassProfile]:
        """Execute Stage 2 Class Profiling (Section 4.3) and Stage 3 Distributed Conformal Calibration (Section 4.4).

        Returns:
            Dict mapping class_id -> GlobalClassProfile with exact pooled covariances and conformal thresholds.
        """
        if not self.encoder_frozen:
            self.freeze_global_encoder()

        latent_dim = self.config.model.latent_dim
        if num_blocks * block_dim != latent_dim:
            num_blocks = min(num_blocks, latent_dim)
            block_dim = latent_dim // num_blocks

        # Stage 2 (Section 4.3): Parallel extraction of moments from eligible nodes K_c^{prof} = {k in M : N_{k,c} >= N_min}
        class_moments_map: Dict[int, List[ClientClassMoment]] = {
            c: [] for c in range(len(self.config.data.known_classes))
        }
        min_cov_samples = getattr(self.config.federated, "min_cov_samples", 3)
        tikhonov_eps = getattr(self.config.federated, "tikhonov_eps", 1e-4)

        for client in self.clients:
            m_list = client.extract_phase_two_moments(
                frozen_encoder=self.global_model.encoder,
                num_blocks=num_blocks,
                block_dim=block_dim,
                alpha=alpha,
                min_cov_samples=min_cov_samples,
            )
            for idx, m_item in enumerate(m_list):
                c = int(m_item.class_id) if m_item.class_id is not None else int(client.local_classes[idx])
                if c in class_moments_map:
                    class_moments_map[c].append(m_item)

        # Section 4.3 Fallback: if a rare class has N_{k,c} < N_min on every node, fall back to K_c = {k in M : N_{k,c} > 0}
        for c, subs in class_moments_map.items():
            if not subs:
                for client in self.clients:
                    if c in client.local_classes:
                        fallback_list = client.extract_phase_two_moments(
                            frozen_encoder=self.global_model.encoder,
                            num_blocks=num_blocks,
                            block_dim=block_dim,
                            alpha=alpha,
                            min_cov_samples=1,
                        )
                        for m_item in fallback_list:
                            if m_item.class_id == c:
                                class_moments_map[c].append(m_item)

        # Law of Total Variance aggregation over K_c^{prof}
        self.global_profiles = aggregate_total_variance_moments(
            class_submissions=class_moments_map,
            num_blocks=num_blocks,
            block_dim=block_dim,
            tikhonov_eps=tikhonov_eps,
            min_cov_samples=min_cov_samples,
        )

        # Stage 3 (Section 4.1 & 4.4): Distributed Mondrian Conformal Calibration over K_{cal, c} = {k in M : n_{cal,k,c} >= 1}
        # Nodes evaluate s_{k,j}^{(c)} = D_block(f_{theta^*}(x_{k,j}), c) on D_{cal, k, c} against global profiles,
        # report scalar maximum s_{max,k,c} so the server defines shared grid [0, S_{max,c}], and upload
        # right-closed empirical CDF histograms h_{k,c} over I_{c,1} = [xi_{c,0}, xi_{c,1}], I_{c,m} = (xi_{c,m-1}, xi_{c,m}].
        enable_conformal = getattr(self.config.osr, "enable_conformal", True)
        if enable_conformal:
            from torch.utils.data import DataLoader
            from src.osr.block_mahalanobis import compute_block_mahalanobis_distances
            from src.osr.conformal import (
                aggregate_distributed_cdfs,
                compute_finite_sample_quantile,
                compute_right_closed_histogram,
            )

            class_keys = sorted(list(self.global_profiles.keys()))
            self.calibration_counts = {}
            self.calibration_thresholds = {}

            if cal_dataset is not None:
                loader = DataLoader(cal_dataset, batch_size=256, shuffle=False)
                all_cal_z, all_cal_y = [], []
                with torch.no_grad():
                    for cx, cy in loader:
                        cx_dev = cx.to(self.device)
                        c_mu = extract_latent_mu(self.global_model.encoder, cx_dev)
                        all_cal_z.append(c_mu)
                        all_cal_y.append(cy.to(self.device))
                cal_z_all = torch.cat(all_cal_z, dim=0)
                cal_y_all = torch.cat(all_cal_y, dim=0)
                class_dists = compute_block_mahalanobis_distances(
                    cal_z_all, self.global_profiles, num_blocks=num_blocks, block_dim=block_dim
                )
                for c in class_keys:
                    true_mask = (cal_y_all == c)
                    n_cal_c = int(true_mask.sum().item())
                    new_tau = (
                        float(compute_finite_sample_quantile(class_dists[c][true_mask], alpha=alpha))
                        if n_cal_c > 0
                        else float("inf")
                    )
                    self.global_profiles[c] = self.global_profiles[c]._replace(conformal_threshold=new_tau)
                    self.calibration_counts[int(c)] = n_cal_c
                    self.calibration_thresholds[int(c)] = new_tau
            else:
                # Collect local calibration score sets per node k in K_{cal, c}, deduplicating shared dataset references
                client_scores_per_class: Dict[int, List[np.ndarray]] = {c: [] for c in class_keys}
                seen_cal_ids: set[int] = set()
                for c_node in self.clients:
                    node_cal = getattr(c_node, "cal_dataset", None)
                    if node_cal is None or len(node_cal) == 0:
                        continue
                    if id(node_cal) in seen_cal_ids:
                        continue
                    seen_cal_ids.add(id(node_cal))

                    scores_map = c_node.compute_calibration_scores(
                        frozen_encoder=self.global_model.encoder,
                        global_profiles=self.global_profiles,
                        num_blocks=num_blocks,
                        block_dim=block_dim,
                    )
                    for c, s_tensor in scores_map.items():
                        if c in client_scores_per_class and len(s_tensor) > 0:
                            client_scores_per_class[c].append(s_tensor.numpy().flatten())

                # Fallback without mutating c_node.cal_dataset if no client was provisioned with cal_dataset
                if all(len(v) == 0 for v in client_scores_per_class.values()):
                    for c_node in self.clients:
                        orig_cal = c_node.cal_dataset
                        try:
                            c_node.cal_dataset = c_node.dataset
                            scores_map = c_node.compute_calibration_scores(
                                frozen_encoder=self.global_model.encoder,
                                global_profiles=self.global_profiles,
                                num_blocks=num_blocks,
                                block_dim=block_dim,
                            )
                            for c, s_tensor in scores_map.items():
                                if c in client_scores_per_class and len(s_tensor) > 0:
                                    client_scores_per_class[c].append(s_tensor.numpy().flatten())
                        finally:
                            c_node.cal_dataset = orig_cal

                num_bins = 2048
                for c in class_keys:
                    c_arrays = client_scores_per_class[c]
                    n_cal_c = int(sum(len(arr) for arr in c_arrays))
                    if n_cal_c > 0:
                        s_max_c = max(max(float(np.max(arr)) for arr in c_arrays), 1e-6)
                        shared_bin_edges = np.linspace(0.0, s_max_c, num_bins + 1)
                        client_hists = [
                            compute_right_closed_histogram(arr, bin_edges=shared_bin_edges)
                            for arr in c_arrays
                        ]
                        new_tau = float(aggregate_distributed_cdfs(client_hists, alpha=alpha))
                    else:
                        new_tau = float("inf")

                    self.global_profiles[c] = self.global_profiles[c]._replace(conformal_threshold=new_tau)
                    self.calibration_counts[int(c)] = n_cal_c
                    self.calibration_thresholds[int(c)] = new_tau
        elif not enable_conformal:
            import scipy.stats as stats
            total_dim = num_blocks * block_dim
            theoretical_tau = float(np.sqrt(stats.chi2.ppf(1.0 - alpha, df=total_dim)))
            self.calibration_counts = {}
            self.calibration_thresholds = {}
            for c in self.global_profiles.keys():
                self.global_profiles[c] = self.global_profiles[c]._replace(conformal_threshold=theoretical_tau)
                self.calibration_counts[int(c)] = int(self.global_profiles[c].total_samples)
                self.calibration_thresholds[int(c)] = theoretical_tau

        # Broadcast synthesized profiles back to all edge clients (Phase III, LaTeX Algorithm 2)
        self.broadcast_profiles_to_clients(
            unknown_label=len(self.config.data.known_classes),
            num_blocks=num_blocks,
            block_dim=block_dim,
        )

        return self.global_profiles

    def broadcast_profiles_to_clients(
        self,
        unknown_label: int = -1,
        num_blocks: int = 4,
        block_dim: int = 16,
    ) -> None:
        """Broadcast global class profiles back to all edge nodes (LaTeX Algorithm 1 Line 25 & Algorithm 2 Line 1)."""
        if not self.global_profiles:
            raise RuntimeError("Profiles must be aggregated before broadcasting to clients.")
        for client in self.clients:
            client.receive_global_profiles(
                profiles=self.global_profiles,
                frozen_encoder=self.global_model.encoder,
                unknown_label=unknown_label,
                num_blocks=num_blocks,
                block_dim=block_dim,
            )

    def get_edge_detector(
        self,
        unknown_label: int = -1,
        num_blocks: int = 4,
        block_dim: int = 16,
    ) -> WireSpeedEdgeDetector:
        """Instantiate the Phase III Wire-Speed Edge Detector (LaTeX Algorithm 2)."""
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

    def evaluate_federated_edge_clients(
        self,
        test_open_ds: Any,
        test_closed_ds: Any,
        client_test_indices: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[str, Any]:
        """Execute Q1-standard dual-level evaluation:
        1. Global consensus evaluation on entire test distribution.
        2. Local edge client evaluation across all individual clients, calculating
           mean, standard deviation, and worst-case client dispersion for fairness auditing.

        Args:
            test_open_ds: Open-set test dataset containing known traffic and zero-day anomalies.
            test_closed_ds: Closed-set test dataset for standard classification utility.
            client_test_indices: Optional mapping client_id -> indices in test_open_ds for local non-IID slices.

        Returns:
            Dict containing 'global', 'edge_clients_dispersion', and 'per_client' detailed metrics.
        """
        num_known = len(self.config.data.known_classes)

        # 1. Global Consensus Evaluation
        global_detector = self.get_edge_detector(
            unknown_label=num_known,
            num_blocks=self.config.federated.num_blocks,
            block_dim=self.config.federated.block_dim,
        )
        g_preds_open, _, g_cands_open, g_scores_open = global_detector.predict_batch(test_open_ds.features)
        g_osr = compute_osr_metrics(
            y_true=test_open_ds.labels.numpy(),
            y_pred=g_preds_open,
            anomaly_scores=g_scores_open,
            num_known_classes=num_known,
            unknown_label=num_known,
            closed_preds=g_cands_open,
        )

        _, _, g_cands_closed, _ = global_detector.predict_batch(test_closed_ds.features)
        g_closed = compute_classification_metrics(
            y_true=test_closed_ds.labels.numpy(),
            y_pred=g_cands_closed,
            num_classes=num_known,
        )

        # 2. Local Edge Client Evaluation (Phase III wire-speed evaluation on edge nodes)
        per_client_metrics: List[Dict[str, float]] = []
        for client in self.clients:
            cid = client.client_id
            if client_test_indices is not None and cid in client_test_indices:
                sub_idx = client_test_indices[cid]
                if len(sub_idx) == 0:
                    continue
                sub_t = torch.tensor(sub_idx, dtype=torch.long)
                c_feats_open = test_open_ds.features[sub_t]
                c_labels_open = test_open_ds.labels.numpy()[sub_idx]
            else:
                c_feats_open = test_open_ds.features
                c_labels_open = test_open_ds.labels.numpy()

            # Client edge inference (Phase III)
            c_preds_open, _, c_cands_open, c_scores_open = client.evaluate_edge_traffic(c_feats_open)
            c_osr = compute_osr_metrics(
                y_true=c_labels_open,
                y_pred=c_preds_open,
                anomaly_scores=c_scores_open,
                num_known_classes=num_known,
                unknown_label=num_known,
                closed_preds=c_cands_open,
            )

            # Local closed set evaluation using candidate class assignment hat{y}(x)
            _, _, c_cands_closed, _ = client.evaluate_edge_traffic(test_closed_ds.features)
            c_closed = compute_classification_metrics(
                y_true=test_closed_ds.labels.numpy(),
                y_pred=c_cands_closed,
                num_classes=num_known,
            )

            per_client_metrics.append({
                "client_id": cid,
                "auroc": float(c_osr["auroc"]),
                "unknown_f1": float(c_osr["unknown_f1"]),
                "kfr": float(c_osr["kfr"]),
                "open_set_macro_f1": float(c_osr["open_set_macro_f1"]),
                "closed_accuracy": float(c_closed["accuracy"]),
                "closed_macro_f1": float(c_closed["macro_f1"]),
            })

        # Calculate dispersion statistics across participating edge clients
        client_aurocs = [m["auroc"] for m in per_client_metrics]
        client_unk_f1s = [m["unknown_f1"] for m in per_client_metrics]
        client_kfrs = [m["kfr"] for m in per_client_metrics]
        client_closed_accs = [m["closed_accuracy"] for m in per_client_metrics]
        client_closed_f1s = [m["closed_macro_f1"] for m in per_client_metrics]

        return {
            "global": {
                "auroc": g_osr["auroc"],
                "aupr": g_osr["aupr"],
                "unknown_recall": g_osr["unknown_recall"],
                "unknown_f1": g_osr["unknown_f1"],
                "open_set_macro_f1": g_osr["open_set_macro_f1"],
                "kfr": g_osr["kfr"],
                "closed_accuracy": g_closed["accuracy"],
                "closed_macro_f1": g_closed["macro_f1"],
            },
            "edge_clients_dispersion": {
                "auroc_mean": float(np.mean(client_aurocs)),
                "auroc_std": float(np.std(client_aurocs)),
                "auroc_worst": float(np.min(client_aurocs)),
                "unknown_f1_mean": float(np.mean(client_unk_f1s)),
                "unknown_f1_std": float(np.std(client_unk_f1s)),
                "unknown_f1_worst": float(np.min(client_unk_f1s)),
                "kfr_mean": float(np.mean(client_kfrs)),
                "kfr_std": float(np.std(client_kfrs)),
                "kfr_worst": float(np.max(client_kfrs)),  # highest false alarm rate
                "closed_accuracy_mean": float(np.mean(client_closed_accs)),
                "closed_accuracy_std": float(np.std(client_closed_accs)),
                "closed_accuracy_worst": float(np.min(client_closed_accs)),
                "closed_macro_f1_mean": float(np.mean(client_closed_f1s)),
                "closed_macro_f1_std": float(np.std(client_closed_f1s)),
                "closed_macro_f1_worst": float(np.min(client_closed_f1s)),
            },
            "per_client": per_client_metrics,
        }
