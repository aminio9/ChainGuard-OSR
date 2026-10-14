"""Federated client implementation executing local training and moment extraction (LaTeX Algorithm 1)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.base import FlowDataset
from src.models.vib_network import VIBClassifierNetwork
from src.losses.composite_loss import CompositeLossCalculator
from src.aggregation.total_variance import ClientClassMoment
from src.osr.conformal import compute_finite_sample_quantile
from src.config import AppConfig


def extract_latent_mu(model_or_encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract deterministic latent mean mu(x; theta) across various encoder/network signatures."""
    if hasattr(model_or_encoder, "extract_features"):
        return model_or_encoder.extract_features(x)
    out = model_or_encoder(x)
    if isinstance(out, tuple):
        if len(out) == 4:
            # (logits, z, mu, logvar) from VIBClassifierNetwork
            return out[2]
        elif len(out) == 5:
            # Legacy 5-tuple
            return out[3]
        elif len(out) == 3:
            # (z, mu, logvar) from VIBEncoder
            return out[1]
        else:
            return out[0]
    return out


class FederatedClient:
    """Represents a blockchain edge node participating in federated learning.

    Executes Phase I local training (LaTeX Equations 86-89) and Phase II moment extraction (LaTeX Algorithm 1 Lines 333-340).
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
        self.global_profiles: Optional[Any] = None
        self.edge_detector: Optional[Any] = None

        # Initialize canonical VIB Classifier network (Encoder E_theta + Classifier C_phi)
        self.model = VIBClassifierNetwork(
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

        # Local optimizer: SGD with momentum 0.9 (LaTeX Algorithm 1 Line 326) or Adam
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

        # Count samples per class in local data for Class-Balanced Prior (Equation L_BLA)
        labels_np = self.dataset.labels.numpy()
        self.local_classes = np.unique(labels_np)
        self.samples_per_class = np.zeros(len(config.data.known_classes), dtype=int)
        for c in self.local_classes:
            if c < len(self.samples_per_class):
                self.samples_per_class[c] = int(np.sum(labels_np == c))

        from src.losses.logit_adjusted import compute_laplace_class_priors
        laplace_eps = getattr(config.loss, "laplace_eps", 0.1)
        self.class_priors = compute_laplace_class_priors(
            self.samples_per_class,
            num_classes=len(config.data.known_classes),
            epsilon=laplace_eps,
            device=self.device,
        )

        # In-VRAM / In-Device direct tensor caching for zero-PCIe training and moment extraction
        self.features_dev = self.dataset.features.to(self.device, non_blocking=True)
        self.labels_dev = self.dataset.labels.to(self.device, non_blocking=True)
        self.num_samples = self.features_dev.size(0)

        if self.cal_dataset is not None and len(self.cal_dataset) > 0:
            self.cal_features_dev = self.cal_dataset.features.to(self.device, non_blocking=True)
            self.cal_labels_dev = self.cal_dataset.labels.to(self.device, non_blocking=True)
            self.num_cal_samples = self.cal_features_dev.size(0)
        else:
            self.cal_features_dev = None
            self.cal_labels_dev = None
            self.num_cal_samples = 0

    def set_learning_rate(self, lr: float) -> None:
        """Update optimizer learning rate according to Cosine Annealing (LaTeX Equation 93)."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def synchronize(
        self,
        global_params: Dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Receive global model parameters w^t = (theta^t, phi^t) from orchestrator (LaTeX Algorithm 1 Line 293)."""
        # Handle full model state dict or encoder parameters dictionary
        if all(k.startswith("encoder.") or k.startswith("classifier.") for k in global_params.keys()):
            self.model.load_state_dict(global_params)
        elif hasattr(self.model, "set_encoder_parameters") and any(k.startswith("network.") for k in global_params.keys()):
            self.model.set_encoder_parameters(global_params)
            if "global_classifier_params" in kwargs and kwargs["global_classifier_params"] is not None:
                self.model.classifier.load_state_dict(kwargs["global_classifier_params"])
        else:
            self.model.load_state_dict(global_params, strict=False)

        # Clear optimizer momentum state to eliminate stale client drift
        if hasattr(self, "optimizer"):
            self.optimizer.state.clear()

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract full model state dict w_k = (theta_k, phi_k)."""
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def get_encoder_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract encoder state dictionary for federated parameter exchange."""
        return self.model.get_encoder_parameters()

    def get_classifier_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract classifier head state dictionary for federated parameter exchange."""
        return self.model.get_classifier_parameters()

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
        """Execute Phase I local training epochs (LaTeX Algorithm 1 Lines 317-330).

        Returns:
            - updated_model_or_encoder_params: State dict of the updated model
            - prototypes_dict: mapping class_id -> (mu_k_c, cov_k_c, sample_count)
            - average_loss_metrics
        """
        self.model.train()
        epochs = epochs or self.config.federated.local_epochs
        batch_size = batch_size or self.config.federated.batch_size

        num_samples = self.num_samples
        if num_samples == 0:
            return self.get_parameters(), {}, {}

        drop_last = num_samples > batch_size
        running_totals = torch.zeros(3, device=self.device)
        total_batches = 0
        end_limit = num_samples - (batch_size if drop_last else 0) + 1

        for _ in range(epochs):
            perm = torch.randperm(num_samples, device=self.device)
            feats = self.features_dev[perm]
            labs = self.labels_dev[perm]

            for i in range(0, end_limit, batch_size):
                x_b = feats[i : i + batch_size]
                y_b = labs[i : i + batch_size]
                if x_b.size(0) <= 1:
                    continue  # BatchNorm/LayerNorm stability

                self.optimizer.zero_grad(set_to_none=True)

                logits, z, mu, logvar = self.model(x_b)

                total_loss, breakdown = self.loss_calculator(
                    logits=logits,
                    targets=y_b,
                    mu=mu,
                    logvar=logvar,
                    class_priors=self.class_priors,
                    return_tensors=True,
                )

                total_loss.backward()
                # Gradient clipping for numerical stability (LaTeX Section 5.3: max_norm=1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_totals[0] += total_loss.detach()
                running_totals[1] += breakdown["loss_bla"].detach()
                running_totals[2] += breakdown["loss_vib"].detach()
                total_batches += 1

        # Compute average metrics across all batches (single GPU->CPU transfer)
        avg_metrics: Dict[str, float] = {}
        if total_batches > 0:
            avg_tensors = (running_totals / float(total_batches)).cpu().tolist()
            avg_metrics = {
                "loss_total": float(avg_tensors[0]),
                "loss_bla": float(avg_tensors[1]),
                "loss_vib": float(avg_tensors[2]),
            }

        return self.get_parameters(), {}, avg_metrics

    @torch.no_grad()
    def extract_phase_two_moments(
        self,
        frozen_encoder: nn.Module,
        num_blocks: int = 4,
        block_dim: int = 16,
        alpha: float = 0.05,
        min_cov_samples: Optional[int] = None,
    ) -> List[ClientClassMoment]:
        """Extract local class means over K_c and sub-block covariances over K_c^{cov} (Section 4.3, Eqs. eq:local_mean & eq:local_block_cov).

        Args:
            frozen_encoder: Permanently frozen global encoder f_{theta^*}.
            num_blocks: Number of sub-blocks B (default: 4).
            block_dim: Dimensionality of each sub-block d_b = d_z / B (default: 16).
            alpha: Mondrian conformal significance level.
            min_cov_samples: Minimum sample count N_min >= 3 required to upload local block covariances.

        Returns:
            List of ClientClassMoment submissions for all present local classes c in Y_k (N_{k,c} > 0).
        """
        frozen_encoder.eval()
        n_min = min_cov_samples if min_cov_samples is not None else getattr(self.config.federated, "min_cov_samples", 3)
        
        all_z = []
        chunk_size = 4096
        for i in range(0, self.num_samples, chunk_size):
            x_chunk = self.features_dev[i : i + chunk_size]
            mu_z = extract_latent_mu(frozen_encoder, x_chunk)
            all_z.append(mu_z)

        z_all = torch.cat(all_z, dim=0) if all_z else torch.empty((0, self.config.model.latent_dim), device=self.device)
        if getattr(self.config.osr, "normalize_features", False):
            norm_z = torch.norm(z_all, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            z_all = z_all / norm_z
        y_all = self.labels_dev

        actual_latent = z_all.size(1)
        if num_blocks * block_dim != actual_latent:
            num_blocks = min(num_blocks, actual_latent)
            block_dim = actual_latent // num_blocks

        use_mp = getattr(self.config.osr, "use_multi_prototype", True)
        m0 = getattr(self.config.osr, "num_prototypes_normal", 4)
        mc_atk = getattr(self.config.osr, "num_prototypes_attack", 1)

        moments_list: List[ClientClassMoment] = []

        for c in self.local_classes:
            c = int(c)
            idx = (y_all == c)
            count = int(idx.sum().item())
            if count < n_min:
                continue

            z_c = z_all[idx]  # Shape: (count, d_z)

            # Section 4.3, Eq. eq:local_mean: Local mean mu_{k,c} uploaded by k in K_c^{prof} (N_{k,c} >= N_min)
            mu_k_c = torch.mean(z_c, dim=0)  # Shape: (d_z,)

            # Privacy-Preserving Local k-means for Multi-Prototype Classes (e.g., M_1 = 4 for normal class)
            target_m = m0 if c == 0 else mc_atk
            local_protos = None
            local_proto_counts = None
            if use_mp and target_m > 1:
                k_actual = min(target_m, count)
                if k_actual > 1:
                    from sklearn.cluster import KMeans
                    import numpy as np
                    feat_np = z_c.cpu().numpy()
                    km = KMeans(n_clusters=k_actual, random_state=42, n_init=3)
                    km.fit(feat_np)
                    local_protos = torch.from_numpy(km.cluster_centers_).to(dtype=z_c.dtype).cpu()
                    counts_arr = np.bincount(km.labels_, minlength=k_actual)
                    local_proto_counts = [int(x) for x in counts_arr]
                else:
                    local_protos = mu_k_c.unsqueeze(0).cpu()
                    local_proto_counts = [count]

            # Section 4.3, Eq. eq:local_block_cov: Sub-block covariances Sigma_{k, c, G_b}
            block_covs: List[torch.Tensor] = []
            for b in range(num_blocks):
                start = b * block_dim
                end = start + block_dim
                z_c_b = z_c[:, start:end]  # (count, d_b)
                mu_c_b = mu_k_c[start:end]  # (d_b,)

                diff_b = z_c_b - mu_c_b.unsqueeze(0)  # (count, d_b)
                cov_b = torch.matmul(diff_b.t(), diff_b) / float(count)
                block_covs.append(cov_b.cpu())

            moments_list.append(
                ClientClassMoment(
                    client_id=self.client_id,
                    sample_count=count,
                    mu=mu_k_c.cpu(),
                    block_covs=block_covs,
                    cal_quantile=None,
                    class_id=c,
                    local_prototypes=local_protos,
                    local_proto_counts=local_proto_counts,
                )
            )

        return moments_list

    @torch.no_grad()
    def compute_calibration_scores(
        self,
        frozen_encoder: nn.Module,
        global_profiles: Dict[int, Any],
        num_blocks: int = 4,
        block_dim: int = 16,
    ) -> Dict[int, torch.Tensor]:
        """Compute local nonconformity scores s_{k,j}^{(c)} = D_block(f_{theta^*}(x_{k,j}), c) on D_{cal, k, c} (Section 4.4).

        Privacy note: Unlike Stage 2 moment profiling (which requires N_{k,c} >= N_min = 3 to prevent
        exact vector reconstruction in R^{d_z}), each scalar distance s_{k,j}^{(c)} in R_{>= 0} is a
        one-dimensional radial projection onto a (d_z - 1)-dimensional ellipsoid around mu_c and cannot
        be inverted to recover z_{k,j} in R^{d_z}; hence all nodes with n_{cal, k, c} >= 1 participate
        in K_{cal, c}.

        Returns:
            Dict mapping class c (with n_{cal, k, c} >= 1, i.e., k in K_{cal, c}) -> 1D tensor of scores.
        """
        from src.osr.block_mahalanobis import compute_block_mahalanobis_distances

        if self.cal_features_dev is None or self.num_cal_samples == 0:
            return {}

        frozen_encoder.eval()
        chunk_size = 4096
        all_cal_z = []
        for i in range(0, self.num_cal_samples, chunk_size):
            cx = self.cal_features_dev[i : i + chunk_size]
            c_mu = extract_latent_mu(frozen_encoder, cx)
            all_cal_z.append(c_mu)

        cal_z_all = torch.cat(all_cal_z, dim=0) if all_cal_z else torch.empty((0, self.config.model.latent_dim), device=self.device)
        cal_y_all = self.cal_labels_dev

        metric_type = getattr(self.config.osr, "metric_type", "mahalanobis")
        normalize_features = getattr(self.config.osr, "normalize_features", False)
        class_dists = compute_block_mahalanobis_distances(
            cal_z_all,
            global_profiles,
            num_blocks=num_blocks,
            block_dim=block_dim,
            normalize_features=normalize_features,
            metric_type=metric_type,
        )

        scores_by_class: Dict[int, torch.Tensor] = {}
        for c in global_profiles.keys():
            mask = (cal_y_all == c)
            if int(mask.sum().item()) > 0:
                scores_by_class[int(c)] = class_dists[c][mask].detach().cpu()

        return scores_by_class

    @torch.no_grad()
    def compute_calibration_histograms(
        self,
        frozen_encoder: nn.Module,
        global_profiles: Dict[int, Any],
        bin_edges: np.ndarray,
        num_blocks: int = 4,
        block_dim: int = 16,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Compute local right-closed empirical CDF histograms hat{F}_{k,c} on D_{cal, k, c} over shared grid [0, S_{max,c}] (Section 4.4)."""
        from src.osr.conformal import compute_right_closed_histogram

        scores_by_class = self.compute_calibration_scores(
            frozen_encoder=frozen_encoder,
            global_profiles=global_profiles,
            num_blocks=num_blocks,
            block_dim=block_dim,
        )
        histograms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for c, scores_t in scores_by_class.items():
            counts, edges = compute_right_closed_histogram(scores_t, bin_edges=bin_edges)
            histograms[int(c)] = (counts, edges)
        return histograms

    def receive_global_profiles(
        self,
        profiles: Dict[int, Any],
        frozen_encoder: nn.Module,
        unknown_label: int = -1,
        num_blocks: int = 4,
        block_dim: int = 16,
    ) -> None:
        """Receive synthesized global profiles from coordinator to configure local edge detector (Phase III, LaTeX Algorithm 2).

        Args:
            profiles: GlobalClassProfile dictionary mapping class_id -> profile.
            frozen_encoder: Permanently frozen global encoder theta_g^*.
            unknown_label: Label for rejected zero-day anomalies.
            num_blocks: Number of sub-blocks B.
            block_dim: Sub-block dimension d_b.
        """
        from src.osr.block_mahalanobis import WireSpeedEdgeDetector

        normalize_features = getattr(self.config.osr, "normalize_features", False)
        metric_type = getattr(self.config.osr, "metric_type", "mahalanobis")
        self.global_profiles = profiles
        self.edge_detector = WireSpeedEdgeDetector(
            frozen_encoder=frozen_encoder,
            global_profiles=profiles,
            unknown_label=unknown_label,
            num_blocks=num_blocks,
            block_dim=block_dim,
            normalize_features=normalize_features,
            metric_type=metric_type,
        )

    def evaluate_edge_traffic(
        self,
        x_stream: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Classify incoming streaming traffic at wire speed at this edge node (Phase III, Algorithm 2).

        Args:
            x_stream: Batch of streaming flow feature vectors.

        Returns:
            Tuple of (predictions, is_unknown_flags, candidate_classes, nonconformity_distances).
        """
        if self.edge_detector is None:
            raise RuntimeError(f"Client {self.client_id} has not received global profiles yet.")
        return self.edge_detector.predict_batch(x_stream)

