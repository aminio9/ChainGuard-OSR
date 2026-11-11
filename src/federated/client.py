"""Federated Client Node (Edge Node).

Executes local node operations:
    - Representation Learning: Trains local encoder and classification head using composite
      loss (Balanced Logit Adjustment + VIB KL regularization).
    - Statistical Profiling: Partitions normal traffic against reference centroids via Voronoi cells
      and extracts local class moments (mean and block covariances).
    - Conformal Calibration: Computes local nonconformity histograms over a shared grid.
    - Edge Inference: Deploys WireSpeedEdgeDetector for real-time open-set detection.
"""

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
    """Represents a blockchain edge node participating in federated learning."""

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

        # Initialize canonical VIB Classifier network (Encoder + Classifier)
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

        # Local optimizer: SGD with momentum 0.9 or Adam
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

        # Count samples per class in local data for Class-Balanced Logit Adjustment
        labels_np = self.dataset.labels.numpy()
        self.local_classes = np.unique(labels_np)
        self.samples_per_class = np.zeros(len(config.data.known_classes), dtype=int)
        for c in self.local_classes:
            if c < len(self.samples_per_class):
                self.samples_per_class[c] = int(np.sum(labels_np == c))

        from src.losses.logit_adjusted import (
            compute_laplace_class_priors,
            compute_local_class_frequencies,
        )
        laplace_eps = getattr(config.loss, "laplace_eps", 0.0)
        if getattr(config.loss, "use_laplace_priors", False) and laplace_eps > 0.0:
            self.class_priors = compute_laplace_class_priors(
                self.samples_per_class,
                num_classes=len(config.data.known_classes),
                epsilon=laplace_eps,
                device=self.device,
            )
        else:
            # Normalized local class frequencies for Balanced Logit Adjustment
            self.class_priors = compute_local_class_frequencies(
                self.samples_per_class,
                num_classes=len(config.data.known_classes),
                device=self.device,
            )

        # In-VRAM tensor caching for zero-copy training and moment extraction
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
        """Update optimizer learning rate according to scheduled decay."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def synchronize(
        self,
        global_params: Dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Receive global model parameters from coordinator."""
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
        """Execute local training epochs.

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
                # Gradient clipping for numerical stability
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
    def extract_representative_centers(
        self,
        frozen_encoder: nn.Module,
        class_id: int = 0,
        num_centers: int = 4,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        """Extract local candidate representative centers for One-Shot Federated K-Means.

        Workflow:
            1. Extracts local latent embeddings for multi-modal normal traffic.
            2. Computes initial local candidate centers and cluster counts via K-Means.
            3. Uploads only candidate centers and scalar weights without exposing raw data.

        Args:
            frozen_encoder: Frozen representation extractor.
            class_id: Class index (0 for normal traffic).
            num_centers: Maximum candidate centers (default: 4).

        Returns:
            Tuple of (candidate_centers, sample_counts_per_center).
        """
        frozen_encoder.eval()
        idx = (self.labels_dev == class_id)
        count = int(idx.sum().item())
        if count == 0:
            return None, None

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

        z_c = z_all[idx]
        k_actual = min(num_centers, count)
        if k_actual <= 1:
            mean_v = torch.mean(z_c, dim=0, keepdim=True).cpu()
            return mean_v, [count]

        from sklearn.cluster import KMeans
        import numpy as np
        feat_np = z_c.cpu().numpy()
        km = KMeans(n_clusters=k_actual, random_state=42, n_init=3)
        km.fit(feat_np)
        centers = torch.from_numpy(km.cluster_centers_).to(dtype=z_c.dtype).cpu()
        counts_arr = np.bincount(km.labels_, minlength=k_actual)
        return centers, [int(x) for x in counts_arr]

    @torch.no_grad()
    def extract_phase_two_moments(
        self,
        frozen_encoder: nn.Module,
        num_blocks: int = 4,
        block_dim: int = 16,
        alpha: float = 0.05,
        min_cov_samples: Optional[int] = None,
        reference_centers: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[ClientClassMoment]:
        """Extract local moments via Voronoi partitioning against global reference centers.

        Methodology:
            1. Normal Traffic: Partition local samples into subclusters via nearest reference center
               and extract local mean and block covariance matrices per subcluster.
            2. Attack Classes: Estimate moments directly on class samples as single-mode distributions.

        Args:
            frozen_encoder: Permanently frozen global encoder.
            num_blocks: Number of sub-blocks B (default: 4).
            block_dim: Dimensionality of each sub-block d_b = latent_dim / B.
            alpha: Mondrian conformal significance level.
            min_cov_samples: Minimum sample count required to upload local block covariances.
            reference_centers: Dict mapping class_id -> global reference centers tensor.

        Returns:
            List of ClientClassMoment submissions for all present local classes and active clusters.
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

        moments_list: List[ClientClassMoment] = []

        for c in self.local_classes:
            c = int(c)
            idx = (y_all == c)
            count = int(idx.sum().item())
            if count < n_min:
                continue

            z_c = z_all[idx]  # Shape: (count, latent_dim)

            # Check if multi-modal reference centers exist for class c
            if reference_centers is not None and c in reference_centers and reference_centers[c] is not None:
                ref_c = reference_centers[c].to(device=self.device)  # Shape: (M_c, d_z)
                M_c = ref_c.size(0)

                # Euclidean Voronoi Partitioning: m = argmin_{j} ||z_i - c_j||_2^2
                dists_to_ref = torch.cdist(z_c, ref_c)  # Shape: (count, M_c)
                assignments = torch.argmin(dists_to_ref, dim=-1)  # Shape: (count,)

                for m in range(M_c):
                    m_mask = (assignments == m)
                    n_m = int(m_mask.sum().item())
                    if n_m < n_min:
                        # Cluster omitted if samples are below support threshold N_min
                        continue

                    z_m = z_c[m_mask]
                    mu_m = torch.mean(z_m, dim=0)

                    block_covs_m: List[torch.Tensor] = []
                    for b in range(num_blocks):
                        start = b * block_dim
                        end = start + block_dim
                        z_m_b = z_m[:, start:end]
                        diff_m_b = z_m_b - mu_m[start:end].unsqueeze(0)
                        cov_m_b = torch.matmul(diff_m_b.t(), diff_m_b) / float(n_m)
                        block_covs_m.append(cov_m_b.cpu())

                    moments_list.append(
                        ClientClassMoment(
                            client_id=self.client_id,
                            sample_count=n_m,
                            mu=mu_m.cpu(),
                            block_covs=block_covs_m,
                            cal_quantile=None,
                            class_id=c,
                            cluster_id=m,
                        )
                    )
            else:
                # Single-Prototype Class (Mc = 1): direct moment estimation on whole class
                mu_k_c = torch.mean(z_c, dim=0)
                block_covs: List[torch.Tensor] = []
                for b in range(num_blocks):
                    start = b * block_dim
                    end = start + block_dim
                    z_c_b = z_c[:, start:end]
                    diff_b = z_c_b - mu_k_c[start:end].unsqueeze(0)
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
                        cluster_id=0,
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
        """Compute local nonconformity scores (Mahalanobis distances) on calibration data.

        Returns:
            Dict mapping class_id -> 1D tensor of nonconformity scores.
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
        """Compute local empirical CDF histograms on calibration data over a shared grid."""
        from src.osr.conformal import compute_right_closed_histogram

        scores_by_class = self.compute_calibration_scores(
            frozen_encoder=frozen_encoder,
            global_profiles=global_profiles,
            num_blocks=num_blocks,
            block_dim=block_dim,
        )
        histograms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for c, scores_t in scores_by_class.items():
            counts, edges = compute_right_closed_histogram(scores_t, bin_edges=bin_edges, include_overflow=True)
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
        """Receive synthesized global profiles from coordinator to configure local edge detector.

        Args:
            profiles: GlobalClassProfile dictionary mapping class_id -> profile.
            frozen_encoder: Permanently frozen global encoder.
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

