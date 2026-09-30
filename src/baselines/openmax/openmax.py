"""OpenMax (Bendale & Boult, 2016) baseline adapted to Federated Learning.

Reference:
    Bendale & Boult, 'Towards Open Set Deep Networks', CVPR 2016.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.baselines.base import BaseBaseline
from src.baselines.fedavg.fedavg import FedAvgBaseline
from src.data.base import FlowDataset


class OpenMaxBaseline(BaseBaseline):
    """OpenMax EVT meta-recognition on top of federated learned representations."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        tail_size: int = 20,
        alpha_rank: int = 3,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            name="OpenMax",
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank
        self.fedavg = FedAvgBaseline(
            in_features=in_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        # Class Mean Activation Vectors (MAVs) and Weibull models
        self.mavs: Dict[int, np.ndarray] = {}
        self.weibull_models: Dict[int, Tuple[float, float, float]] = {}

    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute one round of federated training using FedAvg backbone."""
        return self.fedavg.fit_round(
            round_idx=round_idx,
            client_datasets=client_datasets,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            **kwargs,
        )

    def calibrate(self, cal_dataset: FlowDataset, alpha: float = 0.05) -> float:
        """Fit class MAVs and Weibull EVT distributions on known calibration set."""
        self.fedavg.model.eval()
        x_cal = cal_dataset.features.to(self.device)
        y_cal = cal_dataset.labels.numpy()

        with torch.no_grad():
            logits, z = self.fedavg.model(x_cal)
            z_np = z.cpu().numpy()

        # 1. Compute Class Mean Activation Vectors (MAVs)
        for c in range(self.num_classes):
            idx_c = np.where(y_cal == c)[0]
            if len(idx_c) > 0:
                self.mavs[c] = np.mean(z_np[idx_c], axis=0)
            else:
                self.mavs[c] = np.zeros(self.latent_dim)

        # 2. Fit Weibull distributions to tail distances
        for c in range(self.num_classes):
            idx_c = np.where(y_cal == c)[0]
            if len(idx_c) < 3:
                self.weibull_models[c] = (1.0, 1.0, 0.0)
                continue

            mav = self.mavs[c]
            dists = np.linalg.norm(z_np[idx_c] - mav, axis=1)
            actual_tail = min(self.tail_size, len(dists))
            sorted_dists = np.sort(dists)[-actual_tail:]
            tau_dist = float(sorted_dists[0])
            tail_excess = sorted_dists - tau_dist

            try:
                shape, loc, scale = stats.weibull_min.fit(tail_excess, floc=0)
                self.weibull_models[c] = (float(shape), float(scale), tau_dist)
            except Exception:
                self.weibull_models[c] = (1.0, float(np.mean(tail_excess) + 1e-4), tau_dist)

        # 3. Compute OpenMax probabilities on calibration set
        evt_probs = self._compute_openmax_probs(logits, z)
        p_unk_cal = evt_probs[:, 0].cpu().numpy()

        q_val = 100.0 * (1.0 - alpha)
        tau = float(np.percentile(p_unk_cal, q_val))
        self.threshold = max(0.1, tau)
        return self.threshold

    def _compute_openmax_probs(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute OpenMax revised activation vectors and pseudo-unknown probability."""
        batch_size = logits.size(0)
        z_np = z.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()

        modified_logits = np.zeros((batch_size, self.num_classes + 1), dtype=np.float32)

        for i in range(batch_size):
            ranked_classes = np.argsort(logits_np[i])[::-1]
            revised_v = logits_np[i].copy()
            unk_v = 0.0

            for rank in range(min(self.alpha_rank, self.num_classes)):
                c = ranked_classes[rank]
                weight = 1.0 - (rank / float(self.alpha_rank))
                dist = np.linalg.norm(z_np[i] - self.mavs[c])

                if c in self.weibull_models:
                    shape, scale, tau_d = self.weibull_models[c]
                    if dist > tau_d and scale > 0:
                        scaled_dist = (dist - tau_d) / scale
                        exponent = np.clip(scaled_dist ** min(shape, 20.0), 0.0, 100.0)
                        w_score = 1.0 - np.exp(-exponent)
                    else:
                        w_score = 0.0
                else:
                    w_score = 0.0

                w_score = float(np.clip(w_score, 0.0, 1.0))
                omega = 1.0 - weight * w_score
                revised_v[c] = logits_np[i, c] * omega
                unk_v += logits_np[i, c] * (1.0 - omega)

            modified_logits[i, 0] = unk_v  # Class 0 represents unknown
            modified_logits[i, 1:] = revised_v

        # Softmax over (num_classes + 1)
        mod_t = torch.tensor(modified_logits, device=self.device)
        return torch.softmax(mod_t, dim=-1)

    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and OpenMax anomaly scores."""
        self.fedavg.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits, z = self.fedavg.model(x)
            evt_probs = self._compute_openmax_probs(logits, z)
            p_unk = evt_probs[:, 0].cpu().numpy()
            known_probs = evt_probs[:, 1:].cpu().numpy()
            cand_classes = np.argmax(known_probs, axis=-1)

        preds = np.where(p_unk > self.threshold, self.num_classes, cand_classes)
        return preds, p_unk, cand_classes
