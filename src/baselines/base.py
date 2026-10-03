"""Base abstraction for federated learning baselines with strict zero-leakage compliance.

Every baseline must strictly respect:
1. Shared representation backbone (deterministic or VIB MLP).
2. Separation of closed-set training (C_K) and unseen open-set evaluation (C_U).
3. Calibration set (D_cal) containing strictly known classes (C_K).
4. Unified Scoring Head (MSP, Energy, Centralized Mahalanobis, or Native) & Fair Mondrian Conformal Calibration.
5. Uniform evaluation pipeline returning standard classification, OSR (including FPR95), and spectral rank metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.base import FlowDataset
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics
from src.osr.conformal import compute_finite_sample_quantile


def create_baseline_optimizer(
    model: nn.Module,
    lr: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create optimizer matching the experimental configuration for fair baseline comparison."""
    if optimizer_name.lower() == "sgd":
        eff_lr = lr if lr >= 0.01 else 0.01
        return torch.optim.SGD(model.parameters(), lr=eff_lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def compute_feature_svd_spectrum(z: np.ndarray) -> Dict[str, Any]:
    """Compute Stable Rank and normalized SVD singular value spectrum of latent feature matrix Z.

    Stable Rank (Rudelson & Vershynin):
        r_stable(Z) = ||Z||_F^2 / ||Z||_2^2 = (sum_i sigma_i^2) / sigma_1^2
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 2 or z_arr.shape[0] < 2:
        return {"stable_rank": 1.0, "singular_values_norm": [1.0]}

    # Center features
    z_centered = z_arr - np.mean(z_arr, axis=0, keepdims=True)
    try:
        s_vals = np.linalg.svd(z_centered, full_matrices=False, compute_uv=False)
    except Exception:
        return {"stable_rank": 1.0, "singular_values_norm": [1.0]}

    sigma_max = float(s_vals[0]) if len(s_vals) > 0 else 0.0
    if sigma_max <= 1e-12:
        return {"stable_rank": 1.0, "singular_values_norm": np.ones_like(s_vals).tolist()}

    s_norm = s_vals / sigma_max
    stable_rank = float(np.sum(s_vals ** 2) / (sigma_max ** 2))
    return {
        "stable_rank": stable_rank,
        "singular_values_norm": [float(v) for v in s_norm],
    }


class BaseBaseline(ABC):
    """Abstract base class for all federated baseline methods."""

    def __init__(
        self,
        name: str,
        in_features: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int | List[int] = 128,
        device: Optional[torch.device] = None,
    ):
        self.name = name
        self.in_features = in_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global model components (to be initialized by subclasses)
        self.global_encoder: Optional[nn.Module] = None
        self.global_classifier: Optional[nn.Module] = None
        self.threshold: float = 0.5
        self.mondrian_thresholds: Dict[int, float] = {}
        self.cal_class_counts: Dict[int, int] = {}
        self.is_fitted: bool = False

    @abstractmethod
    def fit_round(
        self,
        round_idx: int,
        client_datasets: List[FlowDataset],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Execute a single communication round of federated training across clients."""
        pass

    def fit(
        self,
        client_datasets: List[FlowDataset],
        num_rounds: int = 15,
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        """Execute full multi-round federated training."""
        round_histories: List[Dict[str, float]] = []
        ckpt_dir_str = kwargs.get("checkpoint_dir", None)
        target_model = getattr(self, "model", None) or getattr(self, "global_model", None)
        latest_ckpt = None
        start_round = 1

        if ckpt_dir_str and target_model is not None:
            ckpt_path_dir = Path(ckpt_dir_str)
            ckpt_path_dir.mkdir(parents=True, exist_ok=True)
            latest_ckpt = ckpt_path_dir / f"{self.name.lower().replace(' ', '_')}_checkpoint.pt"
            if kwargs.get("resume", True) and latest_ckpt.exists():
                try:
                    saved_dict = torch.load(latest_ckpt, map_location=self.device, weights_only=False)
                    target_model.load_state_dict(saved_dict["model_state_dict"])
                    saved_r = saved_dict.get("round", 0)
                    if saved_r < num_rounds:
                        start_round = saved_r + 1
                        print(f"    [{self.name}] Resuming from round {start_round}/{num_rounds} via {latest_ckpt}...", flush=True)
                    else:
                        print(f"    [{self.name}] Checkpoint already completed {saved_r}/{num_rounds} rounds. Skipping loop.", flush=True)
                        start_round = num_rounds + 1
                except Exception as e:
                    print(f"    [{self.name}] Warning: Failed to load {latest_ckpt}: {e}. Starting fresh.", flush=True)
                    start_round = 1

        t_fit_start = time.perf_counter()
        log_interval = max(1, num_rounds // 10) if num_rounds >= 20 else max(1, num_rounds // 5)
        for r in range(start_round, num_rounds + 1):
            t_r0 = time.perf_counter()
            stats = self.fit_round(
                round_idx=r,
                client_datasets=client_datasets,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                **kwargs,
            )
            round_histories.append(stats)

            if latest_ckpt and target_model is not None and (r % log_interval == 0 or r == num_rounds):
                torch.save({
                    "round": r,
                    "model_state_dict": target_model.state_dict(),
                }, latest_ckpt)

            if r == 1 or r % log_interval == 0 or r == num_rounds:
                elapsed = time.perf_counter() - t_fit_start
                r_sec = time.perf_counter() - t_r0
                rounds_done = r - start_round + 1
                avg_sec = elapsed / max(rounds_done, 1)
                eta_sec = avg_sec * (num_rounds - r)
                loss_val = stats.get("train_loss", stats.get("loss", 0.0)) if isinstance(stats, dict) else 0.0
                print(
                    f"    [{self.name}] Round {r:3d}/{num_rounds} | Loss: {loss_val:.4f} | "
                    f"Speed: {r_sec:.2f}s/rnd | Elapsed: {elapsed:.1f}s | ETA: {eta_sec:.1f}s",
                    flush=True,
                )
        self.is_fitted = True
        return round_histories

    def extract_logits_and_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract classification logits (N, C_K) and latent embeddings (N, d_z) from the underlying model."""
        x = x.to(self.device)
        target_model = getattr(self, "model", None) or getattr(self, "global_model", None)
        if target_model is None and hasattr(self, "fedavg"):
            target_model = getattr(self.fedavg, "model", None)

        if target_model is not None:
            target_model.eval()
            with torch.no_grad():
                out = target_model(x)
                if isinstance(out, tuple):
                    logits = out[0][:, : self.num_classes]
                    z = out[-1]
                    return logits, z
        raise NotImplementedError(f"Baseline {self.name} must expose model or override extract_logits_and_features.")

    @abstractmethod
    def calibrate(
        self,
        cal_dataset: FlowDataset,
        alpha: float = 0.05,
    ) -> float:
        """Calibrate rejection threshold on known-class validation data."""
        pass

    @abstractmethod
    def predict_batch(
        self,
        x: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels and anomaly scores for input batch.

        Returns:
            - preds: predicted class indices (unknown_label = self.num_classes)
            - anomaly_scores: scalar anomaly scores (higher = more anomalous)
            - cand_classes: predicted known candidate class indices
        """
        pass

    def calibrate_unified(
        self,
        cal_dataset: FlowDataset,
        alpha: float = 0.05,
        scoring_method: str = "native",
        use_mondrian: bool = False,
        temperature: float = 1.0,
    ) -> float:
        """Unified scoring & Mondrian conformal calibration for fair baseline comparison (Section 2)."""
        y_cal = cal_dataset.labels.cpu().numpy()
        for c in range(self.num_classes):
            self.cal_class_counts[c] = int(np.sum(y_cal == c))

        if scoring_method == "native" and not use_mondrian:
            return self.calibrate(cal_dataset, alpha=alpha)

        # Compute scores on calibration set
        x_cal = cal_dataset.features.to(self.device)
        if scoring_method == "native":
            tau_global = self.calibrate(cal_dataset, alpha=alpha)
            _, cal_scores, _ = self.predict_batch(x_cal)
        else:
            logits_cal, z_cal = self.extract_logits_and_features(x_cal)
            if scoring_method == "msp":
                probs = torch.softmax(logits_cal, dim=-1)
                cal_scores = (1.0 - torch.max(probs, dim=-1)[0]).cpu().numpy()
            elif scoring_method == "energy":
                cal_scores = (-temperature * torch.logsumexp(logits_cal / temperature, dim=-1)).cpu().numpy()
            elif scoring_method == "mahalanobis":
                # Fit pooled class-conditional Gaussian on calibration set
                self._unified_means = {}
                pooled_cov = torch.zeros((z_cal.size(1), z_cal.size(1)), device=self.device)
                for c in range(self.num_classes):
                    idx_c = np.where(y_cal == c)[0]
                    if len(idx_c) > 0:
                        zc = z_cal[idx_c]
                        mc = zc.mean(dim=0)
                        self._unified_means[c] = mc
                        diff = zc - mc
                        pooled_cov += torch.matmul(diff.t(), diff)
                    else:
                        self._unified_means[c] = torch.zeros(z_cal.size(1), device=self.device)
                pooled_cov = pooled_cov / max(1.0, float(len(y_cal))) + 1e-4 * torch.eye(z_cal.size(1), device=self.device)
                self._unified_inv_cov = torch.linalg.pinv(pooled_cov)
                dists = []
                for c in range(self.num_classes):
                    d_c = z_cal - self._unified_means[c].unsqueeze(0)
                    sq = torch.sum(torch.matmul(d_c, self._unified_inv_cov) * d_c, dim=-1)
                    dists.append(torch.sqrt(torch.clamp(sq, min=0.0)))
                cal_scores = torch.min(torch.stack(dists, dim=-1), dim=-1)[0].cpu().numpy()
            else:
                raise ValueError(f"Unknown unified scoring_method '{scoring_method}'")

        tau_global = compute_finite_sample_quantile(cal_scores, alpha=alpha)
        self.threshold = tau_global

        if use_mondrian:
            self.mondrian_thresholds = {}
            for c in range(self.num_classes):
                scores_c = cal_scores[y_cal == c]
                if len(scores_c) >= 2:
                    self.mondrian_thresholds[c] = compute_finite_sample_quantile(scores_c, alpha=alpha)
                else:
                    self.mondrian_thresholds[c] = tau_global

        return tau_global

    def predict_batch_unified(
        self,
        x: torch.Tensor,
        scoring_method: str = "native",
        use_mondrian: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict batch using unified scoring mechanism and optional Mondrian class-conditional thresholds."""
        if scoring_method == "native" and not use_mondrian:
            return self.predict_batch(x)

        if scoring_method == "native":
            _, anomaly_scores, cand_classes = self.predict_batch(x)
        else:
            logits, z = self.extract_logits_and_features(x)
            if scoring_method == "msp":
                probs = torch.softmax(logits, dim=-1)
                max_p, cand_t = torch.max(probs, dim=-1)
                anomaly_scores = (1.0 - max_p).cpu().numpy()
                cand_classes = cand_t.cpu().numpy()
            elif scoring_method == "energy":
                anomaly_scores = (-temperature * torch.logsumexp(logits / temperature, dim=-1)).cpu().numpy()
                cand_classes = torch.argmax(logits, dim=-1).cpu().numpy()
            elif scoring_method == "mahalanobis":
                dists = []
                for c in range(self.num_classes):
                    d_c = z - self._unified_means[c].unsqueeze(0)
                    sq = torch.sum(torch.matmul(d_c, self._unified_inv_cov) * d_c, dim=-1)
                    dists.append(torch.sqrt(torch.clamp(sq, min=0.0)))
                dist_mat = torch.stack(dists, dim=-1)
                min_d, cand_t = torch.min(dist_mat, dim=-1)
                anomaly_scores = min_d.cpu().numpy()
                cand_classes = cand_t.cpu().numpy()
            else:
                raise ValueError(f"Unknown unified scoring_method '{scoring_method}'")

        if use_mondrian and self.mondrian_thresholds:
            thresh_vec = np.array([self.mondrian_thresholds.get(int(c), self.threshold) for c in cand_classes])
            preds = np.where(anomaly_scores > thresh_vec, self.num_classes, cand_classes)
        else:
            preds = np.where(anomaly_scores > self.threshold, self.num_classes, cand_classes)

        return preds, anomaly_scores, cand_classes

    def compute_feature_svd_spectrum(self, x: torch.Tensor) -> Dict[str, Any]:
        """Extract latent features for input tensor x and compute Stable Rank & SVD spectrum."""
        _, z = self.extract_logits_and_features(x)
        spec = compute_feature_svd_spectrum(z.cpu().numpy())
        spec["normalized_singular_values"] = spec["singular_values_norm"]
        return spec

    def evaluate(
        self,
        test_open_ds: FlowDataset,
        test_closed_ds: FlowDataset,
        cal_ds: FlowDataset,
        conformal_alpha: float = 0.05,
        scoring_method: str = "native",
        score_type: Optional[str] = None,
        use_mondrian: bool = False,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation conforming to Q1 audit standards.

        Strict invariants:
        - Unknown classes never enter cal_ds or training.
        - Closed-set accuracy evaluated on test_closed_ds.
        - Open-set AUROC, AUPR, FPR95, Unknown-Recall, KFR evaluated on test_open_ds.
        """
        eff_scoring = score_type if score_type is not None else scoring_method

        # Step 1: Calibration
        tau = self.calibrate_unified(
            cal_ds,
            alpha=conformal_alpha,
            scoring_method=eff_scoring,
            use_mondrian=use_mondrian,
        )
        self.threshold = tau

        # Step 2: Open-set evaluation
        x_open = test_open_ds.features.to(self.device)
        preds_open, anomaly_scores_open, _ = self.predict_batch_unified(
            x_open,
            scoring_method=eff_scoring,
            use_mondrian=use_mondrian,
        )
        y_open_true = test_open_ds.labels.numpy()

        osr_metrics = compute_osr_metrics(
            y_true=y_open_true,
            y_pred=preds_open,
            anomaly_scores=anomaly_scores_open,
            num_known_classes=self.num_classes,
            unknown_label=self.num_classes,
        )

        # Step 3: Closed-set evaluation
        x_closed = test_closed_ds.features.to(self.device)
        preds_closed, _, cand_classes_closed = self.predict_batch_unified(
            x_closed,
            scoring_method=eff_scoring,
            use_mondrian=use_mondrian,
        )
        y_closed_true = test_closed_ds.labels.numpy()
        closed_preds = cand_classes_closed
        closed_metrics = compute_classification_metrics(
            y_true=y_closed_true,
            y_pred=closed_preds,
            num_classes=self.num_classes,
        )

        # Step 4: Spectral rank analysis on closed-set latent features (Ablation A1)
        try:
            _, z_closed = self.extract_logits_and_features(x_closed)
            svd_stats = compute_feature_svd_spectrum(z_closed.cpu().numpy())
        except Exception:
            svd_stats = {"stable_rank": 1.0, "singular_values_norm": [1.0]}

        results = {
            "baseline_name": self.name,
            "scoring_method": eff_scoring,
            "score_type": eff_scoring,
            "use_mondrian": use_mondrian,
            "threshold": float(tau),
            "cal_class_counts": dict(self.cal_class_counts),
            "closed_set_accuracy": float(closed_metrics["accuracy"]),
            "accuracy": float(closed_metrics["accuracy"]),
            "balanced_accuracy": float(closed_metrics["balanced_accuracy"]),
            "closed_set_macro_f1": float(closed_metrics["macro_f1"]),
            "auroc": float(osr_metrics["auroc"]),
            "aupr": float(osr_metrics["aupr"]),
            "fpr95": float(osr_metrics["fpr95"]),
            "unknown_recall": float(osr_metrics["unknown_recall"]),
            "unknown_precision": float(osr_metrics["unknown_precision"]),
            "unknown_f1": float(osr_metrics["unknown_f1"]),
            "open_set_macro_f1": float(osr_metrics["open_set_macro_f1"]),
            "kfr": float(osr_metrics["kfr"]),
            "stable_rank": float(svd_stats["stable_rank"]),
            "singular_values_norm": svd_stats["singular_values_norm"],
        }
        return results

