"""Extreme Value Theory (EVT) Weibull Meta-Recognition and OpenMax Probability Adaptation (Formula 11)."""

from __future__ import annotations

from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
import numpy as np
import scipy.stats as stats
import torch


class WeibullEVTFitter:
    """Fits Weibull distribution to empirical tail distances of known classes (Formula 11)."""

    def __init__(self, tail_size: int = 20):
        self.tail_size = tail_size
        # Mapping class_id -> (shape_xi, scale_kappa, loc_tau)
        self.weibull_models: Dict[int, Tuple[float, float, float]] = {}

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        prototypes: Dict[int, torch.Tensor],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Fit Weibull distribution to the tail_size largest Euclidean distances for each class."""
        if model is not None:
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                _, _, _, mu, _ = model(features.to(device))
                feats_np = mu.cpu().numpy()
        else:
            feats_np = features.detach().cpu().numpy()
        lbls_np = labels.detach().cpu().numpy()

        for c, proto in prototypes.items():
            idx = np.where(lbls_np == c)[0]
            if len(idx) < 3:
                # Not enough points to fit EVT tail reliably; fallback default
                self.weibull_models[c] = (1.0, 1.0, 0.0)
                continue

            proto_np = proto.detach().cpu().numpy()
            dists = np.linalg.norm(feats_np[idx] - proto_np, axis=1)

            # Extract extreme tail of distances
            actual_tail = min(self.tail_size, len(dists))
            sorted_dists = np.sort(dists)[-actual_tail:]

            tau = float(sorted_dists[0])
            tail_excess = sorted_dists - tau

            try:
                # Fit 2-parameter Weibull on excess distances with floc=0
                shape, loc, scale = stats.weibull_min.fit(tail_excess, floc=0)
                self.weibull_models[c] = (float(shape), float(scale), tau)
            except Exception:
                self.weibull_models[c] = (1.0, float(np.mean(tail_excess) + 1e-4), tau)


def compute_evt_probabilities(
    z: torch.Tensor,
    logits: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
    weibull_models: Dict[int, Tuple[float, float, float]],
    alpha_rank: int = 5,
) -> torch.Tensor:
    """Discount known class activations using Weibull CDF and compute unknown class probability (Formula 11).

    Mathematical Formulation:
        omega_c(x) = 1 - exp( - ( max(0, ||z - mu_c|| - tau_c) / kappa_c )^xi_c )
        z_0 = sum_{c=1}^C z_c * (1 - omega_c(x))

    Returns:
        Calibrated probabilities of shape (batch, num_classes + 1) where index 0 is Unknown class.
    """
    batch_size = z.size(0)
    num_classes = logits.size(1)
    device = z.device

    # Rank classes by logit value
    ranked_indices = torch.argsort(logits, dim=-1, descending=True)  # (batch, num_classes)

    modified_logits = logits.clone()
    unknown_logits = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        z_i = z[i]
        for rank in range(min(alpha_rank, num_classes)):
            c = int(ranked_indices[i, rank].item())
            if c not in prototypes or c not in weibull_models:
                continue

            proto = prototypes[c].to(device)
            dist = torch.norm(z_i - proto, p=2).item()
            shape_xi, scale_kappa, loc_tau = weibull_models[c]

            excess = max(0.0, dist - loc_tau)
            # Weibull CDF: 1 - exp(- (excess / scale)^shape)
            cdf = 1.0 - np.exp(-((excess / (scale_kappa + 1e-7)) ** shape_xi))

            # Weight rank discount: highest ranked classes receive full discount
            alpha_weight = 1.0 - (rank / float(alpha_rank))
            omega = cdf * alpha_weight

            original_logit = logits[i, c]
            discounted_logit = original_logit * (1.0 - omega)
            modified_logits[i, c] = discounted_logit
            unknown_logits[i] += original_logit * omega

    # Prepend unknown class logit at index 0
    full_logits = torch.cat([unknown_logits.unsqueeze(1), modified_logits], dim=1)
    return torch.softmax(full_logits, dim=-1)
