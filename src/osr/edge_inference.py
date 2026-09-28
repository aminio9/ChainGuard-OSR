"""Real-Time Edge Inference Engine (Algorithm 2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.osr.weibull_evt import compute_evt_probabilities
from src.osr.energy_ood import compute_free_energy_score
from src.osr.conformal import MondrianConformalCalibrator


class EdgeInferenceEngine:
    """3-Stage Hierarchical Edge Inference Engine for Open-Set Intrusion Detection (Algorithm 2).

    Stage 1: Weibull EVT probability discount on distance tail.
    Stage 2: Free Helmholtz Energy score for out-of-distribution density screening.
    Stage 3: Mondrian Conformal Prediction set filtering with finite-sample bounds.
    """

    def __init__(
        self,
        model: nn.Module,
        prototypes: Dict[int, torch.Tensor],
        weibull_models: Dict[int, Tuple[float, float, float]],
        conformal_calibrator: MondrianConformalCalibrator,
        unknown_class_id: int = -1,
        energy_temperature: float = 1.0,
    ):
        self.model = model
        self.prototypes = prototypes
        self.weibull_models = weibull_models
        self.conformal_calibrator = conformal_calibrator
        self.unknown_class_id = unknown_class_id
        self.energy_temperature = energy_temperature

    @torch.no_grad()
    def predict_flow(
        self, x: torch.Tensor
    ) -> Tuple[int, List[int], float, float]:
        """Execute Algorithm 2 for a single network flow vector x.

        Returns:
            (predicted_class, conformal_prediction_set, free_energy, unknown_probability)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(device)

        # Line 3: Deterministic latent forward pass z = mu(x)
        logits, _, z, mu, _ = self.model(x)

        # Line 4: Free energy score
        energy = float(compute_free_energy_score(logits, temperature=self.energy_temperature).item())

        # Line 5: EVT Weibull discounting
        # Output has shape (1, num_classes + 1) with index 0 being unknown
        evt_probs = compute_evt_probabilities(
            z, logits, self.prototypes, self.weibull_models
        )
        unknown_prob = float(evt_probs[0, 0].item())
        known_probs = evt_probs[0, 1:]  # shape: (num_classes,)

        # Line 6: Construct Mondrian Conformal Prediction Set
        conformal_sets = self.conformal_calibrator.predict_conformal_sets(known_probs.unsqueeze(0))
        c_set = conformal_sets[0]

        # Line 7 & 8: Decision logic
        if len(c_set) == 0:
            # Empty set -> Mathematically certified Unknown Zero-Day Attack
            pred_class = self.unknown_class_id
        else:
            # Argmax over accepted classes in conformal set
            sub_probs = [known_probs[c].item() for c in c_set]
            best_idx = int(np.argmax(sub_probs))
            pred_class = c_set[best_idx]

        return pred_class, c_set, energy, unknown_prob

    @torch.no_grad()
    def predict_batch(
        self, x: torch.Tensor
    ) -> Tuple[np.ndarray, List[List[int]], np.ndarray, np.ndarray]:
        """Execute Algorithm 2 on a batch of flow vectors."""
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        logits, _, z, mu, _ = self.model(x)
        energies = compute_free_energy_score(logits, temperature=self.energy_temperature).cpu().numpy()

        evt_probs = compute_evt_probabilities(z, logits, self.prototypes, self.weibull_models)
        unknown_probs = evt_probs[:, 0].cpu().numpy()
        known_probs = evt_probs[:, 1:]

        conformal_sets = self.conformal_calibrator.predict_conformal_sets(known_probs)

        preds = []
        for i, c_set in enumerate(conformal_sets):
            if len(c_set) == 0:
                preds.append(self.unknown_class_id)
            else:
                sub_probs = [known_probs[i, c].item() for c in c_set]
                best_idx = int(np.argmax(sub_probs))
                preds.append(c_set[best_idx])

        return np.array(preds, dtype=np.int64), conformal_sets, energies, unknown_probs
