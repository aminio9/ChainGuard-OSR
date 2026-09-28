"""Class-Conditional Mondrian Conformal Prediction with Finite-Sample Guarantees (Formulas 13-15 & Theorem 2)."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import torch


class MondrianConformalCalibrator:
    """Implements Class-Conditional Mondrian Conformal Prediction Calibration.

    Provides finite-sample coverage guarantees:
        P( Y in C_{1-alpha}(X) | Y = c ) >= 1 - alpha

    And by Theorem 2, the false alarm rate on known traffic is strictly bounded:
        P( C_{1-alpha}(X) = empty | (X, Y) ~ P_known ) <= alpha
    """

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        # Mapping class_c -> empirical quantile q_hat_c
        self.quantiles: Dict[int, float] = {}

    def calibrate(
        self,
        cal_probabilities: torch.Tensor | np.ndarray,
        cal_labels: torch.Tensor | np.ndarray,
        num_classes: Optional[int] = None,
    ) -> Dict[int, float]:
        """Compute finite-sample corrected conformal quantiles per class (Formulas 13 & 14).

        Args:
            cal_probabilities: Softmax probabilities of shape (N_cal, num_classes).
            cal_labels: Ground truth class indices of shape (N_cal,).
            num_classes: Total known classes.
        """
        if isinstance(cal_probabilities, torch.Tensor):
            probs_np = cal_probabilities.detach().cpu().numpy()
        else:
            probs_np = np.asarray(cal_probabilities)

        if isinstance(cal_labels, torch.Tensor):
            labels_np = cal_labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(cal_labels)

        if num_classes is None:
            num_classes = probs_np.shape[1]

        self.quantiles = {}

        for c in range(num_classes):
            idx = np.where(labels_np == c)[0]
            if len(idx) == 0:
                # Conservative fallback if no calibration sample for this class
                self.quantiles[c] = 1.0
                continue

            c_probs = probs_np[idx, c]
            # Non-conformity score: s_i = 1 - p(y_i)
            scores = 1.0 - c_probs
            n = len(scores)

            # Finite-sample corrected quantile level: ceil((n + 1) * (1 - alpha)) / n
            q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / float(n)
            q_level = min(1.0, max(0.0, q_level))

            q_hat = float(np.quantile(scores, q_level, method="higher"))
            self.quantiles[c] = q_hat

        return self.quantiles

    def predict_conformal_sets(
        self,
        test_probabilities: torch.Tensor | np.ndarray,
    ) -> List[List[int]]:
        """Construct conformal prediction sets for test instances (Formula 15).

        Mathematical Formulation:
            C_{1-alpha}(x) = { c | 1 - p_c(x) <= q_hat_c }

        Returns:
            List of accepted class lists. Empty list [] indicates Unknown Zero-Day attack.
        """
        if isinstance(test_probabilities, torch.Tensor):
            probs_np = test_probabilities.detach().cpu().numpy()
        else:
            probs_np = np.asarray(test_probabilities)

        prediction_sets: List[List[int]] = []
        for prob in probs_np:
            pset = [c for c, q_hat in self.quantiles.items() if (1.0 - prob[c]) <= q_hat]
            prediction_sets.append(pset)

        return prediction_sets
