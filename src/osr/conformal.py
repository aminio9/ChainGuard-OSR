"""Class-Conditional Mondrian Conformal Prediction with Finite-Sample Guarantees (Equations 22-25 & Theorem 2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def compute_finite_sample_quantile(
    scores: np.ndarray | torch.Tensor,
    alpha: float = 0.05,
) -> float:
    """Compute finite-sample corrected quantile (Equations 22 & 24).

    Mathematical Formulation (Equation 22):
        q^{(1-alpha)} = Quantile( S, ceil((n + 1)(1 - alpha)) / n )

    Theorem 2 Guarantee:
        P( S(X_test) > tau_{alpha, c} | Y_test = c ) <= alpha

    Args:
        scores: Array or tensor of nonconformity scores for class c.
        alpha: Significance level (default: 0.05 for 95% coverage).

    Returns:
        Empirical finite-sample corrected quantile float.
    """
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy().flatten()
    else:
        scores_np = np.asarray(scores).flatten()

    n = len(scores_np)
    if n == 0:
        return float("inf")

    # Quantile index: ceil((n + 1) * (1 - alpha)) / n
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / float(n)
    q_level = float(min(1.0, max(0.0, q_level)))

    # Use 'higher' quantile method as standard in conformal prediction literature
    quantile_val = float(np.quantile(scores_np, q_level, method="higher"))
    return quantile_val


def aggregate_distributed_quantiles(
    client_quantiles: List[Tuple[float, int]],
) -> float:
    """Aggregate local client quantiles into global Mondrian conformal threshold (Equation 23).

    Mathematical Formulation (Equation 23):
        tau_{alpha, c} = sum_{k in K_c} ( n_{cal, c}^{(k)} / N_{cal, c} ) * q_k^{(1 - alpha)}(c)

    Args:
        client_quantiles: List of tuples (local_quantile, sample_count).

    Returns:
        Weighted aggregate conformal threshold tau_{alpha, c}.
    """
    if not client_quantiles:
        return float("inf")

    total_samples = sum(count for _, count in client_quantiles)
    if total_samples == 0:
        return float("inf")

    weighted_sum = sum(q * count for q, count in client_quantiles)
    return float(weighted_sum / total_samples)


class MondrianConformalCalibrator:
    """Implements Class-Conditional Mondrian Conformal Prediction Calibration.

    Provides finite-sample coverage guarantees (Theorem 2):
        P( Y in C_{1-alpha}(X) | Y = c ) >= 1 - alpha

    And by Theorem 2, the false alarm rate on known traffic is strictly bounded:
        P( f_OSR(X) = Unknown | Y = c ) <= alpha
    """

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        # Mapping class_c -> empirical quantile / threshold tau_{alpha, c}
        self.quantiles: Dict[int, float] = {}

    def calibrate_scores(
        self,
        scores_by_class: Dict[int, np.ndarray | torch.Tensor],
    ) -> Dict[int, float]:
        """Compute finite-sample quantiles directly from nonconformity scores (e.g. Mahalanobis).

        Args:
            scores_by_class: Dict mapping class_id -> 1D array of scores.

        Returns:
            Dict mapping class_id -> threshold tau_{alpha, c}.
        """
        self.quantiles = {}
        for c, scores in scores_by_class.items():
            self.quantiles[c] = compute_finite_sample_quantile(scores, alpha=self.alpha)
        return self.quantiles

    def calibrate(
        self,
        cal_probabilities: torch.Tensor | np.ndarray,
        cal_labels: torch.Tensor | np.ndarray,
        num_classes: Optional[int] = None,
    ) -> Dict[int, float]:
        """Compute finite-sample corrected conformal quantiles from softmax probabilities.

        Args:
            cal_probabilities: Softmax probabilities of shape (N_cal, num_classes).
            cal_labels: Ground truth class indices of shape (N_cal,).
            num_classes: Total known classes.

        Returns:
            Dict mapping class_id -> threshold q_hat_c.
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

        scores_by_class: Dict[int, np.ndarray] = {}
        for c in range(num_classes):
            idx = np.where(labels_np == c)[0]
            if len(idx) == 0:
                continue
            # Nonconformity score for probability baseline: s_i = 1 - p(y_i)
            scores_by_class[c] = 1.0 - probs_np[idx, c]

        self.quantiles = {}
        for c in range(num_classes):
            if c in scores_by_class:
                self.quantiles[c] = compute_finite_sample_quantile(
                    scores_by_class[c], alpha=self.alpha
                )
            else:
                self.quantiles[c] = 1.0

        return self.quantiles

    def predict_conformal_sets(
        self,
        test_probabilities: torch.Tensor | np.ndarray,
    ) -> List[List[int]]:
        """Construct conformal prediction sets for test instances.

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
