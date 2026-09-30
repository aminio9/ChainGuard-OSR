"""Class-Conditional Mondrian Conformal Prediction with Finite-Sample Guarantees (Equations 22-25 & Theorem 2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def compute_right_closed_histogram(
    scores: np.ndarray | torch.Tensor,
    bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute right-closed histogram counts over shared bin grid [0, S_{max,c}] (Section 4.4).

    Bin partition:
        I_{c, 1} = [xi_{c, 0}, xi_{c, 1}]
        I_{c, m} = (xi_{c, m-1}, xi_{c, m}]  for m = 2, ..., M_bin
    with xi_{c, 0} = 0 and xi_{c, M_bin} = S_{max, c}.
    Guarantees sum_{m=1}^{M_bin} h_{k, c, m} = n_{cal, k, c} and
    hat{F}_{k, c}(xi_{c, m}) = (1 / n_{cal, k, c}) * |{j : s_{k, j}^{(c)} <= xi_{c, m}}|,
    so hat{F}_c(xi_{c, M_bin}) = 1 holds identically.
    """
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy().flatten()
    else:
        scores_np = np.asarray(scores, dtype=np.float64).flatten()

    edges = np.asarray(bin_edges, dtype=np.float64)
    num_bins = len(edges) - 1
    if len(scores_np) == 0 or num_bins <= 0:
        return np.zeros(max(num_bins, 0), dtype=np.int64), edges

    # searchsorted(..., side='left') on edges[1:] maps s <= xi_{c,1} to bin 0 ([xi_0, xi_1])
    # and xi_{c,m-1} < s <= xi_{c,m} to bin m-1 ((xi_{m-1}, xi_m])
    bin_indices = np.searchsorted(edges[1:], scores_np, side="left")
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    counts = np.bincount(bin_indices, minlength=num_bins).astype(np.int64)
    return counts, edges


def compute_finite_sample_quantile(
    scores: np.ndarray | torch.Tensor,
    alpha: float = 0.05,
) -> float:
    """Compute finite-sample corrected quantile (Section 4.4, Eq. eq:conformal_quantile).

    Mathematical Formulation (Eq. eq:conformal_quantile):
        r_{alpha, c} = ceil((N_{cal, c} + 1)(1 - alpha))
        tau_{alpha, c} = min { xi_{c, m} : hat{F}_c(xi_{c, m}) >= r_{alpha, c} / N_{cal, c} }
                       = S_{(r_{alpha, c})}  when r_{alpha, c} <= N_{cal, c}.
        If 1 <= N_{cal, c} < ceil(1/alpha) - 1 (so r_{alpha, c} > N_{cal, c}), setting
        tau_{alpha, c} = +inf would collapse the normalized ratio D_block(z, c) / tau_{alpha, c} = 0
        in open-set point prediction; therefore tau_{alpha, c} = S_{max, c} < inf, guaranteeing
        finite-sample coverage at level N_{cal, c} / (N_{cal, c} + 1) while keeping tau_{alpha, c} in (0, inf).

    Proposition 2 Guarantee (when N_{cal, c} >= ceil(1/alpha) - 1):
        P( D_block(f_{theta^*}(X_test), c) > tau_{alpha, c} | Y_test = c ) <= alpha
        P( hat{y}_OSR(X_test) = Unknown | Y_test = c ) <= alpha

    Args:
        scores: Array or tensor of nonconformity scores for class c.
        alpha: Significance level in (0, 1) (default: 0.05 for 95% coverage).

    Returns:
        Empirical finite-sample corrected quantile in (0, inf) (or +inf only if N_{cal, c} == 0).
    """
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy().flatten()
    else:
        scores_np = np.asarray(scores).flatten()

    n = len(scores_np)
    if n == 0:
        return float("inf")

    target_rank = np.ceil((n + 1) * (1.0 - alpha))
    if target_rank > n:
        # Finite empirical maximum fallback S_{max, c} when 1 <= n < ceil(1/alpha) - 1
        return float(max(float(np.max(scores_np)), 1e-12))

    q_level = float(min(1.0, max(0.0, target_rank / float(n))))

    # Order statistic S_{(ceil((n+1)(1-alpha)))} via 'higher' quantile method
    quantile_val = float(np.quantile(scores_np, q_level, method="higher"))
    return float(max(quantile_val, 1e-12))


def aggregate_distributed_cdfs(
    client_histograms: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.05,
) -> float:
    """Compute pooled finite-sample quantile by aggregating local empirical CDF histograms (Section 4.4).

    Unlike weighted averaging of local quantiles (which fails to recover the pooled quantile
    under Non-IID partitions; Lu et al., ICML 2023; Plassier et al., ICML 2023), empirical CDF
    and right-closed histogram counts on [0, S_{max, c}] are linear and additive across clients k in K_{cal, c}:
        hat{F}_c(xi_{c, m}) = (1 / N_{cal, c}) * sum_{k in K_{cal, c}} sum_{r=1}^m h_{k, c, r}
                            = (1 / N_{cal, c}) * sum_{k in K_{cal, c}} n_{cal, k, c} * hat{F}_{k, c}(xi_{c, m})
    with hat{F}_c(xi_{c, M_bin}) = 1 identically.
    Inverting hat{F}_c at level r_{alpha, c} / N_{cal, c} (or returning S_{max, c} = xi_{c, M_bin}
    when 1 <= N_{cal, c} < ceil(1/alpha) - 1) preserves finite-sample Mondrian coverage and keeps
    0 < tau_{alpha, c} <= S_{max, c} < inf.

    Args:
        client_histograms: List of tuples (bin_counts, bin_edges) sharing a common grid [0, S_{max, c}].
        alpha: Significance level in (0, 1).

    Returns:
        Finite-sample corrected quantile from the pooled empirical CDF (or S_{max, c} if 1 <= N_{cal, c} < ceil(1/alpha) - 1).
    """
    if not client_histograms:
        return float("inf")

    total_counts = np.sum([np.asarray(counts, dtype=np.float64) for counts, _ in client_histograms], axis=0)
    bin_edges = np.asarray(client_histograms[0][1], dtype=np.float64)
    n_total = float(np.sum(total_counts))
    if n_total <= 0:
        return float("inf")

    target_rank = np.ceil((n_total + 1.0) * (1.0 - alpha))
    if target_rank > n_total:
        return float(max(float(bin_edges[-1]), 1e-12))

    cum_counts = np.cumsum(total_counts)
    idx = int(np.searchsorted(cum_counts, target_rank, side="left"))
    # Return upper bin edge xi_{c, m} of the first bin where hat{F}_c(xi_{c, m}) >= r_{alpha, c} / N_{cal, c}
    edge_idx = min(idx + 1, len(bin_edges) - 1)
    return float(max(float(bin_edges[edge_idx]), 1e-12))


def aggregate_distributed_quantiles(
    client_quantiles: List[Tuple[float, int]],
) -> float:
    """Legacy heuristic: weighted average of local client quantiles.

    NOTE: For exact finite-sample conformal coverage under Non-IID partitions, use
    `compute_finite_sample_quantile` on pooled scores or `aggregate_distributed_cdfs` on
    additive local histograms (Lu et al., ICML 2023; Plassier et al., ICML 2023).
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
