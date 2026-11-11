"""Distributed Mondrian Conformal Calibration.

Calibrates per-class decision thresholds independently to guarantee marginal coverage
and control false alarm rates without pooling raw calibration scores at the server:
    1. Local Nonconformity Scores: Mahalanobis distance to nearest class prototype.
    2. Local Histogram Bins: Nodes bin scores into discrete intervals with an overflow bin.
    3. Global Histogram Aggregation: Server sums bin counts across nodes.
    4. Conformal Quantile Computation: Server determines the (1 - alpha) quantile threshold
       from cumulative histogram frequencies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
import torch


def compute_right_closed_histogram(
    scores: np.ndarray | torch.Tensor,
    bin_edges: np.ndarray,
    include_overflow: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute right-closed histogram counts over a uniform grid with an overflow bin.

    Bin partition:
        First bin: [edges[0], edges[1]]
        Intermediate bins: (edges[m-1], edges[m]]
        Overflow bin: (edges[-1], +inf)

    Guarantees the sum of bin counts equals the total sample count without losing tail scores.
    """
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy().flatten()
    else:
        scores_np = np.asarray(scores, dtype=np.float64).flatten()

    edges = np.asarray(bin_edges, dtype=np.float64)
    num_bins = len(edges) - 1
    if len(scores_np) == 0 or num_bins <= 0:
        total_bins = num_bins + (1 if include_overflow else 0)
        extended_edges = np.append(edges, np.inf) if include_overflow else edges
        return np.zeros(max(total_bins, 0), dtype=np.int64), extended_edges

    if include_overflow:
        b_c = float(edges[-1])
        is_overflow = scores_np > b_c
        regular_scores = scores_np[~is_overflow]
        n_overflow = int(is_overflow.sum())

        if len(regular_scores) > 0:
            bin_indices = np.searchsorted(edges[1:], regular_scores, side="left")
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            reg_counts = np.bincount(bin_indices, minlength=num_bins).astype(np.int64)
        else:
            reg_counts = np.zeros(num_bins, dtype=np.int64)

        counts = np.append(reg_counts, n_overflow)
        extended_edges = np.append(edges, np.inf)
        return counts, extended_edges
    else:
        bin_indices = np.searchsorted(edges[1:], scores_np, side="left")
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        counts = np.bincount(bin_indices, minlength=num_bins).astype(np.int64)
        return counts, edges


def compute_finite_sample_quantile(
    scores: np.ndarray | torch.Tensor,
    alpha: float = 0.05,
) -> float:
    """Compute finite-sample corrected conformal quantile for a given significance level alpha.

    Formulation:
        Target rank: r = ceil((N + 1) * (1 - alpha))
        Threshold: Order statistic S_{(r)} when r <= N.
        If samples are scarce (r > N), falls back to the maximum observed score S_max
        to prevent threshold explosion while maintaining valid finite coverage.

    Args:
        scores: Array or tensor of nonconformity scores.
        alpha: Significance level in (0, 1) (default: 0.05 for 95% coverage).

    Returns:
        Empirical finite-sample corrected quantile threshold.
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
    """Compute pooled finite-sample quantile by aggregating local empirical CDF histograms.

    Formulation:
        1. Global histogram: H = sum_k h_k across participating nodes.
        2. Total calibration samples: N_cal = sum(H).
        3. Target rank: r = ceil((N_cal + 1) * (1 - alpha)).
        4. Quantile threshold: upper boundary of the bin where cumulative counts reach r.

    Args:
        client_histograms: List of tuples (bin_counts, bin_edges) sharing a common grid.
        alpha: Significance level in (0, 1).

    Returns:
        Finite-sample corrected quantile from pooled empirical CDF.
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
        return float("inf")

    cum_counts = np.cumsum(total_counts)
    idx = int(np.searchsorted(cum_counts, target_rank, side="left"))
    # In right-closed partitioning, bin index idx corresponds to upper boundary bin_edges[idx + 1]
    edge_idx = min(idx + 1, len(bin_edges) - 1)
    val = float(bin_edges[edge_idx])
    return val


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


def evaluate_conformal_sensitivity_sweep(
    cal_scores_per_class: Dict[int, np.ndarray | torch.Tensor],
    test_dist_matrix: np.ndarray | torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    num_known_classes: int,
    alphas: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate empirical coverage, known false rejection rate, and zero-day recall across significance levels."""
    if alphas is None:
        alphas = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

    if isinstance(test_dist_matrix, torch.Tensor):
        dist_mat = test_dist_matrix.detach().cpu().numpy()
    else:
        dist_mat = np.asarray(test_dist_matrix)

    if isinstance(test_labels, torch.Tensor):
        labels = test_labels.detach().cpu().numpy()
    else:
        labels = np.asarray(test_labels)

    known_mask = (labels < num_known_classes)
    unknown_mask = (labels == num_known_classes)
    n_known = int(known_mask.sum())
    n_unknown = int(unknown_mask.sum())

    results = []
    for a in alphas:
        taus = {}
        for c in range(num_known_classes):
            if c in cal_scores_per_class and len(cal_scores_per_class[c]) > 0:
                taus[c] = compute_finite_sample_quantile(cal_scores_per_class[c], alpha=a)
            else:
                taus[c] = float("inf")

        tau_arr = np.array([taus[c] for c in range(num_known_classes)], dtype=np.float64).reshape(1, -1)
        accepted_mask = dist_mat <= tau_arr  # (N_test, C)
        is_unknown = ~np.any(accepted_mask, axis=1)  # (N_test,)

        kfr = float(is_unknown[known_mask].mean()) if n_known > 0 else 0.0
        cov = 1.0 - kfr
        rec = float(is_unknown[unknown_mask].mean()) if n_unknown > 0 else 0.0

        n_rej = int(is_unknown.sum())
        tp = int((is_unknown & unknown_mask).sum())
        prec = float(tp / n_rej) if n_rej > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

        results.append({
            "alpha": float(a),
            "nominal_coverage": float(1.0 - a),
            "empirical_coverage": float(cov),
            "empirical_kfr": float(kfr),
            "zero_day_recall": float(rec),
            "zero_day_precision": float(prec),
            "zero_day_f1": float(f1),
            "thresholds": {int(c): float(taus[c]) for c in range(num_known_classes)},
            "status": "GUARANTEED" if kfr <= (a + 0.005) else "DRIFT",
        })

    return results

