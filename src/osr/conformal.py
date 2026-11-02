"""Class-Conditional Mondrian Conformal Prediction with Finite-Sample Guarantees (Equations 22-25 & Theorem 2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
import torch


def compute_right_closed_histogram(
    scores: np.ndarray | torch.Tensor,
    bin_edges: np.ndarray,
    include_overflow: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute right-closed histogram counts over uniform grid [0, B_c] with overflow bin (B_c, +inf) (Section 4.4, Eqs. 4.7-4.8).

    Bin partition (LaTeX Eqs. eq:ch4-bin-intervals & eq:ch4-local-hist):
        I_{c, 1} = [xi_{c, 0}, xi_{c, 1}]
        I_{c, m} = (xi_{c, m-1}, xi_{c, m}]  for m = 2, ..., M_bin
        I_{c, M_bin+1} = (B_c, +inf)  (overflow bin, with xi_{c, M_bin+1} = +inf)
    with xi_{c, 0} = 0 and xi_{c, M_bin} = B_c.

    Guarantees sum_{m=1}^{M_bin+1} h_{k, c, m} = N_{k, c}^{cal} identically without dropping tail scores.
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
    """Compute pooled finite-sample quantile by aggregating local empirical CDF histograms (Section 4.4, Eqs. 4.7-4.8).

    Mathematical Formulation:
        H_{c,m} = sum_{k in K_c} h_{k,c,m}   (Eq. eq:ch4-global-hist)
        N_c^{cal} = sum_{m=1}^{M_bin+1} H_{c,m}  (Eq. eq:ch4-total-cal-samples)
        r_c = ceil((N_c^{cal} + 1)(1 - alpha))   (Eq. eq:ch4-conformal-rank)
        q_hat = min { xi_{c,m} | sum_{j=1}^m H_{c,j} >= r_c }  (Eq. eq:ch4-conformal-threshold)
        If r_c > N_c^{cal} or rank falls in overflow bin (m = M_bin+1): q_hat = +inf.

    Args:
        client_histograms: List of tuples (bin_counts, bin_edges) sharing a common grid.
        alpha: Significance level in (0, 1).

    Returns:
        Finite-sample corrected quantile from pooled empirical CDF (or +inf if insufficient samples).
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
    """Evaluate empirical coverage, KFR, and zero-day recall across multiple conformal significance levels alpha.

    Mathematical Formulation (LaTeX Section 4.4 & Section 5.4 Sensitivity Sweep):
        r_{alpha, c} = ceil((N_{cal, c} + 1)(1 - alpha))
        tau_{alpha, c} = S_{(r_{alpha, c})}
        C_{1-alpha}(x) = { c in Y_K : D(z, c) <= tau_{alpha, c} }
        hat{y}(x) = Unknown  iff  C_{1-alpha}(x) == empty
    """
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


class ThreeWayConformalDecisionResult(NamedTuple):
    """Structured output for Conformal Three-Way Decision with Abstention (Bao & Gao 2025; Yao 2010)."""

    actions: np.ndarray  # 'accept', 'abstain', or 'reject'
    predictions: np.ndarray  # candidate class if accepted, unknown_label (-1 or C) if rejected or abstained
    is_accepted: np.ndarray  # True for Positive region (POS): D <= tau_accept
    is_abstained: np.ndarray  # True for Boundary region (BND): tau_accept < D <= tau_reject
    is_rejected: np.ndarray  # True for Negative region (NEG): D > tau_reject
    is_unknown: np.ndarray  # Total flagged non-accepted (is_rejected | is_abstained)
    candidate_classes: np.ndarray  # argmin_c D(z, c)
    scores: np.ndarray  # Nonconformity distance to candidate class
    tau_accept: np.ndarray  # Acceptance threshold used per sample
    tau_reject: np.ndarray  # Rejection threshold used per sample

    # Backward-compatible aliases
    @property
    def is_fast_pass(self) -> np.ndarray:
        return self.is_accepted

    @property
    def is_inspected(self) -> np.ndarray:
        return self.is_abstained

    @property
    def is_instant_drop(self) -> np.ndarray:
        return self.is_rejected

    @property
    def tau_defer(self) -> np.ndarray:
        return self.tau_accept

    @property
    def tau_drop(self) -> np.ndarray:
        return self.tau_reject


def compute_three_way_conformal_decision(
    dist_matrix: np.ndarray | torch.Tensor,
    tau_reject_map: Optional[Dict[int, float]] = None,
    tau_accept_map: Optional[Dict[int, float]] = None,
    unknown_label: int = -1,
    **kwargs: Any,
) -> ThreeWayConformalDecisionResult:
    """Compute Conformal Three-Way Decision with Abstention (POS / BND / NEG).

    Grounded in Decision-Theoretic Rough Sets (Yao 2010; Bao & Gao, Scientific Reports 2025)
    and calibrated via federated empirical score distributions:
        - Positive Region (POS / Accept): D(z, c*) <= tau_accept,c*   --> Action a_accept
        - Boundary Region (BND / Abstain): tau_accept,c* < D(z, c*) <= tau_reject,c* --> Action a_abstain
        - Negative Region (NEG / Reject): D(z, c*) > tau_reject,c*   --> Action a_reject

    Args:
        dist_matrix: Distance matrix of shape (N, C) across known classes.
        tau_reject_map: Dict mapping class_id -> tau_reject (e.g., at nominal alpha_reject = 0.05).
        tau_accept_map: Dict mapping class_id -> tau_accept (e.g., at nominal alpha_accept = 0.10).
        unknown_label: Label assigned to rejected/abstained samples.

    Returns:
        ThreeWayConformalDecisionResult namedtuple with diagnostic decision masks.
    """
    # Support backward-compatible kwargs
    if "tau_drop_map" in kwargs:
        tau_reject_map = kwargs["tau_drop_map"]
    if "tau_defer_map" in kwargs:
        tau_accept_map = kwargs["tau_defer_map"]
    tau_reject_map = tau_reject_map or {}
    tau_accept_map = tau_accept_map or {}

    if isinstance(dist_matrix, torch.Tensor):
        dist_mat = dist_matrix.detach().cpu().numpy()
    else:
        dist_mat = np.asarray(dist_matrix)

    n_samples, num_known = dist_mat.shape
    best_cand = np.argmin(dist_mat, axis=1)
    scores = dist_mat[np.arange(n_samples), best_cand]

    tau_reject_arr = np.array([tau_reject_map.get(c, float("inf")) for c in best_cand], dtype=np.float64)
    tau_accept_arr = np.array([tau_accept_map.get(c, float("inf")) for c in best_cand], dtype=np.float64)

    is_accepted = (scores <= tau_accept_arr)
    is_abstained = (scores > tau_accept_arr) & (scores <= tau_reject_arr)
    is_rejected = (scores > tau_reject_arr)
    is_unknown = is_rejected | is_abstained

    actions = np.empty(n_samples, dtype=object)
    actions[is_accepted] = "accept"
    actions[is_abstained] = "abstain"
    actions[is_rejected] = "reject"

    preds = np.where(is_accepted, best_cand, unknown_label)

    return ThreeWayConformalDecisionResult(
        actions=actions,
        predictions=preds,
        is_accepted=is_accepted,
        is_abstained=is_abstained,
        is_rejected=is_rejected,
        is_unknown=is_unknown,
        candidate_classes=best_cand,
        scores=scores,
        tau_accept=tau_accept_arr,
        tau_reject=tau_reject_arr,
    )

