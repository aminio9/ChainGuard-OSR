"""Rigorous statistical discipline for Q1 IEEE TIFS/TDSC evaluation.

Implements:
1. 10-Seed Canonical Protocol (guaranteeing p_min = 2^(-9) = 0.00195 < 0.01 for two-sided Wilcoxon test).
2. Exact Two-Sided Wilcoxon Signed-Rank Test with p < 0.01 significance indicator (dagger).
3. Non-parametric Cliff's Delta effect size in [-1, 1].
4. Bias-Corrected and Accelerated (BCa) / Percentile Bootstrap 95% Confidence Intervals (1,000 resamples).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import scipy.stats as stats

# 10 canonical random seeds required so two-sided Wilcoxon signed-rank test can achieve p < 0.01
# Since p_min = 2 * (1/2)^n = 2^{-(n-1)}, n=5 yields p_min = 0.0625 (> 0.01), whereas n=10 yields p_min = 0.001953.
CANONICAL_TEN_SEEDS: Tuple[int, ...] = (42, 101, 2024, 7, 999, 123, 555, 777, 888, 1337)
CANONICAL_FIVE_SEEDS: Tuple[int, ...] = (42, 101, 2024, 7, 999)


def compute_cliffs_delta(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict[str, Any]:
    """Compute non-parametric Cliff's Delta effect size between two sample distributions.

    Formula:
        delta = ( #(a_i > b_j) - #(a_i < b_j) ) / (N_1 * N_2)

    Returns:
        Dictionary with 'delta' in [-1, 1] and 'magnitude' interpretation (Romano et al., 2006).
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"delta": 0.0, "magnitude": "negligible"}

    diff_matrix = a[:, None] - b[None, :]
    greater = int(np.sum(diff_matrix > 0))
    less = int(np.sum(diff_matrix < 0))
    delta = float((greater - less) / float(n1 * n2))

    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"

    return {"delta": delta, "magnitude": mag}


def compute_wilcoxon_signed_rank(
    proposed_scores: Sequence[float],
    baseline_scores: Sequence[float],
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """Perform paired two-sided Wilcoxon Signed-Rank Test.

    Note: Requires n >= 8 paired seeds for two-sided p < 0.01 to be mathematically attainable.
    """
    x = np.asarray(proposed_scores, dtype=np.float64)
    y = np.asarray(baseline_scores, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"Paired Wilcoxon test requires equal lengths, got {len(x)} and {len(y)}.")

    n = len(x)
    p_min_theoretical = 2.0 ** (-(n - 1)) if n >= 1 else 1.0
    diffs = x - y

    if np.allclose(diffs, 0.0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "marker": "",
            "n_seeds": n,
            "p_min_theoretical": p_min_theoretical,
        }

    try:
        res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat_val = 0.0
        p_val = 1.0

    is_sig = bool(p_val < alpha and np.mean(x) > np.mean(y))
    return {
        "statistic": stat_val,
        "p_value": p_val,
        "significant": is_sig,
        "significant_at_0_01": bool(p_val < 0.01 and np.mean(x) > np.mean(y)),
        "marker": r"^\dagger" if is_sig else "",
        "n_seeds": n,
        "p_min_theoretical": p_min_theoretical,
    }


def compute_bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute Bootstrap 95% Confidence Interval (BCa with percentile fallback)."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    if n == 1 or np.allclose(arr, arr[0]):
        val = float(arr[0])
        return {"mean": val, "std": 0.0, "ci_low": val, "ci_high": val, "ci_lower": val, "ci_upper": val}

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))

    try:
        res = stats.bootstrap(
            (arr,),
            np.mean,
            confidence_level=confidence,
            n_resamples=n_resamples,
            method="BCa",
            random_state=np.random.default_rng(seed),
        )
        ci_low = float(res.confidence_interval.low)
        ci_high = float(res.confidence_interval.high)
        if np.isnan(ci_low) or np.isnan(ci_high):
            raise ValueError("BCa returned NaN")
    except Exception:
        rng = np.random.default_rng(seed)
        boot_means = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means[i] = np.mean(sample)
        alpha_tail = (1.0 - confidence) / 2.0
        ci_low = float(np.percentile(boot_means, 100.0 * alpha_tail))
        ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha_tail)))

    return {
        "mean": mean_val,
        "std": std_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
    }


def compute_curve_bootstrap_ci(
    curves: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute point-wise mean and 95% bootstrap confidence band across multi-seed curves.

    Args:
        curves: 2D array of shape (num_seeds, num_points).

    Returns:
        (mean_curve, ci_low_curve, ci_high_curve) each of shape (num_points,).
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim == 1:
        return curves, curves, curves

    n_seeds, n_pts = curves.shape
    mean_curve = np.mean(curves, axis=0)
    if n_seeds <= 1:
        return mean_curve, mean_curve, mean_curve

    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_resamples, n_pts), dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[i] = np.mean(curves[idx], axis=0)

    alpha_tail = (1.0 - confidence) / 2.0
    ci_low = np.percentile(boot_means, 100.0 * alpha_tail, axis=0)
    ci_high = np.percentile(boot_means, 100.0 * (1.0 - alpha_tail), axis=0)
    return mean_curve, ci_low, ci_high


def summarize_multi_seed_Comparison(
    proposed_runs: Dict[str, List[float]] | Sequence[float],
    baseline_runs: Dict[str, Dict[str, List[float]]] | Sequence[float],
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """Compare proposed method against baselines with Mean +- Std, Wilcoxon p-value, and Cliff's Delta."""
    # Support direct 1D array pair comparison as well as nested dict comparison
    if not isinstance(proposed_runs, dict) and not isinstance(baseline_runs, dict):
        prop_ci = compute_bootstrap_ci(proposed_runs)
        base_ci = compute_bootstrap_ci(baseline_runs)
        wilcox = compute_wilcoxon_signed_rank(proposed_runs, baseline_runs, alpha=alpha)
        cliffs = compute_cliffs_delta(proposed_runs, baseline_runs)
        return {
            "proposed": prop_ci,
            "baseline": base_ci,
            "wilcoxon": wilcox,
            "cliffs_delta": cliffs,
            "dagger_p001": wilcox["significant_at_0_01"],
        }

    assert isinstance(proposed_runs, dict) and isinstance(baseline_runs, dict)
    summary: Dict[str, Any] = {"proposed": {}, "baselines": {}}

    for metric_name, prop_vals in proposed_runs.items():
        summary["proposed"][metric_name] = compute_bootstrap_ci(prop_vals)

    for base_name, base_metrics in baseline_runs.items():
        summary["baselines"][base_name] = {}
        for metric_name, base_vals in base_metrics.items():
            ci_info = compute_bootstrap_ci(base_vals)
            if metric_name in proposed_runs and len(proposed_runs[metric_name]) == len(base_vals):
                wilcox = compute_wilcoxon_signed_rank(proposed_runs[metric_name], base_vals, alpha=alpha)
                cliffs = compute_cliffs_delta(proposed_runs[metric_name], base_vals)
            else:
                wilcox = {"statistic": 0.0, "p_value": 1.0, "significant": False, "significant_at_0_01": False, "marker": ""}
                cliffs = {"delta": 0.0, "magnitude": "negligible"}

            summary["baselines"][base_name][metric_name] = {
                **ci_info,
                "wilcoxon": wilcox,
                "cliffs_delta": cliffs,
                "dagger_p001": wilcox["significant_at_0_01"],
            }

    return summary


summarize_multi_seed_comparison = summarize_multi_seed_Comparison

