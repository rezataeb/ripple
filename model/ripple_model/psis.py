"""Pareto-smoothed importance sampling, implemented here rather than imported.

Spec Section 5b asks for PSIS and a Pareto k-hat gate. The installed ArviZ
(1.3.0) exposes `arviz_stats.base.array_stats.psislw`, but on this problem's log
weights it returns k-hat = 22.1 and collapses a healthy weight vector (raw ESS
4805 of 20000, recovering the planted truth) to ESS 1 — see DECISIONS.md. A
k-hat of 22 is not a value the diagnostic can take, so it is not usable as a
gate, and `pareto_khat(..., log_weights=True)` reports -0.04 for weights whose
real ESS is 6, so it is not measuring weight degeneracy either.

So: the algorithm, directly.

  Vehtari, Simpson, Gelman, Yao, Gabry — "Pareto smoothed importance sampling"
  Zhang & Stephens (2009) — the empirical-Bayes generalized-Pareto estimator

What PSIS does: the largest importance weights are the unreliable ones, because
a self-normalized estimate is dominated by them. Fit a generalized Pareto to the
tail, replace the tail weights with its quantiles, and report the fitted shape
k-hat. k-hat > 0.7 means the weight distribution has infinite variance and the
estimate cannot be trusted — which is exactly the "Refit needed" signal.

Validated in model/tests/test_psis.py against weights whose behaviour is known.
"""

from __future__ import annotations

import numpy as np

K_HAT_THRESHOLD = 0.7


def gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Empirical-Bayes GPD fit (Zhang & Stephens 2009). x: sorted, positive.

    Returns (k, sigma) — the shape and scale of the generalized Pareto.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(np.sqrt(n))

    # grid of candidate b = -k/sigma values
    b = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]

    k_grid = np.log1p(-b[:, None] * x[None, :]).mean(axis=1)
    log_lik = n * (np.log(-b / k_grid) - k_grid - 1.0)
    # posterior weights over the grid = softmax(log_lik), computed stably:
    # 1 / sum_j exp(ll_j - ll_i) overflows when the grid spans many nats
    w = np.exp(log_lik - log_lik.max())
    w /= w.sum()
    keep = w >= 10.0 * np.finfo(float).eps
    if not np.all(keep):
        w, b = w[keep], b[keep]
    w = w / w.sum()

    b_post = float(np.sum(b * w))
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    # weakly informative prior on k, as in the reference implementation
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, sigma


def gpd_quantiles(p: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """Inverse CDF of the generalized Pareto, numerically safe as k -> 0."""
    p = np.asarray(p, dtype=float)
    if abs(k) < 1e-10:
        return sigma * -np.log1p(-p)
    return sigma / k * np.expm1(-k * np.log1p(-p))


def psis_smooth(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smooth log importance weights.

    Returns (smoothed log weights, k_hat). The smoothed weights are shifted so
    the maximum is 0; only their ratios matter after self-normalization.
    """
    lw = np.asarray(log_weights, dtype=float).ravel().copy()
    n = lw.size
    lw -= lw.max()  # work in ratios; also prevents overflow in exp

    # the tail that gets smoothed: at most a fifth of the sample
    n_tail = int(min(n / 5.0, 3.0 * np.sqrt(n)))
    if n_tail < 5 or n < 20:
        # too few samples to fit anything; report k_hat as not-estimable
        return lw, float("nan")

    order = np.argsort(lw)
    tail_idx = order[-n_tail:]                 # the largest weights
    cutoff = lw[order[-(n_tail + 1)]]          # threshold just below the tail

    exceed = np.exp(lw[tail_idx]) - np.exp(cutoff)
    exceed_sorted_pos = np.argsort(exceed)
    exceed_sorted = exceed[exceed_sorted_pos]
    if exceed_sorted[-1] <= 0 or not np.all(np.isfinite(exceed_sorted)):
        # the tail is flat: nothing heavy to smooth, and the fit is undefined
        return lw, float("-inf") if exceed_sorted[-1] <= 0 else float("nan")

    # drop any non-positive exceedances before fitting
    positive = exceed_sorted > 0
    if positive.sum() < 5:
        return lw, float("-inf")
    k_hat, sigma = gpdfit(exceed_sorted[positive])

    if np.isfinite(k_hat) and sigma > 0:
        m = n_tail
        probs = (np.arange(1, m + 1, dtype=float) - 0.5) / m
        smoothed = np.log(gpd_quantiles(probs, k_hat, sigma) + np.exp(cutoff))
        # put the smoothed values back in the tail's own rank order
        ranks = np.argsort(np.argsort(exceed))
        lw[tail_idx] = smoothed[ranks]
        # never let smoothing push a weight above the largest raw weight
        lw = np.minimum(lw, 0.0)

    return lw, float(k_hat)


def ess(weights: np.ndarray) -> float:
    """Kish effective sample size of (unnormalized) importance weights."""
    w = np.asarray(weights, dtype=float).ravel()
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    return float(1.0 / np.sum(p * p))
