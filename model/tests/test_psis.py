"""PSIS has to be right, because it is the gate that says when not to trust us.

The installed ArviZ returned k-hat = 22.1 on healthy log weights, which is why
ripple_model/psis.py exists. A hand-rolled diagnostic needs its own evidence, so
these tests check it on weight distributions whose behaviour is known in advance,
and against ArviZ on the cases where ArviZ agrees.
"""

from __future__ import annotations

import numpy as np
import pytest
from ripple_model.psis import K_HAT_THRESHOLD, ess, gpd_quantiles, gpdfit, psis_smooth

N = 20000


def norm(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_well_behaved_weights_pass_the_gate(rng) -> None:
    """Mild weight variation: k-hat well under the threshold, ESS preserved."""
    logw = rng.normal(0.0, 1.0, N)
    smoothed, khat = psis_smooth(logw)
    assert khat < K_HAT_THRESHOLD
    assert ess(norm(smoothed)) > 0.95 * ess(norm(logw)), "smoothing must not destroy a good sample"


def test_heavy_tailed_weights_fail_the_gate(rng) -> None:
    """Log weights with a large spread have infinite-variance ratios: k-hat > 0.7."""
    for sd in (3.0, 5.0, 8.0):
        _, khat = psis_smooth(rng.normal(0.0, sd, N))
        assert khat > K_HAT_THRESHOLD, f"sd={sd} should fail the gate, got k-hat {khat:.3f}"


def test_a_single_dominating_sample_fails_the_gate(rng) -> None:
    """One sample carrying all the weight is the worst case and must be caught."""
    logw = np.concatenate([[50.0], rng.normal(0.0, 1.0, N - 1)])
    assert ess(norm(logw)) < 2, "the setup itself should be degenerate"
    _, khat = psis_smooth(logw)
    assert khat > K_HAT_THRESHOLD


def test_khat_increases_with_weight_spread(rng) -> None:
    ks = [psis_smooth(rng.normal(0.0, sd, N))[1] for sd in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert all(a < b for a, b in zip(ks, ks[1:], strict=False)), f"k-hat not monotone: {ks}"


def test_smoothing_never_raises_a_weight_above_the_largest_raw_weight(rng) -> None:
    logw = rng.normal(0.0, 4.0, N)
    smoothed, _ = psis_smooth(logw)
    assert smoothed.max() <= 1e-12


def test_smoothing_preserves_the_weight_ordering(rng) -> None:
    """PSIS replaces the tail values but must not reorder which samples matter."""
    logw = rng.normal(0.0, 2.0, N)
    smoothed, _ = psis_smooth(logw)
    assert np.array_equal(np.argsort(np.argsort(logw)), np.argsort(np.argsort(smoothed)))


def test_agrees_with_arviz_where_arviz_is_well_behaved(rng) -> None:
    """Cross-check the implementation against the reference on easy cases."""
    from arviz_stats.base import array_stats

    for sd in (1.0, 3.0, 8.0):
        logw = rng.normal(0.0, sd, N)
        _, mine = psis_smooth(logw)
        _, theirs = array_stats.psislw(logw.copy(), r_eff=1.0, axis=-1)
        theirs = float(np.asarray(theirs).ravel()[0])
        assert abs(mine - theirs) < 0.1, f"sd={sd}: mine {mine:.3f} vs arviz {theirs:.3f}"


def test_gpdfit_recovers_a_known_generalized_pareto(rng) -> None:
    for k_true, sigma_true in [(0.1, 2.0), (0.5, 1.0), (0.8, 3.0)]:
        u = rng.uniform(0.0, 1.0, 5000)
        x = np.sort(gpd_quantiles(u, k_true, sigma_true))
        k_hat, sigma_hat = gpdfit(x)
        assert abs(k_hat - k_true) < 0.15, f"k: {k_hat:.3f} vs {k_true}"
        assert abs(sigma_hat - sigma_true) / sigma_true < 0.4, f"sigma: {sigma_hat:.3f}"


def test_gpd_quantiles_are_stable_as_k_goes_to_zero() -> None:
    """k -> 0 is the exponential limit; the formula must not divide by zero."""
    p = np.array([0.1, 0.5, 0.9])
    limit = gpd_quantiles(p, 0.0, 2.0)
    near = gpd_quantiles(p, 1e-12, 2.0)
    assert np.allclose(limit, near, rtol=1e-6)
    assert np.allclose(limit, 2.0 * -np.log1p(-p))


def test_ess_matches_the_closed_form() -> None:
    assert ess(np.ones(100)) == pytest.approx(100.0)
    w = np.zeros(100)
    w[0] = 1.0
    assert ess(w) == pytest.approx(1.0)
    assert ess(np.zeros(10)) == 0.0


def test_too_few_samples_reports_not_estimable(rng) -> None:
    """A sample too small to fit a tail must say so, not guess."""
    _, khat = psis_smooth(rng.normal(0.0, 1.0, 10))
    assert np.isnan(khat)
