"""Trust checks (spec Section 4) -> trust.json.

Four groups, and every one of them can fail in public:

  convergence  R-hat <= 1.01, ESS >= 400, 0 divergences
  fit          posterior predictive coverage of the 90% interval >= 85%
  recovery     demo only: the planted truth is inside the 94% HDI
  holdout      last 12 weeks: 90% coverage, and MAPE beats a naive seasonal baseline

Each check carries `plain_english`, so the UI never has to invent the wording,
and a check that cannot run says so in `skipped_reason` instead of silently
passing. The recovery entries map truth to parameters **by name**: data.py sorts
cities and segments alphabetically, so index order does not match synth.py.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import arviz as az
import numpy as np

from ripple_model.data import ModelData
from ripple_model.model import FitResult
from ripple_model.simulate import season_row

RHAT_MAX = 1.01
ESS_MIN = 400.0
PPC_COVERAGE_MIN = 0.85
HOLDOUT_COVERAGE_MIN = 0.80
HDI_PROB = 0.94
HOLDOUT_WEEKS = 12


def _flat(x: np.ndarray) -> np.ndarray:
    """(chain, draw, ...) -> (chain*draw, ...).

    The first dimension is computed rather than passed as -1: a switched-off term
    like `g` on an upload with no city traits has shape (chain, draw, 0), and
    numpy cannot resolve -1 against another zero-length axis.
    """
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def _summ(x: np.ndarray) -> dict[str, float]:
    p05, p95 = np.quantile(x, [0.05, 0.95])
    return {"mean": float(np.mean(x)), "p05": float(p05), "p95": float(p95)}


# --------------------------------------------------------------------------
# 1. convergence
# --------------------------------------------------------------------------
def convergence(arrays: dict[str, np.ndarray], result: FitResult) -> tuple[list[dict], dict]:
    rhats, esss = {}, {}
    for name, v in arrays.items():
        if v.ndim < 2 or v.shape[0] < 2:
            continue  # ADVI output has a single chain; R-hat is undefined
        if v.size == 0:
            continue  # a switched-off term, e.g. `g` on an upload with no city traits
        if np.all(v == v.flat[0]):
            continue  # a zero-filled term is a constant: R-hat is NaN, not a failure
        rhat = float(np.max(az.rhat(v, chain_axis=0, draw_axis=1)))
        ess_v = float(np.min(az.ess(v, chain_axis=0, draw_axis=1)))
        if not (np.isfinite(rhat) and np.isfinite(ess_v)):
            continue
        rhats[name] = rhat
        esss[name] = ess_v

    if not rhats:
        checks = [
            {
                "name": "Convergence",
                "group": "convergence",
                "passed": False,
                "value": None,
                "threshold": None,
                "plain_english": "This run used a fast approximation, so the usual "
                                 "convergence diagnostics do not apply.",
                "skipped_reason": "single-chain approximation (ADVI)",
            }
        ]
        return checks, {"n_chains": result.chains, "n_draws_per_chain": result.draws_per_chain}

    worst_rhat = max(rhats, key=rhats.get)
    worst_ess = min(esss, key=esss.get)
    max_rhat, min_ess = rhats[worst_rhat], esss[worst_ess]

    checks = [
        {
            "name": "R-hat",
            "group": "convergence",
            "passed": bool(max_rhat <= RHAT_MAX),
            "value": round(max_rhat, 4),
            "threshold": RHAT_MAX,
            "comparison": "lte",
            "plain_english": (
                "The independent chains agree with each other, so the sampler found "
                "one answer rather than several."
                if max_rhat <= RHAT_MAX
                else "The chains do not fully agree, so these numbers are not settled yet."
            ),
            "detail": f"worst parameter: {worst_rhat} (R-hat {max_rhat:.4f})",
        },
        {
            "name": "Effective sample size",
            "group": "convergence",
            "passed": bool(min_ess >= ESS_MIN),
            "value": round(min_ess, 0),
            "threshold": ESS_MIN,
            "comparison": "gte",
            "plain_english": (
                "There are enough independent draws for the intervals to be stable."
                if min_ess >= ESS_MIN
                else "Too few independent draws: the interval edges will move if refitted."
            ),
            "detail": f"smallest for {worst_ess} ({min_ess:.0f} draws)",
        },
        {
            "name": "Divergences",
            "group": "convergence",
            "passed": result.divergences == 0,
            "value": int(result.divergences),
            "threshold": 0,
            "comparison": "lte",
            "plain_english": (
                "The sampler explored the whole posterior without getting stuck."
                if result.divergences == 0
                else f"{result.divergences} draws got stuck in a difficult part of the "
                     "posterior. The fit is usable but the tails may be understated."
            ),
        },
    ]
    conv = {
        "max_rhat": round(max_rhat, 4),
        "min_ess": round(min_ess, 1),
        "divergences": int(result.divergences),
        "worst_param": worst_rhat,
        "n_chains": result.chains,
        "n_draws_per_chain": result.draws_per_chain,
        "sampling_seconds": round(result.seconds, 1),
    }
    return checks, conv


# --------------------------------------------------------------------------
# 2. posterior predictive coverage
# --------------------------------------------------------------------------
def ppc_coverage(y_rep: np.ndarray, y_obs: np.ndarray) -> dict:
    """Share of observed counts inside the 90% posterior predictive interval."""
    lo, hi = np.quantile(y_rep, [0.05, 0.95], axis=0)
    inside = float(np.mean((y_obs >= lo) & (y_obs <= hi)))
    return {
        "name": "Posterior predictive coverage",
        "group": "fit",
        "passed": bool(inside >= PPC_COVERAGE_MIN),
        "value": round(inside, 4),
        "threshold": PPC_COVERAGE_MIN,
        "comparison": "gte",
        "plain_english": (
            f"{inside:.0%} of the weeks the model was fitted on fall inside its own 90% "
            "range, so the spread of the predictions matches the spread of reality."
            if inside >= PPC_COVERAGE_MIN
            else f"Only {inside:.0%} of observed weeks fall inside the model's 90% range, "
                 "so it is overconfident about its own data."
        ),
    }


# --------------------------------------------------------------------------
# 3. parameter recovery (demo only)
# --------------------------------------------------------------------------
def recovery(arrays: dict[str, np.ndarray], truth: dict, d: ModelData) -> list[dict]:
    """Planted truth vs. the 94% HDI, matched by NAME not by index."""
    p = truth["params"]
    t_segments: list[str] = truth["segments"]
    t_cities: list[str] = truth["cities"]
    t_traits: list[str] = truth["trait_names"]

    def by_name(values: list[float], names: list[str]) -> dict[str, float]:
        return dict(zip(names, values, strict=True))

    targets: list[tuple[str, str, float, np.ndarray]] = []

    e_truth = by_name(p["e"], t_segments)
    promo_truth = by_name(p["b_promo"], t_segments)
    seg_truth = by_name(p["a_seg"], t_segments)
    for si, seg in enumerate(d.segments):
        targets.append((f"e[{seg}]", f"Price elasticity · {seg}", e_truth[seg],
                        _flat(arrays["e"])[:, si]))
        if d.has_promo:
            targets.append((f"b_promo[{seg}]", f"Promo lift · {seg}", promo_truth[seg],
                            _flat(arrays["b_promo"])[:, si]))
        if d.n_segments > 1:
            targets.append((f"a_seg[{seg}]", f"Segment level · {seg}", seg_truth[seg],
                            _flat(arrays["a_seg"])[:, si]))

    city_truth = by_name(p["a_city"], t_cities)
    for ci, city in enumerate(d.cities):
        targets.append((f"a_city[{city}]", f"City effect · {city}", city_truth[city],
                        _flat(arrays["a_city"])[:, ci]))

    if d.has_traits and "g" in arrays and arrays["g"].size:
        g_truth = by_name(p["g"], t_traits)
        for ti, trait in enumerate(d.trait_names):
            targets.append((f"g[{trait}]", f"Trait coefficient · {trait}", g_truth[trait],
                            _flat(arrays["g"])[:, ti]))

    targets.append(("g0", "Global level", p["g0"], _flat(arrays["g0"])))
    targets.append(("tau_city", "City-to-city spread", p["tau_city"], _flat(arrays["tau_city"])))
    targets.append(("phi", "Overdispersion", p["phi"], _flat(arrays["phi"])))
    if d.has_promo:
        targets.append(("b_dip", "Post-promo dip", p["b_dip"], _flat(arrays["b_dip"])))
    if d.has_feature:
        targets.append(("b_feature", "Feature lift", p["b_feature"], _flat(arrays["b_feature"])))
    for k in range(arrays.get("season", np.zeros((1, 1, 0))).shape[-1]):
        targets.append((f"season[{k}]", f"Seasonality term {k + 1}", p["season"][k],
                        _flat(arrays["season"])[:, k]))

    out = []
    for param, label, true_value, draws in targets:
        lo, hi = az.hdi(np.asarray(draws).reshape(1, -1), prob=HDI_PROB).squeeze()
        out.append(
            {
                "param": param,
                "label": label,
                "truth": float(true_value),
                "posterior_mean": float(np.mean(draws)),
                "hdi_low": float(lo),
                "hdi_high": float(hi),
                "inside": bool(lo <= true_value <= hi),
            }
        )
    return out


def recovery_check(rec: list[dict]) -> dict:
    n_in = sum(r["inside"] for r in rec)
    n = len(rec)
    share = n_in / n if n else 0.0
    misses = [r["param"] for r in rec if not r["inside"]]
    # 94% intervals on n parameters: a couple of misses is expected, not a failure
    expected_misses = max(1, int(round(0.06 * n)))
    passed = len(misses) <= expected_misses
    return {
        "name": "Parameter recovery",
        "group": "recovery",
        "passed": bool(passed),
        "value": round(share, 4),
        "threshold": round(1.0 - expected_misses / n, 4) if n else None,
        "comparison": "gte",
        "plain_english": (
            f"The data was generated with known values, and {n_in} of {n} of them land "
            f"inside the model's 94% interval. The model finds effects that are really there."
            if passed
            else f"Only {n_in} of {n} known values land inside the 94% interval "
                 f"({', '.join(misses[:4])}), so the model is biased on those parameters."
        ),
        "detail": (
            f"missed: {', '.join(misses)}"
            if misses
            else f"all {n} planted values recovered"
        ),
    }


# --------------------------------------------------------------------------
# 4. time holdout
# --------------------------------------------------------------------------
def holdout(
    arrays: dict[str, np.ndarray],
    d: ModelData,
    *,
    holdout_weeks: int = HOLDOUT_WEEKS,
    seed: int = 20261011,
) -> tuple[dict, dict]:
    """Predict the last N weeks from a fit that never saw them.

    Coverage of the 90% predictive interval, and MAPE against the naive seasonal
    baseline (same week last year; the last observed week when there is no year
    of history). Beating a naive baseline is a low bar — which is exactly why
    failing it would matter.
    """
    cutoff = d.n_weeks - holdout_weeks
    test = d.subset(d.t >= cutoff)
    train = d.subset(d.t < cutoff)

    a_city = _flat(arrays["a_city"])[:, test.city_idx]
    a_seg = _flat(arrays["a_seg"])[:, test.seg_idx]
    e = _flat(arrays["e"])[:, test.seg_idx]
    b_promo = _flat(arrays["b_promo"])[:, test.seg_idx]
    b_feature = _flat(arrays["b_feature"])[:, None]
    b_dip = _flat(arrays["b_dip"])[:, None]
    season_c = _flat(arrays["season"])
    phi = _flat(arrays["phi"])

    season_X = season_row(test.t, n_fourier=season_c.shape[1] // 2, period=52.0)
    log_mu = (
        test.log_share[None, :]
        + a_city
        + a_seg
        + e * np.log(test.price / test.base_price)[None, :]
        + b_promo * test.promo[None, :]
        + b_feature * test.feature[None, :]
        + b_dip * test.promo_last_week[None, :]
        + season_c @ season_X.T
    )
    mu = np.exp(log_mu)

    # predictive draws, not just mu: coverage must include observation noise
    rng = np.random.default_rng(seed)
    lam = rng.gamma(shape=phi[:, None], scale=mu / phi[:, None])
    y_rep = rng.poisson(lam)

    lo, hi = np.quantile(y_rep, [0.05, 0.95], axis=0)
    coverage = float(np.mean((test.units >= lo) & (test.units <= hi)))

    pred = mu.mean(axis=0)
    denom = np.where(test.units > 0, test.units, np.nan)
    mape_model = float(np.nanmean(np.abs(pred - test.units) / denom))

    # naive seasonal baseline: same week last year, per city x segment
    naive = np.empty_like(pred)
    for i in range(test.n_rows):
        c, s, t = test.city_idx[i], test.seg_idx[i], test.t[i]
        same = (train.city_idx == c) & (train.seg_idx == s)
        last_year = same & (train.t == t - 52)
        if last_year.any():
            naive[i] = train.units[last_year][0]
        else:
            hist = train.units[same]
            naive[i] = hist[-1] if len(hist) else float(train.units.mean())
    mape_naive = float(np.nanmean(np.abs(naive - test.units) / denom))

    block = {
        "weeks_held_out": holdout_weeks,
        "coverage_90": round(coverage, 4),
        "mape_model": round(mape_model, 4),
        "mape_naive": round(mape_naive, 4),
        "beats_naive": bool(mape_model < mape_naive),
    }
    check = {
        "name": "Time holdout",
        "group": "holdout",
        "passed": bool(coverage >= HOLDOUT_COVERAGE_MIN and mape_model < mape_naive),
        "value": round(coverage, 4),
        "threshold": HOLDOUT_COVERAGE_MIN,
        "comparison": "gte",
        "plain_english": (
            f"The last {holdout_weeks} weeks were hidden from the fit. {coverage:.0%} of them "
            f"landed inside the 90% range, and the average error was {mape_model:.0%} against "
            f"{mape_naive:.0%} for a same-week-last-year guess."
        ),
        "detail": (
            f"MAPE {mape_model:.1%} vs. naive {mape_naive:.1%}"
            + ("" if mape_model < mape_naive else " — the naive baseline wins, which is a fail")
        ),
    }
    return check, block


# --------------------------------------------------------------------------
# 5. model facts for the agent
# --------------------------------------------------------------------------
def model_facts(arrays: dict[str, np.ndarray], d: ModelData) -> dict:
    """Pre-summarized effects with intervals, so model_facts() computes nothing."""
    facts: dict = {
        "elasticity_by_segment": [
            {"segment": seg, **_summ(_flat(arrays["e"])[:, si])}
            for si, seg in enumerate(d.segments)
        ]
    }
    if d.has_promo:
        facts["promo_lift_by_segment"] = [
            {"segment": seg, **_summ(_flat(arrays["b_promo"])[:, si])}
            for si, seg in enumerate(d.segments)
        ]
        facts["post_promo_dip"] = _summ(_flat(arrays["b_dip"]))
    if d.has_feature:
        facts["feature_lift"] = _summ(_flat(arrays["b_feature"]))
    return facts


# --------------------------------------------------------------------------
# assemble trust.json
# --------------------------------------------------------------------------
LIMITS = [
    "The demo company is synthetic, so the planted truth is known. On your own data "
    "there is no truth to check against, and the recovery card is replaced by the holdout.",
    "The model has no competitor reaction: it answers what happens if you change your "
    "price, not what happens after a competitor answers back.",
    "Effects are assumed stable over the forecast horizon. A market that is changing "
    "shape will drift outside these intervals.",
    "A new city is a forecast from its traits. Outside the training trait range the "
    "interval is a lower bound on the real uncertainty, and Ripple says so.",
]


def build_trust(
    arrays: dict[str, np.ndarray],
    result: FitResult,
    d: ModelData,
    *,
    y_rep: np.ndarray | None = None,
    holdout_arrays: dict[str, np.ndarray] | None = None,
    truth: dict | None = None,
) -> dict:
    checks: list[dict] = []
    conv_checks, conv = convergence(arrays, result)
    checks += conv_checks

    if y_rep is not None:
        checks.append(ppc_coverage(y_rep, d.units))

    rec: list[dict] = []
    if truth is not None:
        rec = recovery(arrays, truth, d)
        checks.append(recovery_check(rec))
    else:
        checks.append(
            {
                "name": "Parameter recovery",
                "group": "recovery",
                "passed": False,
                "value": None,
                "threshold": None,
                "plain_english": "Recovery needs data with known true values. Your own data "
                                 "has none, so the time holdout below is the real test here.",
                "skipped_reason": "no planted truth on uploaded data",
            }
        )

    hold_block: dict | None = None
    if holdout_arrays is not None:
        hold_check, hold_block = holdout(holdout_arrays, d)
        checks.append(hold_check)

    counted = [c for c in checks if not c.get("skipped_reason")]
    passed = sum(c["passed"] for c in counted)
    doc = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "method": result.method,
        "dataset_name": d.dataset_name,
        "summary": {
            "passed": passed,
            "total": len(counted),
            "headline": f"{passed} of {len(counted)} checks passed",
        },
        "checks": checks,
        "convergence": conv,
        "model_facts": model_facts(arrays, d),
        "limits": LIMITS,
    }
    if rec:
        doc["recovery"] = rec
    if hold_block:
        doc["holdout"] = hold_block
    return doc


def write_trust(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
