"""Launch tracker: update the forecast with real launch weeks (spec Section 5b).

Exact Bayes by importance sampling, not a heuristic:

  1. Joint prior samples — for each posterior draw d (D of them) draw K values of
     the single unknown. The other parameters stay AT their posterior draw, so the
     prior over the joint parameter vector is the fitted posterior x the unknown's
     prior.
  2. For every sample, compute expected units in the observed launch cells and the
     Negative Binomial log-likelihood of what actually happened.
  3. Self-normalized importance weights w ∝ exp(loglik), via log-sum-exp. Python
     smooths them with PSIS (Vehtari et al.); the browser does not, and reports
     weight ESS instead.
  4. Every summary is a weighted quantile of those samples.
  5. Validity gate — Pareto k-hat > 0.7 (here) or weight ESS < 300 (browser) means
     the approximation cannot be trusted, so the UI says "Refit needed" and the API
     refits the full model with the new rows.

Limit, stated in the UI: the OTHER parameters are not re-learned from the launch
weeks. The refit path does that. This is why the gate exists.

The twin is web/lib/update.ts and the two are held together by
contract/update.parity.fixture.json.
"""

from __future__ import annotations

import numpy as np

from ripple_model.psis import K_HAT_THRESHOLD, psis_smooth
from ripple_model.simulate import season_row
from ripple_model.synth import HOLDOUT_RECOMMENDATION

K_PER_DRAW = 20
ESS_MIN = 300.0                   # the browser's gate
PARETO_K_MAX = K_HAT_THRESHOLD    # the PSIS gate here
DENSITY_POINTS = 60
DEFAULT_SEED = 20261011

# How much a brand-new offer may differ from the average promotion in the history.
# The fitted posterior of b_promo is the centre; this is the spread around it.
NEW_OFFER_SIGMA = 0.25


# --------------------------------------------------------------------------
# weighted summaries
# --------------------------------------------------------------------------
def weighted_quantile(x: np.ndarray, q: float | np.ndarray, w: np.ndarray) -> np.ndarray:
    """Quantiles of x under normalized weights w. Mirrored in update.ts."""
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws)
    total = cum[-1]
    if total <= 0:
        return np.full(np.shape(q), float("nan"))
    # midpoint convention: the same rule as numpy's "linear" on equal weights
    cdf = (cum - 0.5 * ws) / total
    return np.interp(np.atleast_1d(q), cdf, xs)


def w_interval(x: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """mean + 90% interval under weights. Every user-visible number comes through here."""
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    p05, p50, p95 = weighted_quantile(x, [0.05, 0.5, 0.95], w)
    return {
        "mean": float(np.sum(w * x) / np.sum(w)),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


def w_band(x: np.ndarray, w: np.ndarray) -> dict[str, float]:
    q = weighted_quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95], w)
    return {"p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4])}


def weight_ess(w: np.ndarray) -> float:
    """Kish effective sample size of normalized importance weights."""
    w = np.asarray(w, dtype=float).ravel()
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    return float(1.0 / np.sum(p * p))


def normalize_log_weights(logw: np.ndarray) -> np.ndarray:
    """exp-normalize, log-sum-exp style so a very negative loglik cannot underflow."""
    logw = np.asarray(logw, dtype=float)
    m = np.max(logw)
    if not np.isfinite(m):
        return np.full(logw.shape, 1.0 / logw.size)
    w = np.exp(logw - m)
    total = w.sum()
    return w / total if total > 0 else np.full(logw.shape, 1.0 / logw.size)


def weighted_density(x: np.ndarray, w: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Weighted Gaussian KDE on a fixed grid, Silverman bandwidth.

    A fixed shared grid is what lets the UI draw prior and posterior on one axis.
    """
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    w = w / w.sum()
    mean = np.sum(w * x)
    var = max(np.sum(w * (x - mean) ** 2), 1e-12)
    n_eff = weight_ess(w)
    bw = 1.06 * np.sqrt(var) * max(n_eff, 2.0) ** (-0.2)
    z = (grid[:, None] - x[None, :]) / bw
    return (np.exp(-0.5 * z * z) @ w) / (bw * np.sqrt(2.0 * np.pi))


# --------------------------------------------------------------------------
# negative binomial log-likelihood
# --------------------------------------------------------------------------
def nb_loglik(y: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """log p(y | mu, phi) with var = mu + mu^2/phi, summed over the last axis.

    Written out rather than taken from scipy so that update.ts can be an exact
    line-by-line port: the gammaln terms that do not involve mu are dropped
    because a constant shift cancels in the self-normalized weights... except it
    does NOT cancel across samples when phi varies, so phi's terms are kept.
    """
    from scipy.special import gammaln

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mu = np.maximum(mu, 1e-12)
    ll = (
        gammaln(y + phi)
        - gammaln(phi)
        - gammaln(y + 1.0)
        + phi * (np.log(phi) - np.log(phi + mu))
        + y * (np.log(mu) - np.log(phi + mu))
    )
    return ll.sum(axis=-1)


# --------------------------------------------------------------------------
# the unknown
# --------------------------------------------------------------------------
def draw_unknown(
    post: dict,
    launch: dict,
    *,
    k_per_draw: int = K_PER_DRAW,
    seed: int = DEFAULT_SEED,
    prior_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Prior samples of the unknown.

    Returns (values [D*K], draw_index [D*K], sigma_used). The draw index is what
    keeps each sample paired with the posterior draw it was drawn under.
    """
    draws = post["draws"]
    D = int(post["meta"]["n_draws"])
    unknown = launch["unknown"]
    kind = unknown["kind"]
    prior = unknown.get("prior", {})
    rng = np.random.default_rng(seed)

    draw_idx = np.repeat(np.arange(D), k_per_draw)
    z = rng.standard_normal(D * k_per_draw)

    if kind == "z_new_city":
        sigma = float(prior_sigma if prior_sigma is not None else prior.get("sigma", 1.0))
        mu0 = float(prior.get("mu", 0.0))
        return mu0 + sigma * z, draw_idx, sigma

    if kind == "theta":
        sigma = float(prior_sigma if prior_sigma is not None else prior.get("sigma", 0.15))
        mu0 = float(prior.get("mu", 0.0))
        return mu0 + sigma * z, draw_idx, sigma

    if kind in ("b_promo", "elasticity"):
        # Centre on what history taught us, per posterior draw, and allow a new
        # offer to differ from the average past promotion by NEW_OFFER_SIGMA.
        segments = list(post["meta"]["segments"])
        seg = unknown.get("segment")
        si = segments.index(seg) if seg in segments else 0
        key = "b_promo" if kind == "b_promo" else "e"
        centre = np.asarray(draws[key], dtype=float)[:, si][draw_idx]
        sigma = float(
            prior_sigma
            if prior_sigma is not None
            else prior.get("sigma", NEW_OFFER_SIGMA)
        )
        return centre + sigma * z, draw_idx, sigma

    raise ValueError(f"unknown launch unknown kind: {kind!r}")


# --------------------------------------------------------------------------
# log mu for launch cells
# --------------------------------------------------------------------------
def _launch_log_mu(
    post: dict,
    launch: dict,
    *,
    u: np.ndarray,            # [N] unknown values
    draw_idx: np.ndarray,     # [N]
    seg_idx: np.ndarray,      # [n_cells]
    log_share: np.ndarray,    # [n_cells]
    t: np.ndarray,            # [W]
    price: np.ndarray,        # [W][n_cells]
    promo: np.ndarray,
    promo_last: np.ndarray,
    feature: np.ndarray,
    treated: np.ndarray,      # [W][n_cells] 1 where the unknown applies
    a_city_idx: np.ndarray | None,  # [n_cells] training city index, None for a new city
    apply_unknown: bool = True,
) -> np.ndarray:
    """log mu with shape [N][W][n_cells]."""
    meta, draws = post["meta"], post["draws"]
    base_price = float(meta["base_price"])
    kind = launch["unknown"]["kind"]

    a_seg = np.asarray(draws["a_seg"], dtype=float)[draw_idx][:, seg_idx]
    e = np.asarray(draws["e"], dtype=float)[draw_idx][:, seg_idx]
    b_promo = np.asarray(draws["b_promo"], dtype=float)[draw_idx][:, seg_idx]
    b_feature = np.asarray(draws["b_feature"], dtype=float)[draw_idx][:, None]
    b_dip = np.asarray(draws["b_dip"], dtype=float)[draw_idx][:, None]
    season_c = np.asarray(draws["season"], dtype=float)[draw_idx]
    season_X = season_row(t, int(meta["n_fourier"]), float(meta["season_period"]))
    season_term = season_c @ season_X.T                                  # [N][W]

    if a_city_idx is None:
        g0 = np.asarray(draws["g0"], dtype=float)[draw_idx]
        g = np.asarray(draws["g"], dtype=float)[draw_idx]
        tau = np.asarray(draws["tau_city"], dtype=float)[draw_idx]
        tz = np.asarray(launch["new_city"]["traits_z"], dtype=float)
        trait_part = g @ tz if g.shape[1] else np.zeros(len(draw_idx))
        z_value = u if (kind == "z_new_city" and apply_unknown) else 0.0
        a_city = (g0 + trait_part + tau * z_value)[:, None]              # [N][1]
    else:
        a_city = np.asarray(draws["a_city"], dtype=float)[draw_idx][:, a_city_idx]

    if kind == "b_promo" and apply_unknown:
        b_promo = np.repeat(u[:, None], len(seg_idx), axis=1)
    if kind == "elasticity" and apply_unknown:
        e = np.repeat(u[:, None], len(seg_idx), axis=1)

    log_mu = (
        log_share[None, None, :]
        + (a_city + a_seg)[:, None, :]
        + e[:, None, :] * np.log(price / base_price)[None, :, :]
        + b_promo[:, None, :] * (promo * treated)[None, :, :]
        + b_feature[:, :, None] * feature[None, :, :]
        + b_dip[:, :, None] * promo_last[None, :, :]
        + season_term[:, :, None]
    )
    if kind == "theta" and apply_unknown:
        log_mu = log_mu + u[:, None, None] * treated[None, :, :]
    return log_mu


def _cells_and_design(post: dict, launch: dict, weeks: list[int]) -> dict:
    """Observation design for the given launch weeks, as [W][n_cells] arrays."""
    obs = launch["observations"]
    week = np.asarray(obs["week"])
    keep = np.isin(week, weeks)

    # a cell is one (city_idx, segment_idx, treated) combination
    cols = list(
        dict.fromkeys(
            zip(
                np.asarray(obs["city_idx"])[keep].tolist(),
                np.asarray(obs["segment_idx"])[keep].tolist(),
                np.asarray(obs["treated"])[keep].tolist(),
                strict=True,
            )
        )
    )
    cell_pos = {c: i for i, c in enumerate(cols)}
    W, n = len(weeks), len(cols)
    shape = (W, n)
    out = {
        k: np.zeros(shape)
        for k in ("price", "promo", "promo_last", "feature", "treated", "units")
    }
    out["price"][:] = float(post["meta"]["base_price"])
    t = np.zeros(W, dtype=int)
    week_pos = {w: i for i, w in enumerate(weeks)}

    idx = np.where(keep)[0]
    for i in idx:
        key = (obs["city_idx"][i], obs["segment_idx"][i], obs["treated"][i])
        wi, ci = week_pos[obs["week"][i]], cell_pos[key]
        out["price"][wi, ci] = obs["price"][i]
        out["promo"][wi, ci] = obs["promo"][i]
        out["promo_last"][wi, ci] = obs["promo_last_week"][i]
        out["feature"][wi, ci] = obs["feature"][i]
        out["treated"][wi, ci] = obs["treated"][i]
        out["units"][wi, ci] = obs["units"][i]
        t[wi] = obs["t"][i]

    seg_idx = np.array([c[1] for c in cols], dtype=int)
    city_idx = np.array([c[0] for c in cols], dtype=int)
    is_new_city = bool((city_idx < 0).any())

    if is_new_city:
        shares = np.asarray(launch["new_city"]["shares"], dtype=float)
        log_share = np.log(shares[seg_idx])
        a_city_idx = None
    else:
        shares = np.asarray(post["meta"]["shares"], dtype=float)
        # a holdout arm holds a fraction of the city, so its share is already
        # encoded in the observed volume: recover it from the arm's size
        log_share = np.log(shares[city_idx, seg_idx] * _arm_fraction(launch, cols))
        a_city_idx = city_idx

    out.update(
        t=t, seg_idx=seg_idx, city_idx=city_idx, log_share=log_share,
        a_city_idx=a_city_idx, cells=cols, n_cells=n,
    )
    return out


def _arm_fraction(launch: dict, cols: list[tuple]) -> np.ndarray:
    """Share of a city's segment that sits in each arm (1.0 with no holdout)."""
    if not launch.get("has_holdout"):
        return np.ones(len(cols))
    # synth.py splits 90/10 treated/holdout; read it off the arm flag
    return np.array([0.9 if c[2] == 1 else 0.1 for c in cols])


# --------------------------------------------------------------------------
# the update
# --------------------------------------------------------------------------
def update_with_launch(
    post: dict,
    launch: dict,
    weeks_used: int,
    *,
    k_per_draw: int = K_PER_DRAW,
    seed: int = DEFAULT_SEED,
    prior_sigma: float | None = None,
    smooth: bool = True,
    include_density: bool = True,
) -> dict:
    """One slider position. Returns a dict matching contract/update.schema.json.

    `include_density` is turned off by calibration.py, which runs this thousands
    of times and never draws the prior-vs-posterior plot.
    """
    meta, draws = post["meta"], post["draws"]
    D = int(meta["n_draws"])
    horizon = int(launch["horizon"])
    unit_cost = float(launch["unit_cost"])
    fixed_cost = float(launch["fixed_cost"])
    weeks_used = int(max(0, min(weeks_used, int(launch["weeks_observed"]))))

    u, draw_idx, sigma_used = draw_unknown(
        post, launch, k_per_draw=k_per_draw, seed=seed, prior_sigma=prior_sigma
    )
    N = len(u)
    phi = np.asarray(draws["phi"], dtype=float)[draw_idx]

    # ---- likelihood of the revealed weeks -------------------------------
    pareto_k: float | None = None
    if weeks_used > 0:
        obs_weeks = list(range(1, weeks_used + 1))
        des = _cells_and_design(post, launch, obs_weeks)
        log_mu = _launch_log_mu(
            post, launch, u=u, draw_idx=draw_idx, seg_idx=des["seg_idx"],
            log_share=des["log_share"], t=des["t"], price=des["price"],
            promo=des["promo"], promo_last=des["promo_last"], feature=des["feature"],
            treated=des["treated"], a_city_idx=des["a_city_idx"],
        )
        mu = np.exp(log_mu).reshape(N, -1)
        y = des["units"].reshape(-1)
        logw = nb_loglik(y[None, :], mu, phi[:, None])
        n_obs_cells = int(y.size)

        if smooth:
            smoothed, khat = psis_smooth(logw)
            pareto_k = float(khat)
            w = normalize_log_weights(smoothed)
        else:
            w = normalize_log_weights(logw)
    else:
        w = np.full(N, 1.0 / N)
        n_obs_cells = 0

    ess = weight_ess(w)
    w_prior = np.full(N, 1.0 / N)

    # ---- forecast over the horizon --------------------------------------
    fdes = _forecast_design(post, launch, horizon)
    f_args = dict(
        u=u, draw_idx=draw_idx, seg_idx=fdes["seg_idx"], log_share=fdes["log_share"],
        t=fdes["t"], price=fdes["price"], promo=fdes["promo"],
        promo_last=fdes["promo_last"], feature=fdes["feature"],
        treated=fdes["treated"], a_city_idx=fdes["a_city_idx"],
    )
    mu_f = np.exp(_launch_log_mu(post, launch, **f_args))                 # [N][H][cells]
    mu_cf = np.exp(_launch_log_mu(post, launch, **{**f_args, "apply_unknown": False}))
    if launch["type"] == "new_city":
        mu_cf = np.zeros_like(mu_cf)  # with no launch there is no city

    net_price = float(launch.get("net_price", meta["base_price"]))
    units_tot = mu_f.sum(axis=(1, 2))
    revenue_tot = (mu_f * fdes["price"][None, :, :]).sum(axis=(1, 2))
    profit_tot = (mu_f * (fdes["price"] - unit_cost)[None, :, :]).sum(axis=(1, 2)) - fixed_cost

    observed_by_week = _observed_totals(launch, weeks_used)
    weekly = []
    for h in range(horizon):
        weekly.append(
            {
                "week": h + 1,
                "t": int(fdes["t"][h]),
                "units": w_band(mu_f[:, h, :].sum(axis=1), w),
                "observed_units": observed_by_week.get(h + 1),
                "counterfactual_units": w_band(mu_cf[:, h, :].sum(axis=1), w),
            }
        )

    # ---- the unknown, prior vs. posterior -------------------------------
    unknown = launch["unknown"]
    display = unknown.get("display", "raw")
    scale = float(np.asarray(draws["tau_city"], dtype=float)[draw_idx].mean()) \
        if unknown["kind"] == "z_new_city" else 1.0
    u_eff = u * scale  # a z of 1 means tau_city on the log scale, not 100%

    unknown_block = {
        "name": unknown["name"],
        "label": unknown["label"],
        "display": display,
        "prior": w_interval(u, w_prior),
        "posterior": w_interval(u, w),
    }
    if include_density:
        grid = np.linspace(u.min(), u.max(), DENSITY_POINTS)
        unknown_block["density"] = {
            "x": [float(x) for x in grid],
            "prior": [float(x) for x in weighted_density(u, w_prior, grid)],
            "posterior": [float(x) for x in weighted_density(u, w, grid)],
        }
    if display == "percent_lift":
        unknown_block["prior_percent"] = w_interval(np.exp(u_eff) - 1.0, w_prior)
        unknown_block["posterior_percent"] = w_interval(np.exp(u_eff) - 1.0, w)
    truth = launch.get("truth", {}).get("unknown_value")
    unknown_block["truth"] = None if truth is None else float(truth)

    # ---- validity gate ---------------------------------------------------
    if smooth and pareto_k is not None and np.isfinite(pareto_k):
        ok = bool(pareto_k <= PARETO_K_MAX)
        gate = {
            "ok": ok,
            "criterion": "pareto_k",
            "value": round(pareto_k, 4),
            "threshold": PARETO_K_MAX,
            "ess": round(ess, 1),
            "pareto_k": round(pareto_k, 4),
            "smoothed": True,
            "message": (
                f"Fast update reliable (Pareto k̂ {pareto_k:.2f}, ESS {ess:,.0f} of {N:,})"
                if ok
                else f"Refit needed — Pareto k̂ {pareto_k:.2f} is above {PARETO_K_MAX}. "
                     "The launch weeks are too surprising for the fast update; "
                     "Ripple refits the full model with these rows."
            ),
        }
    else:
        ok = bool(ess >= ESS_MIN) if weeks_used else True
        gate = {
            "ok": ok,
            "criterion": "ess",
            "value": round(ess, 1),
            "threshold": ESS_MIN,
            "ess": round(ess, 1),
            "pareto_k": None,
            "smoothed": False,
            "message": (
                f"Fast update reliable (ESS {ess:,.0f} of {N:,})"
                if ok
                else f"Refit needed — only {ess:,.0f} of {N:,} samples carry any weight."
            ),
        }

    # ---- causal reading --------------------------------------------------
    experiment = bool(launch.get("has_holdout"))
    causal = {
        "mode": "experiment" if experiment else "model_based",
        "label": "Experiment (random holdout)" if experiment else "Model-based estimate",
        "assumptions": launch.get("causal_assumptions", []),
        "recommendation": (
            "A random holdout group is in the data, so this lift is measured against a "
            "real control, not against a model prediction."
            if experiment
            else HOLDOUT_RECOMMENDATION
        ),
    }

    return {
        "launch_id": launch["id"],
        "weeks_used": weeks_used,
        "gate": gate,
        "unknown": unknown_block,
        "forecast": {
            "horizon": horizon,
            "total_units": w_interval(units_tot, w),
            "total_revenue": w_interval(revenue_tot, w),
            "total_profit": w_interval(profit_tot, w),
            "p_profit_positive": float(np.sum(w[profit_tot > 0]) / np.sum(w)),
            "weekly": weekly,
        },
        "prior_forecast": {
            "total_units": w_interval(units_tot, w_prior),
            "total_revenue": w_interval(revenue_tot, w_prior),
            "total_profit": w_interval(profit_tot, w_prior),
            "p_profit_positive": float(np.mean(profit_tot > 0)),
        },
        "causal": causal,
        "meta": {
            "n_prior_draws": D,
            "k_per_draw": k_per_draw,
            "n_samples": N,
            "n_obs_cells": n_obs_cells,
            "seed": seed,
            "source": "python",
            "prior_sigma": sigma_used,
        },
        "_net_price": net_price,
    }


def _forecast_design(post: dict, launch: dict, horizon: int) -> dict:
    """Design for the forecast weeks: week 1's cell pattern, continued."""
    des = _cells_and_design(post, launch, [1])
    t0 = int(launch.get("t0", des["t"][0]))
    out = {
        k: np.repeat(des[k][:1], horizon, axis=0)
        for k in ("price", "promo", "promo_last", "feature", "treated")
    }
    out["t"] = t0 + np.arange(horizon)
    for k in ("seg_idx", "city_idx", "log_share", "a_city_idx", "cells", "n_cells"):
        out[k] = des[k]
    return out


def _observed_totals(launch: dict, weeks_used: int) -> dict[int, float]:
    """Observed units per launch week, summed over cells — the dots on the chart."""
    obs = launch["observations"]
    out: dict[int, float] = {}
    for i, w in enumerate(obs["week"]):
        if w <= weeks_used:
            out[int(w)] = out.get(int(w), 0.0) + float(obs["units"][i])
    return out
