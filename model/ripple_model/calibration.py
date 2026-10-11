"""Simulation-based calibration of the launch update (spec Section 5b).

No model is "100% valid". What we can prove is that the method is **calibrated**
when its assumptions hold, and show where it stops being calibrated when they do
not.

For each replicate:
  1. pick one posterior draw — that is the "true world" for the other parameters;
  2. draw the unknown from its prior — that is the true lift / city effect;
  3. generate 8 weeks of Negative Binomial launch data from that truth;
  4. reveal weeks 1..8 one at a time and run the real update each time;
  5. record whether the 90% interval contains the TRUE horizon total and the TRUE
     value of the unknown, how wide the interval is, the bias, and whether the
     validity gate fired.

Because the truth is drawn from the same prior the update uses, coverage of a 90%
interval must come out at 90% if the computation is right. That is the whole point:
this is a test of the machinery, not a flattering demo.

The stress arm deliberately breaks an assumption — the launch ramps up over the
first weeks, which the model has no term for — and coverage is expected to drop.
Reporting that is the honest half of the claim.

Output: contract/calibration.schema.json -> web/public/model/calibration.json.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np

from ripple_model.update import update_with_launch

TARGET_COVERAGE = 0.9
REVEAL_WEEKS = 8
HORIZON = 12
DEFAULT_SEED = 20261011

# Calibration runs the update thousands of times, so it uses a subsampled
# posterior and a smaller K. The machinery under test is identical.
CAL_DRAWS = 200
CAL_K = 10


def subsample_posterior(post: dict, n: int, seed: int = DEFAULT_SEED) -> dict:
    """A posterior with n draws, evenly spaced. Keeps meta, history, candidates."""
    D = int(post["meta"]["n_draws"])
    idx = np.linspace(0, D - 1, min(n, D)).round().astype(int)
    draws = {}
    for key, value in post["draws"].items():
        a = np.asarray(value, dtype=float)
        draws[key] = a[idx].tolist()
    out = {
        "meta": {**post["meta"], "n_draws": int(len(idx))},
        "draws": draws,
        "history": post["history"],
        "candidates": post.get("candidates", []),
    }
    return out


def _nb_sample(rng: np.random.Generator, mu: np.ndarray, phi: float) -> np.ndarray:
    lam = rng.gamma(shape=phi, scale=np.asarray(mu, dtype=float) / phi)
    return rng.poisson(lam).astype(float)


def _true_log_mu(
    post: dict,
    *,
    draw: int,
    kind: str,
    u_true: float,
    traits_z: np.ndarray | None,
    city_idx: np.ndarray,
    seg_idx: np.ndarray,
    log_share: np.ndarray,
    t: np.ndarray,
    price: np.ndarray,
    treated: np.ndarray,
) -> np.ndarray:
    """log mu of the true world for one replicate. [W][n_cells]."""
    from ripple_model.simulate import season_row

    meta, d = post["meta"], post["draws"]
    base_price = float(meta["base_price"])
    a_seg = np.asarray(d["a_seg"], dtype=float)[draw][seg_idx]
    e = np.asarray(d["e"], dtype=float)[draw][seg_idx]
    season_c = np.asarray(d["season"], dtype=float)[draw]
    season_X = season_row(t, int(meta["n_fourier"]), float(meta["season_period"]))

    if kind == "z_new_city":
        g0 = float(np.asarray(d["g0"], dtype=float)[draw])
        g = np.asarray(d["g"], dtype=float)[draw]
        tau = float(np.asarray(d["tau_city"], dtype=float)[draw])
        a_city = g0 + float(g @ traits_z) + tau * u_true
        a_city_vec = np.full(len(seg_idx), a_city)
    else:
        a_city_vec = np.asarray(d["a_city"], dtype=float)[draw][city_idx]

    log_mu = (
        log_share[None, :]
        + (a_city_vec + a_seg)[None, :]
        + e[None, :] * np.log(price / base_price)
        + (season_X @ season_c)[:, None]
    )
    if kind == "theta":
        log_mu = log_mu + u_true * treated
    return log_mu


def _make_launch_template(post: dict, kind: str) -> dict:
    """The launch dict shape the update expects, for a calibration replicate."""
    meta = post["meta"]
    segments = list(meta["segments"])
    if kind == "z_new_city":
        return {
            "id": "cal-new-city",
            "label": "calibration new city",
            "type": "new_city",
            "unknown": {
                "name": "z_new_city",
                "label": "city effect",
                "kind": "z_new_city",
                "prior": {"dist": "normal", "mu": 0.0, "sigma": 1.0},
                "display": "raw",
            },
            "weeks_observed": REVEAL_WEEKS,
            "horizon": HORIZON,
            "t0": int(meta["weeks"]),
            "cities": ["Calibration City"],
            "segments": segments,
            "new_city": {
                "name": "Calibration City",
                "traits": [0.0] * len(meta["trait_names"]),
                "traits_z": [0.0] * len(meta["trait_names"]),
                "shares": list(np.mean(np.asarray(meta["shares"], dtype=float), axis=0)),
            },
            "has_holdout": False,
            "unit_cost": float(meta["unit_cost"]),
            "fixed_cost": 0.0,
            "net_price": float(meta["base_price"]),
        }
    return {
        "id": "cal-feature",
        "label": "calibration feature",
        "type": "feature",
        "unknown": {
            "name": "theta",
            "label": "feature lift",
            "kind": "theta",
            "prior": {"dist": "normal", "mu": 0.0, "sigma": 0.15},
            "display": "raw",
        },
        "weeks_observed": REVEAL_WEEKS,
        "horizon": HORIZON,
        "t0": int(meta["weeks"]),
        "cities": [meta["cities"][0]],
        "segments": segments,
        "has_holdout": False,
        "unit_cost": float(meta["unit_cost"]),
        "fixed_cost": 0.0,
        "net_price": float(meta["base_price"]),
    }


def _ramp(week: np.ndarray, strength: float = 0.45, rate: float = 0.45) -> np.ndarray:
    """The broken assumption: demand ramps up instead of starting at full level."""
    return 1.0 - strength * np.exp(-rate * (np.asarray(week, dtype=float) - 1.0))


def run_arm(
    post: dict,
    *,
    kind: str,
    n_launches: int,
    assumptions_hold: bool,
    seed: int,
) -> dict:
    """One calibration arm: n_launches replicates, each revealed week by week."""
    meta = post["meta"]
    segments = list(meta["segments"])
    n_seg = len(segments)
    rng = np.random.default_rng(seed)
    template = _make_launch_template(post, kind)
    D = int(meta["n_draws"])

    weeks = np.arange(1, REVEAL_WEEKS + 1)
    t_obs = int(template["t0"]) + weeks - 1
    t_fore = int(template["t0"]) + np.arange(HORIZON)
    price_obs = np.full((REVEAL_WEEKS, n_seg), float(meta["base_price"]))
    price_fore = np.full((HORIZON, n_seg), float(meta["base_price"]))
    seg_idx = np.arange(n_seg)
    treated = np.ones((REVEAL_WEEKS, n_seg))
    treated_f = np.ones((HORIZON, n_seg))

    if kind == "z_new_city":
        shares = np.asarray(template["new_city"]["shares"], dtype=float)
        log_share = np.log(shares)
        city_idx = np.full(n_seg, -1)
    else:
        shares = np.asarray(meta["shares"], dtype=float)[0]
        log_share = np.log(shares)
        city_idx = np.zeros(n_seg, dtype=int)

    records: list[list[dict]] = [[] for _ in range(REVEAL_WEEKS)]

    for _ in range(n_launches):
        draw = int(rng.integers(0, D))
        phi = float(np.asarray(post["draws"]["phi"], dtype=float)[draw])
        prior = template["unknown"]["prior"]
        u_true = float(rng.normal(prior["mu"], prior["sigma"]))
        traits_z = (
            np.asarray(template["new_city"]["traits_z"], dtype=float)
            if kind == "z_new_city"
            else None
        )

        common = dict(
            draw=draw, kind=kind, u_true=u_true, traits_z=traits_z,
            city_idx=city_idx, seg_idx=seg_idx, log_share=log_share,
        )
        mu_obs = np.exp(
            _true_log_mu(post, **common, t=t_obs, price=price_obs, treated=treated)
        )
        mu_fore = np.exp(
            _true_log_mu(post, **common, t=t_fore, price=price_fore, treated=treated_f)
        )
        if not assumptions_hold:
            mu_obs = mu_obs * _ramp(weeks)[:, None]
            mu_fore = mu_fore * _ramp(np.arange(1, HORIZON + 1))[:, None]

        true_total = float(mu_fore.sum())
        y = _nb_sample(rng, mu_obs, phi)

        launch = dict(template)
        launch["observations"] = _columnar(
            weeks, t_obs, city_idx, seg_idx, y, price_obs, treated
        )
        launch["truth"] = {"unknown_value": u_true, "total_units": true_total}

        for wi, w in enumerate(weeks):
            res = update_with_launch(
                post, launch, int(w), k_per_draw=CAL_K, seed=seed + wi,
                include_density=False,
            )
            tot = res["forecast"]["total_units"]
            unk = res["unknown"]["posterior"]
            records[wi].append(
                {
                    "inside_total": bool(tot["p05"] <= true_total <= tot["p95"]),
                    "inside_unknown": bool(unk["p05"] <= u_true <= unk["p95"]),
                    "width": float(tot["p95"] - tot["p05"]),
                    "error": float(tot["mean"] - true_total),
                    "refit": not res["gate"]["ok"],
                    "ess": float(res["gate"]["ess"]),
                    "pareto_k": float(res["gate"]["pareto_k"] or np.nan),
                }
            )

    by_week = []
    for wi, w in enumerate(weeks):
        r = records[wi]
        n = len(r)
        cov = float(np.mean([x["inside_total"] for x in r]))
        by_week.append(
            {
                "week": int(w),
                "coverage_total": round(cov, 4),
                "coverage_total_mcse": round(float(np.sqrt(cov * (1 - cov) / n)), 4),
                "coverage_unknown": round(
                    float(np.mean([x["inside_unknown"] for x in r])), 4
                ),
                "mean_width_total": round(float(np.mean([x["width"] for x in r])), 2),
                "median_width_total": round(float(np.median([x["width"] for x in r])), 2),
                "bias_total": round(float(np.mean([x["error"] for x in r])), 2),
                "rmse_total": round(
                    float(np.sqrt(np.mean([x["error"] ** 2 for x in r]))), 2
                ),
                "refit_rate": round(float(np.mean([x["refit"] for x in r])), 4),
                "mean_ess": round(float(np.mean([x["ess"] for x in r])), 1),
                "mean_pareto_k": round(float(np.nanmean([x["pareto_k"] for x in r])), 4),
                "n": n,
            }
        )

    label = {
        ("z_new_city", True): "New city",
        ("theta", True): "New feature",
        ("theta", False): "New feature with a ramp-up (assumption broken)",
        ("z_new_city", False): "New city with a ramp-up (assumption broken)",
    }[(kind, assumptions_hold)]
    arm = {
        "id": f"{'new_city' if kind == 'z_new_city' else 'feature'}"
              f"{'' if assumptions_hold else '_rampup'}",
        "label": label,
        "kind": "new_city" if kind == "z_new_city" else "feature",
        "assumptions_hold": assumptions_hold,
        "by_week": by_week,
        "verdict": _verdict(by_week, assumptions_hold),
    }
    if not assumptions_hold:
        arm["broken_assumption"] = (
            "the launch ramps up over the first weeks, which the model has no term for"
        )
    return arm


def _columnar(weeks, t, city_idx, seg_idx, y, price, treated) -> dict:
    obs = {
        k: []
        for k in ("week", "t", "city_idx", "segment_idx", "units", "price",
                  "promo", "promo_last_week", "feature", "treated")
    }
    for wi, w in enumerate(weeks):
        for ci in range(len(seg_idx)):
            obs["week"].append(int(w))
            obs["t"].append(int(t[wi]))
            obs["city_idx"].append(int(city_idx[ci]))
            obs["segment_idx"].append(int(seg_idx[ci]))
            obs["units"].append(float(y[wi, ci]))
            obs["price"].append(float(price[wi, ci]))
            obs["promo"].append(0.0)
            obs["promo_last_week"].append(0.0)
            obs["feature"].append(0.0)
            obs["treated"].append(float(treated[wi, ci]))
    return obs


def _verdict(by_week: list[dict], assumptions_hold: bool) -> str:
    cov = np.array([b["coverage_total"] for b in by_week])
    worst = float(np.min(np.abs(cov - TARGET_COVERAGE)))
    gap = float(np.max(np.abs(cov - TARGET_COVERAGE)))
    first, last = by_week[0], by_week[-1]
    shrink = (
        1.0 - last["mean_width_total"] / first["mean_width_total"]
        if first["mean_width_total"]
        else 0.0
    )
    if assumptions_hold:
        return (
            f"Coverage stays within {gap * 100:.0f} points of {TARGET_COVERAGE:.0%} across "
            f"all 8 weeks (closest {worst * 100:.0f} points), and the 90% interval narrows "
            f"{shrink:.0%} from week 1 to week 8."
        )
    return (
        f"With the ramp-up the model does not have, coverage falls to "
        f"{cov.min():.0%} against a target of {TARGET_COVERAGE:.0%}. "
        "This is the failure mode to watch for: a launch that builds slowly looks like "
        "a weak launch to Ripple."
    )


def run_calibration(
    post: dict,
    *,
    n_launches: int = 200,
    seed: int = DEFAULT_SEED,
    include_stress: bool = True,
    cal_draws: int = CAL_DRAWS,
) -> dict:
    t0 = time.perf_counter()
    small = subsample_posterior(post, cal_draws, seed=seed)
    arms = [
        run_arm(small, kind="z_new_city", n_launches=n_launches,
                assumptions_hold=True, seed=seed),
        run_arm(small, kind="theta", n_launches=n_launches,
                assumptions_hold=True, seed=seed + 1000),
    ]
    if include_stress:
        arms.append(
            run_arm(small, kind="theta", n_launches=n_launches,
                    assumptions_hold=False, seed=seed + 2000)
        )

    ok = [a for a in arms if a["assumptions_hold"]]
    worst = min(
        (b["coverage_total"] for a in ok for b in a["by_week"]), default=0.0
    )
    headline = (
        f"Across {n_launches} synthetic launches per arm, the 90% interval covers the "
        f"true 12-week total at least {worst:.0%} of the time when the assumptions hold."
    )
    if include_stress:
        stress = [a for a in arms if not a["assumptions_hold"]][0]
        s_worst = min(b["coverage_total"] for b in stress["by_week"])
        headline += (
            f" Break one assumption — a launch that ramps up — and coverage drops to "
            f"{s_worst:.0%}, which is the limit we want you to know about."
        )

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n_launches": n_launches,
        "target_coverage": TARGET_COVERAGE,
        "arms": arms,
        "headline": headline,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
