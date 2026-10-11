"""Reference scenario simulator (spec Section 5).

This is the Python twin of web/lib/simulate.ts. Both read the SAME exported
posterior JSON and the same scenario JSON, and model/tests/test_parity.py asserts
they agree on mu to 1e-6 — so the browser is not a re-derivation, it is a port.

Everything here follows contract/README.md:
  * horizon week w uses absolute t = meta.weeks + w, so the forecast continues
    the calendar instead of restarting it;
  * g0 is already inside a_city and is used only for a new city;
  * baseline and scenario are differenced INSIDE each draw, then summarized.
"""

from __future__ import annotations

from typing import Any

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


# --------------------------------------------------------------------------
# helpers shared with the exporter and the checks
# --------------------------------------------------------------------------
def season_row(t: np.ndarray | float, n_fourier: int, period: float) -> np.ndarray:
    """Fourier design row(s) for week index t. Mirrored exactly in simulate.ts."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    cols = []
    for k in range(1, n_fourier + 1):
        ang = 2.0 * np.pi * k * t / period
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def interval(x: np.ndarray) -> dict[str, float]:
    """mean + the 90% interval. Every user-visible number goes through here."""
    p05, p50, p95 = np.quantile(x, [0.05, 0.5, 0.95])
    return {
        "mean": float(np.mean(x)),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


def band(x: np.ndarray) -> dict[str, float]:
    """Fan-chart band for one week: 90% and 50% intervals plus the median."""
    q = np.quantile(x, QUANTILES)
    return {"p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4])}


def money(units: np.ndarray, revenue: np.ndarray, profit: np.ndarray) -> dict[str, Any]:
    return {"units": interval(units), "revenue": interval(revenue), "profit": interval(profit)}


# --------------------------------------------------------------------------
# the mean function
# --------------------------------------------------------------------------
def mu_for_cells(
    post: dict,
    *,
    a_city_draws: np.ndarray,     # [D][n_cells] city effect per cell
    seg_idx: np.ndarray,          # [n_cells]
    log_share: np.ndarray,        # [n_cells]
    t: np.ndarray,                # [W] absolute week indices
    price: np.ndarray,            # [W][n_cells]
    promo: np.ndarray,            # [W][n_cells]
    promo_last: np.ndarray,       # [W][n_cells]
    feature: np.ndarray,          # [W][n_cells]
) -> np.ndarray:
    """mu with shape [D][W][n_cells]. The one place the mean function is written."""
    meta, draws = post["meta"], post["draws"]
    base_price = float(meta["base_price"])

    a_seg = np.asarray(draws["a_seg"], dtype=float)[:, seg_idx]        # [D][n_cells]
    e = np.asarray(draws["e"], dtype=float)[:, seg_idx]               # [D][n_cells]
    b_promo = np.asarray(draws["b_promo"], dtype=float)[:, seg_idx]    # [D][n_cells]
    b_feature = np.asarray(draws["b_feature"], dtype=float)[:, None]   # [D][1]
    b_dip = np.asarray(draws["b_dip"], dtype=float)[:, None]           # [D][1]

    season_X = season_row(t, int(meta["n_fourier"]), float(meta["season_period"]))  # [W][K]
    season_c = np.asarray(draws["season"], dtype=float)                             # [D][K]
    season_term = season_c @ season_X.T                                             # [D][W]

    log_price_ratio = np.log(np.asarray(price, dtype=float) / base_price)            # [W][n_cells]

    log_mu = (
        log_share[None, None, :]
        + (a_city_draws + a_seg)[:, None, :]
        + e[:, None, :] * log_price_ratio[None, :, :]
        + b_promo[:, None, :] * np.asarray(promo, dtype=float)[None, :, :]
        + b_feature[:, :, None] * np.asarray(feature, dtype=float)[None, :, :]
        + b_dip[:, :, None] * np.asarray(promo_last, dtype=float)[None, :, :]
        + season_term[:, :, None]
    )
    return np.exp(log_mu)


def new_city_z(post: dict, seed: int) -> np.ndarray:
    """The new-city z draws for this scenario, rotated by the seed.

    The draws come from the posterior export (`draws.z_new`), not from an RNG here.
    numpy's generator and a browser PRNG cannot produce the same numbers, so
    generating them at simulation time would make web/lib/simulate.ts disagree with
    this file on every new-city scenario while the parity test still looked green.
    Rotating a shipped array by the seed keeps `scenario.seed` meaningful and is
    exactly reproducible in both languages.
    """
    z = np.asarray(post["draws"]["z_new"], dtype=float)
    n = len(z)
    if n == 0:
        return np.zeros(int(post["meta"]["n_draws"]))
    return np.roll(z, -(int(seed) % n))


def new_city_a(post: dict, traits_z: np.ndarray, seed: int) -> np.ndarray:
    """a_new = g0 + traits_z·g + tau_city·z_new (spec Section 5)."""
    draws = post["draws"]
    g0 = np.asarray(draws["g0"], dtype=float)
    g = np.asarray(draws["g"], dtype=float)
    tau = np.asarray(draws["tau_city"], dtype=float)
    z = new_city_z(post, seed)
    trait_part = g @ np.asarray(traits_z, dtype=float) if g.shape[1] else np.zeros(len(g0))
    return g0 + trait_part + tau * z


# --------------------------------------------------------------------------
# scenario -> cells and the weekly design
# --------------------------------------------------------------------------
def _resolve_candidate(post: dict, scenario: dict) -> dict | None:
    name = scenario.get("candidate")
    if not name:
        return None
    for cand in post.get("candidates", []):
        if cand["name"] == name:
            return cand
    raise ValueError(f"unknown candidate city: {name!r}")


def _selected(names: list[str], chosen: list[str] | None) -> np.ndarray:
    """Boolean mask over names. Empty or missing selection means all of them."""
    if not chosen:
        return np.ones(len(names), dtype=bool)
    unknown = [c for c in chosen if c not in names]
    if unknown:
        raise ValueError(f"unknown name(s) in scenario: {unknown}")
    return np.array([n in set(chosen) for n in names], dtype=bool)


def simulate(post: dict, scenario: dict) -> dict:
    """Run one scenario. Returns a dict matching contract/result.schema.json."""
    meta, draws = post["meta"], post["draws"]
    cities: list[str] = list(meta["cities"])
    segments: list[str] = list(meta["segments"])
    weeks0 = int(meta["weeks"])
    base_price = float(meta["base_price"])
    n_draws = int(meta["n_draws"])

    stype = scenario["type"]
    horizon = int(scenario["horizon"])
    unit_cost = float(scenario.get("unit_cost", meta.get("unit_cost", 0.0)))
    fixed_cost = float(scenario.get("fixed_cost", 0.0))
    seed = int(scenario.get("seed", 7))
    t = weeks0 + np.arange(horizon)

    seg_mask = _selected(segments, scenario.get("segments"))
    warnings: list[str] = []
    flags: dict[str, Any] = {"extrapolation": False, "out_of_range_traits": [], "warnings": []}

    if stype == "new_city":
        cand = _resolve_candidate(post, scenario)
        if cand is not None:
            traits_z = np.asarray(cand["traits_z"], dtype=float)
            shares = np.asarray(cand["shares"], dtype=float)
            city_label = cand["name"]
            flags["extrapolation"] = bool(cand["extrapolation"])
            flags["out_of_range_traits"] = list(cand["out_of_range_traits"])
        else:
            raw = np.asarray(scenario["traits"], dtype=float)
            t_mean = np.asarray(meta["trait_mean"], dtype=float)
            t_std = np.asarray(meta["trait_std"], dtype=float)
            traits_z = (raw - t_mean) / t_std
            lo = np.asarray(meta["trait_min"], dtype=float)
            hi = np.asarray(meta["trait_max"], dtype=float)
            out = [
                meta["trait_names"][i]
                for i in range(len(traits_z))
                if traits_z[i] < lo[i] - 1e-12 or traits_z[i] > hi[i] + 1e-12
            ]
            flags["extrapolation"] = bool(out)
            flags["out_of_range_traits"] = out
            shares = np.asarray(
                scenario.get("new_city_shares") or np.mean(meta["shares"], axis=0), dtype=float
            )
            city_label = scenario.get("label") or "New city"

        if scenario.get("new_city_shares"):
            shares = np.asarray(scenario["new_city_shares"], dtype=float)
        if not meta.get("has_traits", True):
            raise ValueError(
                "This dataset has no city traits, so a new city cannot be forecast. "
                "Upload a cities file with log_population, median_income_k and share_age_18_34."
            )
        if flags["extrapolation"]:
            warnings.append(
                f"{city_label} is outside the training range on "
                f"{', '.join(flags['out_of_range_traits'])}. The forecast extrapolates; "
                "treat the interval as a lower bound on the real uncertainty."
            )

        # the new city's cells are the selected segments of one city
        sel = np.where(seg_mask)[0]
        seg_idx = sel
        market = float(scenario.get("market_size", 1.0))
        log_share = np.log(shares[sel] * market)
        a_new = new_city_a(post, traits_z, seed)                   # [D]
        a_city_draws = np.repeat(a_new[:, None], len(sel), axis=1)  # [D][n_cells]
        cell_city = [city_label] * len(sel)

        price_s = float(scenario.get("new_city_price", base_price))
        ones = np.ones((horizon, len(sel)))
        price_sc = ones * price_s
        promo_sc = np.zeros_like(ones)
        promo_last_sc = np.zeros_like(ones)
        feature_sc = ones if scenario.get("feature") else np.zeros_like(ones)

        mu_sc = mu_for_cells(
            post, a_city_draws=a_city_draws, seg_idx=seg_idx, log_share=log_share, t=t,
            price=price_sc, promo=promo_sc, promo_last=promo_last_sc, feature=feature_sc,
        )
        # no launch means no city and no sales: the baseline is zero, not "the old city"
        mu_bl = np.zeros_like(mu_sc)
        price_bl = price_sc
        promo_active = np.zeros(horizon, dtype=bool)
        dip_active = np.zeros(horizon, dtype=bool)

    else:
        city_mask = _selected(cities, scenario.get("cities"))
        # every training cell is simulated; the change applies only to selected cells,
        # so unaffected cells cancel out of the incremental exactly
        grid_c, grid_s = np.meshgrid(
            np.arange(len(cities)), np.arange(len(segments)), indexing="ij"
        )
        city_idx = grid_c.ravel()
        seg_idx = grid_s.ravel()
        shares_m = np.asarray(meta["shares"], dtype=float)
        log_share = np.log(shares_m[city_idx, seg_idx])
        a_city_draws = np.asarray(draws["a_city"], dtype=float)[:, city_idx]
        cell_city = [cities[i] for i in city_idx]
        affected = city_mask[city_idx] & seg_mask[seg_idx]          # [n_cells]

        ones = np.ones((horizon, len(city_idx)))
        price_bl = ones * base_price
        promo_bl = np.zeros_like(ones)
        promo_last_bl = np.zeros_like(ones)
        feature_bl = np.zeros_like(ones)

        price_sc = price_bl.copy()
        promo_sc = promo_bl.copy()
        promo_last_sc = promo_last_bl.copy()
        feature_sc = feature_bl.copy()

        if stype == "promo":
            if not meta.get("has_promo", True):
                raise ValueError(
                    "This dataset has no promo column, so promotion scenarios are off."
                )
            discount = float(scenario["discount"])
            n_promo = min(int(scenario["weeks"]), horizon)
            promo_active = np.arange(horizon) < n_promo
            dip_active = np.arange(horizon) == n_promo  # the week right after the promo
            for w in range(horizon):
                if promo_active[w]:
                    price_sc[w, affected] = base_price * (1.0 - discount)
                    promo_sc[w, affected] = 1.0
                if dip_active[w]:
                    promo_last_sc[w, affected] = 1.0
        elif stype == "price":
            change = float(scenario["price_change"])
            price_sc[:, affected] = base_price * (1.0 + change)
            promo_active = np.zeros(horizon, dtype=bool)
            dip_active = np.zeros(horizon, dtype=bool)
        else:
            raise ValueError(f"unknown scenario type: {stype!r}")

        if scenario.get("feature"):
            if not meta.get("has_feature", True):
                raise ValueError("This dataset has no feature column.")
            feature_sc[:, affected] = 1.0

        common = dict(a_city_draws=a_city_draws, seg_idx=seg_idx, log_share=log_share, t=t)
        mu_bl = mu_for_cells(
            post, **common, price=price_bl, promo=promo_bl,
            promo_last=promo_last_bl, feature=feature_bl,
        )
        mu_sc = mu_for_cells(
            post, **common, price=price_sc, promo=promo_sc,
            promo_last=promo_last_sc, feature=feature_sc,
        )

    # ---- money, per draw ------------------------------------------------
    margin_bl = price_bl - unit_cost            # [W][n_cells]
    margin_sc = price_sc - unit_cost
    rev_bl = mu_bl * price_bl[None, :, :]
    rev_sc = mu_sc * price_sc[None, :, :]
    prof_bl = mu_bl * margin_bl[None, :, :]
    prof_sc = mu_sc * margin_sc[None, :, :]

    tot = lambda a: a.sum(axis=(1, 2))  # noqa: E731 - [D][W][cells] -> [D]
    units_bl, units_sc = tot(mu_bl), tot(mu_sc)
    revenue_bl, revenue_sc = tot(rev_bl), tot(rev_sc)
    # the fixed cost is charged once over the whole horizon, on the scenario side only
    profit_bl, profit_sc = tot(prof_bl), tot(prof_sc) - fixed_cost

    inc_units = units_sc - units_bl
    inc_revenue = revenue_sc - revenue_bl
    inc_profit = profit_sc - profit_bl

    # ---- weekly fan chart -----------------------------------------------
    weekly = []
    for w in range(horizon):
        weekly.append(
            {
                "week": w,
                "t": int(t[w]),
                "baseline_units": band(mu_bl[:, w, :].sum(axis=1)),
                "scenario_units": band(mu_sc[:, w, :].sum(axis=1)),
                "incremental_units": band((mu_sc[:, w, :] - mu_bl[:, w, :]).sum(axis=1)),
                "promo_active": bool(promo_active[w]),
                "dip_active": bool(dip_active[w]),
            }
        )

    # ---- per segment and per city ---------------------------------------
    seg_out = []
    if meta.get("has_segments", True):
        for si, name in enumerate(segments):
            cols = np.where(seg_idx == si)[0]
            if not len(cols):
                continue
            du = (mu_sc[:, :, cols] - mu_bl[:, :, cols]).sum(axis=(1, 2))
            dr = (rev_sc[:, :, cols] - rev_bl[:, :, cols]).sum(axis=(1, 2))
            dp = (prof_sc[:, :, cols] - prof_bl[:, :, cols]).sum(axis=(1, 2))
            total = float(np.mean(inc_units))
            seg_out.append(
                {
                    "segment": name,
                    "incremental": money(du, dr, dp),
                    "share_of_total": float(np.mean(du) / total) if total else 0.0,
                }
            )

    city_out = []
    for name in dict.fromkeys(cell_city):
        cols = np.array([i for i, c in enumerate(cell_city) if c == name])
        du = (mu_sc[:, :, cols] - mu_bl[:, :, cols]).sum(axis=(1, 2))
        dr = (rev_sc[:, :, cols] - rev_bl[:, :, cols]).sum(axis=(1, 2))
        dp = (prof_sc[:, :, cols] - prof_bl[:, :, cols]).sum(axis=(1, 2))
        city_out.append({"city": name, "incremental": money(du, dr, dp)})

    # ---- flags the UI has to show ---------------------------------------
    if not meta.get("has_promo", True):
        flags["promo_disabled"] = True
    if meta.get("no_price_variation"):
        flags["no_price_variation"] = True
        warnings.append(
            "Price never varied in the fitted data, so the elasticity is driven by the "
            "prior, not by evidence. Price scenarios are illustrative."
        )
    flags["warnings"] = warnings

    net_price = float(np.max(price_sc)) if stype == "price" else float(np.min(price_sc))

    return {
        "scenario": dict(scenario),
        "horizon": horizon,
        "baseline": money(units_bl, revenue_bl, profit_bl),
        "with_change": money(units_sc, revenue_sc, profit_sc),
        "incremental": money(inc_units, inc_revenue, inc_profit),
        "p_profit_positive": float(np.mean(inc_profit > 0.0)),
        "weekly": weekly,
        "segments": seg_out,
        "cities": city_out,
        "flags": flags,
        "meta": {
            "n_draws": n_draws,
            "seed": seed,
            "source": "python",
            "unit_cost": unit_cost,
            "fixed_cost": fixed_cost,
            "net_price": net_price,
        },
    }


def break_even(post: dict, scenario: dict, *, grid: int = 19) -> dict:
    """The largest discount / price change that still has at least a 50% chance of paying.

    A scan, not calculus: the simulator is cheap and a scan cannot be wrong about
    the model it is scanning.
    """
    stype = scenario["type"]
    if stype == "promo":
        key, values = "discount", np.linspace(0.05, 0.9, grid)
    elif stype == "price":
        key, values = "price_change", np.linspace(-0.5, 1.0, grid)
    else:
        return {"metric": "fixed_cost", "value": None}

    best = None
    for v in values:
        trial = dict(scenario)
        trial[key] = float(v)
        if simulate(post, trial)["p_profit_positive"] >= 0.5:
            best = float(v)
    return {"metric": key, "value": best}
