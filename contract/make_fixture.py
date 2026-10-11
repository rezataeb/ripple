"""Build contract/posterior.fixture.json — the shared tiny posterior.

50 draws, 2 cities, 2 segments, 12 weeks. Session B builds web/lib/simulate.ts
against this file and the Python reference simulator is tested on the same file,
so the parity test compares both implementations on identical input.

Numbers here are plausible but made up: this is a contract fixture, not a fit.
Run: uv run python contract/make_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261011
OUT = Path(__file__).parent / "posterior.fixture.json"

D = 50  # draws
CITIES = ["New York", "Boston"]
SEGMENTS = ["students", "retirees"]
TRAIT_NAMES = ["log_population", "median_income_k", "share_age_18_34"]
WEEKS = 12
N_FOURIER = 2
SEASON_PERIOD = 52.0
BASE_PRICE = 24.0
UNIT_COST = 9.0

C, S, P, K = len(CITIES), len(SEGMENTS), len(TRAIT_NAMES), 2 * N_FOURIER


def r(x) -> list:
    """numpy -> plain python lists, rounded so the JSON stays small and exact."""
    return np.round(np.asarray(x, dtype=float), 10).tolist()


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- city traits (raw), standardized with the training mean/sd -------------
    traits_raw = np.array(
        [
            [15.90, 78.0, 0.30],  # New York
            [14.10, 86.0, 0.35],  # Boston
        ]
    )
    trait_mean = traits_raw.mean(axis=0)
    trait_std = traits_raw.std(axis=0, ddof=0)
    traits_z = (traits_raw - trait_mean) / trait_std
    trait_min = traits_z.min(axis=0)
    trait_max = traits_z.max(axis=0)

    shares = np.array([[0.45, 0.55], [0.60, 0.40]])  # rows sum to 1

    # --- posterior draws ------------------------------------------------------
    # g0 is the global intercept. It must be folded INTO a_city for the training
    # cities, because contract/README.md says the simulator adds g0 only for a new
    # city — an a_city without it would make every mu ~400x too small.
    g0 = rng.normal(6.40, 0.06, D)
    g = np.column_stack(
        [
            rng.normal(0.22, 0.05, D),  # bigger city -> more demand
            rng.normal(0.10, 0.05, D),  # richer city -> slightly more
            rng.normal(0.14, 0.05, D),  # younger city -> more
        ]
    )
    tau_city = np.abs(rng.normal(0.18, 0.04, D))
    # a_city[d][c] = g0[d] + traits_z[c].g[d] + tau_city[d]*z_city[c] — the same
    # identity the fitted model exports, so the fixture is a valid posterior.
    z_city = np.array([0.55, -0.40])  # fixed per city, so the fixture is reproducible
    a_city = g0[:, None] + (g @ traits_z.T) + tau_city[:, None] * z_city[None, :]
    a_seg = np.column_stack([rng.normal(0.05, 0.04, D), rng.normal(-0.05, 0.04, D)])
    e = np.column_stack(
        [
            rng.normal(-2.20, 0.18, D),  # students: price sensitive
            rng.normal(-0.80, 0.15, D),  # retirees: not
        ]
    )
    b_promo = np.column_stack([rng.normal(0.30, 0.06, D), rng.normal(0.12, 0.05, D)])
    b_feature = rng.normal(0.07, 0.03, D)
    b_dip = rng.normal(-0.12, 0.04, D)
    season = np.column_stack(
        [
            rng.normal(0.10, 0.03, D),
            rng.normal(-0.06, 0.03, D),
            rng.normal(0.03, 0.02, D),
            rng.normal(0.02, 0.02, D),
        ]
    )
    phi = np.abs(rng.normal(18.0, 2.0, D))

    # --- history: every city x segment x week, with one 3-week promo ----------
    hist: dict[str, list] = {
        k: []
        for k in (
            "week_start",
            "t",
            "city_idx",
            "segment_idx",
            "units",
            "price",
            "promo",
            "promo_last_week",
            "feature",
        )
    }
    promo_weeks = {4, 5, 6}
    week0 = np.datetime64("2026-01-05")  # a Monday
    for t in range(WEEKS):
        promo = 1.0 if t in promo_weeks else 0.0
        promo_prev = 1.0 if (t - 1) in promo_weeks else 0.0
        feature = 1.0 if t % 4 == 0 else 0.0
        price = BASE_PRICE * (0.8 if promo else 1.0)
        for ci in range(C):
            for si in range(S):
                sr = season.mean(axis=0)
                basis = season_row(t)
                # exactly the mean function in contract/README.md — a_city already
                # carries g0, so g0 must NOT be added again here
                log_mu = (
                    np.log(shares[ci, si])
                    + a_city[:, ci].mean()
                    + a_seg[:, si].mean()
                    + e[:, si].mean() * np.log(price / BASE_PRICE)
                    + b_promo[:, si].mean() * promo
                    + b_feature.mean() * feature
                    + b_dip.mean() * promo_prev
                    + float(basis @ sr)
                )
                units = float(np.round(np.exp(log_mu) * rng.normal(1.0, 0.05)))
                hist["week_start"].append(str(week0 + np.timedelta64(7 * t, "D")))
                hist["t"].append(t)
                hist["city_idx"].append(ci)
                hist["segment_idx"].append(si)
                hist["units"].append(max(units, 0.0))
                hist["price"].append(price)
                hist["promo"].append(promo)
                hist["promo_last_week"].append(promo_prev)
                hist["feature"].append(feature)

    # --- candidate cities: one in range, one out of range --------------------
    cand_raw = {
        "Austin": [14.60, 80.0, 0.33],
        "Miami": [14.80, 62.0, 0.28],  # income below the training range
    }
    cand_shares = {"Austin": [0.55, 0.45], "Miami": [0.40, 0.60]}
    candidates = []
    for name, raw in cand_raw.items():
        z = (np.array(raw) - trait_mean) / trait_std
        out = [
            TRAIT_NAMES[i]
            for i in range(P)
            if z[i] < trait_min[i] - 1e-12 or z[i] > trait_max[i] + 1e-12
        ]
        candidates.append(
            {
                "name": name,
                "traits": r(raw),
                "traits_z": r(z),
                "shares": r(cand_shares[name]),
                "extrapolation": bool(out),
                "out_of_range_traits": out,
            }
        )

    doc = {
        "meta": {
            "dataset_name": "Fixture · 2 cities, 2 segments",
            "cities": CITIES,
            "segments": SEGMENTS,
            "weeks": WEEKS,
            "n_draws": D,
            "base_price": BASE_PRICE,
            "unit_cost": UNIT_COST,
            "shares": r(shares),
            "trait_names": TRAIT_NAMES,
            "traits_z": r(traits_z),
            "trait_mean": r(trait_mean),
            "trait_std": r(trait_std),
            "trait_min": r(trait_min),
            "trait_max": r(trait_max),
            "has_promo": True,
            "has_feature": True,
            "has_traits": True,
            "has_segments": True,
            "season_period": SEASON_PERIOD,
            "n_fourier": N_FOURIER,
            "method": "fixture (not a fit) — fixed seed",
            "fast_approximation": False,
            "seed": SEED,
            "fitted_at": "2026-10-11T00:00:00Z",
        },
        "draws": {
            "g0": r(g0),
            "g": r(g),
            "tau_city": r(tau_city),
            "a_city": r(a_city),
            "a_seg": r(a_seg),
            "e": r(e),
            "b_promo": r(b_promo),
            "b_feature": r(b_feature),
            "b_dip": r(b_dip),
            "season": r(season),
            "phi": r(phi),
        },
        "history": hist,
        "candidates": candidates,
    }

    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


def season_row(t: float) -> np.ndarray:
    """Fourier design row for week t — must match web/lib/simulate.ts exactly."""
    out = []
    for k in range(1, N_FOURIER + 1):
        ang = 2.0 * np.pi * k * t / SEASON_PERIOD
        out += [np.sin(ang), np.cos(ang)]
    return np.array(out)


if __name__ == "__main__":
    main()
