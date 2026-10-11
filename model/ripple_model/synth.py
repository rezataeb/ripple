"""Brightside Coffee Subscriptions — the synthetic demo company.

Fictional, fixed seed, and the truth is planted so we can PROVE recovery:
model/data/truth.json holds the values the fit has to find.

Outputs (all under model/data/):
  brightside_weekly.csv  — the demo dataset, in the exact format a customer uploads
  brightside_cities.csv  — city traits, the optional second upload
  truth.json             — planted parameter values + metadata
  launches.json          — 3 post-launch datasets with known truth (spec 5b)

Run: uv run python -m ripple_model.synth
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261011
DATA = Path(__file__).resolve().parents[1] / "data"

WEEKS = 104
WEEK0 = np.datetime64("2024-01-01")  # a Monday
BASE_PRICE = 24.0
UNIT_COST = 9.0
SEASON_PERIOD = 52.0
N_FOURIER = 2

SEGMENTS = ["students", "young professionals", "families", "retirees"]
TRAIT_NAMES = ["log_population", "median_income_k", "share_age_18_34"]

# Census ACS-shaped values. Illustrative: close to the real figures but not
# pulled from the API during a 3-hour build, so the README labels them as such.
CITIES: dict[str, list[float]] = {
    #              log_population, median_income_k, share_age_18_34
    "New York":   [15.90, 76.6, 0.311],
    "Chicago":    [14.78, 71.7, 0.300],
    "Boston":     [13.42, 89.2, 0.355],
    "Seattle":    [13.52, 110.8, 0.325],
    "Atlanta":    [13.08, 77.7, 0.318],
    "Denver":     [13.44, 88.2, 0.306],
}

# Segment mix per city — sums to 1 per row. Boston/Seattle skew young.
CITY_SHARES: dict[str, list[float]] = {
    #              students, young prof, families, retirees
    "New York":   [0.18, 0.37, 0.28, 0.17],
    "Chicago":    [0.16, 0.32, 0.33, 0.19],
    "Boston":     [0.30, 0.34, 0.22, 0.14],
    "Seattle":    [0.20, 0.40, 0.26, 0.14],
    "Atlanta":    [0.19, 0.33, 0.31, 0.17],
    "Denver":     [0.17, 0.35, 0.30, 0.18],
}

CANDIDATES: dict[str, list[float]] = {
    "Austin":   [13.56, 86.6, 0.330],   # in range
    "Portland": [13.26, 85.4, 0.308],   # in range
    "Miami":    [13.09, 54.9, 0.286],   # income far below the training range
}
# Shown next to a model-based lift; update.py puts it in causal.recommendation.
HOLDOUT_RECOMMENDATION = (
    "Keep a random holdout group next time (for example 10% of users get no offer). "
    "Then this is an experiment, not a model-based estimate."
)

CANDIDATE_SHARES: dict[str, list[float]] = {
    "Austin":   [0.24, 0.38, 0.26, 0.12],
    "Portland": [0.19, 0.36, 0.29, 0.16],
    "Miami":    [0.15, 0.30, 0.30, 0.25],
}


# --------------------------------------------------------------------------
# planted truth
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Truth:
    """The values the fit must recover. Everything here is on the log-mu scale."""

    g0: float
    g: np.ndarray            # [P] trait coefficients
    tau_city: float
    z_city: np.ndarray       # [C] standardized city deviations
    a_seg: np.ndarray        # [S]
    e: np.ndarray            # [S] elasticities, negative
    b_promo: np.ndarray      # [S] deal-signal lift
    b_feature: float
    b_dip: float             # negative: pull-forward
    season: np.ndarray       # [2*N_FOURIER] Fourier coefficients
    phi: float               # NegBin dispersion

    def a_city(self) -> np.ndarray:
        """City effects implied by the traits plus the planted deviations."""
        return self.g0 + traits_z() @ self.g + self.tau_city * self.z_city


def season_row(t: np.ndarray | float) -> np.ndarray:
    """Fourier design row(s) for week index t. Must match contract/README.md."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    cols = []
    for k in range(1, N_FOURIER + 1):
        ang = 2.0 * np.pi * k * t / SEASON_PERIOD
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


def traits_raw() -> np.ndarray:
    return np.array([CITIES[c] for c in CITIES])


def trait_mean_std() -> tuple[np.ndarray, np.ndarray]:
    X = traits_raw()
    return X.mean(axis=0), X.std(axis=0, ddof=0)


def traits_z() -> np.ndarray:
    mu, sd = trait_mean_std()
    return (traits_raw() - mu) / sd


def standardize(raw: list[float] | np.ndarray) -> np.ndarray:
    mu, sd = trait_mean_std()
    return (np.asarray(raw, dtype=float) - mu) / sd


def make_truth(rng: np.random.Generator) -> Truth:
    return Truth(
        # exp(6.6) ~ 735 units/week for an average city before the share offset
        g0=6.60,
        g=np.array([0.55, 0.18, 0.30]),
        tau_city=0.16,
        z_city=rng.normal(0.0, 1.0, len(CITIES)),
        # students buy fewer per head than families; the share offset carries volume
        a_seg=np.array([-0.15, 0.20, 0.10, -0.25]),
        e=np.array([-2.20, -1.60, -1.10, -0.80]),
        b_promo=np.array([0.42, 0.30, 0.22, 0.10]),
        b_feature=0.08,
        b_dip=-0.18,
        season=np.array([0.12, -0.07, 0.04, 0.02]),
        phi=22.0,
    )


# --------------------------------------------------------------------------
# promotion calendar
# --------------------------------------------------------------------------
def make_promo_calendar(rng: np.random.Generator) -> pd.DataFrame:
    """One row per city x week: discount, which segment it targets, feature flag.

    Promotions are random in timing, depth and targeting, and price also drifts a
    little between cities — without price variation the elasticity cannot be learned.
    """
    rows = []
    for ci, city in enumerate(CITIES):
        # a small permanent price difference per city, so price varies across cells
        city_price = BASE_PRICE * (1.0 + rng.normal(0.0, 0.03))
        t = 0
        while t < WEEKS:
            if rng.random() < 0.22:  # start a promotion
                discount = float(rng.choice([0.10, 0.20, 0.30]))
                length = int(rng.integers(1, 5))  # 1..4 weeks
                # a third of promotions target one segment only
                target = str(rng.choice(SEGMENTS)) if rng.random() < 0.34 else "all"
                for k in range(length):
                    if t + k < WEEKS:
                        rows.append((ci, city, t + k, discount, target, city_price))
                t += length + int(rng.integers(1, 4))  # a gap after every promo
            else:
                rows.append((ci, city, t, 0.0, "all", city_price))
                t += 1
    df = pd.DataFrame(rows, columns=["city_idx", "city", "t", "discount", "target", "city_price"])
    # fill any week a promo block skipped over
    full = pd.MultiIndex.from_product([range(len(CITIES)), range(WEEKS)], names=["city_idx", "t"])
    df = df.drop_duplicates(["city_idx", "t"]).set_index(["city_idx", "t"]).reindex(full)
    df["city"] = [list(CITIES)[i] for i, _ in full]
    df["discount"] = df["discount"].fillna(0.0)
    df["target"] = df["target"].fillna("all")
    df["city_price"] = df.groupby("city_idx")["city_price"].transform(
        lambda s: s.fillna(s.dropna().iloc[0] if s.notna().any() else BASE_PRICE)
    )
    df = df.reset_index()

    # email feature campaign: on for bursts, independent of promotions
    feat = np.zeros((len(CITIES), WEEKS))
    for ci in range(len(CITIES)):
        t = 0
        while t < WEEKS:
            if rng.random() < 0.18:
                length = int(rng.integers(1, 4))
                feat[ci, t : t + length] = 1.0
                t += length + 2
            else:
                t += 1
    df["feature"] = [feat[int(r.city_idx), int(r.t)] for r in df.itertuples()]
    return df


# --------------------------------------------------------------------------
# the generative model
# --------------------------------------------------------------------------
def log_mu(
    truth: Truth,
    a_city_value: float,
    seg_idx: int,
    share: float,
    t: int,
    price: float,
    promo: float,
    promo_last: float,
    feature: float,
) -> float:
    """The mean function from spec Section 3, for one cell. One source of truth."""
    return (
        np.log(share)
        + a_city_value
        + truth.a_seg[seg_idx]
        + truth.e[seg_idx] * np.log(price / BASE_PRICE)
        + truth.b_promo[seg_idx] * promo
        + truth.b_feature * feature
        + truth.b_dip * promo_last
        + float(season_row(t)[0] @ truth.season)
    )


def negbin(rng: np.random.Generator, mu: np.ndarray | float, phi: float) -> np.ndarray:
    """NegBin with mean mu and var mu + mu^2/phi, via the gamma-Poisson mixture."""
    mu = np.asarray(mu, dtype=float)
    lam = rng.gamma(shape=phi, scale=mu / phi)
    return rng.poisson(lam)


def generate_panel(rng: np.random.Generator, truth: Truth) -> pd.DataFrame:
    cal = make_promo_calendar(rng)
    a_city = truth.a_city()
    rows = []
    for r in cal.itertuples():
        ci, t = int(r.city_idx), int(r.t)
        prev = cal[(cal.city_idx == ci) & (cal.t == t - 1)]
        promo_last_any = float(prev.discount.iloc[0] > 0) if len(prev) else 0.0
        prev_target = str(prev.target.iloc[0]) if len(prev) else "all"
        for si, seg in enumerate(SEGMENTS):
            targeted = r.target in ("all", seg)
            promo = 1.0 if (r.discount > 0 and targeted) else 0.0
            price = r.city_price * (1.0 - (r.discount if targeted else 0.0))
            prev_targeted = prev_target in ("all", seg)
            promo_last = promo_last_any if prev_targeted else 0.0
            share = CITY_SHARES[r.city][si]
            mu = np.exp(
                log_mu(truth, a_city[ci], si, share, t, price, promo, promo_last, r.feature)
            )
            rows.append(
                {
                    "week": str(WEEK0 + np.timedelta64(7 * t, "D")),
                    "city": r.city,
                    "segment": seg,
                    "units": int(negbin(rng, mu, truth.phi)),
                    "price": round(float(price), 2),
                    "promo": int(promo),
                    "feature": int(r.feature),
                    # not part of the customer CSV format; dropped before writing
                    "_t": t,
                    "_mu": float(mu),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the 3 demo launches (spec 5b)
# --------------------------------------------------------------------------
def generate_launches(rng: np.random.Generator, truth: Truth) -> dict:
    """Austin launch (new city), student coupon (new offer), gift boxes (new feature).

    8 observed weeks each, 12-week horizon, truth planted and known.
    """
    a_city = truth.a_city()
    t0 = WEEKS  # launches start the week after the training data ends
    out = []

    CAUSAL = [
        {
            "id": "no_cointervention",
            "text": "No other change happened in the same weeks.",
            "holds": None,
        },
        {
            "id": "no_spillover",
            "text": "Untreated cities and groups were not affected.",
            "holds": None,
        },
        {
            "id": "stable",
            "text": "The demand pattern is stable — nothing shifted the baseline.",
            "holds": None,
        },
    ]

    # ---- 1. Austin launch: the unknown is the new city's deviation z --------
    name = "Austin"
    z_true = 0.85  # Austin does better than its traits predict
    a_new = truth.g0 + standardize(CANDIDATES[name]) @ truth.g + truth.tau_city * z_true
    obs = _empty_obs()
    for w in range(1, 9):
        t = t0 + w - 1
        for si in range(len(SEGMENTS)):
            share = CANDIDATE_SHARES[name][si]
            mu = np.exp(log_mu(truth, a_new, si, share, t, BASE_PRICE, 0.0, 0.0, 0.0))
            _push(obs, w, t, -1, si, int(negbin(rng, mu, truth.phi)), BASE_PRICE, 0, 0, 0, 1)
    out.append(
        {
            "id": "austin-launch",
            "label": "Austin launch",
            "type": "new_city",
            "description": (
                "Brightside opened in Austin. How big is this city really, and does the launch pay?"
            ),
            "unknown": {
                "name": "z_new_city",
                "label": "Austin city effect",
                "kind": "z_new_city",
                "prior": {"dist": "normal", "mu": 0.0, "sigma": 1.0, "editable_sigma": False},
                "display": "percent_lift",
            },
            "weeks_observed": 8,
            "horizon": 12,
            "t0": t0,
            "cities": [name],
            "segments": SEGMENTS,
            "new_city": {
                "name": name,
                "traits": CANDIDATES[name],
                "traits_z": standardize(CANDIDATES[name]).round(10).tolist(),
                "shares": CANDIDATE_SHARES[name],
                "extrapolation": False,
                "out_of_range_traits": [],
            },
            "observations": obs,
            "has_holdout": False,
            "unit_cost": UNIT_COST,
            "fixed_cost": 60000.0,
            "net_price": BASE_PRICE,
            "truth": _truth_block(
                truth, a_new, CANDIDATE_SHARES[name], t0, 12, BASE_PRICE, 60000.0, z_true
            ),
            "causal_assumptions": CAUSAL,
        }
    )

    # ---- 2. Student coupon in Boston: the unknown is b_promo[students] ------
    city, ci, si = "Boston", list(CITIES).index("Boston"), SEGMENTS.index("students")
    promo_price = BASE_PRICE * 0.8  # 20% off
    b_true = 0.62  # the real coupon beats what history taught us about promos
    obs = _empty_obs()
    for w in range(1, 9):
        t = t0 + w - 1
        for s2 in range(len(SEGMENTS)):
            treated = 1 if s2 == si else 0
            price = promo_price if treated else BASE_PRICE
            b = b_true if treated else 0.0
            share = CITY_SHARES[city][s2]
            mu = np.exp(
                np.log(share)
                + a_city[ci]
                + truth.a_seg[s2]
                + truth.e[s2] * np.log(price / BASE_PRICE)
                + b * treated
                + float(season_row(t)[0] @ truth.season)
            )
            _push(obs, w, t, ci, s2, int(negbin(rng, mu, truth.phi)), price, treated, 0, 0, treated)
    out.append(
        {
            "id": "student-coupon",
            "label": "Student coupon · Boston",
            "type": "offer",
            "description": (
                "20% off for students in Boston for 8 weeks. "
                "Is the deal signal stronger than history suggested?"
            ),
            "unknown": {
                "name": "b_promo[students]",
                "label": "Student deal-signal lift",
                "kind": "b_promo",
                "segment": "students",
                "prior": {"dist": "posterior", "editable_sigma": False},
                "display": "percent_lift",
            },
            "weeks_observed": 8,
            "horizon": 12,
            "t0": t0,
            "cities": [city],
            "segments": ["students"],
            "observations": obs,
            "has_holdout": False,
            "unit_cost": UNIT_COST,
            "fixed_cost": 0.0,
            "net_price": promo_price,
            "truth": {"unknown_value": b_true, "unknown_percent": float(np.exp(b_true) - 1.0)},
            "causal_assumptions": CAUSAL,
        }
    )

    # ---- 3. Gift boxes: a never-seen feature, with a random holdout --------
    theta_true = 0.11  # +11.6%
    obs = _empty_obs()
    for w in range(1, 9):
        t = t0 + w - 1
        for city2 in ("New York", "Seattle"):
            cidx = list(CITIES).index(city2)
            for s2 in range(len(SEGMENTS)):
                for arm in (1, 0):  # treated arm and a 10% random holdout
                    share = CITY_SHARES[city2][s2] * (0.9 if arm else 0.1)
                    mu = np.exp(
                        np.log(share)
                        + a_city[cidx]
                        + truth.a_seg[s2]
                        + theta_true * arm
                        + float(season_row(t)[0] @ truth.season)
                    )
                    _push(
                        obs, w, t, cidx, s2, int(negbin(rng, mu, truth.phi)),
                        BASE_PRICE, 0, 0, arm, arm, holdout=1 - arm,
                    )
    out.append(
        {
            "id": "gift-boxes",
            "label": "New feature · gift boxes",
            "type": "feature",
            "description": (
                "A feature the model has never seen, launched in New York and Seattle "
                "with a random 10% holdout."
            ),
            "unknown": {
                "name": "theta",
                "label": "Lift from gift boxes",
                "kind": "theta",
                "prior": {"dist": "normal", "mu": 0.0, "sigma": 0.15, "editable_sigma": True},
                "display": "percent_lift",
            },
            "weeks_observed": 8,
            "horizon": 12,
            "t0": t0,
            "cities": ["New York", "Seattle"],
            "segments": SEGMENTS,
            "observations": obs,
            "has_holdout": True,
            "unit_cost": UNIT_COST,
            "fixed_cost": 25000.0,
            "net_price": BASE_PRICE,
            "truth": {
                "unknown_value": theta_true,
                "unknown_percent": float(np.exp(theta_true) - 1.0),
            },
            "causal_assumptions": [
                {**a, "holds": True} if a["id"] == "no_spillover" else a for a in CAUSAL
            ],
        }
    )

    return {"generated_at": "2026-10-11T00:00:00Z", "seed": SEED, "launches": out}


def _empty_obs() -> dict[str, list]:
    return {
        k: []
        for k in (
            "week", "week_start", "t", "city_idx", "segment_idx", "units",
            "price", "promo", "promo_last_week", "feature", "treated", "holdout",
        )
    }


def _push(o, week, t, ci, si, units, price, promo, promo_last, feature, treated, holdout=0):
    o["week"].append(week)
    o["week_start"].append(str(WEEK0 + np.timedelta64(7 * t, "D")))
    o["t"].append(t)
    o["city_idx"].append(ci)
    o["segment_idx"].append(si)
    o["units"].append(units)
    o["price"].append(round(float(price), 2))
    o["promo"].append(promo)
    o["promo_last_week"].append(promo_last)
    o["feature"].append(feature)
    o["treated"].append(treated)
    o["holdout"].append(holdout)


def _truth_block(truth, a_value, shares, t0, horizon, price, fixed_cost, unknown_value):
    """True expected totals over the horizon — what the launch tracker should converge to."""
    units = 0.0
    for w in range(horizon):
        t = t0 + w
        for si in range(len(SEGMENTS)):
            units += float(np.exp(log_mu(truth, a_value, si, shares[si], t, price, 0.0, 0.0, 0.0)))
    profit = units * (price - UNIT_COST) - fixed_cost
    return {
        "unknown_value": float(unknown_value),
        "unknown_percent": float(np.exp(truth.tau_city * unknown_value) - 1.0),
        "total_units": round(units, 4),
        "total_profit": round(profit, 4),
        "pays": bool(profit > 0),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    truth = make_truth(rng)

    panel = generate_panel(rng, truth)
    # the customer CSV format from spec Section 6 — nothing extra
    csv = panel[["week", "city", "segment", "units", "price", "promo", "feature"]]
    csv.to_csv(DATA / "brightside_weekly.csv", index=False)

    cities = pd.DataFrame(
        [{"city": c, **dict(zip(TRAIT_NAMES, v, strict=True))} for c, v in CITIES.items()]
    )
    cities.to_csv(DATA / "brightside_cities.csv", index=False)

    mu_, sd_ = trait_mean_std()
    tz = traits_z()
    truth_doc = {
        "dataset_name": "Demo · Brightside Coffee Subscriptions",
        "note": "Fictional company. City traits are ACS-shaped but illustrative, not pulled live.",
        "seed": SEED,
        "weeks": WEEKS,
        "week0": str(WEEK0),
        "base_price": BASE_PRICE,
        "unit_cost": UNIT_COST,
        "season_period": SEASON_PERIOD,
        "n_fourier": N_FOURIER,
        "cities": list(CITIES),
        "segments": SEGMENTS,
        "trait_names": TRAIT_NAMES,
        "trait_mean": mu_.round(10).tolist(),
        "trait_std": sd_.round(10).tolist(),
        "trait_min": tz.min(axis=0).round(10).tolist(),
        "trait_max": tz.max(axis=0).round(10).tolist(),
        "shares": [CITY_SHARES[c] for c in CITIES],
        "candidates": {
            name: {
                "traits": raw,
                "traits_z": standardize(raw).round(10).tolist(),
                "shares": CANDIDATE_SHARES[name],
            }
            for name, raw in CANDIDATES.items()
        },
        "params": {
            "g0": truth.g0,
            "g": truth.g.tolist(),
            "tau_city": truth.tau_city,
            "z_city": truth.z_city.round(10).tolist(),
            "a_city": truth.a_city().round(10).tolist(),
            "a_seg": truth.a_seg.tolist(),
            "e": truth.e.tolist(),
            "b_promo": truth.b_promo.tolist(),
            "b_feature": truth.b_feature,
            "b_dip": truth.b_dip,
            "season": truth.season.tolist(),
            "phi": truth.phi,
        },
    }
    (DATA / "truth.json").write_text(json.dumps(truth_doc, indent=1) + "\n")

    launches = generate_launches(rng, truth)
    (DATA / "launches.json").write_text(json.dumps(launches, indent=1) + "\n")

    print(f"brightside_weekly.csv  {len(csv):>6} rows  "
          f"({len(CITIES)} cities x {len(SEGMENTS)} segments x {WEEKS} weeks)")
    print(f"  units/week per cell   mean {panel.units.mean():.0f}  "
          f"min {panel.units.min()}  max {panel.units.max()}")
    print(f"  promo weeks           {panel.promo.mean():.1%} of cells")
    print(f"  price range           ${panel.price.min():.2f} – ${panel.price.max():.2f}")
    print(f"brightside_cities.csv  {len(cities)} cities, traits {TRAIT_NAMES}")
    print("truth.json             planted parameters written")
    for L in launches["launches"]:
        print(f"launch {L['id']:<16} {len(L['observations']['t']):>4} obs rows, "
              f"unknown {L['unknown']['name']}")


if __name__ == "__main__":
    main()
