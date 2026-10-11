"""Build contract/parity.fixture.json — the Python↔TypeScript agreement test.

Three scenarios (promo, price, new city) are run through
model/ripple_model/simulate.py on contract/posterior.fixture.json. For each we
record:

  * mu for BOTH baseline and scenario, cell by cell, at selected weeks — the
    1e-6 tolerance in spec Section 5 is on mu, and cell-level values are what
    catch an indexing or broadcasting mistake;
  * the complete result object, so web/tests can diff its own output field by
    field instead of only checking a headline.

web/lib/simulate.ts must reproduce both. Run after any change to simulate.py:
    uv run python contract/make_parity_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ripple_model.simulate import mu_for_cells, new_city_a, simulate

HERE = Path(__file__).parent
POSTERIOR = HERE / "posterior.fixture.json"
OUT = HERE / "parity.fixture.json"

MU_TOLERANCE = 1e-6
SUMMARY_TOLERANCE = 1e-6

# Weeks at which mu is recorded. Chosen to hit the promo window (0, 3), the
# post-promo dip week (4), the week after it (5) and a late week (11).
CHECK_WEEKS = [0, 3, 4, 5, 11]

CASES = [
    {
        "name": "promo-students-boston",
        "about": "20% off for students in Boston for 4 weeks. Exercises b_promo, the "
                 "price term, the b_dip week, and the rule that unselected cells "
                 "stay at baseline and cancel out of the incremental.",
        "scenario": {
            "type": "promo",
            "label": "20% student coupon in Boston",
            "cities": ["Boston"],
            "segments": ["students"],
            "discount": 0.2,
            "weeks": 4,
            "unit_cost": 9.0,
            "fixed_cost": 2000.0,
            "horizon": 12,
            "seed": 7,
        },
    },
    {
        "name": "price-plus-8pct-all",
        "about": "A permanent +8% price change everywhere. Exercises the elasticity "
                 "term on every cell and a scenario with no promo weeks.",
        "scenario": {
            "type": "price",
            "label": "Raise price 8% in all cities",
            "price_change": 0.08,
            "unit_cost": 9.0,
            "horizon": 12,
            "seed": 7,
        },
    },
    {
        "name": "new-city-miami",
        "about": "Launch in Miami, which is out of the training trait range. Exercises "
                 "a_new = g0 + traits_z.g + tau_city*z with the seeded z draw, the "
                 "zero baseline, the candidate's segment mix, and the extrapolation flag.",
        "scenario": {
            "type": "new_city",
            "label": "Launch in Miami",
            "candidate": "Miami",
            "unit_cost": 9.0,
            "fixed_cost": 50000.0,
            "horizon": 12,
            "seed": 7,
        },
    },
]


def flat(a: np.ndarray) -> list[float]:
    """Row-major flatten. Flat lists keep the JSON a third of the nested size."""
    return [float(x) for x in np.asarray(a).ravel()]


def recompute_mu(post: dict, scenario: dict) -> dict:
    """Recompute mu for the recorded weeks, mirroring simulate() exactly.

    Deliberately re-derived here rather than returned by simulate(): if the two
    disagree, the fixture generator is the thing that is wrong, and a silent
    agreement between a function and itself would prove nothing.
    """
    meta, draws = post["meta"], post["draws"]
    cities, segments = list(meta["cities"]), list(meta["segments"])
    weeks0 = int(meta["weeks"])
    base_price = float(meta["base_price"])
    horizon = int(scenario["horizon"])
    weeks = [w for w in CHECK_WEEKS if w < horizon]
    t = weeks0 + np.array(weeks)

    if scenario["type"] == "new_city":
        cand = next(c for c in post["candidates"] if c["name"] == scenario["candidate"])
        sel = np.arange(len(segments))
        cells = [{"city": cand["name"], "segment": segments[s]} for s in sel]
        log_share = np.log(np.asarray(cand["shares"])[sel])
        a_new = new_city_a(post, np.asarray(cand["traits_z"]), int(scenario.get("seed", 7)))
        a_city_draws = np.repeat(a_new[:, None], len(sel), axis=1)
        price = np.ones((len(weeks), len(sel))) * base_price
        zeros = np.zeros_like(price)
        mu_sc = mu_for_cells(
            post, a_city_draws=a_city_draws, seg_idx=sel, log_share=log_share, t=t,
            price=price, promo=zeros, promo_last=zeros, feature=zeros,
        )
        mu_bl = np.zeros_like(mu_sc)
    else:
        grid_c, grid_s = np.meshgrid(
            np.arange(len(cities)), np.arange(len(segments)), indexing="ij"
        )
        city_idx, seg_idx = grid_c.ravel(), grid_s.ravel()
        cells = [
            {"city": cities[c], "segment": segments[s]}
            for c, s in zip(city_idx, seg_idx, strict=True)
        ]
        shares = np.asarray(meta["shares"])
        log_share = np.log(shares[city_idx, seg_idx])
        a_city_draws = np.asarray(draws["a_city"])[:, city_idx]

        city_sel = scenario.get("cities") or cities
        seg_sel = scenario.get("segments") or segments
        affected = np.array(
            [cities[c] in set(city_sel) and segments[s] in set(seg_sel)
             for c, s in zip(city_idx, seg_idx, strict=True)]
        )

        base = np.ones((len(weeks), len(city_idx))) * base_price
        zeros = np.zeros_like(base)
        price_sc, promo_sc, promo_last_sc = base.copy(), zeros.copy(), zeros.copy()

        if scenario["type"] == "promo":
            n_promo = min(int(scenario["weeks"]), horizon)
            for i, w in enumerate(weeks):
                if w < n_promo:
                    price_sc[i, affected] = base_price * (1.0 - float(scenario["discount"]))
                    promo_sc[i, affected] = 1.0
                if w == n_promo:
                    promo_last_sc[i, affected] = 1.0
        else:
            price_sc[:, affected] = base_price * (1.0 + float(scenario["price_change"]))

        common = dict(a_city_draws=a_city_draws, seg_idx=seg_idx, log_share=log_share, t=t)
        mu_bl = mu_for_cells(
            post, **common, price=base, promo=zeros, promo_last=zeros, feature=zeros
        )
        mu_sc = mu_for_cells(
            post, **common, price=price_sc, promo=promo_sc,
            promo_last=promo_last_sc, feature=zeros,
        )

    return {
        "weeks": weeks,
        "t": [int(x) for x in t],
        "shape": list(mu_sc.shape),
        "shape_meaning": "[draw][week_in_weeks][cell]",
        "cells": cells,
        "baseline_flat": flat(mu_bl),
        "scenario_flat": flat(mu_sc),
    }


def main() -> None:
    post = json.loads(POSTERIOR.read_text())
    meta = post["meta"]

    cases = []
    for case in CASES:
        scenario = case["scenario"]
        result = simulate(post, scenario)
        mu = recompute_mu(post, scenario)

        # the generator and the simulator must agree before anything is written
        weeks = mu["weeks"]
        for i, w in enumerate(weeks):
            got = np.array(mu["scenario_flat"]).reshape(mu["shape"])[:, i, :].sum(axis=1)
            want_band = result["weekly"][w]["scenario_units"]
            assert abs(float(np.quantile(got, 0.5)) - want_band["p50"]) < 1e-9, (
                f"{case['name']}: week {w} disagrees with simulate()"
            )

        cases.append(
            {
                "name": case["name"],
                "about": case["about"],
                "scenario": scenario,
                "mu": mu,
                "expected": result,
            }
        )

    doc = {
        "about": (
            "Python/TypeScript parity fixture. web/lib/simulate.ts must reproduce every "
            "mu and every field of expected, reading the same contract/posterior.fixture.json."
        ),
        "posterior": "contract/posterior.fixture.json",
        "generated_by": "contract/make_parity_fixture.py",
        "tolerance": {"mu": MU_TOLERANCE, "summary": SUMMARY_TOLERANCE},
        "ignore_fields": ["meta.source", "meta.computed_ms"],
        "check_weeks_note": (
            "mu is recorded only at mu.weeks (promo window, the dip week, and a late "
            "week). expected.weekly covers all horizon weeks."
        ),
        "posterior_meta": {
            "cities": meta["cities"],
            "segments": meta["segments"],
            "n_draws": meta["n_draws"],
            "weeks": meta["weeks"],
            "base_price": meta["base_price"],
        },
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    for c in cases:
        inc = c["expected"]["incremental"]["profit"]
        print(
            f"  {c['name']:<24} mu{c['mu']['shape']}  "
            f"inc profit {inc['mean']:>12,.0f}  "
            f"90% [{inc['p05']:>11,.0f}, {inc['p95']:>11,.0f}]  "
            f"P(pays) {c['expected']['p_profit_positive']:.0%}"
        )


if __name__ == "__main__":
    main()
