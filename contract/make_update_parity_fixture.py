"""Build contract/update_parity.fixture.json — the parity test for the launch update.

web/lib/update.ts is a line-by-line port of model/ripple_model/update.py, so it needs
the same inputs to produce the same outputs. One of those inputs cannot be ported:
the prior samples `u` of the unknown come from numpy's Generator, which no browser
PRNG reproduces. So they are shipped here, once per launch, so the web port can be tested against
them — the same reason `draws.z_new` ships in the posterior.

Everything downstream of `u` is then compared field by field: the importance weights,
the PSIS gate, the weighted quantiles, the unknown's prior-vs-posterior summaries, the
incremental verdict, and every week of the chart.

k_per_draw is 5 rather than the production 20 purely to keep this file small; the
machinery under test is identical.

Run: uv run python contract/make_update_parity_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ripple_model.update import DEFAULT_SEED, draw_unknown, update_with_launch

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web" / "public" / "model"
OUT = ROOT / "contract" / "update_parity.fixture.json"

K_PER_DRAW = 5
WEEKS = [0, 1, 3, 8]
TOLERANCE = 1e-6


def main() -> None:
    post = json.loads((PUBLIC / "posterior.json").read_text())
    launches = json.loads((PUBLIC / "launches.json").read_text())["launches"]

    cases = []
    priors = {}
    for launch in launches:
        u, draw_idx, sigma = draw_unknown(
            post, launch, k_per_draw=K_PER_DRAW, seed=DEFAULT_SEED
        )
        # draw_index is not shipped: it is exactly i // k_per_draw by construction,
        # and 15,000 derivable integers is a third of this file.
        assert all(int(d) == i // K_PER_DRAW for i, d in enumerate(draw_idx))
        priors[launch["id"]] = {
            "u": [round(float(x), 10) for x in u],
            "prior_sigma": float(sigma),
            "kind": launch["unknown"]["kind"],
        }
        for weeks_used in WEEKS:
            result = update_with_launch(
                post, launch, weeks_used, k_per_draw=K_PER_DRAW, seed=DEFAULT_SEED
            )
            result.pop("_net_price", None)
            cases.append(
                {
                    "launch_id": launch["id"],
                    "weeks_used": weeks_used,
                    "expected": result,
                }
            )

    doc = {
        "about": (
            "Python/TypeScript parity for the launch update. web/lib/update.ts must "
            "reproduce every field of each case's `expected`, reading the same "
            "posterior.json and launches.json and the prior samples in `priors`."
        ),
        "posterior": "web/public/model/posterior.json",
        "launches": "web/public/model/launches.json",
        "generated_by": "contract/make_update_parity_fixture.py",
        "k_per_draw": K_PER_DRAW,
        "seed": DEFAULT_SEED,
        "tolerance": {"summary": TOLERANCE},
        "ignore_fields": ["meta.source", "meta.computed_ms"],
        "u_note": (
            "priors[launch_id].u holds the prior samples of the unknown, length "
            "n_draws * k_per_draw. Sample i was drawn under posterior draw "
            "floor(i / k_per_draw) — that mapping is by construction, so it is not "
            "shipped. Feed u in rather than generating it: numpy's Generator is not "
            "reproducible in a browser, and every weighted summary downstream depends "
            "on the exact values."
        ),
        "sort_note": (
            "weighted_quantile sorts with kind='stable'. K samples share each posterior "
            "draw so counterfactual values tie exactly, and an unstable sort makes the "
            "interpolated quantile depend on tie order (~5e-5 relative)."
        ),
        "headline_note": (
            "forecast.p_profit_positive is computed from incremental_profit — "
            "(mu_forecast - mu_counterfactual) x (price - unit_cost) summed over the "
            "horizon minus fixed_cost, differenced inside each sample. total_profit "
            "carries the baseline margin of existing cells and is not a verdict for an "
            "offer or feature launch."
        ),
        "priors": priors,
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(launches)} launches x {len(WEEKS)} slider positions = {len(cases)} cases")
    for launch in launches:
        n = len(priors[launch["id"]]["u"])
        print(f"  {launch['id']:<16} {n:>6,} prior samples  kind={priors[launch['id']]['kind']}")
    for c in cases:
        if c["weeks_used"] in (0, 8):
            f = c["expected"]["forecast"]
            print(f"    {c['launch_id']:<16} wk{c['weeks_used']}: "
                  f"incremental profit {f['incremental_profit']['mean']:>10,.0f} "
                  f"pays {f['p_profit_positive']:>5.0%} "
                  f"gate {'OK' if c['expected']['gate']['ok'] else 'REFIT'}")


if __name__ == "__main__":
    main()
