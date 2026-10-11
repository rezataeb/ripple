# contract/

The one thing both sessions agree on. Session A (`model/`, `api/`) writes these files; Session B (`web/`) builds against them.

| File | What it is |
|---|---|
| `posterior.schema.json` | the fitted model, exported for the browser |
| `posterior.fixture.json` | a tiny real example of it — 50 draws, 2 cities, 2 segments, 12 weeks |
| `scenario.schema.json` | the question (UI controls or the agent's `build_scenario`) |
| `result.schema.json` | the answer |
| `trust.schema.json` | the checks, plus pre-summarized `model_facts` for the agent |
| `launch.schema.json` | the real weekly data that arrives after a launch (3 demo launches, truth known) |
| `update.schema.json` | the result of the Bayesian update, including the validity gate |
| `calibration.schema.json` | the calibration study: coverage by week, plus the stress arm |
| `parity.fixture.json` | **the Python↔TS agreement test** — 3 scenarios with expected `mu` arrays and full results |
| `make_fixture.py` | regenerates the posterior fixture (`uv run python contract/make_fixture.py`) |
| `make_parity_fixture.py` | regenerates the parity fixture (run it after any change to `simulate.py`) |

The real export lands at `web/public/model/posterior.json` + `trust.json`. Until it does, point the UI at the fixture — the shapes are identical.

## The two rules that make the parity test pass

**1. Mean function.** For draw `d`, city `c`, segment `s`, week `t`:

```
log mu = log(shares[c][s])
       + a_city[d][c]                       (or a_new for a new city, below)
       + a_seg[d][s]
       + e[d][s] * log(price / base_price)
       + b_promo[d][s] * promo
       + b_feature[d] * feature
       + b_dip[d] * promo_last_week
       + dot(season_row(t), season[d])
mu = exp(log mu)
```

`g0` is already inside the fitted `a_city`. It is used **only** for a new city:

```
a_new[d] = g0[d] + dot(traits_z, g[d]) + tau_city[d] * z[d],   z[d] ~ N(0, 1)
```

Draw `z` once per posterior draw with a fixed seed (`scenario.seed`, default 7) so the same scenario always returns the same numbers.

**2. Season basis.** `n_fourier` pairs, period `season_period` (52):

```
season_row(t) = [sin(2*pi*1*t/P), cos(2*pi*1*t/P), ..., sin(2*pi*n*t/P), cos(2*pi*n*t/P)]
```

Length `K = 2 * n_fourier`. Horizon week `w` (0-based) uses `t = meta.weeks + w`, so the forecast continues the calendar instead of restarting it.

## Scenario math

Baseline and scenario run over the **same** cities, segments and weeks. Difference them **inside each draw**, then summarize — never summarize first and subtract means.

```
units   = mu                                   (expected units; the band is posterior uncertainty in mu)
revenue = units * net_price
profit  = units * (net_price - unit_cost) - fixed_cost
```

`fixed_cost` is charged once over the whole horizon, on the scenario side only. Cells outside `scenario.cities` / `scenario.segments` stay at baseline, so they cancel out of the incremental — selecting one segment does not silently change the others.

`p_profit_positive` is the share of draws with incremental profit > 0.

## The parity fixture

`parity.fixture.json` is how `web/lib/simulate.ts` proves it agrees with
`model/ripple_model/simulate.py`. Three cases:

| case | what it pins down |
|---|---|
| `promo-students-boston` | `b_promo`, the price term, the `b_dip` week, and the rule that unselected cells cancel out of the incremental |
| `price-plus-8pct-all` | the elasticity term on every cell, with no promo weeks |
| `new-city-miami` | `a_new = g0 + traits_z·g + tau_city·z` with the seeded `z`, the zero baseline, the candidate's segment mix, and the extrapolation flag |

Each case has:

- `scenario` — feed this to your `simulate()`.
- `mu.baseline_flat` / `mu.scenario_flat` — flat row-major arrays of shape
  `mu.shape` = `[draw][week_in_mu.weeks][cell]`, with `mu.cells` naming the cells
  in order and `mu.t` giving the absolute week index. Reshape and compare
  **element by element**: tolerance **1e-6 on `mu`**. This is the test that catches
  an indexing or broadcasting mistake, which a headline total would hide.
- `expected` — the complete result object. Diff every field except those listed in
  `ignore_fields` (`meta.source`, `meta.computed_ms`).

`mu` is recorded only at `mu.weeks` (`[0, 3, 4, 5, 11]`: inside the promo window,
the last promo week, the dip week, the week after, and a late week) to keep the file
small. `expected.weekly` covers every horizon week, so nothing goes unchecked.

The Python side of this test lives in `model/tests/test_parity.py` — the TS side
should assert the same things.

## Hard rules

- Every number shown to a user carries its 90% interval (`p05`/`p95`).
- Terms that do not apply are present and zero-filled (`has_promo: false` ⇒ `b_promo` all zeros). The simulator never branches on a missing key.
- A new city with a standardized trait outside `[trait_min, trait_max]` sets `flags.extrapolation` and names the traits.

Changes to these files go in `HANDOFF.md` first.
