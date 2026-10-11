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
| `make_fixture.py` | regenerates the fixture (`uv run python contract/make_fixture.py`) |

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

Tolerance for the Python↔TS parity test: **1e-6 on `mu`**.

## Hard rules

- Every number shown to a user carries its 90% interval (`p05`/`p95`).
- Terms that do not apply are present and zero-filled (`has_promo: false` ⇒ `b_promo` all zeros). The simulator never branches on a missing key.
- A new city with a standardized trait outside `[trait_min, trait_max]` sets `flags.extrapolation` and names the traits.

Changes to these files go in `HANDOFF.md` first.
