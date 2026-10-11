# HANDOFF

Cross-session notes. Session A owns `contract/`, `model/`, `api/`. Session B owns `web/`.
Need a change in the other session's folder? Write it here instead of editing it.

## A → B

**[T+0:15] The contract is ready.** `contract/` has all schemas, `posterior.fixture.json` (50 draws, 2 cities, 2 segments, 12 weeks), plus `launch.schema.json` / `update.schema.json` for the launch tracker. **Read `contract/README.md`** — the mean function, the `season_row(t)` basis and the within-draw differencing rule are written out there, and `web/lib/simulate.ts` has to match them exactly or the parity test fails.

Things worth knowing before you write `simulate.ts`:
- Horizon week `w` uses absolute `t = meta.weeks + w`. Do not restart `t` at 0.
- `g0` is already inside `a_city`; only a new city adds `g0 + traits_z·g + tau_city·z`.
- Difference baseline and scenario **inside each draw**, then take quantiles.
- The fixture's `Miami` candidate has `extrapolation: true` with two out-of-range traits — use it to build the warning banner.
- The fixture's history has a 3-week promo at `t = 4,5,6`, so the dip at `t = 7` is visible in the chart.

**The real export** will appear at `web/public/model/posterior.json` + `trust.json` with identical shapes. Load from `/model/posterior.json` and fall back to the fixture, so nothing breaks before the fit lands.

**[T+0:35] Demo data is generated.** `model/data/` now has `brightside_weekly.csv` (2496 rows, 6 cities x 4 segments x 104 weeks, price $16.17–$25.79, 21% promo cells), `brightside_cities.csv`, `truth.json` (planted parameters), and `launches.json` (3 launches, 8 observed weeks each, truth known). `launches.json` already validates against `contract/launch.schema.json`, so you can build `/launch` against it now — copy it to `web/public/model/launches.json` or wait for the export step, which will put it there.

The three launches, and what the slider should show:
- `austin-launch` (new city, unknown `z_new_city`) — true z is high, so the band should move **up** as weeks reveal. Fixed cost $60k; it does pay. `city_idx` is `-1` for the new city.
- `student-coupon` (offer, unknown `b_promo[students]`) — the real coupon is stronger than history taught the model, so the band moves up. No fixed cost.
- `gift-boxes` (feature, unknown `theta`, skeptical prior N(0, 0.15)) — truth +11.6%. This one **has a random holdout** (`holdout` column, 0/1), so its causal badge is "Experiment", not "Model-based estimate". Its prior scale is `editable_sigma: true` — that's the slider the spec asks for.

**[T+1:10] `simulate.py` and the parity fixture are in.** Two things you need:

1. **The posterior fixture changed — re-pull it.** `contract/posterior.fixture.json` had a bug: `a_city` did not include `g0`, so every `mu` came out ~400x too small. It is fixed and regenerated. If you already read the old file, numbers you saw were wrong (baseline was ~0.5 units/week/cell; it is now ~380). `meta` also gained an optional `traits_z` field (the training cities' standardized traits) — ignore it unless you want to plot candidates against training cities.

2. **`contract/parity.fixture.json` is the agreement test for `web/lib/simulate.ts`.** Three cases — promo, price, new city. For each: `scenario` to feed your simulator, `mu.baseline_flat` / `mu.scenario_flat` (flat row-major, shape in `mu.shape` = `[draw][week][cell]`, cells named in `mu.cells`) to compare **element by element at 1e-6**, and `expected`, the complete result object, to diff field by field (skip `ignore_fields`). `contract/README.md` has the details and `model/tests/test_parity.py` is the Python side doing exactly this — mirror its assertions.

Three traps the fixture is built to catch:
- horizon week `w` uses absolute `t = meta.weeks + w`, not `t = w`;
- `g0` is added **only** for a new city (`a_new = g0 + traits_z·g + tau_city·z`, `z` seeded from `scenario.seed`, default 7, one `z` per draw);
- difference baseline and scenario **inside each draw**, then take quantiles. The promo case asserts the non-Boston cities' incremental is exactly 0.

Headline numbers you should reproduce on the fixture: promo +$9,196 profit 90% [$4,626, $13,444] P(pays) 100%; price +$310 90% [−$2,464, +$3,652] P(pays) 50%; Miami +$2,353 90% [−$21,610, +$26,009] P(pays) 54% with `extrapolation: true`.

**[T+2:30] The real model is exported. All four files are live in `web/public/model/`:**

| file | what |
|---|---|
| `posterior.json` | 389 KB, 1000 draws, 6 cities x 4 segments x 104 weeks, 3 candidates |
| `trust.json` | 11 KB, **6 of 6 checks pass** |
| `launches.json` | 26 KB, the 3 launches |
| `calibration.json` | 9 KB, 3 arms x 8 weeks — the `/method` coverage chart |

Switch the simulator from the fixture to `/model/posterior.json`. The shapes are identical, so it should be a one-line change; keep the fixture as the fallback so nothing breaks offline.

**Numbers you can quote in the UI and the README** (all from `trust.json`):
- R-hat 1.0100, min ESS 612, **0 divergences**, posterior predictive coverage 91%.
- **Parameter recovery 30/30** planted values inside the 94% HDI.
- Time holdout: 92% coverage, MAPE **18.4% vs. 44.7%** for a same-week-last-year guess.
- Method string: `nutpie NUTS, 4 chains x 1000 draws, target_accept 0.99`.

**`calibration.json` for the `/method` chart.** Three arms, `by_week` has 8 entries each. Plot `coverage_total` against week with a reference line at `target_coverage` (0.9), and use `coverage_total_mcse` for the error bars — the chart should show its own uncertainty. The arm with `assumptions_hold: false` is the **stress arm**: draw it dashed and print its `broken_assumption` next to it. Headline numbers:
- New city: 90% -> 94% coverage, interval narrows 56% over 8 weeks.
- New feature: 88-92% coverage.
- **Stress arm (a launch that ramps up): coverage falls to 6%.** This is deliberate and it is the most honest thing on the page — please do not hide it.

`trust.json` also carries `model_facts` (elasticity and promo lift per segment with intervals, the dip, the feature lift) pre-summarized, so the agent's `model_facts()` tool returns them directly and never computes a number.

`docs/method/{model-card,checks,limits}.md` are written — they are the RAG source for `explain_method` and the source for the README's model section.

## B → A

(nothing yet)

## Blocked / errors

(nothing yet)

## [B → A] parity fixture: new-city z draws (not blocking)
`simulate.py` draws z with `numpy.random.default_rng(seed).standard_normal`, which the browser cannot reproduce. The
new-city parity case therefore passes only because `web/tests/parity.test.ts` backs z out of `mu.scenario_flat`
(week 0, first cell) and then compares every other field. Request: add `"z": [...]` (the 50 draws used) to the
`new-city-miami` case in `contract/parity.fixture.json`. The test already uses `case.z` when present. All 3 cases
(promo, price, new city) otherwise match at 1e-6: cell-level mu and every result field.

Runtime consequence: the browser's `scenario.seed` z draws differ from Python's for the same seed, so a new-city
interval in the browser can differ slightly from the API's. If you want identical numbers, export the z draws for seed 7
in `posterior.json` and I will use them.
