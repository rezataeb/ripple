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

**[T+1:25] Cached market scan + the API are ready.**

`web/public/model/market_scan.demo.json` — the cached scan for demo mode, so the Ask panel works with no Nimble key. It is a map from a normalized query to one result: keys are `coffee subscription austin` / `boston` / `miami` and a no-city `coffee subscription`. Shape is `contract/market_scan.schema.json`.

Two things to carry into the UI:
- **`comparison.sentence` is pre-computed — print it verbatim.** The Austin one is exactly the spec's example: *"Your price $25.92 is above 2 of 3 competitors in Austin."* The counts and the median are in the same object, so the agent never does arithmetic on web numbers.
- **Every entry is `source: "demo"` and carries a `note` saying the figures are illustrative and the competitors are fictional (example.com URLs).** Please surface that note on the card. Brightside is a fictional company, so inventing real-looking competitors with real-looking links would be presenting fabricated evidence as checkable — the honest label is the whole reason the card is trustworthy when it *is* live. There are tests in `model/tests/test_demo_scan.py` that fail if the label is dropped.

When `NIMBLE_API_KEY` is set, fetch live results into this same shape with `source: "nimble"` and cache by the same normalized query.

**The API is up** for the `/data` page: `POST /datasets` (returns a preview plus a `detected_mapping` of exact column-name matches, so you can pre-fill the mapping table before calling Gemini), `POST /datasets/{id}/validate` with `{mapping}`, `POST /datasets/{id}/fit` → `{job_id, poll}`, `GET /jobs/{id}` → `{status, step, progress, posterior, trust}`, `DELETE /datasets/{id}` for the delete button. `make api` runs it on :8000, `make up` runs it in Docker, and `/docs` has the live OpenAPI page. Poll `/jobs/{id}` and show `step` — it moves through reading the file → checking the data → fitting the model → running the trust checks → exporting.

Note for the trust cards on a customer upload: **recovery and holdout come back with `skipped_reason` set**, not `passed: true`. There is no planted truth on real data and no time for a holdout refit inside the 90-second budget. Please render a skipped check as "not applicable" rather than as a failure.

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

## [B → A] update.py: make weighted_quantile's sort stable (small fix, affects parity)
`update.weighted_quantile` uses `np.argsort(x)` (unstable). K samples share each posterior draw, so
counterfactual values tie, and the interpolated quantile then depends on tie order (diffs ~5e-5 relative).
`np.argsort(x, kind="stable")` makes update.ts match update.py to 1e-6 on every field, for all 3 launches
at weeks 0 / 3 / 8 (verified with `web/tests/fixtures/make_update_reference.py`, which patches this in).
`update.ts` also needs the unknown's prior samples `u` (numpy's RNG is not reproducible in JS), so the web
test feeds `u` from that snapshot. If you write `contract/update.parity.fixture.json`, include `u` per launch
and I will point the test at it instead; until then the web snapshot (k_per_draw=5) guards the port.

## [B → A] launch update: "chance it pays" is 100% in every launch at every week (semantics, needs your call)
`update.py` returns `forecast.total_profit` = profit of ALL launch cells over the horizon, minus fixed cost.
For `new_city` that is incremental (counterfactual 0). For `offer` and `feature` launches the cells are an
existing business, so the profit includes the baseline margin and P(profit>0) is trivially 1.00 (student-coupon
and gift-boxes: profit ~$210k and ~$520k, interval nowhere near 0). The spec's "Chance it pays: 61% → 78%"
needs the INCREMENTAL profit: (mu_forecast − mu_counterfactual) × (price − unit_cost) − fixed_cost, differenced
inside each sample, for offer and feature launches. Austin is 1.00 for real (truth pays, +$77k vs $60k fixed).
Suggested: add `incremental_units / incremental_revenue / incremental_profit` + `p_profit_positive` computed that
way for every launch type (equal to the totals for new_city) and use it as the headline. update.ts is a
line-by-line port, so tell me when the Python changes and I will mirror it (parity test is already in place).
Not blocking: the /launch page uses the current fields until then.
