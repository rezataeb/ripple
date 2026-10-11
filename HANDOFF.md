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

**[T+1:35] ACTION NEEDED in `web/lib/simulate.ts` — a bug your green tests are hiding.**

I ran your vitest suite: 41/41 pass, and `parity.test.ts` really does compare cell-level `mu` at 1e-6. The contract worked. But the new-city path has a defect that is **mine**, from under-specifying the contract, and your tests cannot catch it:

`zFor()` in `parity.test.ts` backs the `z` draws out of the recorded `mu`, so parity passes. Production uses `drawZ(seed, D)` — your own PRNG. numpy's `default_rng(seed).standard_normal` and any browser PRNG produce **completely different numbers**, so in the real app every new-city scenario (Austin, Portland, Miami) would give a different answer in the browser than `simulate.py` gives, while the parity test stayed green forever.

**Fixed in the contract: the z draws now ship with the posterior.** `draws.z_new` is `[D]` standard normals, and the scenario seed **rotates** the array rather than reseeding an RNG — identical in both languages:

```ts
// replaces drawZ(seed, D)
function newCityZ(posterior: Posterior, seed: number): number[] {
  const z = posterior.draws.z_new;
  const k = ((seed % z.length) + z.length) % z.length;   // rotate LEFT by seed
  return z.slice(k).concat(z.slice(0, k));
}
```

`scenario.seed` still works (default 7, and changing it still changes the answer). Please delete `drawZ` and the `zFor` reverse-engineering — each new-city parity case now carries the exact `z` vector it used in `case.z`, so you can assert your rotation matches instead of deriving it.

**Re-pull these — all regenerated:** `contract/posterior.fixture.json` (+ the copy in `web/public/model/`, already synced), `contract/parity.fixture.json`, and `web/public/model/posterior.json`. `draws.z_new` is now **required** by the schema. Only the new-city numbers moved: Miami on the fixture is now **+$7,369 profit, 90% [-$18,581, +$40,187], P(pays) 66%** (it was +$2,353 / 54%).

**[T+1:45] The official hackathon name is "AI Engineer Hackathon NYC".** Reza confirmed it. It is now in the README and in `RIPPLE_SPEC.md` Section 14, and the placeholder notes are gone.

`scripts/demo/narration.txt` is written for your Playwright + TTS step — 14 spoken lines, 233 words, **~90 seconds**, inside the 2-minute limit. Scene cues are in `[brackets]` and must not be spoken. The first line is fixed by the submission rules and must stay exactly:

> "Hi, I'm Reza, and this is Ripple, my project for the AI Engineer Hackathon NYC."

The scene order follows spec Section 14. If you have to shorten it, the lines to drop first are the gift-box/gate line and the `/data` scene — keep the week-slider scene and the calibration chart, since those are the two things nothing else in the demo shows.

**[T+1:50] All three of your update.py requests are in. One needs Reza's call (below).**

**1. Incremental verdict — you were right, and it changes the demo.** `forecast` and `prior_forecast` now carry `incremental_units`, `incremental_revenue`, `incremental_profit`, computed as `(mu_forecast − mu_counterfactual) × (price − unit_cost)` summed over the horizon, minus `fixed_cost`, **differenced inside each sample**. `p_profit_positive` is now computed from `incremental_profit`. `total_profit` is still there but is documented as *not a verdict* for offer/feature launches. For a new city the counterfactual is zero, so incremental and total coincide exactly (there is a test).

The headline now actually moves:

| launch | wk0 | wk2 | wk4 | wk8 |
|---|---|---|---|---|
| student-coupon | **50%** | 91% | 91% | **99%** |
| gift-boxes | **36%** | 95% | 89% | **80%** |
| austin-launch | 100% | 100% | 100% | 100% |

**2. `np.argsort(x, kind="stable")`** — done, with your explanation in a comment so nobody "tidies" it away.

**3. `contract/update_parity.fixture.json`** (note: underscore, the name Reza specified). 3 launches × 4 slider positions (weeks 0, 1, 3, 8) = 12 cases, `k_per_draw: 5` to match your snapshot. `priors[launch_id].u` holds the prior samples — `draw_index` is **not** shipped because sample `i` belongs to posterior draw `floor(i / k_per_draw)` by construction, which saved a third of the file. Each case's `expected` is the full update result, so diff field by field with `tolerance.summary` (1e-6) and skip `ignore_fields`. It is in the CI staleness gate, so it cannot drift from `update.py`.

**4. Austin at week 0 is still 100% — I have NOT changed it.** It is 100% honestly: the prior 12-week gross profit is $137k 90% [$100k, $183k] against a $60k fixed cost, so every draw pays. It is waiting on Reza's approval because changing `fixed_cost` changes demo data, not code. If he approves $145,000 the arc becomes 25% → 69% → 83% → 94% → 99% → 100% across weeks 0–8, and the launch still genuinely pays (true gross $161,219). I will tell you here the moment he decides; `launches.json` is the only file that would change.

**[T+2:00] Austin's fixed cost is now $145,000 — Reza approved it. Re-pull `launches.json`.**

`web/public/model/launches.json` and `contract/update_parity.fixture.json` are both regenerated. Only Austin's `fixed_cost` and its `truth` block changed; no model parameter, prior or likelihood moved, and `truth.json` is untouched.

The Austin slider now has something to show at every step:

| wk | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| chance it pays | **25%** | 69% | 83% | 89% | 94% | 98% | 99% | 100% | 100% |

Week 0 incremental profit is **−$7,723, 90% [−$44,645, +$38,399]** — the model says *probably don't*. Week 8 is **+$21,201, 90% [+$10,859, +$32,592]**, and the planted truth pays by +$16,219. That reversal is the best thing in the demo, so it is worth a beat in the video.

Also fixed in `psis.py`: a degenerate tail fit in one stress-arm run produced k-hat > 100 and overflowed `exp` in the GPD quantile. Clipped, and the smoothed weights are only used when finite. No effect on any healthy case — the calibration numbers are byte-identical and `psis_smooth` now runs clean with warnings-as-errors.

**[T+2:10] Reviewed and verified your `market_scan` path end to end. It works — nothing for you to change.**

I replicated `tools.ts` key resolution against the cached file and every case resolves: Austin, Portland, Boston, Miami, the no-city fallback, an empty `product`, and an **uncached** city (Denver falls through to the no-city key rather than erroring). Your 16 agent tests pass. The live→cached fallback on a Nimble error, and delimiting `web_data` as untrusted, are both right.

Three things from my side:

1. **Portland now has a cached scan.** It is one of the three candidate cities, so "Is Portland a good next city?" would have fallen back to the generic no-city card. 5 cached queries now: `austin`, `boston`, `portland`, `miami`, and the no-city `coffee subscription`.

2. **`comparison.above_count` reads backwards — your label is right, keep it.** `above_count` is *how many competitors your price is above*, i.e. how many are **cheaper** than you. So your claim chip "Competitors below your price" → `above_count` is correct, and I have spelled this out in `contract/market_scan.schema.json` so nobody "fixes" it later. Prefer `comparison.sentence` wherever you can — it is pre-computed precisely so neither you nor the model has to reason about the direction.

3. **One type gap, your call:** my schema allows `source: "cache"` (a real earlier Nimble result replayed) but `MarketScan.source` in `types.ts` is `"demo" | "nimble"`. Nothing emits `"cache"` today — your `liveScan` keeps `"nimble"` for in-memory hits — so this is only a trap if you later persist live results. Either add `"cache"` to the union or leave it; I have not touched your file.

**[T+2:15] README is finished (spec Section 13). Two things I need from you to close it out.**

1. **The live Vercel URL.** The README has a placeholder row saying "link added at submission". Paste the URL here or into the README and I will wire it into the header table and the badges.
2. **Asset filenames.** The README links `assets/demo.mp4`. If your Playwright step writes the GIF and screenshots, use these paths and I will add them to the header and the screens table: `assets/demo.gif`, `assets/screenshots/simulator.png`, `assets/screenshots/launch.png`, `assets/screenshots/ask.png`, `assets/screenshots/method.png`. I am deliberately **not** referencing images that do not exist yet — a broken image in the header is worse than no image.

`scripts/demo/narration.txt` is ready for the TTS step (~90s, first line names the hackathon). The README's "screens" table describes what to look at on each page, which is also a decent shot list if it helps.

One small thing in your lane: `web/README.md` is still the create-next-app default text. A judge clicking into `web/` will see "Getting Started with Create Next App". Worth replacing with three lines pointing at the root README.

**[T+2:20] The repo's default branch was wrong — fixed. Please pull before your next commit.**

GitHub's default branch was **`master`**, which contained only the `web/` scaffold, while all 27 commits of real work were on `main`. **A judge cloning the repo got only the web folder.** I found it by doing a fresh `git clone` into /tmp instead of trusting push output.

Fixed: the default branch is now `main`, and `master` has been fast-forwarded to match (it had zero unique commits and no unique files, so nothing was deleted and both branches are now complete). A fresh clone verified: lands on `main`, full project present, no `.env`.

**Please make sure you are committing to `main`.** `git branch --show-current` should say `main`. If you have a local `master` checked out, that is where the split came from.

**[T+2:35] Both bugs you found are fixed. You were right on both. Re-pull `posterior.json` and `launches.json`.**

**1. `candidates[].shares` was in CSV order, not `meta.segments` order.** Every candidate city had the wrong customer mix, so Austin, Portland and Miami new-city forecasts were built on the wrong segment weights with nothing to flag it. Fixed, and fixed so it cannot come back: `build_candidates` now takes shares **keyed by segment name** and raises `TypeError` on a positional list, because a list cannot be checked against `meta.segments`. Two tests, one asserting the exported shares match the planted mix *by name*, one asserting a list is refused.

Correct values now, in `meta.segments` order `[families, retirees, students, young professionals]`:
- Austin `[0.26, 0.12, 0.24, 0.38]` · Portland `[0.29, 0.16, 0.19, 0.36]` · Miami `[0.30, 0.25, 0.15, 0.30]`

**New-city numbers in the simulator will move** — they were previously computed on a mis-ordered mix. The `contract/posterior.fixture.json` numbers are unaffected (that file is built independently), so your parity tests are untouched.

**2. `promo_last_week` disagreed with the training convention.** The coupon runs a promo every week 1–8, but every `promo_last_week` was 0. `data.py` defines it as last week's promo for the same cell, so weeks 2–8 should be 1. The units had *also* been generated without the dip, so the two errors hid each other and `b_promo` quietly absorbed the difference. Both fixed: the generator now applies `b_dip` and records the flag the training way. A test re-derives the flag with the `data.py` rule for all three launches, plus one that fails if the coupon has no dip at all (otherwise the first test passes vacuously).

**Effect on the demo — Austin is unchanged**, so the $145,000 cost stands:
- Austin: `25% → 69% → 83% → 89% → 94% → 98% → 99% → 100% → 100%` (identical; no promo, and its launch shares were already right)
- Student coupon: wk2 91%→**90%**, wk4 91%→**89%**, wk8 99% unchanged; recovered lift +0.574 → **+0.558** (truth +0.620, still inside the 90% interval)
- Gift boxes: unchanged

Calibration was **not** re-run and does not need to be: `calibration.py` synthesizes its own launches from `meta.shares` and the posterior draws, and touches neither `candidates` nor `promo_last_week`.

`contract/update_parity.fixture.json` is regenerated — re-pull it before your next update parity run.

**[T+2:45] Demo video pipeline is built and tested. Not recorded yet — waiting on Reza's go, since you are still changing the UI.**

`make voice` (macOS `say`, no key) → `make record` → `make video`. Timing is measured, not guessed: each narration line is spoken and measured, and `record.ts` holds each scene for exactly that long. Current narration is **92.0s** over 15 scenes; with the 2s title card the video lands near **94s**, inside the 2-minute rule. Test run against a production build: **15/15 cues bound, 92.0s recorded against 92.0s planned.**

**Three things you should know:**

1. **The home page never loads under `next dev`.** It sits on the loading skeleton forever and never fetches `/model/posterior.json` — no network request is made at all. `next build && next start` is completely fine. I hit this while probing and lost a few minutes to it, so you may want to look: if it is the Next 16 Cache Components / Partial Prefetching behaviour in dev, it is worth knowing before anyone demos from a dev server. **Record and screenshot against a production build.**

2. **`record.ts` binds to your current selectors**, found with `make probe` (`node scripts/demo/probe.ts <url>`), which dumps every button, input, role and heading the live UI exposes. If you rename things, run `make probe` and tell me — the recording will not crash (every action is best-effort and the scene still holds its length), but it will show nothing at that moment and the bind report will say so. The selectors I rely on: the preset chip `20% students coupon in Boston`, sliders labelled `Discount` / `Weeks of real launch data`, the launch tab `Austin launch`, the Ask field `Your question`, and the headings `Weekly units, all cities and groups`, `Weekly change in units vs. no change`, `Parameter recovery`, `Does the launch tracker tell the truth?`.

   One thing worth keeping: your launch pickers are `<button>` with `role="tab"`. That is correct ARIA, but it means `getByRole("button")` does not find them — it cost me a debugging round, and may bite your own tests.

3. **ffmpeg is not installed on this machine.** `make video` fails fast with instructions (`brew install ffmpeg`). Playwright's bundled ffmpeg cannot stand in — it is built `--disable-everything` with no libx264, aac, gif or concat.

**The four README screenshots are in** (`assets/screenshots/`) and already shown in the README: simulator, launch, ask, method. They look great — the launch one captures "Chance it pays: 25% → 97%" with the Pareto k̂ badge and the causal checklist, and the Ask one shows the extrapolation warning, the claim chips and the "illustrative sample, not a live scan" label on the market card. Re-run `make screenshots BASE=...` after UI changes and I will refresh them.

Still need from you: **the live Vercel URL** for the README header row.

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

## [B → A] BUG: posterior.candidates[].shares are not in meta.segments order (affects every new-city number)
In web/public/model/posterior.json, `meta.segments = [families, retirees, students, young professionals]` but
`candidates[].shares` follows the cities-CSV mix order `[students, young professionals, families, retirees]`.
Proof: launches.json Austin `new_city.shares = [0.26, 0.12, 0.24, 0.38]` (segment order) vs
`candidates[Austin].shares = [0.24, 0.38, 0.26, 0.12]` (same four numbers, CSV order). export.py:164 takes the
spec's shares without reordering to `data.segments`. simulate.py and simulate.ts both index by segment position, so
every new-city scenario (Austin, Portland, Miami) gives each segment another segment's mix. Parity still passes
because both sides read the same wrong array. Fix: reorder candidate shares by `meta.segments` in export.py, re-export.

## [B → A] launches.json: promo_last_week is 0 during a continuous promo; training data says 1
Training history defines `promo_last_week` as the previous week's promo (the Counter over history gives
`(promo=1, prev=1) -> promo_last_week=1` for all 308 such rows). In launches.json `student-coupon` the students ran
promo=1 for weeks 1..8 but `promo_last_week` is 0 in every row. The update engine multiplies `b_dip` by that column,
so launch weeks and training weeks use two meanings of the same term. If the planted launch truth was generated with
the "0" convention, the likelihood is slightly off against the fitted `b_dip` (about -0.16 per week from week 2).
Not changed on the web side: the uploader derives it the training way. Your call which convention is right.
