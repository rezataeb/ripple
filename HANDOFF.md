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

## B → A

(nothing yet)

## Blocked / errors

(nothing yet)
