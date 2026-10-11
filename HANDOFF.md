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

## B → A

(nothing yet)

## Blocked / errors

(nothing yet)
