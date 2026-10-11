# Decisions

Choices the spec left open. One line each.

| T+ | Decision | Why |
|---|---|---|
| 0:15 | One `uv` project at the repo root manages both `model/ripple_model` and `api/app`, instead of two projects. | One `uv sync`, one `pytest` run, one lockfile; the API imports `ripple_model` directly with no path dependency. |
| 0:15 | Seasonality is exported as Fourier **coefficients** (`draws.season`, shape `[D][2*n_fourier]`) rather than a per-week effect. | The browser can then extrapolate `season(t)` to any future week instead of being limited to the 104 training weeks. |
| 0:15 | `history` is exported columnar (parallel arrays), not as row objects. | ~3x smaller JSON for 6 cities x 4 segments x 104 weeks, which the browser downloads on first paint. |
| 0:15 | `g0` is folded into the fitted `a_city` and used only for new cities. | Keeps the simulator's mean function to one line per term, with no "is this a training city" branch except for `a_new`. |
| 0:15 | Took the T+15 cut: `posterior.schema.json` + fixture first, then the other three schemas in the same step. | The other schemas are small; writing them now unblocks Session B's verdict and trust cards. |
| 0:15 | Wrote `launch.schema.json` and `update.schema.json` at T+15 instead of T+1:15. | Session B builds `update.ts` in parallel from 1:15 and would otherwise be blocked waiting on Session A. |
