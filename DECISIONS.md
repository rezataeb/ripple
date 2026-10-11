# Decisions

Choices the spec left open. One line each.

| T+ | Decision | Why |
|---|---|---|
| 0:15 | One `uv` project at the repo root manages both `model/ripple_model` and `api/app`, instead of two projects. | One `uv sync`, one `pytest` run, one lockfile; the API imports `ripple_model` directly with no path dependency. |
| 0:15 | Seasonality is exported as Fourier **coefficients** (`draws.season`, shape `[D][2*n_fourier]`) rather than a per-week effect. | The browser can then extrapolate `season(t)` to any future week instead of being limited to the 104 training weeks. |
| 0:15 | `history` is exported columnar (parallel arrays), not as row objects. | ~3x smaller JSON for 6 cities x 4 segments x 104 weeks, which the browser downloads on first paint. |
| 0:15 | `g0` is folded into the fitted `a_city` and used only for new cities. | Keeps the simulator's mean function to one line per term, with no "is this a training city" branch except for `a_new`. |
| 0:15 | Wrote `posterior.schema.json` + fixture first (the T+15 cut order), then the other three schemas in the same step. **Nothing was cut.** | The other schemas are small; writing them now unblocks Session B's verdict and trust cards. |
| 0:15 | Wrote `launch.schema.json` and `update.schema.json` at T+15 instead of T+1:15. | Session B builds `update.ts` in parallel from 1:15 and would otherwise be blocked waiting on Session A. |
- [B] web: shadcn init wired `cn` to an npm package named `cn`; replaced with clsx + tailwind-merge in `web/lib/utils.ts` (same helper, standard shadcn).
- [B] web: UI loads `/model/posterior.json`, falls back to `/model/posterior.fixture.json` (a copy of `contract/posterior.fixture.json`) until Session A's export lands.
- [B] web: no result is shown until `web/lib/simulate.ts` exists; the verdict is a skeleton, never a placeholder number.
| 1:55 | Implemented PSIS ourselves in `model/ripple_model/psis.py` instead of using ArviZ's. | ArviZ 1.3.0's `array_stats.psislw` returned **k-hat = 22.1** on a healthy weight vector (raw ESS 4,805 of 20,000, recovering the planted truth) and collapsed it to ESS 1. A k-hat of 22 is not a value the diagnostic can take. `pareto_khat(..., log_weights=True)` was no good either — it reported −0.04 for weights whose real ESS was 6. Ours follows Vehtari et al. / Zhang–Stephens, agrees with ArviZ to within 0.1 on the cases where ArviZ behaves, and is tested against weight distributions with known behaviour. |
| 1:55 | A new offer's prior is the fitted `b_promo[s]` posterior **widened** by `NEW_OFFER_SIGMA = 0.25`, not the posterior itself. | The spec says the prior is "the posterior of the existing model". Taken literally with `K` draws per posterior draw there is nothing to draw, and a coupon stronger than any past promotion would collapse the weights and trip the gate every time. A new offer is a draw from the distribution of offers, not a copy of the average past one. |
| 2:05 | Took `target_accept` to 0.99 for the demo fit rather than shipping 4 divergences. | At 0.95 the fit had 4 divergences and R-hat 1.0101 against a 1.01 threshold — both on `tau_city`. 0.99 gives 0 divergences for 336s instead of 161s, which is affordable once. The alternative was an amber trust card for a problem that had a one-line fix. |
