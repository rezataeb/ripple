# ripple

**"What happens to sales if…?"** — a Bayesian what-if simulator, a launch tracker that
learns from real data, and a market-aware AI agent.

<!-- TODO(Reza): assets/demo.gif, live Vercel link, assets/demo.mp4 -->

> Built in 3 hours at the **AI Engineer Hackathon NYC**, 2026-10-11.

---

## The problem

Promotions, prices and new markets get decided on **one number**, with no uncertainty
and no market context. "The 20% student coupon will add $18k." Will it? How sure are
you? What happens the week after, when the people who would have bought anyway have
already bought? And is $25.92 even a sensible price when two of your three competitors
in that city charge less?

Ripple answers those questions with a **distribution**, never a point:

> Incremental profit **+$18.4k** · 90% **−$3.1k to +$41.0k** · **84%** chance it pays

Three layers, and the division of labour is the whole design:

**The model makes the numbers. The agent explains them. The web gives context.**

| Layer | What it does |
|---|---|
| **Model** | A hierarchical Bayesian demand model fitted in PyMC. The posterior ships to the browser, so sliders update instantly with no server round trip. |
| **Agent** | "Ask Ripple" — a Gemini tool-calling agent that turns plain English into a scenario, runs it, and explains the result. **It never writes a number a tool did not return.** |
| **Web** | Nimble searches live competitor prices and current offers, cited, as *context* — never as a model input. |
| **Reality** | After a launch, each new week of real data updates the forecast with Bayes' rule. The interval narrows. A calibration study proves the intervals are honest. |

---

## The model

One row of data is one week × city × customer segment.

```
y[c,s,t] ~ NegativeBinomial(mu, phi)

log mu = log(share[c,s])                 customer mix of the city (fixed offset)
       + a_city[c]                        how big this city is
       + a_seg[s]                         how much this segment buys per head
       + e[s]       * log(price / base)   price elasticity, per segment
       + b_promo[s] * promo               the deal signal of being on promotion
       + b_feature  * feature             email campaign flag
       + b_dip      * promo_last_week     pull-forward: last week's promo borrowed from this one
       + season(t)                        2 Fourier pairs, 52-week period

a_city[c] = g0 + X_city[c]·g + tau_city · z_city[c]
e[s] ~ Normal(mu_e, sigma_e),  mu_e ~ Normal(-1.5, 0.5),  sigma_e ~ HalfNormal(0.5)
```

### Why each part

| Part | Why it is there | What breaks without it |
|---|---|---|
| **Negative Binomial** | Weekly counts are overdispersed — variance far exceeds the mean. | Poisson calls normal variation a surprise; every interval comes out far too narrow. |
| **log–log price term** | `e[s]` *is* the elasticity: +1% price → `e[s]`% demand. | A linear term means nothing at a different price level and can predict negative demand. |
| **Elasticity per segment, partially pooled** | Students and retirees do not respond to price alike; pooling lets thin segments borrow strength. | Pooled: targeted offers look identical for everyone. Unpooled: small segments chase noise. |
| **Partially pooled promo lift** | Being on promotion lifts demand beyond the price cut — the deal is a signal. | The price term alone understates a promotion, then overstates the discount needed next time. |
| **Post-promo dip** | A promotion pulls purchases forward from the following week. | Promotions look better than they are: the borrowed sales are never paid back. |
| **Hierarchical city effect on traits** | `a_city` is predicted from population, income and age, so a city with **no history** still gets a forecast, with honest uncertainty from `tau_city`. | New-city questions become unanswerable — a third of the product. |
| **Customer-mix offset** | Volume scales with who lives there, so a new city with a different mix needs no refit. | A student-heavy city is forecast as if it had the average mix. |
| **Fourier seasonality** | Yearly shape with few enough parameters to extrapolate past the training weeks. | Season is confounded with the promo calendar and `b_promo` absorbs it. |

**What the model found** (posterior mean, 90% interval — planted truth in brackets):

| Segment | Price elasticity | Promo lift |
|---|---|---|
| students | **−2.25** [−2.51, −1.98] *(truth −2.20)* | **+0.39** [+0.31, +0.47] *(truth +0.42)* |
| young professionals | **−1.60** [−1.85, −1.35] *(truth −1.60)* | **+0.30** [+0.22, +0.37] *(truth +0.30)* |
| families | **−1.15** [−1.41, −0.88] *(truth −1.10)* | **+0.20** [+0.12, +0.27] *(truth +0.22)* |
| retirees | **−0.76** [−1.02, −0.50] *(truth −0.80)* | **+0.06** [−0.02, +0.14] *(truth +0.10)* |

Post-promo dip **−0.164** [−0.186, −0.143] *(truth −0.18)*. Feature lift **+0.094**
[+0.076, +0.112] *(truth +0.08)*.

### Scenario math

Per posterior draw: baseline `mu` against scenario `mu` over the **same** cities,
segments and weeks → `revenue = units × net_price` → `profit = units × (net_price −
unit_cost) − fixed_cost`. Difference the two **inside each draw**, then summarize. That
ordering is the only honest way to get the interval, and cells outside the selection
cancel out exactly — so selecting one segment does not silently move the others.

The same math runs twice: `model/ripple_model/simulate.py` as the reference and
`web/lib/simulate.ts` in the browser. `contract/parity.fixture.json` holds three
scenarios with cell-level `mu` arrays and complete expected results, and both
implementations are tested against it to **1e-6 on `mu`**. The browser is a port, not a
re-derivation.

---

## Learning from a launch

After a launch, real weeks arrive. Ripple updates with Bayes' rule: **the fitted model
is the prior, the new weeks are the likelihood.**

1. For each posterior draw (1000), draw K = 20 values of the one unknown → 20,000 joint
   prior samples. The other parameters stay at their posterior draw, so the prior over
   the joint parameter vector *is* the fitted posterior × the unknown's prior.
2. Compute expected units in the observed cells and the Negative Binomial log-likelihood
   of what actually happened.
3. Self-normalized importance weights `w ∝ exp(loglik)`, via log-sum-exp.
   **Pareto-smoothed** (PSIS) on the server; the browser reports weight ESS.
4. Every summary is a weighted quantile.

| Launch type | The unknown | Its prior |
|---|---|---|
| New city | `a_new = g0 + X_new·g + tau_city·z` | `z ~ N(0,1)` — the hierarchical prior |
| New offer | `b_promo[s]` for *this* offer | the fitted posterior, widened for a new offer |
| A feature never seen before | log-lift `theta` | **skeptical** `N(0, 0.15)`, adjustable in the UI |

**The gate.** This is exact Bayes only while the weights behave. When the launch weeks
are too surprising, a handful of samples carry all the weight and the answer is noise.
So Ripple measures it: **Pareto k̂ > 0.7** (server) or **weight ESS < 300** (browser)
flips the badge to **"Refit needed"** and the full model is refitted with the new rows.

The limit it cannot fix: the *other* parameters are not re-learned from the launch weeks.
The refit path does that.

### Is the lift causal?

The lift is observed outcome vs. the model's counterfactual for the same cells and weeks.
That is causal only if (1) nothing else changed in those weeks, (2) untreated cities and
groups were unaffected, (3) the demand pattern was stable. The `/launch` page shows all
three as a checklist instead of hiding them, because the model cannot verify any of them.

**The fix is in your hands:** keep a random holdout group — say 10% of users get no
offer. With a `holdout` column, Ripple uses those cells as the control and labels the
result **"Experiment"** instead of **"Model-based estimate"**. The gift-boxes demo launch
has one, so you can see both labels side by side.

---

## Evidence

### Trust checks — 6 of 6 pass

`nutpie NUTS, 4 chains × 1000 draws, target_accept 0.99`, fixed seed.

| Check | Threshold | Result |
|---|---|---|
| R-hat | ≤ 1.01 | **1.0100** (worst: `tau_city`) |
| Effective sample size | ≥ 400 | **612** |
| Divergences | 0 | **0** |
| Posterior predictive coverage | ≥ 85% | **91%** |
| **Parameter recovery** | truth in the 94% HDI | **30 of 30** |
| **Time holdout** (last 12 weeks) | ≥ 80% coverage, beat naive | **92%** coverage, **18.4%** MAPE vs **44.7%** naive |

`target_accept` is 0.99 deliberately. At the PyMC default the fit had 28 divergences out
of 4000; non-centring `e` and `b_promo` took it to 4, and 0.99 took it to 0.

**Parameter recovery is the check real data cannot give you**, and it is why the demo
company is synthetic rather than scraped: every planted value — all four elasticities,
all four promo lifts, the dip, the trait coefficients, the overdispersion — lands inside
its 94% interval.

### Calibration of the launch update — 200 synthetic launches per arm

The truth is drawn from the *same prior the update uses*, so a 90% interval **must**
cover it 90% of the time if the computation is right. This tests the machinery, not the
model's flattering side.

| Arm | Coverage, week 1 → 8 | Interval width | Bias |
|---|---|---|---|
| New city | **90% → 94%** (target 90%, MCSE ±2%) | −56% | ≈ 0 |
| New feature | **90% → 92%** | −56% | ≈ 0 |
| **New feature with a ramp-up** *(assumption broken)* | **28% → 69%, as low as 6%** | −44% | **−1,040 units** |

The third arm is the point. The model has no ramp-up term, so a launch that builds
slowly looks like a *weak* launch, and we publish exactly how bad that gets rather than
only showing the arms that pass. Worth noting: the refit gate fires on **14–42%** of
stress runs against **0–7%** of healthy ones — it partially catches the broken
assumption without being told about it.

---

## The agent

Opens from a ⌘K command bar. Max 5 tool calls, then it answers.

| Tool | Does | Source of truth |
|---|---|---|
| `build_scenario(question)` | fills the scenario schema, zod-validated; asks one question if ambiguous | — |
| `run_scenario(scenario)` | runs `simulate.ts` on the active posterior | model |
| `model_facts()` | elasticities, promo lift, dip, trust checks | model |
| `market_scan(product, city?)` | Nimble search → extracts `{competitor, price, offer, url, seen_at}` | **web, cited** |
| `explain_method(question)` | RAG over `docs/method/*.md` | **docs, cited** |

**The number guard.** Every number in the answer text must match a `claim` returned by a
tool. If one does not, the template answer is shown instead. `model_facts` is served
from pre-summarized values in `trust.json`, so the agent never computes a figure either.

**Nimble is context, not an input.** After the model answers, `market_scan` adds *"your
new price $25.92 is above 2 of 3 competitors in Austin"* with links, labelled as web
data. A price decision is not made in isolation — but letting scraped text into the
likelihood would be indefensible, so it stays out.

**Prompt-injection guard.** Web text is passed inside delimiters as data, the agent is
instructed to ignore instructions found in it, and there are no write tools.

---

## Use your own data

Long-format CSV, one row per week × city × segment:

| Column | Required | Notes |
|---|---|---|
| `week` | yes | date of week start |
| `units` | yes | count |
| `price` | yes | price paid per unit |
| `city` | no | defaults to "All" |
| `segment` | no | defaults to "All" |
| `promo` | no | 0/1 |
| `feature` | no | 0/1 |
| `holdout` | no | 0/1 — present ⇒ the lift is an **experiment** |

Optional `cities.csv` with `log_population, median_income_k, share_age_18_34` enables
new-city scenarios, and `mix_<segment>` columns supply the customer mix.

**The model builds itself from the columns present.** No `segment` → one group. No
`promo` → promo terms off and promo scenarios disabled. No traits → new-city scenarios
disabled, and the UI says why. Validation warns about the things that silently ruin a
fit — above all **"price never changes → elasticity cannot be learned"**.

**Privacy:** no customer rows go to the LLM by default. The mapping assistant gets column
names only, unless you tick "share sample rows". Uploaded rows live in the API's temp
storage for the session, and `DELETE /datasets/{id}` removes them from disk — a test
asserts the file is actually gone, not just unindexed.

### The API

```
GET    /health                    status, and whether the model can be imported
POST   /datasets                  upload CSV(s) -> dataset_id + preview + detected mapping
POST   /datasets/{id}/validate    a column mapping -> validation report
POST   /datasets/{id}/fit         start a background fit -> job_id
GET    /jobs/{id}                 status, step, progress, then posterior + trust
DELETE /datasets/{id}             the delete button
```

```bash
make up            # docker compose, then http://localhost:8000/docs
make api           # or run it locally with reload
```

It imports `ripple_model` directly, so a customer's fit, checks and export are the **same
code** that produced the demo — there is no second implementation to drift. On an upload
there is no planted truth and no holdout refit inside the 90-second budget, so those two
cards are marked **skipped** rather than quietly passing.

---

## Design decisions and limits

- **Synthetic demo data**, fixed seed, planted truth — a deliberate trade that makes
  parameter recovery provable. City traits are ACS-shaped but **illustrative**, not
  pulled live.
- **No competitor reaction.** Ripple answers what happens if *you* change price.
- **Effects assumed stable** over the horizon.
- **Extrapolation is flagged, not hidden.** Miami is out of the training range on income
  and age; Ripple names the traits and says the interval is a *lower bound* on the real
  uncertainty. It is in the demo to show the warning, not as a recommendation.
- **We wrote our own PSIS.** The installed ArviZ returned k̂ = 22.1 on log weights whose
  raw ESS was 4,805 of 20,000 and which recovered the planted truth, collapsing them to
  ESS 1. A k̂ of 22 is not a value the diagnostic can take, so it could not be the gate.
  `model/ripple_model/psis.py` follows Vehtari et al. with the Zhang–Stephens GPD fit,
  agrees with ArviZ to within 0.1 wherever ArviZ behaves, and has 11 tests of its own.
  See `DECISIONS.md`.

---

## Quickstart

```bash
uv sync                               # Python: model + API
uv run pytest                         # 73 tests: contract, data, parity, PSIS, launch update

uv run python -m ripple_model.synth           # regenerate the demo company
uv run python model/scripts/run_fits.py       # fit (~5 min, 4 chains x 1000)
uv run python model/scripts/export_demo.py    # -> web/public/model/
uv run python model/scripts/run_calibration.py

cd web && npm install && npm run dev
```

**No API key is needed.** The live demo runs in demo mode with a cached agent and a
cached market scan. Add keys to `.env` (see `.env.example`) to turn on live Gemini and
Nimble.

## Continuous integration

`.github/workflows/ci.yml` runs on every push:

- **ruff** over `model/`, `api/`, `contract/`;
- **`scripts/check_fixtures.py`** — regenerates every fixture and fails if a committed one
  no longer matches its generator. A fixture edited by hand, or a `simulate.py` change
  that was not followed by regenerating the parity fixture, would leave the parity test
  comparing new code against expectations derived from new code: green and worthless. The
  comparison is **numeric, not byte-for-byte** — float64 differs in the last digit or two
  between CPUs and BLAS builds, and a 1e-16 relative difference is not a stale fixture;
- **pytest**, including the slow end-to-end BYOD upload → fit → poll flow, which is a
  real (tiny) PyMC fit;
- **tsc + vitest** in `web/`, including the Python↔TS parity tests and the number guard.

## Repository

```
contract/     schemas + the posterior and parity fixtures — the agreement between sessions
model/        ripple_model/{synth,data,model,checks,simulate,update,psis,calibration,export}.py
api/          FastAPI BYOD endpoints
web/          Next.js app, browser simulator, agent route
docs/method/  model card, checks, limits — RAG source and the source of this README
```

## AI tools used

All code in this repository was written during the AI Engineer Hackathon NYC on 2026-10-11. No code was
carried over from earlier projects.

- **Claude Code** (Claude Opus 5) — wrote all the code during the event, in two parallel
  sessions: one on `contract/` + `model/` + `api/`, one on `web/`.
- **Google Gemini** — the "Ask Ripple" agent (function calling), the CSV mapping
  assistant, and embeddings for the method RAG.
- **Nimble** — live web search for competitor prices and current offers.

## License

MIT — see [LICENSE](LICENSE).
