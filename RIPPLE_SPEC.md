# Ripple — 3-Hour Hackathon Build Spec (v4)
### "What happens to sales if…?" · Bayesian what-if simulator + launch tracker that learns from real data + market-aware AI agent + bring your own data

> **For the coding agent (Claude Code):** Read all of this file before you write code. Total time: **3 hours.** The work runs in **two parallel Claude Code sessions** (Section 12): Session A owns `model/` and `api/`; Session B owns `web/`. They share one contract (Section 7). Commit after each step. When this spec does not decide something, choose the simplest option and write one line in `DECISIONS.md`. **Do not invent API calls:** check the installed versions and official docs of PyMC, nutpie, FastAPI, Next.js, `@google/genai`, and Nimble first. Write all code new. **Never cut or skip anything without Reza's approval** (see CLAUDE.md, "Approval rule").

---

## 1. Product

Ripple answers business "what if" questions with a Bayesian demand model:

- **Promotion / coupon:** "What if we give students in Boston 20% off for 4 weeks?"
- **Price:** "What if we raise the price 8% in all cities?"
- **New city / new customers:** "What if we launch in Austin? Which customer groups will buy?"

Each answer is a **distribution**: incremental units, revenue, and profit vs. a no-change baseline, a 90% interval, and the **probability that the change pays**.

Three layers:
1. **Model (the numbers):** hierarchical Bayesian demand model, fitted in Python. Simulation runs in the browser, so sliders update instantly.
2. **Agent (the conversation):** "Ask Ripple" — a Gemini tool-calling agent turns a plain-English question into a scenario, runs it, checks the live market with **Nimble** (competitor prices and current offers, with sources), and explains the result. **The agent never writes a number that a tool did not return.**
3. **Your data:** a company uploads a CSV; an LLM mapping assistant maps its columns; the server validates and fits the model; the same UI then works on the company's own data.
4. **Launch tracker (learn from reality):** after a launch, each new week of real data updates the forecast with Bayes' rule (Section 5b). The interval narrows, the verdict updates, and a calibration study proves the intervals are honest.

Principle shown in the UI: **"The model makes the numbers. The agent explains them. The web gives context."**

## 2. Demo data — synthetic company with planted truth

`model/synth.py` → "Brightside Coffee Subscriptions" (fictional), fixed seed, truth in `model/data/truth.json`.
- 6 training cities (New York, Chicago, Boston, Seattle, Atlanta, Denver) with traits `log_population`, `median_income_k`, `share_age_18_34` (Census ACS values if quick; else mark *illustrative*).
- 4 segments: students, young professionals, families, retirees; segment shares per city.
- 104 weeks, base price $24. Random promotions (discount 0/10/20/30%, 1–4 weeks, sometimes one segment only), `feature` email flag.
- Planted effects: segment elasticities (students ≈ −2.2 … retirees ≈ −0.8), promo deal-signal lift, post-promo dip, yearly seasonality, trait-driven city effects + noise.
- 3 candidate cities: Austin, Portland (in range), Miami (one trait out of range → extrapolation warning).
- The output is written in the **same CSV format a customer uploads** (Section 6), so demo and BYOD share one path.

Why synthetic: true values are known, so we **prove** recovery. Real public data is a stretch only.

## 3. Model (`model/ripple_model/model.py`, PyMC)

```
y[c,s,t] ~ NegativeBinomial(mu, phi)
log mu = log(share[c,s]) + a_city[c] + a_seg[s]
       + e[s] * log(price/base_price)          # elasticity per segment (partially pooled)
       + b_promo[s] * promo                    # deal signal (partially pooled)
       + b_feature * feature
       + b_dip * promo_last_week               # pull-forward dip
       + season(t)                             # 2 Fourier pairs
a_city[c] = g0 + X_city[c]·g + tau_city * z_city[c]
e[s] ~ Normal(mu_e, sigma_e);  mu_e ~ Normal(-1.5, 0.5);  sigma_e ~ HalfNormal(0.5)
```
- The model builds itself from the columns present: no `segment` column → one segment; no `promo` → no promo terms; no city traits → new-city scenarios disabled (the UI says why).
- Demo fit: nutpie (if installed) or NUTS, 4 chains × 1000. BYOD fit: 2 chains × 500, time limit 90 s; if over the limit, use Pathfinder/ADVI and label the run "fast approximation".
- README table "why each part" (NegBin for overdispersed counts; log-log = elasticity; partial pooling; dip prevents overrating promos; hierarchical city effect enables new cities; share offset handles new customer mix).

## 4. Trust checks (`model/ripple_model/checks.py`) → `trust.json`
Convergence (R-hat ≤ 1.01, ESS ≥ 400, 0 divergences) · posterior predictive coverage ≥ 85% · **parameter recovery** (demo only: truth inside 94% HDI) · **time holdout** (last 12 weeks: 90% coverage + MAPE vs. naive seasonal baseline). Each check: `{name, passed, value, threshold, plain_english}`.

## 5. Scenario math
Per posterior draw: baseline `mu` vs. scenario `mu` over the same cities/segments/weeks → units → `revenue = units × net_price`, `profit = units × (net_price − unit_cost) − fixed_cost` → incremental per draw → mean, 90% interval, `P(profit > 0)`, segment and weekly breakdown. New city: `a_city = g0 + X_new·g + tau_city·z`, `z ~ N(0,1)`; extrapolation flag if a standardized trait is outside training min–max.
Implemented in `model/ripple_model/simulate.py` (reference) and `web/lib/simulate.ts` (browser). **Parity test** on a shared fixture (tolerance 1e-6 on `mu`).

## 5b. Learn from launch data — "Launch tracker" (Bayesian updating)

After a launch (new city, new offer, or new feature), real weekly data arrives. Ripple updates the forecast **with Bayes' rule**: the fitted model is the prior; the new weeks are the likelihood; the result is the posterior. The interval gets narrower each week and moves toward the truth.

### Unknown parameter per launch type
| Launch | Unknown | Prior (from the fitted model) |
|---|---|---|
| New city | city effect `a_new = g0 + X_new·g + tau_city·z` | `z ~ N(0,1)` per posterior draw → the hierarchical prior |
| New offer / promo | segment promo lift `b_promo[s]` and elasticity `e[s]` | posterior of the existing model (learned from history) |
| New feature (never seen) | log-lift `theta` on demand in the treated cells | **skeptical prior** `theta ~ Normal(0, 0.15)` (shown in the UI; the user can change the scale) |

### Update engine (`web/lib/update.ts` + `model/ripple_model/update.py`, same math, parity test)
1. Joint prior samples: for each posterior draw `d` (1000), draw `K = 20` values of the unknown → 20,000 samples.
2. For each sample, compute expected units for the observed launch weeks and cells, and the **Negative Binomial log-likelihood** of the observed units.
3. **Self-normalized importance weights** `w ∝ exp(loglik)` (log-sum-exp for stability). In Python, smooth them with **PSIS** (Pareto-smoothed importance sampling, Vehtari et al.; `arviz` provides it). In the browser, compute the effective sample size of the weights.
4. Posterior summaries = weighted quantiles: updated 12- or 26-week forecast, updated `P(profit > 0)`, and the posterior of the unknown (for example "the feature lifts demand by +6% (90%: +2% to +10%)").
5. **Validity gate:** if weight ESS < 300 (browser) or Pareto `k̂ > 0.7` (Python), the fast update is not reliable → show "Refit needed", and the API refits the full model with the new rows (same code path as BYOD).

Why this is exact Bayes, not a heuristic: the other parameters stay at their posterior draws, the unknown comes from its prior, and importance weighting by the likelihood targets the posterior of the joint model. A standard diagnostic says when the approximation fails. (Limit: the existing parameters are not re-learned from the launch weeks; the refit path does that.)

### Causal note for offers and features (in the UI, short)
The "lift" is **observed vs. counterfactual** (the model's prediction with no change, for the same cells and weeks). It is causal only if: (1) no other change happened at the same time, (2) untreated cities and groups were not affected, (3) the demand pattern is stable. The UI shows these 3 assumptions as a checklist and recommends the gold standard: **keep a random holdout group** (for example 10% of users get no offer). If the CSV has a `holdout` column (0/1), Ripple uses holdout cells as the control and labels the result "experiment" instead of "model-based estimate".

### Proof of validity — calibration study (`model/ripple_model/calibration.py`)
No model is "100% valid". What we **prove** is that the method is **calibrated** when its assumptions hold, and we **show where it fails**:
- Simulation-based calibration: generate 200 synthetic launches with known truth (new city and new feature), reveal weeks 1…8, run the update each week.
- Report: **coverage** of the 90% interval for the true 12-week total and the true lift at each week (target ≈ 90%), interval width by week, bias, and the share of runs that hit the refit gate.
- **Stress test:** break one assumption (a launch with a ramp-up that the model does not have) and show that coverage drops. This shows that we know the limits.
- Output `calibration.json` → chart on `/method` and in the README.

### UI — `/launch` page
- Pick a launch: "Austin launch", "Student coupon", "New feature: gift boxes" (synthetic, truth known), or "Upload launch weeks" (CSV, same columns).
- **Week slider "Real data so far: week 0 … 8".** As it moves: observed points appear on the chart, the forecast band narrows and shifts, the verdict line updates ("Chance it pays: 61% → 78%"), and a small "prior vs. posterior" density plot of the unknown animates.
- Badges: gate status ("Fast update reliable" / "Refit needed"), causal checklist, "Show truth" toggle (demo only) that draws the true value as a line.
- Agent tool `update_with_launch(launch_id, weeks)` so "Ask Ripple" can answer "After 4 weeks in Austin, is the launch on track?"

## 6. Bring your own data (BYOD)

**CSV (long format), one row per week × city × segment:**
| Column | Required | Notes |
|---|---|---|
| `week` | yes | date of week start |
| `units` | yes | count |
| `price` | yes | price paid per unit |
| `city` | no | default "All" |
| `segment` | no | default "All" |
| `promo` | no | 0/1 |
| `feature` | no | 0/1 |
Optional second CSV `cities.csv`: `city, log_population, median_income_k, share_age_18_34` (enables new-city scenarios). Download links for templates + the demo CSV in the UI.

**Flow (`/data` page):** drop file → **Mapping assistant** (Gemini gets column names + 3 sample rows *only if the user ticks "share sample rows"*, else names only) returns `{mapping, confidence, notes}` → user confirms in a mapping table → validation report (row count, weeks, missing values, negative units, price variation; "price never changes → elasticity cannot be learned" warning) → **Fit** (POST to API, progress steps) → trust cards → "Open in simulator".
Privacy: files stay in the API's temp storage for the session; no rows go to the LLM by default; delete button.

## 7. Contract between sessions (write this first, in `contract/`)
- `contract/posterior.schema.json` and a small fixture `contract/posterior.fixture.json` (50 draws, 2 cities, 2 segments). Fields: `meta {cities, segments, weeks, base_price, unit_cost, trait_names, trait_mean, trait_std, trait_min, trait_max, has_promo, has_traits, method}`, `draws {g0[], g[][], tau_city[], a_city[][], a_seg[][], e[][], b_promo[][], b_feature[], b_dip[], season[][], phi[]}`, `history`, `candidates`.
- `contract/trust.schema.json`, `contract/scenario.schema.json` (`{type: promo|price|new_city, cities, segments, discount, weeks, price_change, candidate | traits, unit_cost, fixed_cost, horizon}`), `contract/result.schema.json`.
- Session B builds against the fixture and switches to the real export later.

## 8. API (`api/`, FastAPI, Python) — needed only for BYOD
| Method | Path | Does |
|---|---|---|
| GET | `/health` | status |
| POST | `/datasets` | upload CSV(s) → `dataset_id`, preview |
| POST | `/datasets/{id}/validate` | mapping → validation report |
| POST | `/datasets/{id}/fit` | background fit → `job_id` |
| GET | `/jobs/{id}` | status, step, then `posterior.json` + `trust.json` |
Docker image. Deploy to Render, Fly.io, or Hugging Face Spaces if time allows; otherwise BYOD runs locally (`make up`) and the live link is demo mode. The mapping assistant runs in a Next.js route (Gemini), not in the API.

## 9. Agent — "Ask Ripple" (`web/app/api/agent/route.ts`, Gemini function calling)
Opens from a **⌘K command bar** and a floating "Ask" button. Max 5 tool calls, then answer.

| Tool | Does | Source of truth |
|---|---|---|
| `build_scenario(question)` | Gemini fills the scenario schema; zod-validated; ambiguous → ask one question | — |
| `run_scenario(scenario)` | Runs `simulate.ts` on the active posterior | Model |
| `model_facts()` | Elasticities by segment, promo lift, dip, trust checks | Model |
| `market_scan(product, city?)` | **Nimble** search for competitor prices and current offers → Gemini extracts `{competitor, price, offer, url, seen_at}` (zod) | Web, cited |
| `explain_method(question)` | Small RAG over `docs/method/*.md` (model card, checks, limits; ~20 chunks, Gemini embeddings, cosine top-3) | Docs, cited |

**Answer schema:** `{text, claims: [{value, label, tool_call_id}], sources: [{title, url}], scenario, open_question?}`.
**Number guard:** every number in `text` must match a `claim` from a tool result; else the template answer.
**How Nimble is meaningful:** a price or promo decision is not made in isolation. After the model answers, `market_scan` shows *"Your new price $25.92 is above 2 of 3 competitors in Austin"* with links, and flags it as context (the model does not use it as an input). Results are cached by query; demo mode uses a cached scan.
**Safety:** web text is data inside delimiters; the agent ignores instructions in it; no write tools.
Example prompts (chips): "Should we run a 20% student coupon in Boston?" · "What if we raise prices 8%?" · "Is Miami a good next city?" · "Why should I trust this?"

## 10. UI (`web/`, Next.js App Router + TypeScript + Tailwind + shadcn/ui + Recharts + Framer Motion + KaTeX)

### 10.1 Design system — minimal and clean
- **Color:** background `#FFFFFF`, surface `#FAFAFA`, border `#EAEAEA`, text `#0A0A0A`, muted `#737373`. **One accent:** `#4F46E5`. Gain `#16A34A`, loss `#DC2626`, warning `#D97706` — used only on numbers and badges. Interval bands = accent at 10% / 20% opacity.
- **Type:** Geist Sans (or Inter) for UI, Geist Mono for numbers (tabular). Sizes: 12 / 14 / 16 / 20 / 32 / 48. Big numbers 48px, weight 600.
- **Layout:** max width 1200px, 8px grid, generous white space, 1px borders, radius 12px, no heavy shadows, no gradients. Light mode first; dark mode via tokens if free.
- **Motion:** 150–250 ms ease-out; numbers count up; chart bands morph between states; no decorative animation.
- **Accessibility:** keyboard navigation, visible focus ring, contrast AA, labels on all sliders.

### 10.2 Pages
1. **`/` Simulator (home).**
   - Top bar: wordmark "ripple", dataset switcher ("Demo · Brightside Coffee" / your datasets), "⌘K Ask", "Use your data".
   - Left column (320px): segmented control **Promotion · Price · New city**, then only the controls for that type. Preset chips at the top.
   - Main: **verdict line** ("Incremental profit **+$18.4k** · 90% −$3.1k to +$41.0k · **84%** chance it pays"), a thin probability bar, then the **fan chart** (baseline band in grey, scenario band in accent, promo weeks shaded, dip visible), then **segment bars** with intervals. Extrapolation banner when flagged. Market context card (from the last `market_scan`) under the chart, with sources.
2. **Ask panel** (right sheet / ⌘K). Chat with live step chips ("building scenario · running model · scanning market"), claim chips (hover → tool + value), source links, "Open this scenario" button (loads it into the simulator).
3. **`/data` Your data.** Drop zone → mapping table with confidence dots → validation checklist → fit progress (steps) → trust cards → "Open in simulator".
4. **`/method` How it works.** Equation (KaTeX), "why each part" table, trust cards, parameter-recovery chart (true value vs. 94% HDI), limits. Short and readable.

### 10.3 States
Skeletons, empty states with one clear action, error toasts with a fix ("price never varies — add weeks with different prices"), offline/demo pill.

## 11. Repository
```
ripple/
├── README.md  LICENSE  DECISIONS.md  RIPPLE_SPEC.md  Makefile  docker-compose.yml  .env.example
├── contract/        schemas + fixture
├── model/           uv project: ripple_model/{synth,model,simulate,update,calibration,checks,export}.py, data/, tests/
├── api/             FastAPI app (imports ripple_model), Dockerfile
├── web/             Next.js: app/(/, /launch, /data, /method, api/agent, api/map), components/, lib/{simulate,update,gemini,nimble,rag}.ts, public/model/, tests/
├── docs/method/     model card, checks, limits (RAG source + README source)
├── scripts/demo/    record.ts, narration.txt
├── assets/          demo.gif, demo.mp4, screenshots/
└── .github/workflows/ci.yml   ruff + pytest (tiny fit) + vitest (simulate + update parity, number guard)
```

## 12. Plan — 3 hours, two parallel sessions

| Time | Session A — `model/` + `api/` | Session B — `web/` |
|---|---|---|
| 0:00–0:15 | Repo, `contract/` schemas + fixture (A writes, B reviews) | Next.js + Tailwind + shadcn + fonts + tokens |
| 0:15–0:40 | `synth.py`, demo CSV, truth, **3 synthetic launches** (Austin, student coupon, gift-box feature: 8 weeks each, truth known) | Simulator page layout on the fixture |
| 0:40–1:15 | Model fit, checks, export → `web/public/model/` | `simulate.ts` + parity test; verdict, fan chart, segment bars |
| 1:15–1:45 | `update.py` (PSIS gate) + `calibration.py` (vectorized, 200 launches) → `calibration.json`; export launch data | `update.ts` + parity test; `/launch` page with week slider |
| 1:45–2:10 | FastAPI BYOD endpoints + Dockerfile; `docs/method/*.md` | Agent route (`build_scenario`, `run_scenario`, `model_facts`, `update_with_launch`, `market_scan`); Ask panel |
| 2:10–2:25 | Help B: Nimble cached scan; README draft | `/data` page (mapping, validation); `/method` page with calibration chart |
| 2:25–2:35 | Deploy API (optional) | Deploy web to Vercel (demo mode, cached scan) |
| 2:35–3:00 | README final, screenshots | Playwright video + TTS + `ffmpeg`; submit |

**Cut list (cut in this order):** API deploy (BYOD local only) → `explain_method` RAG → dark mode → BYOD fit (keep upload + mapping + validation) → "Upload launch weeks" (keep 3 demo launches) → stress test → live Nimble (keep cached scan) → `/data` page.
**Never cut:** intervals, incremental vs. baseline, recovery + holdout checks, **launch tracker with the PSIS/ESS gate**, **calibration coverage chart**, number guard, live demo link, README, video.

## 13. README outline
1. Wordmark, one-line pitch, GIF, **live link**, video, badges.
2. Problem: promotions, prices, and new markets get decided on one number with no uncertainty and no market context.
3. Screens: simulator, Ask Ripple with market sources, your-data flow, method.
4. **Model** (equation + why-each-part table), **scenario math**, **launch updating** (Bayes' rule, importance sampling + PSIS gate, causal assumptions, holdout groups), **evidence** (trust table, recovery chart, holdout vs. naive, **calibration coverage by week**).
5. **Agent design:** tools table; "numbers only from tools" + number guard; Nimble as cited context; prompt-injection guard.
6. **Use your own data:** CSV format, privacy, run with Docker.
7. Design decisions and limits (synthetic demo; no competitor reaction in the model; stable effects; extrapolation).
8. Quickstart. **AI tools used** (Claude Code, Gemini). Built at the AI Engineer Hackathon NYC, 2026-10-11. Roadmap. MIT.

## 14. Demo video with no manual recording (≈ 100 s)
Playwright (1440×900, video on): home → preset "Student coupon in Boston" → drag discount 10% → 30% (numbers and bands move) → `/launch` "Austin launch": drag the week slider 0 → 8 (band narrows, "chance it pays" updates, prior → posterior animates, truth line) → ⌘K "Is Miami a good next city?" (steps, verdict, extrapolation warning, market sources) → `/data` with the sample CSV (mapping, validation, trust cards) → `/method`. TTS narration, **line 1: "Hi, I'm Reza, and this is Ripple, my project for the AI Engineer Hackathon NYC."** `ffmpeg` → `assets/demo.mp4` + 15 s `assets/demo.gif`.

## 15. Rules for the coding agent
- Fixed seeds. Never show a number without its interval.
- The LLM never creates a number; numbers come from the posterior or from cited web results (labeled as web).
- No keys in the repo; the live demo needs no key (cached agent answers for the 4 example prompts if no key).
- No customer rows to the LLM by default.
- Write code only after the hackathon start time.
