// Bayesian launch update in the browser. Twin of model/ripple_model/update.py (spec Section 5b).
//
// prior = fitted posterior draws x K values of ONE unknown; likelihood = Negative Binomial of the
// observed launch weeks; posterior = self-normalized importance weights. Every summary is a
// weighted quantile. Weights are Pareto-smoothed (PSIS) as in update.py; gate: k-hat > 0.7 or weight ESS < 300
// means "Refit needed".
import { drawZ, seasonRow } from "./simulate";
import { K_HAT_THRESHOLD, psisSmooth } from "./psis";
import type { Interval, Launch, Posterior, UpdateGate, UpdateResult, UpdateWeeklyRow } from "./types";

export const K_PER_DRAW = 20;
export const ESS_MIN = 300;
export const DEFAULT_SEED = 20261011;
export const NEW_OFFER_SIGMA = 0.25;
const DENSITY_POINTS = 60;

const HOLDOUT_RECOMMENDATION =
  "Keep a random holdout group next time (for example 10% of users get no offer). " +
  "Then this is an experiment, not a model-based estimate.";

// ---- special functions ----------------------------------------------------
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/** log Gamma(x), Lanczos (g=7). Accurate to ~1e-14 for x > 0. */
export function gammaln(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - gammaln(1 - x);
  const xm = x - 1;
  let a = LANCZOS[0];
  const t = xm + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (xm + i);
  return 0.5 * Math.log(2 * Math.PI) + (xm + 0.5) * Math.log(t) - t + Math.log(a);
}

/** log p(y | mu, phi), NegBin with var = mu + mu^2/phi (same terms as nb_loglik in update.py). */
export function nbLogPmf(y: number, mu: number, phi: number): number {
  const m = Math.max(mu, 1e-12);
  const lphim = Math.log(phi + m);
  return (
    gammaln(y + phi) -
    gammaln(phi) -
    gammaln(y + 1) +
    phi * (Math.log(phi) - lphim) +
    y * (Math.log(m) - lphim)
  );
}

// ---- weights and weighted summaries --------------------------------------
export function normalizeLogWeights(logw: Float64Array): Float64Array {
  let m = -Infinity;
  for (let i = 0; i < logw.length; i++) if (logw[i] > m) m = logw[i];
  const w = new Float64Array(logw.length);
  if (!Number.isFinite(m)) return w.fill(1 / logw.length);
  let total = 0;
  for (let i = 0; i < logw.length; i++) {
    w[i] = Math.exp(logw[i] - m);
    total += w[i];
  }
  if (!(total > 0)) return w.fill(1 / logw.length);
  for (let i = 0; i < w.length; i++) w[i] /= total;
  return w;
}

/** Kish effective sample size of the weights. */
export function weightEss(w: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i];
  if (!(s > 0)) return 0;
  let q = 0;
  for (let i = 0; i < w.length; i++) q += (w[i] / s) ** 2;
  return 1 / q;
}

/** numpy.interp with clamped ends. */
function interp(q: number, xp: Float64Array, fp: Float64Array): number {
  const n = xp.length;
  if (q <= xp[0]) return fp[0];
  if (q >= xp[n - 1]) return fp[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= q) lo = mid;
    else hi = mid;
  }
  const dx = xp[lo + 1] - xp[lo];
  if (dx === 0) return fp[lo];
  return fp[lo] + ((fp[lo + 1] - fp[lo]) * (q - xp[lo])) / dx;
}

/** Weighted quantiles with the midpoint-CDF convention of update.py (equals numpy "linear" at equal weights). */
export function weightedQuantiles(x: ArrayLike<number>, w: ArrayLike<number>, qs: number[]): number[] {
  const n = x.length;
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => x[a] - x[b]);
  const xs = new Float64Array(n);
  const cdf = new Float64Array(n);
  let cum = 0;
  let total = 0;
  for (let i = 0; i < n; i++) total += w[i];
  if (!(total > 0)) return qs.map(() => NaN);
  for (let i = 0; i < n; i++) {
    const wi = w[order[i]];
    cum += wi;
    xs[i] = x[order[i]];
    cdf[i] = (cum - 0.5 * wi) / total;
  }
  return qs.map((q) => interp(q, cdf, xs));
}

export function wInterval(x: ArrayLike<number>, w: ArrayLike<number>): Interval {
  let sw = 0;
  let swx = 0;
  for (let i = 0; i < x.length; i++) {
    sw += w[i];
    swx += w[i] * x[i];
  }
  const [p05, p50, p95] = weightedQuantiles(x, w, [0.05, 0.5, 0.95]);
  return { mean: swx / sw, p05, p50, p95 };
}

export function wBand(x: ArrayLike<number>, w: ArrayLike<number>) {
  const [p05, p25, p50, p75, p95] = weightedQuantiles(x, w, [0.05, 0.25, 0.5, 0.75, 0.95]);
  return { p05, p25, p50, p75, p95 };
}

/** Weighted Gaussian KDE on a fixed grid, Silverman bandwidth (weighted_density in update.py). */
export function weightedDensity(x: ArrayLike<number>, wRaw: ArrayLike<number>, grid: number[]): number[] {
  const n = x.length;
  let sw = 0;
  for (let i = 0; i < n; i++) sw += wRaw[i];
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = wRaw[i] / sw;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += w[i] * x[i];
  let varr = 0;
  for (let i = 0; i < n; i++) varr += w[i] * (x[i] - mean) ** 2;
  varr = Math.max(varr, 1e-12);
  const bw = 1.06 * Math.sqrt(varr) * Math.max(weightEss(w), 2) ** -0.2;
  const norm = 1 / (bw * Math.sqrt(2 * Math.PI));
  return grid.map((g) => {
    let acc = 0;
    for (let i = 0; i < n; i++) {
      if (w[i] === 0) continue;
      const z = (g - x[i]) / bw;
      acc += Math.exp(-0.5 * z * z) * w[i];
    }
    return acc * norm;
  });
}

// ---- design ---------------------------------------------------------------
interface Design {
  W: number;
  n: number;
  price: Float64Array; // [W*n]
  promo: Float64Array;
  promoLast: Float64Array;
  feature: Float64Array;
  treated: Float64Array;
  units: Float64Array;
  t: number[]; // [W]
  segIdx: number[]; // [n]
  logShare: Float64Array; // [n]
  aCityIdx: number[] | null; // [n], null for a new city
}

function armFraction(launch: Launch, treated: number): number {
  if (!launch.has_holdout) return 1;
  return treated === 1 ? 0.9 : 0.1;
}

function cellsAndDesign(post: Posterior, launch: Launch, weeks: number[]): Design {
  const o = launch.observations;
  const wanted = new Set(weeks);
  const keep: number[] = [];
  for (let i = 0; i < o.week.length; i++) if (wanted.has(o.week[i])) keep.push(i);

  const cols: [number, number, number][] = [];
  const pos = new Map<string, number>();
  for (const i of keep) {
    const key = `${o.city_idx[i]}|${o.segment_idx[i]}|${o.treated[i]}`;
    if (!pos.has(key)) {
      pos.set(key, cols.length);
      cols.push([o.city_idx[i], o.segment_idx[i], o.treated[i]]);
    }
  }
  const W = weeks.length;
  const n = cols.length;
  const mk = () => new Float64Array(W * n);
  const d: Design = {
    W,
    n,
    price: mk().fill(post.meta.base_price),
    promo: mk(),
    promoLast: mk(),
    feature: mk(),
    treated: mk(),
    units: mk(),
    t: new Array(W).fill(0),
    segIdx: cols.map((c) => c[1]),
    logShare: new Float64Array(n),
    aCityIdx: null,
  };
  const weekPos = new Map(weeks.map((w, i) => [w, i]));
  for (const i of keep) {
    const wi = weekPos.get(o.week[i])!;
    const ci = pos.get(`${o.city_idx[i]}|${o.segment_idx[i]}|${o.treated[i]}`)!;
    const k = wi * n + ci;
    d.price[k] = o.price[i];
    d.promo[k] = o.promo[i];
    d.promoLast[k] = o.promo_last_week[i];
    d.feature[k] = o.feature[i];
    d.treated[k] = o.treated[i];
    d.units[k] = o.units[i];
    d.t[wi] = o.t[i];
  }
  const isNew = cols.some((c) => c[0] < 0);
  if (isNew) {
    const shares = launch.new_city!.shares;
    cols.forEach((c, j) => (d.logShare[j] = Math.log(shares[c[1]])));
  } else {
    cols.forEach((c, j) => {
      d.logShare[j] = Math.log(post.meta.shares[c[0]][c[1]] * armFraction(launch, c[2]));
    });
    d.aCityIdx = cols.map((c) => c[0]);
  }
  return d;
}

function forecastDesign(post: Posterior, launch: Launch, horizon: number): Design {
  const d1 = cellsAndDesign(post, launch, [1]);
  const t0 = launch.t0 ?? d1.t[0];
  const n = d1.n;
  const rep = (a: Float64Array) => {
    const out = new Float64Array(horizon * n);
    for (let h = 0; h < horizon; h++) out.set(a.subarray(0, n), h * n);
    return out;
  };
  return {
    ...d1,
    W: horizon,
    price: rep(d1.price),
    promo: rep(d1.promo),
    promoLast: rep(d1.promoLast),
    feature: rep(d1.feature),
    treated: rep(d1.treated),
    units: new Float64Array(horizon * n),
    t: Array.from({ length: horizon }, (_, h) => t0 + h),
  };
}

// ---- the unknown ----------------------------------------------------------
export function drawUnknown(
  post: Posterior,
  launch: Launch,
  opts: { kPerDraw: number; seed: number; priorSigma?: number },
): { u: Float64Array; drawIdx: Uint32Array; sigma: number } {
  const D = post.meta.n_draws;
  const N = D * opts.kPerDraw;
  const { kind, prior } = launch.unknown;
  const z = drawZ(opts.seed, N);
  const drawIdx = new Uint32Array(N);
  for (let i = 0; i < N; i++) drawIdx[i] = Math.floor(i / opts.kPerDraw);
  const u = new Float64Array(N);

  if (kind === "z_new_city" || kind === "theta") {
    const sigma = opts.priorSigma ?? prior.sigma ?? (kind === "theta" ? 0.15 : 1);
    const mu0 = prior.mu ?? 0;
    for (let i = 0; i < N; i++) u[i] = mu0 + sigma * z[i];
    return { u, drawIdx, sigma };
  }
  if (kind === "b_promo" || kind === "elasticity") {
    const segs = post.meta.segments;
    const si = Math.max(0, segs.indexOf(launch.unknown.segment ?? ""));
    const src = kind === "b_promo" ? post.draws.b_promo : post.draws.e;
    const sigma = opts.priorSigma ?? prior.sigma ?? NEW_OFFER_SIGMA;
    for (let i = 0; i < N; i++) u[i] = src[drawIdx[i]][si] + sigma * z[i];
    return { u, drawIdx, sigma };
  }
  throw new Error(`unknown launch unknown kind: ${String(kind)}`);
}

/** log mu for every sample x week x cell, flat [N][W][n]. Mirrors _launch_log_mu. */
function launchLogMu(
  post: Posterior,
  launch: Launch,
  u: Float64Array,
  drawIdx: Uint32Array,
  des: Design,
  applyUnknown: boolean,
): Float64Array {
  const { meta, draws } = post;
  const kind = launch.unknown.kind;
  const base = meta.base_price;
  const N = u.length;
  const { W, n } = des;
  const out = new Float64Array(N * W * n);
  const rows = des.t.map((t) => seasonRow(t, meta.season_period, meta.n_fourier));
  const lp = Float64Array.from(des.price, (p) => Math.log(p / base));
  const seasonByDraw: Float64Array[] = [];
  const seasonFor = (d: number) =>
    (seasonByDraw[d] ??= Float64Array.from(rows, (r) => r.reduce((a, v, k) => a + v * draws.season[d][k], 0)));

  for (let i = 0; i < N; i++) {
    const d = drawIdx[i];
    const season = seasonFor(d);
    let aNew = 0;
    if (des.aCityIdx === null) {
      const tz = launch.new_city!.traits_z;
      let tp = 0;
      for (let p = 0; p < tz.length; p++) tp += draws.g[d][p] * tz[p];
      const zv = kind === "z_new_city" && applyUnknown ? u[i] : 0;
      aNew = draws.g0[d] + tp + draws.tau_city[d] * zv;
    }
    for (let w = 0; w < W; w++) {
      for (let c = 0; c < n; c++) {
        const s = des.segIdx[c];
        const k = w * n + c;
        const aCity = des.aCityIdx === null ? aNew : draws.a_city[d][des.aCityIdx[c]];
        let e = draws.e[d][s];
        let bPromo = draws.b_promo[d][s];
        if (kind === "b_promo" && applyUnknown) bPromo = u[i];
        if (kind === "elasticity" && applyUnknown) e = u[i];
        let lm =
          des.logShare[c] +
          aCity +
          draws.a_seg[d][s] +
          e * lp[k] +
          bPromo * (des.promo[k] * des.treated[k]) +
          draws.b_feature[d] * des.feature[k] +
          draws.b_dip[d] * des.promoLast[k] +
          season[w];
        if (kind === "theta" && applyUnknown) lm += u[i] * des.treated[k];
        out[(i * W + w) * n + c] = lm;
      }
    }
  }
  return out;
}

// ---- the update -------------------------------------------------------------
export interface UpdateOptions {
  kPerDraw?: number;
  seed?: number;
  priorSigma?: number;
  /** Override the unknown's prior samples (parity tests; numpy's RNG is not reproducible here). */
  u?: ArrayLike<number>;
  includeDensity?: boolean;
}

export function updateWithLaunch(
  post: Posterior,
  launch: Launch,
  weeksUsedIn: number,
  opts: UpdateOptions = {},
): UpdateResult {
  const started = typeof performance !== "undefined" ? performance.now() : Date.now();
  const kPerDraw = opts.kPerDraw ?? K_PER_DRAW;
  const seed = opts.seed ?? DEFAULT_SEED;
  const D = post.meta.n_draws;
  const horizon = launch.horizon;
  const weeksUsed = Math.max(0, Math.min(Math.trunc(weeksUsedIn), launch.weeks_observed));

  const drawn = drawUnknown(post, launch, { kPerDraw, seed, priorSigma: opts.priorSigma });
  const { drawIdx, sigma } = drawn;
  let u = drawn.u;
  if (opts.u) {
    if (opts.u.length !== u.length) throw new Error(`u override has ${opts.u.length} values, need ${u.length}`);
    u = Float64Array.from(opts.u);
  }
  const N = u.length;

  // likelihood of the revealed weeks
  let w: Float64Array;
  let paretoK: number | null = null;
  let nObsCells = 0;
  if (weeksUsed > 0) {
    const obsWeeks = Array.from({ length: weeksUsed }, (_, i) => i + 1);
    const des = cellsAndDesign(post, launch, obsWeeks);
    const lm = launchLogMu(post, launch, u, drawIdx, des, true);
    const per = des.W * des.n;
    const logw = new Float64Array(N);
    // gammaln terms depend only on (draw, observation): cache them, K samples share each draw.
    const gy1 = Float64Array.from(des.units, (y) => gammaln(y + 1));
    const LOG_FLOOR = Math.log(1e-12);
    let cachedDraw = -1;
    const gyPhi = new Float64Array(per);
    let gPhi = 0;
    for (let i = 0; i < N; i++) {
      const d = drawIdx[i];
      const phi = post.draws.phi[d];
      if (d !== cachedDraw) {
        cachedDraw = d;
        gPhi = gammaln(phi);
        for (let j = 0; j < per; j++) gyPhi[j] = gammaln(des.units[j] + phi);
      }
      const logPhi = Math.log(phi);
      let acc = 0;
      for (let j = 0; j < per; j++) {
        const lmu = Math.max(lm[i * per + j], LOG_FLOOR); // mu clamped at 1e-12, as in nb_loglik
        const lphim = Math.log(phi + Math.exp(lmu));
        acc += gyPhi[j] - gPhi - gy1[j] + phi * (logPhi - lphim) + des.units[j] * (lmu - lphim);
      }
      logw[i] = acc;
    }
    const sm = psisSmooth(logw);
    paretoK = sm.kHat;
    w = normalizeLogWeights(sm.logw);
    nObsCells = per;
  } else {
    w = new Float64Array(N).fill(1 / N);
  }
  const ess = weightEss(w);
  const wPrior = new Float64Array(N).fill(1 / N);

  // forecast
  const fdes = forecastDesign(post, launch, horizon);
  const lmF = launchLogMu(post, launch, u, drawIdx, fdes, true);
  const isNewCity = launch.type === "new_city";
  const lmCf = isNewCity ? null : launchLogMu(post, launch, u, drawIdx, fdes, false);
  const { W: H, n } = fdes;
  const unitCost = launch.unit_cost;
  const unitsTot = new Float64Array(N);
  const revTot = new Float64Array(N);
  const profTot = new Float64Array(N);
  const weekUnits = Array.from({ length: H }, () => new Float64Array(N));
  const weekCf = Array.from({ length: H }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) {
    let tu = 0, tr = 0, tp = 0;
    for (let h = 0; h < H; h++) {
      let wu = 0, wc = 0;
      for (let c = 0; c < n; c++) {
        const price = fdes.price[h * n + c];
        const m = Math.exp(lmF[(i * H + h) * n + c]);
        wu += m;
        tr += m * price;
        tp += m * (price - unitCost);
        if (lmCf) wc += Math.exp(lmCf[(i * H + h) * n + c]);
      }
      tu += wu;
      weekUnits[h][i] = wu;
      weekCf[h][i] = wc;
    }
    unitsTot[i] = tu;
    revTot[i] = tr;
    profTot[i] = tp - launch.fixed_cost;
  }

  const observed = new Map<number, number>();
  launch.observations.week.forEach((wk, i) => {
    if (wk <= weeksUsed) observed.set(wk, (observed.get(wk) ?? 0) + launch.observations.units[i]);
  });
  const weekly: UpdateWeeklyRow[] = Array.from({ length: H }, (_, h) => ({
    week: h + 1,
    t: fdes.t[h],
    units: wBand(weekUnits[h], w),
    observed_units: observed.get(h + 1) ?? null,
    counterfactual_units: wBand(weekCf[h], w),
  }));

  // the unknown, prior vs. posterior
  const { kind, display = "raw" } = launch.unknown;
  let scale = 1;
  if (kind === "z_new_city") {
    let s = 0;
    for (let i = 0; i < N; i++) s += post.draws.tau_city[drawIdx[i]];
    scale = s / N;
  }
  const unknown: UpdateResult["unknown"] = {
    name: launch.unknown.name,
    label: launch.unknown.label,
    display,
    prior: wInterval(u, wPrior),
    posterior: wInterval(u, w),
  };
  if (opts.includeDensity !== false) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < N; i++) {
      if (u[i] < lo) lo = u[i];
      if (u[i] > hi) hi = u[i];
    }
    const grid = Array.from({ length: DENSITY_POINTS }, (_, j) => lo + ((hi - lo) * j) / (DENSITY_POINTS - 1));
    unknown.density = { x: grid, prior: weightedDensity(u, wPrior, grid), posterior: weightedDensity(u, w, grid) };
  }
  if (display === "percent_lift") {
    const pct = Float64Array.from(u, (v) => Math.exp(v * scale) - 1);
    unknown.prior_percent = wInterval(pct, wPrior);
    unknown.posterior_percent = wInterval(pct, w);
  }
  unknown.truth = launch.truth?.unknown_value ?? null;

  // validity gate: the browser reports weight ESS (no PSIS smoothing)
  const ok = weeksUsed ? ess >= ESS_MIN : true;
  const fmt = (x: number) => Math.round(x).toLocaleString("en-US");
  const gate: UpdateGate = {
    ok,
    criterion: "ess",
    value: Math.round(ess * 10) / 10,
    threshold: ESS_MIN,
    ess: Math.round(ess * 10) / 10,
    pareto_k: null,
    smoothed: false,
    message: ok
      ? `Fast update reliable (ESS ${fmt(ess)} of ${fmt(N)})`
      : `Refit needed — only ${fmt(ess)} of ${fmt(N)} samples carry any weight.`,
  };

  let pPost = 0, pPrior = 0, sw = 0;
  for (let i = 0; i < N; i++) {
    sw += w[i];
    if (profTot[i] > 0) {
      pPost += w[i];
      pPrior += 1;
    }
  }
  const experiment = !!launch.has_holdout;
  const now = typeof performance !== "undefined" ? performance.now() : Date.now();
  return {
    launch_id: launch.id,
    weeks_used: weeksUsed,
    gate,
    unknown,
    forecast: {
      horizon,
      total_units: wInterval(unitsTot, w),
      total_revenue: wInterval(revTot, w),
      total_profit: wInterval(profTot, w),
      p_profit_positive: pPost / sw,
      weekly,
    },
    prior_forecast: {
      total_units: wInterval(unitsTot, wPrior),
      total_revenue: wInterval(revTot, wPrior),
      total_profit: wInterval(profTot, wPrior),
      p_profit_positive: pPrior / N,
    },
    causal: {
      mode: experiment ? "experiment" : "model_based",
      label: experiment ? "Experiment (random holdout)" : "Model-based estimate",
      assumptions: launch.causal_assumptions ?? [],
      recommendation: experiment
        ? "A random holdout group is in the data, so this lift is measured against a real control, not against a model prediction."
        : HOLDOUT_RECOMMENDATION,
    },
    meta: {
      n_prior_draws: D,
      k_per_draw: kPerDraw,
      n_samples: N,
      n_obs_cells: nObsCells,
      seed,
      source: "browser",
      computed_ms: now - started,
      prior_sigma: sigma,
    },
  };
}
