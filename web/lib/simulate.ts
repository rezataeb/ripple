// Browser scenario simulator. Reference: model/ripple_model/simulate.py.
// Math is specified in contract/README.md ("The two rules" + "Scenario math").
import type {
  Band,
  Interval,
  Money,
  Posterior,
  Scenario,
  ScenarioResult,
  WeeklyRow,
} from "./types";

const TWO_PI = 2 * Math.PI;

/** Fourier design row for absolute week t: [sin(2π·1t/P), cos(2π·1t/P), …]. */
export function seasonRow(t: number, period: number, nFourier: number): number[] {
  const row: number[] = [];
  for (let k = 1; k <= nFourier; k++) {
    row.push(Math.sin((TWO_PI * k * t) / period), Math.cos((TWO_PI * k * t) / period));
  }
  return row;
}

export interface MuInput {
  /** log of segment share (offset) */
  logShare: number;
  /** a_city[d][c], or a_new[d] for a new city */
  aCity: number;
  aSeg: number;
  elasticity: number;
  price: number;
  basePrice: number;
  bPromo: number;
  promo: number;
  bFeature: number;
  feature: number;
  bDip: number;
  promoLastWeek: number;
  /** dot(season_row(t), season[d]) */
  season: number;
}

/** Expected units for one draw / cell / week. Parity target: tolerance 1e-6 on mu. */
export function mu(i: MuInput): number {
  return Math.exp(
    i.logShare +
      i.aCity +
      i.aSeg +
      i.elasticity * Math.log(i.price / i.basePrice) +
      i.bPromo * i.promo +
      i.bFeature * i.feature +
      i.bDip * i.promoLastWeek +
      i.season,
  );
}

/** Deterministic standard-normal draws (mulberry32 + Box–Muller). */
export function drawZ(seed: number, n: number): number[] {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out: number[] = [];
  while (out.length < n) {
    const u1 = Math.max(uniform(), 1e-12);
    const u2 = uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(TWO_PI * u2));
    if (out.length < n) out.push(r * Math.sin(TWO_PI * u2));
  }
  return out;
}

/** Linear-interpolation quantile of an ascending-sorted array (numpy default). */
export function quantileSorted(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  if (n === 1) return sorted[0];
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function summarize(values: ArrayLike<number>): Interval {
  const sorted = Float64Array.from(values as ArrayLike<number>).sort();
  let sum = 0;
  for (let i = 0; i < sorted.length; i++) sum += sorted[i];
  return {
    mean: sum / sorted.length,
    p05: quantileSorted(sorted, 0.05),
    p50: quantileSorted(sorted, 0.5),
    p95: quantileSorted(sorted, 0.95),
  };
}

function band(values: ArrayLike<number>): Band {
  const sorted = Float64Array.from(values as ArrayLike<number>).sort();
  return {
    p05: quantileSorted(sorted, 0.05),
    p25: quantileSorted(sorted, 0.25),
    p50: quantileSorted(sorted, 0.5),
    p75: quantileSorted(sorted, 0.75),
    p95: quantileSorted(sorted, 0.95),
  };
}

/** Per-draw totals; summarized only after differencing inside the draw. */
class MoneyAcc {
  units: Float64Array;
  revenue: Float64Array;
  profit: Float64Array;
  constructor(d: number) {
    this.units = new Float64Array(d);
    this.revenue = new Float64Array(d);
    this.profit = new Float64Array(d);
  }
  summary(): Money {
    return { units: summarize(this.units), revenue: summarize(this.revenue), profit: summarize(this.profit) };
  }
}

export interface SimulateOptions {
  /** Override the new-city z draws (parity tests, one value per posterior draw). */
  z?: number[];
}

function selectIdx(all: string[], chosen?: string[]): number[] {
  if (!chosen || chosen.length === 0) return all.map((_, i) => i);
  return all.map((n, i) => (chosen.includes(n) ? i : -1)).filter((i) => i >= 0);
}

/** Scenario levers per horizon week, shared by simulate() and muGrid(). */
function levers(posterior: Posterior, scenario: Scenario) {
  const { meta } = posterior;
  const base = meta.base_price;
  const H = scenario.horizon;
  const promoWeeks =
    scenario.type === "promo" && meta.has_promo ? Math.min(scenario.weeks ?? 0, H) : 0;
  const discount = scenario.type === "promo" && meta.has_promo ? (scenario.discount ?? 0) : 0;
  const priceChange = scenario.type === "price" ? (scenario.price_change ?? 0) : 0;
  return {
    promoWeeks,
    featureOn: scenario.feature ? 1 : 0,
    scenPrice: (w: number) =>
      scenario.type === "price"
        ? base * (1 + priceChange)
        : scenario.type === "promo" && w < promoWeeks
          ? base * (1 - discount)
          : base,
    promoFlag: (w: number) => (w < promoWeeks ? 1 : 0),
    promoLast: (w: number) => (promoWeeks > 0 && w === promoWeeks ? 1 : 0),
  };
}

export function simulate(
  posterior: Posterior,
  scenario: Scenario,
  opts: SimulateOptions = {},
): ScenarioResult {
  const started = typeof performance !== "undefined" ? performance.now() : Date.now();
  const { meta, draws, candidates } = posterior;
  const D = meta.n_draws;
  const C = meta.cities.length;
  const S = meta.segments.length;
  const H = scenario.horizon;
  const base = meta.base_price;
  const cost = scenario.unit_cost ?? meta.unit_cost;
  const fixed = scenario.fixed_cost ?? 0;
  const seed = scenario.seed ?? 7;
  const isNew = scenario.type === "new_city";

  const cityIdx = selectIdx(meta.cities, scenario.cities);
  const segIdx = selectIdx(meta.segments, scenario.segments);
  const citySel = new Set(cityIdx);
  const segSel = new Set(segIdx);

  // Scenario levers per horizon week.
  const { promoWeeks, scenPrice, promoFlag, promoLast, featureOn } = levers(posterior, scenario);

  // New city definition.
  const cand = isNew ? candidates.find((c) => c.name === scenario.candidate) : undefined;
  const traitsRaw = isNew ? (cand?.traits ?? scenario.traits ?? []) : [];
  const traitsZ = isNew
    ? (cand?.traits_z ?? traitsRaw.map((v, p) => (v - meta.trait_mean[p]) / meta.trait_std[p]))
    : [];
  const outOfRange: string[] = [];
  if (isNew) {
    traitsZ.forEach((z, p) => {
      if (z < meta.trait_min[p] || z > meta.trait_max[p]) outOfRange.push(meta.trait_names[p]);
    });
  }
  const trainingMix = Array.from({ length: S }, (_, s) => {
    let m = 0;
    for (let c = 0; c < C; c++) m += meta.shares[c][s];
    return m / C;
  });
  const newShares = isNew ? (scenario.new_city_shares ?? cand?.shares ?? trainingMix) : [];
  const newPrice = scenario.new_city_price ?? base;
  const marketSize = scenario.market_size ?? 1;
  const zDraws = isNew ? (opts.z ?? drawZ(seed, D)) : [];

  // Accumulators (per draw).
  const baseAcc = new MoneyAcc(D);
  const scenAcc = new MoneyAcc(D);
  const incAcc = new MoneyAcc(D);
  const weekBase = new Float64Array(D * H);
  const weekScen = new Float64Array(D * H);
  const weekInc = new Float64Array(D * H);
  const segAcc = meta.segments.map(() => new MoneyAcc(D));
  const cityAcc = (isNew ? [scenario.candidate ?? "New city"] : meta.cities).map(() => new MoneyAcc(D));

  for (let d = 0; d < D; d++) {
    let sBaseU = 0, sBaseR = 0, sBaseP = 0, sScenU = 0, sScenR = 0, sScenP = 0;
    const seasonW = new Float64Array(H);
    for (let w = 0; w < H; w++) {
      const row = seasonRow(meta.weeks + w, meta.season_period, meta.n_fourier);
      let acc = 0;
      for (let k = 0; k < row.length; k++) acc += row[k] * draws.season[d][k];
      seasonW[w] = acc;
    }
    const aNew = isNew
      ? draws.g0[d] +
        traitsZ.reduce((a, z, p) => a + z * draws.g[d][p], 0) +
        draws.tau_city[d] * zDraws[d]
      : 0;

    for (let w = 0; w < H; w++) {
      let bU = 0, bR = 0, bP = 0, xU = 0, xR = 0, xP = 0;

      // Existing business: baseline everywhere; scenario where selected.
      // A new city is reported on its own (baseline 0), as in simulate.py.
      for (let c = 0; c < (isNew ? 0 : C); c++) {
        for (let s = 0; s < S; s++) {
          const common = {
            logShare: Math.log(meta.shares[c][s]),
            aCity: draws.a_city[d][c],
            aSeg: draws.a_seg[d][s],
            elasticity: draws.e[d][s],
            basePrice: base,
            bPromo: draws.b_promo[d][s],
            bFeature: draws.b_feature[d],
            bDip: draws.b_dip[d],
            season: seasonW[w],
          };
          const baseline = mu({ ...common, price: base, promo: 0, feature: 0, promoLastWeek: 0 });
          const treated = !isNew && citySel.has(c) && segSel.has(s);
          const p = treated ? scenPrice(w) : base;
          const scen = treated
            ? mu({
                ...common,
                price: p,
                promo: promoFlag(w),
                feature: featureOn,
                promoLastWeek: promoLast(w),
              })
            : baseline;
          bU += baseline;
          bR += baseline * base;
          bP += baseline * (base - cost);
          xU += scen;
          xR += scen * p;
          xP += scen * (p - cost);
          if (!isNew && treated) {
            const a = segAcc[s];
            a.units[d] += scen - baseline;
            a.revenue[d] += scen * p - baseline * base;
            a.profit[d] += scen * (p - cost) - baseline * (base - cost);
            const ca = cityAcc[c];
            ca.units[d] += scen - baseline;
            ca.revenue[d] += scen * p - baseline * base;
            ca.profit[d] += scen * (p - cost) - baseline * (base - cost);
          }
        }
      }

      // New city: its volume is the whole scenario; its no-launch baseline is zero.
      if (isNew) {
        for (let s = 0; s < S; s++) {
          const units =
            marketSize *
            mu({
              logShare: Math.log(newShares[s]),
              aCity: aNew,
              aSeg: draws.a_seg[d][s],
              elasticity: draws.e[d][s],
              price: newPrice,
              basePrice: base,
              bPromo: 0,
              promo: 0,
              bFeature: draws.b_feature[d],
              feature: featureOn,
              bDip: 0,
              promoLastWeek: 0,
              season: seasonW[w],
            });
          xU += units;
          xR += units * newPrice;
          xP += units * (newPrice - cost);
          const a = segAcc[s];
          a.units[d] += units;
          a.revenue[d] += units * newPrice;
          a.profit[d] += units * (newPrice - cost);
          const ca = cityAcc[0];
          ca.units[d] += units;
          ca.revenue[d] += units * newPrice;
          ca.profit[d] += units * (newPrice - cost);
        }
      }

      weekBase[d * H + w] = bU;
      weekScen[d * H + w] = xU;
      weekInc[d * H + w] = xU - bU;
      sBaseU += bU; sBaseR += bR; sBaseP += bP;
      sScenU += xU; sScenR += xR; sScenP += xP;
    }

    sScenP -= fixed; // fixed cost: scenario side only, once over the horizon
    baseAcc.units[d] = sBaseU; baseAcc.revenue[d] = sBaseR; baseAcc.profit[d] = sBaseP;
    scenAcc.units[d] = sScenU; scenAcc.revenue[d] = sScenR; scenAcc.profit[d] = sScenP;
    incAcc.units[d] = sScenU - sBaseU;
    incAcc.revenue[d] = sScenR - sBaseR;
    incAcc.profit[d] = sScenP - sBaseP;
  }

  let positive = 0;
  for (let d = 0; d < D; d++) if (incAcc.profit[d] > 0) positive++;

  const weekly: WeeklyRow[] = [];
  for (let w = 0; w < H; w++) {
    const col = (arr: Float64Array) => {
      const out = new Float64Array(D);
      for (let d = 0; d < D; d++) out[d] = arr[d * H + w];
      return out;
    };
    weekly.push({
      week: w,
      t: meta.weeks + w,
      baseline_units: band(col(weekBase)),
      scenario_units: band(col(weekScen)),
      incremental_units: band(col(weekInc)),
      promo_active: w < promoWeeks,
      dip_active: promoWeeks > 0 && w === promoWeeks,
    });
  }

  const totalIncUnits = incAcc.units.reduce((a, b) => a + b, 0) / D;
  const segments = meta.has_segments
    ? meta.segments.map((segment, s) => {
        const incremental = segAcc[s].summary();
        return {
          segment,
          incremental,
          share_of_total: totalIncUnits !== 0 ? incremental.units.mean / totalIncUnits : 0,
        };
      })
    : [];
  const cityNames = isNew ? [scenario.candidate ?? "New city"] : meta.cities;
  const cities = cityNames
    .map((city, i) => ({ city, incremental: cityAcc[i].summary() }));

  const prices = posterior.history.price;
  const noPriceVariation = prices.length > 0 && Math.min(...prices) === Math.max(...prices);
  const warnings: string[] = [];
  if (isNew && outOfRange.length > 0) {
    warnings.push(
      `${cand?.name ?? scenario.candidate ?? "This city"} is outside the training range on ` +
        `${outOfRange.join(", ")}. The forecast extrapolates; ` +
        "treat the interval as a lower bound on the real uncertainty.",
    );
  }
  if (noPriceVariation) {
    warnings.push(
      "Price never varied in the fitted data, so the elasticity is driven by the " +
        "prior, not by evidence. Price scenarios are illustrative.",
    );
  }

  const now = typeof performance !== "undefined" ? performance.now() : Date.now();
  return {
    scenario,
    horizon: H,
    baseline: baseAcc.summary(),
    with_change: scenAcc.summary(),
    incremental: incAcc.summary(),
    p_profit_positive: positive / D,
    weekly,
    segments,
    cities,
    flags: {
      extrapolation: outOfRange.length > 0,
      out_of_range_traits: outOfRange,
      no_price_variation: noPriceVariation,
      promo_disabled: scenario.type === "promo" && !meta.has_promo,
      warnings,
    },
    meta: {
      n_draws: D,
      seed,
      source: "browser",
      computed_ms: now - started,
      unit_cost: cost,
      fixed_cost: fixed,
      net_price: isNew ? newPrice : scenPrice(0),
    },
  };
}

/**
 * Cell-level mu for the parity test: flat [draw][week-in-`weeks`][cell], same cell order as
 * simulate.py (city-major then segment; a new city's cells are its segments).
 */
export function muGrid(
  posterior: Posterior,
  scenario: Scenario,
  weeks: number[],
  z?: number[],
): { baseline: number[]; scenario: number[]; cells: number } {
  const { meta, draws, candidates } = posterior;
  const D = meta.n_draws;
  const S = meta.segments.length;
  const base = meta.base_price;
  const { scenPrice, promoFlag, promoLast, featureOn } = levers(posterior, scenario);
  const cityIdx = selectIdx(meta.cities, scenario.cities);
  const segIdx = selectIdx(meta.segments, scenario.segments);
  const isNew = scenario.type === "new_city";
  const cand = isNew ? candidates.find((c) => c.name === scenario.candidate) : undefined;
  const zs = isNew ? (z ?? drawZ(scenario.seed ?? 7, D)) : [];
  const cells: { c: number; s: number }[] = isNew
    ? Array.from({ length: S }, (_, s) => ({ c: -1, s }))
    : meta.cities.flatMap((_, c) => meta.segments.map((__, s) => ({ c, s })));
  const outB: number[] = [];
  const outS: number[] = [];
  for (let d = 0; d < D; d++) {
    const aNew = isNew
      ? draws.g0[d] +
        (cand?.traits_z ?? []).reduce((a, v, p) => a + v * draws.g[d][p], 0) +
        draws.tau_city[d] * zs[d]
      : 0;
    for (const w of weeks) {
      const row = seasonRow(meta.weeks + w, meta.season_period, meta.n_fourier);
      const season = row.reduce((a, r, k) => a + r * draws.season[d][k], 0);
      for (const { c, s } of cells) {
        const common = {
          logShare: Math.log(isNew ? (cand?.shares[s] ?? 0) : meta.shares[c][s]),
          aCity: isNew ? aNew : draws.a_city[d][c],
          aSeg: draws.a_seg[d][s],
          elasticity: draws.e[d][s],
          basePrice: base,
          bPromo: draws.b_promo[d][s],
          bFeature: draws.b_feature[d],
          bDip: draws.b_dip[d],
          season,
        };
        const treated = isNew || (cityIdx.includes(c) && segIdx.includes(s));
        outB.push(isNew ? 0 : mu({ ...common, price: base, promo: 0, feature: 0, promoLastWeek: 0 }));
        outS.push(
          mu({
            ...common,
            price: treated ? (isNew ? (scenario.new_city_price ?? base) : scenPrice(w)) : base,
            promo: treated && !isNew ? promoFlag(w) : 0,
            feature: treated ? featureOn : 0,
            promoLastWeek: treated && !isNew ? promoLast(w) : 0,
            ...(isNew ? { bPromo: 0, bDip: 0 } : {}),
          }),
        );
      }
    }
  }
  return { baseline: outB, scenario: outS, cells: cells.length };
}
