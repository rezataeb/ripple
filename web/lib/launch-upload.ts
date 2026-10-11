// "Upload launch weeks": turn a user's CSV of real post-launch weeks into a Launch the update engine can use.
import { parseCsv, ruleMapping, type RippleKey } from "./data-flow";
import type { Launch, Posterior } from "./types";

export type UploadKind = "new_city" | "offer" | "feature";

export interface UploadOptions {
  kind: UploadKind;
  /** new_city: a name from posterior.candidates */
  candidate?: string;
  /** offer: the training city and the customer group that got the offer */
  city?: string;
  segment?: string;
  horizon: number;
  unitCost: number;
  fixedCost: number;
  fileName: string;
}

export type UploadResult = { launch: Launch; notes: string[] } | { errors: string[] };

const CAUSAL: Launch["causal_assumptions"] = [
  { id: "no-other-change", text: "No other change happened in the same weeks.", holds: null },
  { id: "no-spillover", text: "Untreated cities and groups were not affected.", holds: null },
  { id: "stable-pattern", text: "The demand pattern is stable: nothing shifted the baseline.", holds: null },
];

const num = (s: string) => (s.trim() === "" ? NaN : Number(s));

export function buildLaunchFromCsv(post: Posterior, csvText: string, o: UploadOptions): UploadResult {
  const { meta, candidates } = post;
  const { columns, rows } = parseCsv(csvText);
  const errors: string[] = [];
  if (columns.length === 0 || rows.length === 0) return { errors: ["The file has no data rows."] };

  const m = ruleMapping(columns).mapping as Partial<Record<RippleKey, string>>;
  for (const req of ["week", "units", "price"] as const) {
    if (!m[req]) errors.push(`Missing a "${req}" column. Expected: week, units, price (plus city, segment, promo, holdout if you have them).`);
  }
  if (o.kind === "new_city") {
    if (!meta.has_traits || candidates.length === 0) errors.push("This model has no city traits, so a new-city launch cannot be forecast.");
    else if (!candidates.some((c) => c.name === o.candidate)) errors.push("Choose which candidate city launched.");
  }
  if (o.kind === "offer") {
    if (!meta.cities.includes(o.city ?? "")) errors.push("Choose the city where the offer ran.");
    if (!meta.segments.includes(o.segment ?? "")) errors.push("Choose the customer group that got the offer.");
    if (!meta.has_promo) errors.push("This model has no promotion term, so an offer's lift cannot be learned.");
  }
  if (errors.length) return { errors };

  // weeks: 1-based launch weeks by order of appearance, whether the file has dates or numbers
  const rawWeeks = [...new Set(rows.map((r) => r[m.week!]))];
  const sortKey = (w: string) => (Number.isFinite(Number(w)) ? Number(w) : Date.parse(w));
  rawWeeks.sort((a, b) => sortKey(a) - sortKey(b));
  if (rawWeeks.some((w) => !Number.isFinite(sortKey(w)))) return { errors: ["Some week values are neither numbers nor dates."] };
  const weekIdx = new Map(rawWeeks.map((w, i) => [w, i + 1]));

  const cityIdxOf = (name: string): number => {
    if (o.kind === "new_city") return -1;
    return meta.cities.indexOf(name);
  };
  const segIdxOf = (name: string): number => (meta.has_segments ? meta.segments.indexOf(name) : 0);

  const obs = {
    week: [] as number[],
    week_start: [] as string[],
    t: [] as number[],
    city_idx: [] as number[],
    segment_idx: [] as number[],
    units: [] as number[],
    price: [] as number[],
    promo: [] as number[],
    promo_last_week: [] as number[],
    feature: [] as number[],
    treated: [] as number[],
    holdout: [] as number[],
  };
  // a holdout column of all zeros is no control group
  const hasHoldout = !!m.holdout && rows.some((r) => (num(r[m.holdout!]) || 0) !== 0);
  const bad: string[] = [];
  const unknownCities = new Set<string>();
  const unknownSegs = new Set<string>();

  for (const r of rows) {
    const cityName = m.city ? r[m.city] : o.kind === "offer" ? (o.city ?? "") : "";
    const segName = m.segment ? r[m.segment] : meta.segments[0];
    const ci = cityIdxOf(cityName);
    const si = segIdxOf(segName);
    if (o.kind !== "new_city" && ci < 0) {
      unknownCities.add(cityName);
      continue;
    }
    if (si < 0) {
      unknownSegs.add(segName);
      continue;
    }
    if (o.kind === "offer" && cityName !== o.city) continue; // offer data for other cities is not part of this launch
    if (o.kind === "new_city" && m.city && r[m.city] !== o.candidate && cities1(rows, m.city) > 1) continue;
    const units = num(r[m.units!]);
    const price = num(r[m.price!]);
    if (!(units >= 0) || !(price > 0)) {
      bad.push(`${r[m.week!]} · ${cityName} · ${segName}`);
      continue;
    }
    const w = weekIdx.get(r[m.week!])!;
    const holdout = hasHoldout ? num(r[m.holdout!]) || 0 : 0;
    const promo = m.promo ? num(r[m.promo]) || 0 : 0;
    // who gets the effect: offer → the chosen group; feature/new city → everyone not held out
    const treated = o.kind === "offer" ? (segName === o.segment ? 1 : 0) : hasHoldout ? 1 - holdout : 1;
    obs.week.push(w);
    obs.week_start.push(r[m.week!]);
    obs.t.push(meta.weeks + w - 1);
    obs.city_idx.push(ci);
    obs.segment_idx.push(si);
    obs.units.push(units);
    obs.price.push(price);
    obs.promo.push(promo);
    obs.promo_last_week.push(0);
    obs.feature.push(0); // a feature launch's effect is theta; the fitted feature flag is a different thing
    obs.treated.push(treated);
    obs.holdout.push(holdout);
  }
  if (unknownCities.size > 0)
    errors.push(`City not in the fitted model: ${[...unknownCities].slice(0, 4).join(", ")}. Known: ${meta.cities.join(", ")}.`);
  if (unknownSegs.size > 0)
    errors.push(`Customer group not in the fitted model: ${[...unknownSegs].slice(0, 4).join(", ")}. Known: ${meta.segments.join(", ")}.`);
  if (bad.length > 0) errors.push(`${bad.length} row(s) have missing or invalid units/price, e.g. ${bad[0]}.`);
  if (obs.week.length === 0) errors.push("No usable rows were found for this launch.");
  if (errors.length) return { errors };

  // promo_last_week from the same cell's previous week
  const seen = new Map<string, number>();
  obs.week.forEach((w, i) => seen.set(`${obs.city_idx[i]}|${obs.segment_idx[i]}|${obs.treated[i]}|${w}`, obs.promo[i]));
  obs.week.forEach((w, i) => {
    obs.promo_last_week[i] = seen.get(`${obs.city_idx[i]}|${obs.segment_idx[i]}|${obs.treated[i]}|${w - 1}`) ?? 0;
  });

  const weeksObserved = rawWeeks.length;
  const cand = candidates.find((c) => c.name === o.candidate);
  const base = {
    id: `upload-${Date.now()}`,
    label: `Your launch · ${o.fileName}`,
    horizon: Math.max(o.horizon, weeksObserved),
    weeks_observed: weeksObserved,
    t0: meta.weeks,
    observations: { ...obs, ...(hasHoldout ? {} : { holdout: undefined }) },
    has_holdout: hasHoldout,
    unit_cost: o.unitCost,
    fixed_cost: o.fixedCost,
    net_price: obs.price[0],
    causal_assumptions: CAUSAL,
  };
  if (!hasHoldout) delete (base.observations as { holdout?: number[] }).holdout;

  const notes: string[] = [];
  if (hasHoldout) notes.push("A holdout column was found, so the lift is read against a real control group (assumes a 90/10 split).");
  if (o.kind === "feature") notes.push("The effect is modelled as a log-lift with a skeptical prior; you can change its scale on the slider.");

  let launch: Launch;
  if (o.kind === "new_city") {
    launch = {
      ...base,
      type: "new_city",
      description: `Real weeks from your ${cand!.name} launch.`,
      cities: [cand!.name],
      new_city: { name: cand!.name, traits: cand!.traits, traits_z: cand!.traits_z, shares: cand!.shares, extrapolation: cand!.extrapolation, out_of_range_traits: cand!.out_of_range_traits },
      unknown: {
        name: "z_new_city",
        label: `${cand!.name} city effect`,
        kind: "z_new_city",
        prior: { dist: "normal", mu: 0, sigma: 1, editable_sigma: false },
        display: "percent_lift",
      },
    } as Launch;
  } else if (o.kind === "offer") {
    launch = {
      ...base,
      type: "offer",
      description: `Real weeks from your ${o.segment} offer in ${o.city}.`,
      cities: [o.city!],
      segments: [o.segment!],
      unknown: {
        name: `b_promo[${o.segment}]`,
        label: `${o.segment} deal-signal lift`,
        kind: "b_promo",
        segment: o.segment,
        prior: { dist: "posterior", editable_sigma: false },
        display: "percent_lift",
      },
    } as Launch;
  } else {
    launch = {
      ...base,
      type: "feature",
      description: "Real weeks from your feature launch.",
      unknown: {
        name: "theta",
        label: "Lift from the new feature",
        kind: "theta",
        prior: { dist: "normal", mu: 0, sigma: 0.15, editable_sigma: true },
        display: "percent_lift",
      },
    } as Launch;
  }
  return { launch, notes };
}

function cities1(rows: Record<string, string>[], col: string): number {
  return new Set(rows.map((r) => r[col])).size;
}
