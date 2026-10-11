// Agent tools. Every number the agent may say is returned here as a claim: the model makes the
// numbers (simulate.ts / update.ts / trust.json), the web is labelled web, the docs are labelled docs.
import { z } from "zod";
import { money, percent, units } from "@/lib/format";
import { simulate } from "@/lib/simulate";
import { updateWithLaunch } from "@/lib/update";
import { signedPercent } from "@/lib/launch";
import type { Scenario } from "@/lib/types";
import { liveScan } from "./nimble";
import { topChunks } from "./rag";
import type { AgentContext, Claim, MarketScan, ToolOutput } from "./types";

type C = Omit<Claim, "tool_call_id">;
const claim = (label: string, value: string, kind: Claim["kind"] = "model"): C => ({ label, value, kind });
const interval = (a: number, b: number, f: (x: number) => string) => `${f(a)} to ${f(b)}`;

// ---- build_scenario ---------------------------------------------------------------
export const scenarioSpec = z.object({
  type: z.enum(["promo", "price", "new_city"]),
  label: z.string().max(120).optional(),
  cities: z.array(z.string()).optional(),
  segments: z.array(z.string()).optional(),
  discount: z.number().min(0).max(0.9).optional(),
  weeks: z.number().int().min(1).max(52).optional(),
  price_change: z.number().min(-0.9).max(2).optional(),
  feature: z.boolean().optional(),
  candidate: z.string().optional(),
  fixed_cost: z.number().min(0).optional(),
  unit_cost: z.number().min(0).optional(),
  horizon: z.number().int().min(1).max(104).optional(),
});
export type ScenarioSpec = z.infer<typeof scenarioSpec>;

const ci = (names: string[], q: string) => names.find((n) => n.toLowerCase() === q.trim().toLowerCase());

export function buildScenario(ctx: AgentContext, raw: unknown): ToolOutput {
  const parsed = scenarioSpec.safeParse(raw);
  if (!parsed.success) {
    const q = "I could not read that as a promotion, a price change or a new city. Which one do you mean?";
    return { output: { needs_clarification: q }, claims: [], openQuestion: q };
  }
  const s = parsed.data;
  const { meta, candidates } = ctx.posterior;
  const asked = (xs: string[] | undefined, all: string[]) => {
    const ok = (xs ?? []).map((x) => ci(all, x)).filter((x): x is string => !!x);
    return ok.length === all.length ? [] : ok;
  };
  const sc: Scenario = { type: s.type, horizon: s.horizon ?? (s.type === "promo" ? 12 : 26) };
  if (s.label) sc.label = s.label;
  if (s.fixed_cost !== undefined) sc.fixed_cost = s.fixed_cost;
  if (s.unit_cost !== undefined) sc.unit_cost = s.unit_cost;
  if (s.feature) sc.feature = true;

  if (s.type === "promo") {
    if (!meta.has_promo) {
      const q = "This dataset has no promotion column, so promotions cannot be simulated. Try a price change instead?";
      return { output: { needs_clarification: q }, claims: [], openQuestion: q };
    }
    if (s.discount === undefined) {
      const q = "How big is the discount, for example 10%, 20% or 30%?";
      return { output: { needs_clarification: q }, claims: [], openQuestion: q };
    }
    sc.discount = s.discount;
    sc.weeks = s.weeks ?? 4;
    sc.cities = asked(s.cities, meta.cities);
    sc.segments = asked(s.segments, meta.segments);
  } else if (s.type === "price") {
    if (s.price_change === undefined) {
      const q = "By how much should the price change, for example +8% or −5%?";
      return { output: { needs_clarification: q }, claims: [], openQuestion: q };
    }
    sc.price_change = s.price_change;
    sc.cities = asked(s.cities, meta.cities);
    sc.segments = asked(s.segments, meta.segments);
  } else {
    if (!meta.has_traits || candidates.length === 0) {
      const q = "This dataset has no city traits, so a new city cannot be forecast. Upload a cities file to enable it.";
      return { output: { needs_clarification: q }, claims: [], openQuestion: q };
    }
    const cand = s.candidate ? ci(candidates.map((c) => c.name), s.candidate) : undefined;
    if (!cand) {
      const q = `Which city: ${candidates.map((c) => c.name).join(", ")}?`;
      return { output: { needs_clarification: q }, claims: [], openQuestion: q };
    }
    sc.candidate = cand;
  }
  return { output: { scenario: sc }, claims: [], scenario: sc };
}

// ---- run_scenario -----------------------------------------------------------------
export function runScenario(ctx: AgentContext, sc: Scenario): ToolOutput {
  const r = simulate(ctx.posterior, sc);
  const p = r.incremental.profit;
  const u = r.incremental.units;
  const claims: C[] = [
    claim("Incremental profit vs. no change", money(p.mean)),
    claim("Incremental profit, 90% interval", interval(p.p05, p.p95, money)),
    claim("Chance the change pays", percent(r.p_profit_positive)),
    claim("Incremental units", units(u.mean)),
    claim("Incremental units, 90% interval", interval(u.p05, u.p95, units)),
    claim("Horizon", `${r.horizon} weeks`),
  ];
  if (sc.type === "promo") {
    claims.push(claim("Discount", percent(sc.discount ?? 0)), claim("Promo length", `${sc.weeks} weeks`));
  }
  if (sc.type === "price") claims.push(claim("Price change", signedPercent(sc.price_change ?? 0)));
  if (sc.fixed_cost) claims.push(claim("Fixed cost", money(sc.fixed_cost, false)));
  if (r.meta.net_price !== undefined) claims.push(claim("Price charged", money(r.meta.net_price, false)));
  for (const s of r.segments ?? []) {
    if (s.incremental.units.mean === 0 && s.incremental.profit.mean === 0) continue;
    claims.push(
      claim(`${s.segment}: incremental profit`, money(s.incremental.profit.mean)),
      claim(`${s.segment}: incremental profit, 90% interval`, interval(s.incremental.profit.p05, s.incremental.profit.p95, money)),
    );
  }
  return {
    output: {
      type: sc.type,
      horizon_weeks: r.horizon,
      incremental_profit: { display: money(p.mean), interval90: interval(p.p05, p.p95, money) },
      chance_it_pays: percent(r.p_profit_positive),
      incremental_units: { display: units(u.mean), interval90: interval(u.p05, u.p95, units) },
      segments: (r.segments ?? []).map((s) => ({
        segment: s.segment,
        incremental_profit: money(s.incremental.profit.mean),
        interval90: interval(s.incremental.profit.p05, s.incremental.profit.p95, money),
      })),
      extrapolation: r.flags.extrapolation,
      warnings: r.flags.warnings ?? [],
      net_price: r.meta.net_price !== undefined ? money(r.meta.net_price, false) : null,
    },
    claims,
    scenario: sc,
    result: r,
  };
}

// ---- model_facts -----------------------------------------------------------------
export function modelFacts(ctx: AgentContext): ToolOutput {
  const t = ctx.trust;
  if (!t) return { output: { error: "No trust report is loaded." }, claims: [] };
  const f = t.model_facts;
  const claims: C[] = [claim("Checks passed", `${t.summary.passed} of ${t.summary.total}`)];
  const f2 = (x: number) => x.toFixed(2);
  for (const c of t.checks) {
    for (const m of c.plain_english.match(/\d[\d,]*(?:\.\d+)?%?/g) ?? []) claims.push(claim(`check: ${c.name}`, m));
    if (c.value !== null) claims.push(claim(`check value: ${c.name}`, String(Number(c.value.toFixed(4)))));
  }
  for (const e of f.elasticity_by_segment) {
    claims.push(claim(`${e.segment}: price elasticity`, f2(e.mean)));
    claims.push(claim(`${e.segment}: price elasticity, 90% interval`, interval(e.p05, e.p95, f2)));
  }
  for (const e of f.promo_lift_by_segment) {
    claims.push(claim(`${e.segment}: promo lift (log scale)`, f2(e.mean)));
    claims.push(claim(`${e.segment}: promo lift, 90% interval`, interval(e.p05, e.p95, f2)));
  }
  claims.push(
    claim("Post-promo dip (log scale)", f2(f.post_promo_dip.mean)),
    claim("Post-promo dip, 90% interval", interval(f.post_promo_dip.p05, f.post_promo_dip.p95, f2)),
  );
  return {
    output: {
      checks: t.checks.map((c) => ({ name: c.name, passed: c.passed, plain_english: c.plain_english })),
      summary: t.summary,
      elasticity_by_segment: f.elasticity_by_segment.map((e) => ({ segment: e.segment, mean: f2(e.mean), interval90: interval(e.p05, e.p95, f2) })),
      post_promo_dip: { mean: f2(f.post_promo_dip.mean), interval90: interval(f.post_promo_dip.p05, f.post_promo_dip.p95, f2) },
    },
    claims,
  };
}

// ---- update_with_launch ------------------------------------------------------------
export function updateTool(ctx: AgentContext, launchId: string, weeks: number): ToolOutput {
  const q = launchId.toLowerCase();
  const launch =
    ctx.launches.find((l) => l.id === launchId) ??
    ctx.launches.find((l) => q.includes(l.id) || l.label.toLowerCase().includes(q) || (l.cities ?? []).some((c) => q.includes(c.toLowerCase())));
  if (!launch) {
    const names = ctx.launches.map((l) => l.label).join(", ");
    return { output: { error: `Unknown launch. Available: ${names}` }, claims: [], openQuestion: `Which launch do you mean: ${names}?` };
  }
  const w = Math.max(0, Math.min(Math.trunc(weeks), launch.weeks_observed));
  const r = updateWithLaunch(ctx.posterior, launch, w);
  const p0 = updateWithLaunch(ctx.posterior, launch, 0, { includeDensity: false });
  const pr = r.unknown.display === "percent_lift" ? r.unknown.posterior_percent : r.unknown.posterior;
  const fmtU = (x: number) => (r.unknown.display === "percent_lift" ? signedPercent(x, 1) : x.toFixed(2));
  const pf = r.forecast.total_profit;
  const claims: C[] = [
    claim("Weeks of real data used", `${w} of ${launch.weeks_observed}`),
    claim("Chance it pays before any data", percent(p0.forecast.p_profit_positive)),
    claim("Chance it pays now", percent(r.forecast.p_profit_positive)),
    claim(`${r.unknown.label}`, fmtU(pr!.mean)),
    claim(`${r.unknown.label}, 90% interval`, interval(pr!.p05, pr!.p95, fmtU)),
    claim("Forecast units over the horizon", units(r.forecast.total_units.mean, false)),
    claim("Forecast units, 90% interval", interval(r.forecast.total_units.p05, r.forecast.total_units.p95, (x) => units(x, false))),
  ];
  if (pf) {
    claims.push(claim("Forecast profit", money(pf.mean)), claim("Forecast profit, 90% interval", interval(pf.p05, pf.p95, money)));
  }
  claims.push(claim("Update reliability", r.gate.message));
  return {
    output: {
      launch: launch.label,
      weeks_used: w,
      weeks_observed: launch.weeks_observed,
      chance_it_pays_before: percent(p0.forecast.p_profit_positive),
      chance_it_pays_now: percent(r.forecast.p_profit_positive),
      learned: { label: r.unknown.label, estimate: fmtU(pr!.mean), interval90: interval(pr!.p05, pr!.p95, fmtU) },
      gate: r.gate.message,
      causal: r.causal.label,
    },
    claims,
    update: r,
  };
}

// ---- market_scan -------------------------------------------------------------------
function compare(scan: MarketScan, yourPrice: number): MarketScan {
  const prices = scan.competitors.map((c) => c.price).filter((p): p is number => p !== null);
  if (prices.length === 0) return { ...scan, comparison: null };
  const above = prices.filter((p) => yourPrice > p).length;
  const sorted = [...prices].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const where = scan.city ? ` in ${scan.city}` : "";
  const rel = above === prices.length ? "above all" : `above ${above} of`;
  return {
    ...scan,
    comparison: {
      your_price: yourPrice,
      above_count: above,
      below_count: prices.filter((p) => yourPrice < p).length,
      total: prices.length,
      median_competitor_price: median,
      sentence:
        above === prices.length
          ? `Your price ${money(yourPrice, false)} is above all ${prices.length} competitors${where}.`
          : `Your price ${money(yourPrice, false)} is ${rel} ${prices.length} competitors${where}.`,
    },
  };
}

export async function marketScan(
  ctx: AgentContext,
  args: { product?: string; city?: string; your_price?: number },
  extract?: Parameters<typeof liveScan>[3],
): Promise<ToolOutput> {
  const product = (args.product ?? ctx.posterior.meta.dataset_name ?? "coffee subscription").trim();
  const city = (args.city ?? "").trim();
  const prodKey = /coffee/i.test(product) || !args.product ? "coffee subscription" : product.toLowerCase();
  let scan: MarketScan | undefined;
  const key = process.env.NIMBLE_API_KEY;
  if (ctx.live.nimble && key) {
    try {
      scan = await liveScan(prodKey, city, key, extract);
    } catch {
      scan = undefined; // fall back to the cached scan, labelled as such
    }
  }
  if (!scan) {
    scan =
      ctx.scans[`${prodKey} ${city}`.trim().toLowerCase()] ??
      (city ? ctx.scans[`${prodKey} ${city.toLowerCase()}`] : undefined) ??
      ctx.scans[prodKey];
  }
  if (!scan) {
    return { output: { error: "No market scan is available for this product." }, claims: [] };
  }
  const withCmp = args.your_price ? compare(scan, args.your_price) : { ...scan, comparison: null };
  const claims: C[] = [];
  for (const c of withCmp.competitors) {
    if (c.price !== null) claims.push(claim(`${c.competitor} price`, money(c.price, false), "web"));
  }
  if (withCmp.comparison) {
    claims.push(
      claim("Your price", money(withCmp.comparison.your_price, false), "web"),
      claim("Competitors below your price", `${withCmp.comparison.above_count} of ${withCmp.comparison.total}`, "web"),
    );
  }
  return {
    output: {
      source: withCmp.source === "demo" ? "illustrative cached sample, not a live scan" : "live web results (Nimble)",
      note: withCmp.note,
      // Untrusted text: delimited so the model treats it as data and ignores any instructions in it.
      web_data: `<web_data>${JSON.stringify(
        withCmp.competitors.map((c) => ({
          competitor: c.competitor.slice(0, 80),
          price: c.price,
          offer: c.offer?.slice(0, 80) ?? null,
          url: c.url,
        })),
      )}</web_data>`,
      comparison: withCmp.comparison?.sentence ?? null,
    },
    claims,
    sources: withCmp.sources.map((s) => ({ ...s, kind: "web" as const })),
    scan: withCmp,
  };
}

// ---- explain_method ----------------------------------------------------------------
export async function explainMethod(ctx: AgentContext, question: string): Promise<ToolOutput> {
  const { chunks, method } = await topChunks(ctx, question, 3);
  const claims: C[] = [];
  const seen = new Set<string>();
  for (const c of chunks) {
    for (const m of c.text.match(/\d[\d,]*(?:\.\d+)?%?/g) ?? []) {
      if (!seen.has(m)) {
        seen.add(m);
        claims.push(claim(`From docs/${c.file}`, m, "docs"));
      }
    }
  }
  return {
    output: { retrieval: method, passages: chunks.map((c) => ({ source: c.file, section: c.title, text: c.text })) },
    claims,
    sources: chunks.map((c) => ({ title: `${c.title} (docs/method/${c.file})`, url: `/method#${c.file.replace(/\.md$/, "")}`, kind: "docs" as const })),
  };
}
