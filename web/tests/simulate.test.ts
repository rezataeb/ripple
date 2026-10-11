import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { drawZ, mu, quantileSorted, seasonRow, simulate } from "@/lib/simulate";
import type { Posterior, Scenario } from "@/lib/types";

const root = path.resolve(__dirname, "../..");
const posterior = JSON.parse(
  readFileSync(path.join(root, "contract/posterior.fixture.json"), "utf8"),
) as Posterior;

const promo: Scenario = {
  type: "promo", cities: ["Boston"], segments: ["students"], discount: 0.2, weeks: 4, horizon: 12,
};

describe("mean function", () => {
  it("matches a hand computation", () => {
    const v = mu({
      logShare: Math.log(0.5), aCity: 0.1, aSeg: 0.2, elasticity: -2, price: 12, basePrice: 24,
      bPromo: 0.3, promo: 1, bFeature: 0.05, feature: 1, bDip: -0.1, promoLastWeek: 0, season: 0.02,
    });
    const log = Math.log(0.5) + 0.1 + 0.2 + -2 * Math.log(0.5) + 0.3 + 0.05 + 0 + 0.02;
    expect(v).toBeCloseTo(Math.exp(log), 12);
  });
  it("season row is [sin, cos] per harmonic", () => {
    const r = seasonRow(13, 52, 2);
    expect(r).toHaveLength(4);
    expect(r[0]).toBeCloseTo(Math.sin((2 * Math.PI * 13) / 52), 12);
    expect(r[3]).toBeCloseTo(Math.cos((2 * Math.PI * 2 * 13) / 52), 12);
  });
});

describe("simulate", () => {
  it("no change gives exactly zero incremental", () => {
    const r = simulate(posterior, { type: "price", price_change: 0, horizon: 12 });
    expect(r.incremental.units.mean).toBe(0);
    expect(r.incremental.profit.p05).toBe(0);
    expect(r.incremental.profit.p95).toBe(0);
    expect(r.p_profit_positive).toBe(0);
  });

  it("is deterministic", () => {
    expect(simulate(posterior, promo)).toEqual({
      ...simulate(posterior, promo),
      meta: expect.anything(),
    });
    const a = simulate(posterior, { type: "new_city", candidate: "Austin", horizon: 12 });
    const b = simulate(posterior, { type: "new_city", candidate: "Austin", horizon: 12 });
    expect(a.incremental).toEqual(b.incremental);
  });

  it("intervals are ordered and every headline carries one", () => {
    const r = simulate(posterior, promo);
    for (const k of ["units", "revenue", "profit"] as const) {
      const i = r.incremental[k];
      expect(i.p05).toBeLessThanOrEqual(i.p50);
      expect(i.p50).toBeLessThanOrEqual(i.p95);
    }
    expect(r.weekly).toHaveLength(12);
    expect(r.weekly.filter((w) => w.promo_active)).toHaveLength(4);
    expect(r.weekly.filter((w) => w.dip_active)).toHaveLength(1);
  });

  it("segment increments add up to the total (means are linear)", () => {
    const r = simulate(posterior, { ...promo, cities: [], segments: [] });
    const sum = r.segments!.reduce((a, s) => a + s.incremental.units.mean, 0);
    expect(sum).toBeCloseTo(r.incremental.units.mean, 6);
  });

  it("untreated segment stays at baseline", () => {
    const r = simulate(posterior, promo);
    const retirees = r.segments!.find((s) => s.segment === "retirees")!;
    expect(retirees.incremental.units.mean).toBe(0);
  });

  it("fixed cost lowers profit by exactly that amount, scenario side only", () => {
    const a = simulate(posterior, promo);
    const b = simulate(posterior, { ...promo, fixed_cost: 1000 });
    expect(a.incremental.profit.mean - b.incremental.profit.mean).toBeCloseTo(1000, 6);
    expect(b.incremental.units.mean).toBeCloseTo(a.incremental.units.mean, 9);
  });

  it("raising price cuts units when elasticities are negative", () => {
    const r = simulate(posterior, { type: "price", price_change: 0.1, horizon: 12 });
    expect(r.incremental.units.mean).toBeLessThan(0);
  });

  it("flags extrapolation for Miami but not Austin", () => {
    const miami = simulate(posterior, { type: "new_city", candidate: "Miami", horizon: 12 });
    const austin = simulate(posterior, { type: "new_city", candidate: "Austin", horizon: 12 });
    expect(miami.flags.extrapolation).toBe(true);
    expect(miami.flags.out_of_range_traits).toEqual(["median_income_k", "share_age_18_34"]);
    expect(austin.flags.extrapolation).toBe(false);
  });

  it("new city is pure addition: positive units, one city row", () => {
    const r = simulate(posterior, { type: "new_city", candidate: "Austin", horizon: 12 });
    expect(r.incremental.units.p05).toBeGreaterThan(0);
    expect(r.cities).toHaveLength(1);
    expect(r.cities![0].city).toBe("Austin");
  });

  it("a stronger z-draw raises new-city volume", () => {
    const D = posterior.meta.n_draws;
    const lo = simulate(posterior, { type: "new_city", candidate: "Austin", horizon: 4 }, { z: Array(D).fill(-2) });
    const hi = simulate(posterior, { type: "new_city", candidate: "Austin", horizon: 4 }, { z: Array(D).fill(2) });
    expect(hi.incremental.units.mean).toBeGreaterThan(lo.incremental.units.mean);
  });
});

describe("helpers", () => {
  it("quantile matches numpy linear interpolation", () => {
    const s = [1, 2, 3, 4];
    expect(quantileSorted(s, 0.5)).toBeCloseTo(2.5, 12);
    expect(quantileSorted(s, 0.05)).toBeCloseTo(1.15, 12);
  });
  it("drawZ is seeded and roughly standard normal", () => {
    const z = drawZ(7, 4000);
    expect(drawZ(7, 4000)).toEqual(z);
    const m = z.reduce((a, b) => a + b, 0) / z.length;
    const v = z.reduce((a, b) => a + (b - m) ** 2, 0) / z.length;
    expect(Math.abs(m)).toBeLessThan(0.06);
    expect(Math.abs(v - 1)).toBeLessThan(0.08);
  });
});
