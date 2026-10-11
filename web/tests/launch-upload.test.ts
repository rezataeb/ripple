import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildLaunchFromCsv, type UploadKind } from "@/lib/launch-upload";
import { updateWithLaunch } from "@/lib/update";
import type { Launch, LaunchesFile, Posterior } from "@/lib/types";

const pub = path.resolve(__dirname, "../public/model");
const post = JSON.parse(readFileSync(path.join(pub, "posterior.json"), "utf8")) as Posterior;
const launches = (JSON.parse(readFileSync(path.join(pub, "launches.json"), "utf8")) as LaunchesFile).launches;

/** Re-create the CSV a user would upload from a demo launch's own observations. */
function toCsv(l: Launch): string {
  const o = l.observations;
  const city = (i: number) => (i < 0 ? l.new_city!.name : post.meta.cities[o.city_idx[i]]);
  const hasHoldout = !!o.holdout;
  const head = ["week", "city", "segment", "units", "price", "promo", ...(hasHoldout ? ["holdout"] : [])].join(",");
  const lines = o.week.map((w, i) =>
    [w, city(i), post.meta.segments[o.segment_idx[i]], o.units[i], o.price[i], o.promo[i], ...(hasHoldout ? [o.holdout![i]] : [])].join(","),
  );
  return [head, ...lines].join("\n");
}

describe("Upload launch weeks", () => {
  const cases: [string, UploadKind, Parameters<typeof buildLaunchFromCsv>[2]][] = [
    ["austin-launch", "new_city", { kind: "new_city", candidate: "Austin", horizon: 12, unitCost: 9, fixedCost: 145000, fileName: "a.csv" }],
    ["student-coupon", "offer", { kind: "offer", city: "Boston", segment: "students", horizon: 12, unitCost: 9, fixedCost: 0, fileName: "b.csv" }],
    ["gift-boxes", "feature", { kind: "feature", horizon: 12, unitCost: 9, fixedCost: 25000, fileName: "c.csv" }],
  ];

  for (const [id, , opts] of cases) {
    it(`round-trips ${id}: the uploaded launch gives the same update as the built-in one`, () => {
      let original = launches.find((l) => l.id === id)!;
      if (original.new_city) {
        // an upload can only use the candidate's mix from the fitted model, so compare like with like
        const c = post.candidates.find((x) => x.name === original.new_city!.name)!;
        original = { ...original, new_city: { ...original.new_city, shares: c.shares, traits_z: c.traits_z, traits: c.traits } };
      }
      {
        // The training data defines promo_last_week as the PREVIOUS week's promo (so 1 during a continuous
        // promo). launches.json carries 0 there; an upload has no such column, so Ripple derives it the
        // training way. Compare like with like.
        const o = original.observations;
        const key = (i: number, w: number) => `${o.city_idx[i]}|${o.segment_idx[i]}|${o.treated[i]}|${w}`;
        const prev = new Map(o.week.map((w, i) => [key(i, w), o.promo[i]]));
        original = {
          ...original,
          observations: { ...o, promo_last_week: o.week.map((w, i) => prev.get(key(i, w - 1)) ?? 0) },
        };
      }
      const built = buildLaunchFromCsv(post, toCsv(original), opts);
      if ("errors" in built) throw new Error(built.errors.join(" | "));
      expect(built.launch.weeks_observed).toBe(original.weeks_observed);
      for (const wk of [0, 3, 8]) {
        const a = updateWithLaunch(post, original, wk, { kPerDraw: 5, includeDensity: false });
        const b = updateWithLaunch(post, built.launch, wk, { kPerDraw: 5, includeDensity: false });
        expect(b.forecast.incremental_profit.mean).toBeCloseTo(a.forecast.incremental_profit.mean, 6);
        expect(b.forecast.p_profit_positive).toBeCloseTo(a.forecast.p_profit_positive, 9);
        expect(b.unknown.posterior.mean).toBeCloseTo(a.unknown.posterior.mean, 9);
      }
    });
  }

  it("explains what is wrong instead of failing silently", () => {
    const r = buildLaunchFromCsv(post, "when,sold\n1,5\n", { kind: "feature", horizon: 12, unitCost: 9, fixedCost: 0, fileName: "x.csv" });
    expect("errors" in r && r.errors.join(" ")).toMatch(/Missing a "week" column/);
    const r2 = buildLaunchFromCsv(post, "week,city,units,price\n1,Atlantis,5,24\n", { kind: "feature", horizon: 12, unitCost: 9, fixedCost: 0, fileName: "x.csv" });
    expect("errors" in r2 && r2.errors.join(" ")).toMatch(/City not in the fitted model: Atlantis/);
    const r3 = buildLaunchFromCsv(post, "week,units,price\n1,5,24\n", { kind: "new_city", horizon: 12, unitCost: 9, fixedCost: 0, fileName: "x.csv" });
    expect("errors" in r3 && r3.errors.join(" ")).toMatch(/Choose which candidate city/);
  });

  it("accepts dates for weeks and numbers them 1..N in order", () => {
    const csv = "week,units,price\n2026-01-12,500,24\n2026-01-05,480,24\n";
    const r = buildLaunchFromCsv(post, csv, { kind: "new_city", candidate: "Austin", horizon: 12, unitCost: 9, fixedCost: 0, fileName: "d.csv" });
    if ("errors" in r) throw new Error(r.errors.join(" "));
    expect(r.launch.weeks_observed).toBe(2);
    expect(r.launch.observations.week).toEqual([2, 1].map((x) => x));
  });
});
