import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  gammaln,
  nbLogPmf,
  normalizeLogWeights,
  updateWithLaunch,
  weightEss,
  weightedQuantiles,
} from "@/lib/update";
import type { LaunchesFile, Posterior } from "@/lib/types";

const pub = path.resolve(__dirname, "../public/model");
const posterior = JSON.parse(readFileSync(path.join(pub, "posterior.json"), "utf8")) as Posterior;
const launches = (JSON.parse(readFileSync(path.join(pub, "launches.json"), "utf8")) as LaunchesFile).launches;
const parity = JSON.parse(readFileSync(path.resolve(__dirname, "../../contract/update_parity.fixture.json"), "utf8")) as {
  k_per_draw: number;
  tolerance: { summary: number };
  ignore_fields: string[];
  priors: Record<string, { u: number[] }>;
  cases: { launch_id: string; weeks_used: number; expected: Record<string, unknown> }[];
};

function diff(got: unknown, want: unknown, tol: number, at: string, out: string[], ignore: Set<string> = new Set()) {
  if (ignore.has(at.replace(/^\./, ""))) return;
  if (typeof want === "number") {
    if (typeof got !== "number" || Math.abs(got - want) > tol * Math.max(1, Math.abs(want)))
      out.push(`${at}: got ${got} want ${want}`);
  } else if (Array.isArray(want)) {
    if (!Array.isArray(got) || got.length !== want.length) out.push(`${at}: length`);
    else want.forEach((w, i) => diff(got[i], w, tol, `${at}[${i}]`, out, ignore));
  } else if (want && typeof want === "object") {
    for (const [k, v] of Object.entries(want)) {
      diff((got as Record<string, unknown> | undefined)?.[k], v, tol, `${at}.${k}`, out, ignore);
    }
  } else if (got !== want) out.push(`${at}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
}

describe("update.ts parity with update.py (contract/update_parity.fixture.json)", () => {
  const ignore = new Set(parity.ignore_fields);
  for (const c of parity.cases) {
    it(`${c.launch_id} @ week ${c.weeks_used}: every field within ${parity.tolerance.summary}`, () => {
      const launch = launches.find((l) => l.id === c.launch_id)!;
      const got = updateWithLaunch(posterior, launch, c.weeks_used, {
        kPerDraw: parity.k_per_draw,
        u: parity.priors[c.launch_id].u,
      });
      const out: string[] = [];
      diff(got, c.expected, parity.tolerance.summary, "", out, ignore);
      expect(out.slice(0, 8)).toEqual([]);
    });
  }
});

describe("update math", () => {
  it("gammaln matches known values", () => {
    expect(gammaln(1)).toBeCloseTo(0, 12);
    expect(gammaln(5)).toBeCloseTo(Math.log(24), 12);
    expect(gammaln(0.5)).toBeCloseTo(0.5 * Math.log(Math.PI), 12);
    expect(gammaln(100.5)).toBeCloseTo(361.4355404677776, 9); // math.lgamma
  });

  it("negative binomial pmf sums to 1", () => {
    let s = 0;
    for (let y = 0; y < 400; y++) s += Math.exp(nbLogPmf(y, 12, 7));
    expect(s).toBeCloseTo(1, 8);
  });

  it("normalizes huge negative log-likelihoods without underflow", () => {
    const w = normalizeLogWeights(Float64Array.from([-90000, -90001, -90002]));
    expect(w[0] + w[1] + w[2]).toBeCloseTo(1, 12);
    expect(w[0]).toBeGreaterThan(w[1]);
  });

  it("ESS is N for equal weights and 1 for a single point mass", () => {
    expect(weightEss(new Float64Array(100).fill(0.01))).toBeCloseTo(100, 9);
    expect(weightEss(Float64Array.from([1, 0, 0, 0]))).toBeCloseTo(1, 12);
  });

  it("weighted quantiles equal numpy linear at equal weights", () => {
    const x = [4, 1, 3, 2];
    const [q50, q05] = weightedQuantiles(x, [1, 1, 1, 1], [0.5, 0.05]);
    expect(q50).toBeCloseTo(2.5, 12);
    expect(q05).toBeCloseTo(1, 12); // midpoint CDF clamps below 0.125
  });
});

describe("launch tracker behaviour", () => {
  const austin = launches.find((l) => l.id === "austin-launch")!;

  it("week 0 equals the prior forecast", () => {
    const r = updateWithLaunch(posterior, austin, 0, { kPerDraw: 5 });
    expect(r.forecast.total_profit!.mean).toBeCloseTo(r.prior_forecast.total_profit!.mean, 9);
    expect(r.forecast.p_profit_positive).toBeCloseTo(r.prior_forecast.p_profit_positive, 12);
    expect(r.gate.ok).toBe(true);
  });

  it("more real weeks narrow the interval on the total", () => {
    const width = (w: number) => {
      const t = updateWithLaunch(posterior, austin, w, { kPerDraw: 5 }).forecast.total_units;
      return t.p95 - t.p05;
    };
    expect(width(8)).toBeLessThan(width(0));
    expect(width(4)).toBeLessThan(width(0));
  });

  it("is deterministic for a fixed seed", () => {
    const a = updateWithLaunch(posterior, austin, 3, { kPerDraw: 5 });
    const b = updateWithLaunch(posterior, austin, 3, { kPerDraw: 5 });
    expect(a.forecast).toEqual(b.forecast);
  });

  it("clamps weeks to what was observed", () => {
    const r = updateWithLaunch(posterior, austin, 99, { kPerDraw: 5, includeDensity: false });
    expect(r.weeks_used).toBe(austin.weeks_observed);
  });
});
