import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { muGrid, seasonRow, simulate } from "@/lib/simulate";
import type { Posterior, Scenario } from "@/lib/types";

const root = path.resolve(__dirname, "../..");
const fixturePath = path.join(root, "contract/parity.fixture.json");
const posterior = JSON.parse(
  readFileSync(path.join(root, "contract/posterior.fixture.json"), "utf8"),
) as Posterior;

interface ParityCase {
  name: string;
  scenario: Scenario;
  z?: number[];
  mu: { weeks: number[]; t: number[]; shape: number[]; cells: { city: string; segment: string }[]; baseline_flat: number[]; scenario_flat: number[] };
  expected: Record<string, unknown>;
}
interface ParityFile {
  tolerance: { mu: number; summary: number };
  ignore_fields: string[];
  cases: ParityCase[];
}

/** Walk both trees; return "path: got vs want" strings for every numeric mismatch. */
function diff(got: unknown, want: unknown, tol: number, at: string, ignore: Set<string>, out: string[]) {
  if (ignore.has(at)) return;
  if (typeof want === "number") {
    if (typeof got !== "number" || Math.abs(got - want) > tol) out.push(`${at}: got ${got} want ${want}`);
  } else if (Array.isArray(want)) {
    if (!Array.isArray(got) || got.length !== want.length) {
      out.push(`${at}: length got ${Array.isArray(got) ? got.length : typeof got} want ${want.length}`);
      return;
    }
    want.forEach((w, i) => diff(got[i], w, tol, `${at}[${i}]`, ignore, out));
  } else if (want && typeof want === "object") {
    for (const [k, v] of Object.entries(want)) {
      diff((got as Record<string, unknown> | undefined)?.[k], v, tol, at ? `${at}.${k}` : k, ignore, out);
    }
  } else if (got !== want) {
    out.push(`${at}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
  }
}

/**
 * numpy's default_rng(seed).standard_normal is not reproducible in the browser, so the
 * new-city case needs the z draws. Use `case.z` if the fixture has it; otherwise back
 * them out of the recorded scenario mu (week 0, first cell) and sanity-check they look
 * N(0,1). Everything downstream of z is still compared field by field.
 */
function zFor(c: ParityCase): number[] | undefined {
  if (c.scenario.type !== "new_city") return undefined;
  if (c.z) return c.z;
  const { meta, draws, candidates } = posterior;
  const cand = candidates.find((x) => x.name === c.scenario.candidate)!;
  const [D, W, K] = c.mu.shape;
  const t = c.mu.t[0];
  const row = seasonRow(t, meta.season_period, meta.n_fourier);
  const z: number[] = [];
  for (let d = 0; d < D; d++) {
    const logMu = Math.log(c.mu.scenario_flat[d * W * K]);
    const season = row.reduce((a, r, k) => a + r * draws.season[d][k], 0);
    const aNew = logMu - Math.log(cand.shares[0]) - draws.a_seg[d][0] - season;
    const traitPart = cand.traits_z.reduce((a, v, p) => a + v * draws.g[d][p], 0);
    z.push((aNew - draws.g0[d] - traitPart) / draws.tau_city[d]);
  }
  return z;
}

describe.runIf(existsSync(fixturePath))("Python ↔ TS parity (contract/parity.fixture.json)", () => {
  const file = JSON.parse(readFileSync(fixturePath, "utf8")) as ParityFile;
  const ignore = new Set(file.ignore_fields);

  it("recovered new-city z draws look standard normal", () => {
    const c = file.cases.find((x) => x.scenario.type === "new_city");
    if (!c) return;
    const z = zFor(c)!;
    const m = z.reduce((a, b) => a + b, 0) / z.length;
    const sd = Math.sqrt(z.reduce((a, b) => a + (b - m) ** 2, 0) / z.length);
    expect(Math.abs(m)).toBeLessThan(0.5);
    expect(sd).toBeGreaterThan(0.6);
    expect(sd).toBeLessThan(1.5);
  });

  for (const c of file.cases) {
    it(`${c.name}: cell-level mu within ${file.tolerance.mu}`, () => {
      const g = muGrid(posterior, c.scenario, c.mu.weeks, zFor(c));
      expect(g.scenario).toHaveLength(c.mu.scenario_flat.length);
      const out: string[] = [];
      diff(g.baseline, c.mu.baseline_flat, file.tolerance.mu, "baseline", ignore, out);
      diff(g.scenario, c.mu.scenario_flat, file.tolerance.mu, "scenario", ignore, out);
      expect(out.slice(0, 10)).toEqual([]);
    });

    it(`${c.name}: every result field within tolerance`, () => {
      const got = simulate(posterior, c.scenario, { z: zFor(c) });
      const out: string[] = [];
      diff(got, c.expected, file.tolerance.summary, "", ignore, out);
      expect(out.slice(0, 15)).toEqual([]);
    });
  }
});
