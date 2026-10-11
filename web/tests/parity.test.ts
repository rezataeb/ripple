import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { muGrid, newCityZ, simulate } from "@/lib/simulate";
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

describe("served fixture", () => {
  it("web/public/model/posterior.fixture.json equals contract/posterior.fixture.json (npm run sync:contract)", () => {
    const served = path.join(__dirname, "../public/model/posterior.fixture.json");
    expect(readFileSync(served, "utf8")).toBe(
      readFileSync(path.join(root, "contract/posterior.fixture.json"), "utf8"),
    );
  });
});

describe.runIf(existsSync(fixturePath))("Python ↔ TS parity (contract/parity.fixture.json)", () => {
  const file = JSON.parse(readFileSync(fixturePath, "utf8")) as ParityFile;
  const ignore = new Set(file.ignore_fields);

  it("new-city z is derived from draws.z_new exactly as the fixture's z", () => {
    for (const c of file.cases.filter((x) => x.scenario.type === "new_city")) {
      expect(c.z, `${c.name} must carry z`).toBeDefined();
      expect(newCityZ(posterior.draws, c.scenario.seed ?? 7, posterior.meta.n_draws)).toEqual(c.z);
    }
  });

  for (const c of file.cases) {
    it(`${c.name}: cell-level mu within ${file.tolerance.mu}`, () => {
      const g = muGrid(posterior, c.scenario, c.mu.weeks);
      expect(g.scenario).toHaveLength(c.mu.scenario_flat.length);
      const out: string[] = [];
      diff(g.baseline, c.mu.baseline_flat, file.tolerance.mu, "baseline", ignore, out);
      diff(g.scenario, c.mu.scenario_flat, file.tolerance.mu, "scenario", ignore, out);
      expect(out.slice(0, 10)).toEqual([]);
    });

    it(`${c.name}: every result field within tolerance`, () => {
      const got = simulate(posterior, c.scenario);
      const out: string[] = [];
      diff(got, c.expected, file.tolerance.summary, "", ignore, out);
      expect(out.slice(0, 15)).toEqual([]);
    });
  }
});
