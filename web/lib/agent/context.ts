import { readFile } from "node:fs/promises";
import path from "node:path";
import type { LaunchesFile, Posterior } from "@/lib/types";
import type { AgentContext, MarketScan, TrustFile } from "./types";

let cached: Omit<AgentContext, "live" | "geminiModel" | "embedModel"> | null = null;

async function readJson<T>(rel: string, origin?: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(path.join(process.cwd(), "public", rel), "utf8")) as T;
  } catch {
    if (!origin) return null;
    try {
      const res = await fetch(`${origin}/${rel}`);
      return res.ok ? ((await res.json()) as T) : null;
    } catch {
      return null;
    }
  }
}

export interface LoadOptions {
  origin?: string;
  /** BYOD: use the user's fitted posterior instead of the demo one. */
  posterior?: Posterior;
  trust?: TrustFile;
}

export async function loadContext(opts: LoadOptions = {}): Promise<AgentContext> {
  if (!cached) {
    const posterior =
      (await readJson<Posterior>("model/posterior.json", opts.origin)) ??
      (await readJson<Posterior>("model/posterior.fixture.json", opts.origin));
    if (!posterior) throw new Error("No posterior found in public/model/");
    cached = {
      posterior,
      trust: await readJson<TrustFile>("model/trust.json", opts.origin),
      launches: (await readJson<LaunchesFile>("model/launches.json", opts.origin))?.launches ?? [],
      scans: (await readJson<Record<string, MarketScan>>("model/market_scan.demo.json", opts.origin)) ?? {},
      docs: (await readJson<AgentContext["docs"]>("method/docs.json", opts.origin)) ?? [],
    };
  }
  // DEMO_MODE=1 (or no key) means cached/template answers; keys alone switch the live paths on.
  const demo = process.env.DEMO_MODE === "1";
  return {
    ...cached,
    posterior: opts.posterior ?? cached.posterior,
    trust: opts.posterior ? (opts.trust ?? null) : cached.trust,
    // the demo's launches and cached scans describe the demo company, not an upload
    launches: opts.posterior ? [] : cached.launches,
    scans: opts.posterior ? {} : cached.scans,
    live: {
      gemini: !demo && !!process.env.GEMINI_API_KEY,
      nimble: !demo && !!process.env.NIMBLE_API_KEY,
    },
    geminiModel: process.env.GEMINI_MODEL || "gemini-2.5-flash",
    embedModel: process.env.GEMINI_EMBED_MODEL || "gemini-embedding-001",
  };
}
