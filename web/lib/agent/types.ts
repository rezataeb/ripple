import type { Interval, Launch, Posterior, Scenario, ScenarioResult, UpdateResult } from "@/lib/types";

export type ClaimKind = "model" | "web" | "docs";

/** A number (or short quantity) the answer may quote. `value` is the exact display string. */
export interface Claim {
  value: string;
  label: string;
  tool_call_id: string;
  kind: ClaimKind;
}

export interface Source {
  title: string;
  url: string;
  kind?: ClaimKind;
}

export interface Competitor {
  competitor: string;
  price: number | null;
  unit?: string;
  offer: string | null;
  url: string;
  seen_at: string;
  confidence?: number;
}

export interface MarketScan {
  query: string;
  product: string;
  city: string;
  source: "demo" | "nimble";
  fetched_at: string;
  competitors: Competitor[];
  comparison: {
    your_price: number;
    above_count: number;
    below_count: number;
    total: number;
    median_competitor_price: number | null;
    sentence: string;
  } | null;
  note?: string;
  sources: Source[];
}

export interface TrustFile {
  summary: { passed: number; total: number; headline?: string };
  checks: { name: string; passed: boolean; value: number | null; threshold: number | string | null; plain_english: string }[];
  model_facts: {
    elasticity_by_segment: ({ segment: string } & Interval)[];
    promo_lift_by_segment: ({ segment: string } & Interval)[];
    post_promo_dip: Interval;
    feature_lift: Interval;
  };
  limits?: string[];
}

export interface DocChunk {
  id: string;
  file: string;
  title: string;
  text: string;
}

export interface AgentContext {
  posterior: Posterior;
  trust: TrustFile | null;
  launches: Launch[];
  scans: Record<string, MarketScan>;
  docs: { file: string; title: string; text: string }[];
  live: { gemini: boolean; nimble: boolean };
  geminiModel: string;
  embedModel: string;
}

export interface ToolOutput {
  /** JSON given back to the model */
  output: Record<string, unknown>;
  claims: Omit<Claim, "tool_call_id">[];
  sources?: Source[];
  scenario?: Scenario;
  result?: ScenarioResult;
  update?: UpdateResult;
  scan?: MarketScan;
  openQuestion?: string;
}

export interface Step {
  id: string;
  label: string;
  status: "running" | "done" | "error";
}

export interface AgentAnswer {
  text: string;
  claims: Claim[];
  sources: Source[];
  scenario?: Scenario;
  open_question?: string;
  mode: "gemini" | "template";
  guard: { passed: boolean; offending: string[] };
  result_summary?: { profit: Interval; p_profit_positive: number };
}
