// Bring-your-own-data helpers: CSV header reading, a rule-based column mapper (the fallback for the
// Gemini mapping assistant), and a small client for the Ripple API.
export const RIPPLE_COLUMNS = [
  { key: "week", required: true, hint: "date the week starts" },
  { key: "units", required: true, hint: "units sold (a count)" },
  { key: "price", required: true, hint: "price paid per unit" },
  { key: "city", required: false, hint: "market or city; 'All' if absent" },
  { key: "segment", required: false, hint: "customer group; 'All' if absent" },
  { key: "promo", required: false, hint: "1 if a promotion ran" },
  { key: "feature", required: false, hint: "1 if the feature or campaign was on" },
  { key: "holdout", required: false, hint: "1 if in a random control group" },
] as const;

export type RippleKey = (typeof RIPPLE_COLUMNS)[number]["key"];

export interface MappingResult {
  mapping: Partial<Record<RippleKey, string>>;
  confidence: Partial<Record<RippleKey, number>>;
  notes: string;
  source: "gemini" | "rules";
}

const SYNONYMS: Record<RippleKey, string[]> = {
  week: ["week", "week_start", "weekstart", "date", "period", "week_begin", "week_of", "ds", "wk"],
  units: ["units", "qty", "quantity", "orders", "sales", "units_sold", "count", "subscriptions", "volume", "sold"],
  price: ["price", "unit_price", "avg_price", "net_price", "selling_price", "price_paid", "asp"],
  city: ["city", "market", "region", "location", "metro", "store", "area", "dma"],
  segment: ["segment", "customer_segment", "cohort", "group", "tier", "customer_type", "audience"],
  promo: ["promo", "promotion", "is_promo", "on_promo", "discount_flag", "on_sale", "coupon", "deal"],
  feature: ["feature", "featured", "email", "campaign", "email_flag", "newsletter", "marketing"],
  holdout: ["holdout", "hold_out", "control", "is_control", "control_group"],
};

const norm = (s: string) => s.trim().toLowerCase().replace(/[\s-]+/g, "_");

/** Names-only mapping: exact match 1.0, synonym 0.8, substring 0.55. */
export function ruleMapping(columns: string[]): MappingResult {
  const mapping: MappingResult["mapping"] = {};
  const confidence: MappingResult["confidence"] = {};
  const used = new Set<string>();
  for (const { key } of RIPPLE_COLUMNS) {
    let best: { col: string; score: number } | null = null;
    for (const col of columns) {
      if (used.has(col)) continue;
      const n = norm(col);
      const syn = SYNONYMS[key];
      const score = n === key ? 1 : syn.includes(n) ? 0.8 : syn.some((s) => s.length > 3 && n.includes(s)) ? 0.55 : 0;
      if (score > 0 && (!best || score > best.score)) best = { col, score };
    }
    if (best) {
      mapping[key] = best.col;
      confidence[key] = best.score;
      used.add(best.col);
    }
  }
  return {
    mapping,
    confidence,
    notes: "Matched by column names only. Check each row before continuing.",
    source: "rules",
  };
}

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (q && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else q = !q;
    } else if (ch === "," && !q) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

/** Header and all rows (quoted fields supported; no multi-line cells). */
export function parseCsv(text: string): { columns: string[]; rows: Record<string, string>[] } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return { columns: [], rows: [] };
  const columns = parseCsvLine(lines[0]);
  const rows = lines.slice(1).map((l) => {
    const v = parseCsvLine(l);
    return Object.fromEntries(columns.map((c, i) => [c, v[i] ?? ""]));
  });
  return { columns, rows };
}

/** Header and a few rows from the CSV text. */
export function parseCsvHead(text: string, rows = 3): { columns: string[]; sample: Record<string, string>[] } {
  const all = parseCsv(text);
  return { columns: all.columns, sample: all.rows.slice(0, rows) };
}

// ---- API client --------------------------------------------------------------------
export const API_URL = (process.env.NEXT_PUBLIC_RIPPLE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export interface ApiPreview {
  dataset_id: string;
  name: string;
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
  detected_mapping: Record<string, string>;
}

export interface ApiValidation {
  ok: boolean;
  rows: number;
  missing_required: string[];
  errors: string[];
  warnings: string[];
  info: string[];
  weeks?: number | null;
  week_range?: string[] | null;
  cities?: number | null;
  segments?: number | null;
  missing_values?: number | null;
  negative_units?: number | null;
  price_cv?: number | null;
}

export interface ApiJob {
  job_id: string;
  status: "queued" | "running" | "done" | "error" | string;
  step: string;
  progress: number;
  error?: string | null;
  posterior?: unknown;
  trust?: unknown;
}

async function ok<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      const d = j.detail;
      detail = typeof d === "string" ? d : d?.errors ? `${d.message} ${d.errors.join(" ")}` : JSON.stringify(d);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(2500) }).then((r) => ok<{ status: string; model_available: boolean }>(r)),
  upload: (file: File, cities?: File | null) => {
    const fd = new FormData();
    fd.append("file", file);
    if (cities) fd.append("cities", cities);
    return fetch(`${API_URL}/datasets`, { method: "POST", body: fd }).then((r) => ok<ApiPreview>(r));
  },
  validate: (id: string, mapping: Record<string, string>) =>
    fetch(`${API_URL}/datasets/${id}/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mapping }),
    }).then((r) => ok<ApiValidation>(r)),
  fit: (id: string, body: { mapping: Record<string, string>; unit_cost: number; base_price?: number }) =>
    fetch(`${API_URL}/datasets/${id}/fit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => ok<{ job_id: string }>(r)),
  job: (id: string) => fetch(`${API_URL}/jobs/${id}`).then((r) => ok<ApiJob>(r)),
  remove: (id: string) => fetch(`${API_URL}/datasets/${id}`, { method: "DELETE" }).then((r) => ok<{ deleted: string }>(r)),
};
