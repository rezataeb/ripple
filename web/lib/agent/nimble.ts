// Live competitor scan through Nimble's Search API.
// Docs: POST https://sdk.nimbleway.com/v2/search, Authorization: Bearer <key>,
// body {query, max_results, focus, output_format}; response {results:[{title, description, url, content, ...}]}.
import type { Competitor, MarketScan } from "./types";

const ENDPOINT = "https://sdk.nimbleway.com/v2/search";
const liveCache = new Map<string, MarketScan>();

interface NimbleResult {
  title?: string;
  description?: string;
  url?: string;
  content?: string;
  additional_data?: Record<string, unknown>;
}

const PRICE = /\$\s?(\d{1,4}(?:\.\d{2})?)/g;
const OFFER = /(\d{1,2}\s?%\s*off[^.;|]{0,40}|free shipping[^.;|]{0,30}|first (?:month|box|order)[^.;|]{0,30})/i;

/** Plain extraction from search snippets, used when no LLM is available to read them. */
export function extractCompetitors(results: NimbleResult[], now: string): Competitor[] {
  const out: Competitor[] = [];
  for (const r of results) {
    if (!r.url) continue;
    const text = `${r.title ?? ""} ${r.description ?? ""}`;
    const found = [...text.matchAll(PRICE)].map((m) => Number(m[1])).find((p) => p > 0);
    const ap = r.additional_data?.price;
    const price = found ?? (typeof ap === "number" && ap > 0 ? ap : null);
    if (price === null) continue;
    let host = r.url;
    try {
      host = new URL(r.url).hostname.replace(/^www\./, "");
    } catch {
      /* keep raw */
    }
    out.push({
      competitor: (r.title ?? host).slice(0, 80),
      price,
      offer: text.match(OFFER)?.[1]?.trim() ?? null,
      url: r.url,
      seen_at: now,
      confidence: 0.5,
    });
  }
  return out;
}

export async function nimbleSearch(query: string, apiKey: string): Promise<NimbleResult[]> {
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ query, max_results: 8, focus: "general", output_format: "plain_text" }),
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error(`Nimble search failed: HTTP ${res.status}`);
  const json = (await res.json()) as { results?: NimbleResult[] };
  return json.results ?? [];
}

export async function liveScan(
  product: string,
  city: string,
  apiKey: string,
  extract?: (results: NimbleResult[]) => Promise<Competitor[]>,
): Promise<MarketScan> {
  const query = `${product} price ${city}`.trim();
  const hit = liveCache.get(query);
  if (hit) return hit;
  const now = new Date().toISOString();
  const results = await nimbleSearch(query, apiKey);
  const competitors = (extract ? await extract(results) : extractCompetitors(results, now)).filter(
    (c) => c.price !== null,
  );
  const scan: MarketScan = {
    query,
    product,
    city,
    source: "nimble",
    fetched_at: now,
    competitors,
    comparison: null,
    note: "Live web results via Nimble. Treat as context, not as a model input.",
    sources: competitors.map((c) => ({ title: c.competitor, url: c.url, kind: "web" as const })),
  };
  liveCache.set(query, scan);
  return scan;
}
