// Template answers. Used with no key, and whenever the number guard rejects a model answer.
// Every number is pasted from a claim, so the guard passes by construction.
import type { Claim, MarketScan } from "./types";

const get = (claims: Claim[], label: string) => claims.find((c) => c.label === label)?.value;

export interface TemplateInput {
  kind: "scenario" | "trust" | "launch" | "clarify";
  claims: Claim[];
  scenarioType?: "promo" | "price" | "new_city";
  cityLabel?: string;
  segments?: string[];
  scan?: MarketScan;
  warnings?: string[];
  question?: string;
  launchLabel?: string;
  launchUnknownLabel?: string;
  gateOk?: boolean;
  checkSentences?: string[];
  docSentences?: string[];
  pPays?: number;
  extrapolation?: boolean;
  segmentLines?: { segment: string }[];
}

function verdict(p: number): string {
  if (p >= 0.8) return "Likely yes";
  if (p >= 0.5) return "Possibly";
  return "Probably not";
}

export function composeTemplate(i: TemplateInput): string {
  const c = i.claims;
  if (i.kind === "clarify") return i.question ?? "Could you say a bit more?";

  if (i.kind === "trust") {
    const lines = [
      `Ripple passed ${get(c, "Checks passed") ?? "its"} checks. In plain words:`,
      ...(i.checkSentences ?? []).map((s) => `- ${s}`),
    ];
    if (i.docSentences?.length) lines.push("", ...i.docSentences);
    lines.push("", "The model makes the numbers. This answer only repeats what the checks measured.");
    return lines.join("\n");
  }

  if (i.kind === "launch") {
    const out = [
      `${i.launchLabel}: with ${get(c, "Weeks of real data used")} weeks of real data, the chance it pays moved from ${get(c, "Chance it pays before any data")} to ${get(c, "Chance it pays now")}.`,
      `What the launch has taught us: ${i.launchUnknownLabel} is ${get(c, i.launchUnknownLabel ?? "")} (90%: ${get(c, `${i.launchUnknownLabel}, 90% interval`)}).`,
    ];
    const fp = get(c, "Forecast profit");
    if (fp) out.push(`Added profit over the horizon, against no change: ${fp} (90%: ${get(c, "Forecast profit, 90% interval")}).`);
    out.push(`Update status: ${get(c, "Update reliability")}.`);
    return out.join("\n");
  }

  // scenario
  const profit = get(c, "Incremental profit vs. no change");
  const iv = get(c, "Incremental profit, 90% interval");
  const pays = get(c, "Chance the change pays");
  const horizon = get(c, "Horizon");
  const p = i.pPays ?? 0;
  const lines: string[] = [];
  if (i.scenarioType === "promo") {
    lines.push(
      `${verdict(p)}. A ${get(c, "Discount")} discount for ${get(c, "Promo length")} adds ${profit} of profit over ${horizon} (90% interval ${iv}), with a ${pays} chance it pays.`,
    );
  } else if (i.scenarioType === "price") {
    lines.push(
      `${verdict(p)}. A price change of ${get(c, "Price change")} changes profit by ${profit} over ${horizon} (90% interval ${iv}), with a ${pays} chance it pays.`,
    );
  } else {
    lines.push(
      `${verdict(p)}. Launching in ${i.cityLabel} adds ${profit} of profit over ${horizon} (90% interval ${iv}), with a ${pays} chance it pays.`,
    );
  }
  lines.push(`Incremental units: ${get(c, "Incremental units")} (90%: ${get(c, "Incremental units, 90% interval")}).`);
  const segs = (i.segmentLines ?? [])
    .map((s) => `${s.segment} ${get(c, `${s.segment}: incremental profit`)} (90%: ${get(c, `${s.segment}: incremental profit, 90% interval`)})`)
    .filter((s) => !s.includes("undefined"));
  if (segs.length > 0) lines.push(`By group: ${segs.join("; ")}.`);
  if (i.extrapolation && i.warnings?.length) lines.push(i.warnings[0]);
  if (i.scan?.comparison) {
    const tag = i.scan.source === "demo" ? "illustrative sample, not a live scan" : "from the web";
    lines.push(`Market context (${tag}): ${i.scan.comparison.sentence} It is context only; the model did not use it.`);
  }
  return lines.join("\n");
}
