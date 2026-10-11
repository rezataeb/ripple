// Number guard: every number in the answer text must be a number some tool returned.
import type { Claim } from "./types";

const ISO_DATE = /\b\d{4}-\d{2}-\d{2}\b/g;
// identifiers such as share_age_18_34 or e[students] carry digits that are names, not numbers
const IDENT = /\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b/g;
const NUM = /\d[\d,]*(?:\.\d+)?\s?(?:[kKmM]\b|%)?/g;

interface Parsed {
  value: number;
  unit: string;
}

function parseToken(raw: string): Parsed | null {
  const t = raw.replace(/\s/g, "");
  const m = t.match(/^([\d,]*\.?\d+)([kKmM%]?)$/);
  if (!m) return null;
  const n = Number(m[1].replace(/,/g, ""));
  if (!Number.isFinite(n)) return null;
  const u = m[2].toLowerCase();
  return { value: u === "k" ? n * 1e3 : u === "m" ? n * 1e6 : n, unit: u === "%" ? "%" : "" };
}

export function numbersIn(text: string): Parsed[] {
  const out: Parsed[] = [];
  for (const m of text.replace(ISO_DATE, " ").replace(IDENT, " ").match(NUM) ?? []) {
    const p = parseToken(m);
    if (p) out.push(p);
  }
  return out;
}

const same = (a: Parsed, b: Parsed) =>
  a.unit === b.unit && Math.abs(a.value - b.value) <= 1e-9 * Math.max(1, Math.abs(b.value));

/** Fixed reporting convention, not a result: every interval Ripple shows is a 90% interval. */
const CONSTANTS: Parsed[] = [{ value: 90, unit: "%" }];

export function numberGuard(text: string, claims: Pick<Claim, "value">[]): { passed: boolean; offending: string[] } {
  const allowed = [...CONSTANTS, ...claims.flatMap((c) => numbersIn(c.value))];
  const offending = numbersIn(text)
    .filter((n) => !allowed.some((a) => same(n, a)))
    .map((n) => `${n.value}${n.unit}`);
  return { passed: offending.length === 0, offending };
}
