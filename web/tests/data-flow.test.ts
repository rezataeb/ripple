import { describe, expect, it } from "vitest";
import { parseCsvHead, ruleMapping } from "@/lib/data-flow";

describe("rule-based column mapping", () => {
  it("maps exact names with full confidence", () => {
    const m = ruleMapping(["week", "city", "segment", "units", "price", "promo", "feature"]);
    expect(m.mapping).toMatchObject({ week: "week", units: "units", price: "price", city: "city" });
    expect(m.confidence.week).toBe(1);
  });
  it("maps common synonyms with lower confidence and never reuses a column", () => {
    const m = ruleMapping(["Week Start", "Market", "Qty Sold", "Unit Price", "On Promo"]);
    expect(m.mapping).toMatchObject({ week: "Week Start", city: "Market", units: "Qty Sold", price: "Unit Price", promo: "On Promo" });
    expect(m.confidence.units).toBeLessThan(1);
    expect(new Set(Object.values(m.mapping)).size).toBe(Object.values(m.mapping).length);
  });
  it("leaves unknown columns unmapped instead of guessing", () => {
    const m = ruleMapping(["foo", "bar"]);
    expect(m.mapping).toEqual({});
  });
});

describe("CSV head", () => {
  it("reads quoted fields and a few sample rows", () => {
    const { columns, sample } = parseCsvHead('week,city,units\n2025-01-06,"Boston, MA",120\n2025-01-13,Austin,98\n');
    expect(columns).toEqual(["week", "city", "units"]);
    expect(sample[0].city).toBe("Boston, MA");
    expect(sample).toHaveLength(2);
  });
});
