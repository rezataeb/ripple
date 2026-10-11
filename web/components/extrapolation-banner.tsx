import type { ScenarioResult } from "@/lib/types";

export function Warnings({ result }: { result: ScenarioResult | null }) {
  const f = result?.flags;
  if (!f) return null;
  const items: string[] = [];
  if (f.extrapolation)
    items.push(
      `Extrapolation: ${(f.out_of_range_traits ?? []).join(", ") || "a trait"} is outside the range of the training cities. Treat this forecast with caution.`,
    );
  if (f.no_price_variation) items.push("Price never varied in this data, so elasticity comes from the prior, not the data.");
  if (f.promo_disabled) items.push("This dataset has no promo column, so promotions are disabled.");
  items.push(...(f.warnings ?? []));
  if (items.length === 0) return null;
  return (
    <div className="space-y-1 rounded-xl border border-warning/40 bg-warning/5 px-4 py-3 text-sm" role="status">
      {items.map((t) => (
        <p key={t}>
          <span className="num mr-2 text-warning">!</span>
          {t}
        </p>
      ))}
    </div>
  );
}
