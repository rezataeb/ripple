import { money } from "@/lib/format";
import type { ScenarioResult } from "@/lib/types";

export function SegmentBars({ result }: { result: ScenarioResult | null }) {
  const rows = result?.segments ?? [];
  if (!result || rows.length === 0) return null;
  const extent = Math.max(...rows.flatMap((r) => [Math.abs(r.incremental.profit.p05), Math.abs(r.incremental.profit.p95)]), 1);
  const x = (v: number) => 50 + (v / extent) * 50;
  return (
    <div className="rounded-xl border p-4">
      <h2 className="mb-4 text-base font-medium">Incremental profit by customer group <span className="text-xs font-normal text-muted-foreground">(before fixed cost)</span></h2>
      <ul className="space-y-4">
        {rows.map((r) => {
          const pr = r.incremental.profit;
          return (
            <li key={r.segment} className="grid grid-cols-[110px_1fr_190px] items-center gap-3 text-sm">
              <span className="capitalize">{r.segment}</span>
              <div className="relative h-5">
                <div className="absolute inset-y-2 left-0 right-0 rounded-full bg-muted" />
                <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
                <div
                  className="absolute inset-y-1 rounded bg-brand-mid"
                  style={{ left: `${x(pr.p05)}%`, width: `${Math.max(x(pr.p95) - x(pr.p05), 0.5)}%` }}
                />
                <div
                  className="absolute inset-y-0 w-0.5 bg-brand"
                  style={{ left: `${x(pr.mean)}%` }}
                />
              </div>
              <span className="num text-right text-xs text-muted-foreground">
                <span className="text-foreground">{money(pr.mean)}</span> · {money(pr.p05)} to {money(pr.p95)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
