import type { RecoveryRow } from "@/lib/method";
import { cn } from "@/lib/utils";

/** Planted truth (dot) against the 94% HDI (bar), one row per parameter, each on its own scale. */
export function RecoveryChart({ rows }: { rows: RecoveryRow[] }) {
  const fmt = (x: number) => x.toFixed(2);
  return (
    <ul className="divide-y rounded-xl border" aria-label="Parameter recovery: true value against the 94% credible interval">
      {rows.map((r) => {
        const pad = (r.hdi_high - r.hdi_low) * 0.35 + 1e-9;
        const lo = Math.min(r.hdi_low, r.truth) - pad;
        const hi = Math.max(r.hdi_high, r.truth) + pad;
        const x = (v: number) => ((v - lo) / (hi - lo)) * 100;
        return (
          <li key={r.param} className="grid grid-cols-[minmax(120px,200px)_1fr_minmax(150px,210px)] items-center gap-3 px-4 py-2 text-sm">
            <span className="truncate" title={r.param}>
              {r.label ?? r.param}
            </span>
            <div className="relative h-4" aria-hidden>
              <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
              <div
                className={cn("absolute top-1 h-2 rounded-full", r.inside ? "bg-brand-mid" : "bg-loss/30")}
                style={{ left: `${x(r.hdi_low)}%`, width: `${Math.max(x(r.hdi_high) - x(r.hdi_low), 0.8)}%` }}
              />
              <div
                className={cn("absolute top-0 size-4 -translate-x-1/2 rounded-full border-2 bg-background", r.inside ? "border-gain" : "border-loss")}
                style={{ left: `${x(r.truth)}%` }}
                title={`truth ${fmt(r.truth)}`}
              />
            </div>
            <span className="num text-right text-xs text-muted-foreground">
              truth <span className={r.inside ? "text-gain" : "text-loss"}>{fmt(r.truth)}</span> · 94%: {fmt(r.hdi_low)} to {fmt(r.hdi_high)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
