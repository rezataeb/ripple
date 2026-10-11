"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { money, percent } from "@/lib/format";
import type { ScenarioResult } from "@/lib/types";
import { cn } from "@/lib/utils";

export function Verdict({ result }: { result: ScenarioResult | null }) {
  if (!result) {
    return (
      <section aria-busy="true" className="space-y-3">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-12 w-64" />
        <Skeleton className="h-4 w-80" />
        <Skeleton className="h-1.5 w-full" />
      </section>
    );
  }
  const profit = result.incremental.profit;
  const p = result.p_profit_positive;
  const tone = profit.mean >= 0 ? "text-gain" : "text-loss";
  return (
    <section className="space-y-3" aria-live="polite">
      <p className="text-sm text-muted-foreground">Incremental profit vs. no change</p>
      <p className={cn("num-hero", tone)}>{money(profit.mean)}</p>
      <p className="num text-sm text-muted-foreground">
        90% interval {money(profit.p05)} to {money(profit.p95)} ·{" "}
        <span className="font-medium text-foreground">{percent(p)}</span> chance it pays
      </p>
      <div
        className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label={`${percent(p)} chance the change pays`}
      >
        <div
          className="h-full rounded-full bg-brand transition-[width] duration-200 ease-out"
          style={{ width: `${p * 100}%` }}
        />
      </div>
    </section>
  );
}
