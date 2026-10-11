"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ScenarioResult } from "@/lib/types";

export function IncrementalChart({ result }: { result: ScenarioResult | null }) {
  if (!result) return null;
  const data = result.weekly.map((w) => ({
    label: `+${w.week + 1}`,
    band: [w.incremental_units?.p05 ?? 0, w.incremental_units?.p95 ?? 0] as [number, number],
    inner: [w.incremental_units?.p25 ?? 0, w.incremental_units?.p75 ?? 0] as [number, number],
    median: w.incremental_units?.p50 ?? 0,
  }));
  const promo = result.weekly.filter((w) => w.promo_active);
  const dip = result.weekly.find((w) => w.dip_active);
  return (
    <div className="rounded-xl border p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-base font-medium">Weekly change in units vs. no change</h2>
        <p className="text-xs text-muted-foreground">
          Line: median · light band 90% · dark band 50%
          {promo.length > 0 && " · shaded: promo weeks"}
          {dip && " · after the promo: pull-forward dip"}
        </p>
      </div>
      <div className="h-[200px] w-full" role="img" aria-label="Weekly incremental units with 90% band">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} minTickGap={24} />
            <YAxis tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} width={48} />
            <Tooltip
              contentStyle={{ borderRadius: 12, border: "1px solid var(--border)", fontSize: 12 }}
              formatter={(v) => (Array.isArray(v) ? `${Math.round(v[0])} to ${Math.round(v[1])}` : Math.round(Number(v)))}
            />
            {promo.length > 0 && (
              <ReferenceArea
                x1={`+${promo[0].week + 1}`}
                x2={`+${promo[promo.length - 1].week + 1}`}
                fill="var(--brand)"
                fillOpacity={0.06}
              />
            )}
            <ReferenceLine y={0} stroke="var(--muted-text)" strokeDasharray="3 3" />
            <Area dataKey="band" stroke="none" fill="var(--brand)" fillOpacity={0.12} isAnimationActive={false} name="90% interval" />
            <Area dataKey="inner" stroke="none" fill="var(--brand)" fillOpacity={0.2} isAnimationActive={false} name="50% interval" />
            <Line dataKey="median" stroke="var(--brand)" strokeWidth={2} dot={false} isAnimationActive={false} name="Median" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
