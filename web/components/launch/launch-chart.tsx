"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { UpdateResult } from "@/lib/types";

export function LaunchChart({ result, prior }: { result: UpdateResult; prior: UpdateResult }) {
  const data = result.forecast.weekly.map((w, i) => {
    const p = prior.forecast.weekly[i];
    return {
      week: w.week,
      band: [w.units.p05, w.units.p95] as [number, number],
      median: w.units.p50,
      cf: w.counterfactual_units ? [w.counterfactual_units.p05, w.counterfactual_units.p95] : undefined,
      priorLo: result.weeks_used > 0 ? p.units.p05 : undefined,
      priorHi: result.weeks_used > 0 ? p.units.p95 : undefined,
      observed: w.observed_units ?? undefined,
    };
  });
  const used = result.weeks_used;
  return (
    <div className="rounded-xl border p-4">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-medium">Weekly units: forecast and what really happened</h2>
        <p className="text-xs text-muted-foreground">
          Indigo band: updated forecast (90%) · dashed: forecast before any data · grey: no-change expectation · dots: real weeks
        </p>
      </div>
      <div className="h-[340px] w-full" role="img" aria-label="Updated weekly forecast band with observed points">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="week" tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} tickFormatter={(v) => `wk ${v}`} />
            <YAxis tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} width={52} domain={[0, "auto"]} />
            <Tooltip
              contentStyle={{ borderRadius: 12, border: "1px solid var(--border)", fontSize: 12 }}
              formatter={(v) => (Array.isArray(v) ? `${Math.round(v[0])} to ${Math.round(v[1])}` : Math.round(Number(v)))}
            />
            <Area dataKey="cf" stroke="none" fill="#a3a3a3" fillOpacity={0.18} isAnimationActive={false} name="No change, 90%" />
            <Area dataKey="band" stroke="none" fill="var(--brand)" fillOpacity={0.2} animationDuration={250} name="Forecast, 90%" />
            <Line dataKey="priorLo" stroke="var(--brand)" strokeOpacity={0.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} name="Before data, low" />
            <Line dataKey="priorHi" stroke="var(--brand)" strokeOpacity={0.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} name="Before data, high" />
            <Line dataKey="median" stroke="var(--brand)" strokeWidth={2} dot={false} animationDuration={250} name="Forecast, median" />
            <Line dataKey="observed" stroke="none" dot={{ r: 4, fill: "var(--foreground)", stroke: "var(--background)", strokeWidth: 1 }} isAnimationActive={false} name="Observed" />
            {used > 0 && <ReferenceLine x={used} stroke="var(--muted-text)" strokeDasharray="2 4" label={{ value: "now", position: "insideTopRight", fontSize: 12, fill: "var(--muted-text)" }} />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
