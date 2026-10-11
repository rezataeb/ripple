"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Posterior, ScenarioResult } from "@/lib/types";

interface Point {
  label: string;
  observed?: number;
  base?: number;
  base_band?: [number, number];
  scen?: number;
  scen_band?: [number, number];
}

function historyPoints(p: Posterior): Point[] {
  const h = p.history;
  const byT = new Map<number, { label: string; units: number }>();
  h.t.forEach((t, i) => {
    const cur = byT.get(t) ?? { label: h.week_start[i], units: 0 };
    cur.units += h.units[i];
    byT.set(t, cur);
  });
  return [...byT.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => ({ label: v.label, observed: v.units }));
}

export function FanChart({
  posterior,
  result,
}: {
  posterior: Posterior;
  result: ScenarioResult | null;
}) {
  const hist = historyPoints(posterior);
  const future: Point[] = (result?.weekly ?? []).map((w) => ({
    label: `+${w.week + 1}`,
    base: w.baseline_units.p50,
    base_band: [w.baseline_units.p05, w.baseline_units.p95],
    scen: w.scenario_units.p50,
    scen_band: [w.scenario_units.p05, w.scenario_units.p95],
  }));
  const data = [...hist, ...future];
  const promoWeeks = (result?.weekly ?? []).filter((w) => w.promo_active);
  const first = promoWeeks[0]?.week;
  const last = promoWeeks[promoWeeks.length - 1]?.week;

  return (
    <div className="rounded-xl border p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-base font-medium">Weekly units, all selected cells</h2>
        <p className="text-xs text-muted-foreground">
          {result ? "Grey: no change · Indigo: with change · bands are 90%" : "Observed history (the forecast appears when you pick a scenario)"}
        </p>
      </div>
      <div className="h-[320px] w-full" role="img" aria-label="Weekly units: history and forecast bands">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} minTickGap={32} />
            <YAxis tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} width={48} />
            <Tooltip
              contentStyle={{ borderRadius: 12, border: "1px solid var(--border)", fontSize: 12 }}
              formatter={(v) => (Array.isArray(v) ? `${Math.round(v[0])} to ${Math.round(v[1])}` : Math.round(Number(v)))}
            />
            {first !== undefined && last !== undefined && (
              <ReferenceArea
                x1={`+${first + 1}`}
                x2={`+${last + 1}`}
                fill="var(--brand)"
                fillOpacity={0.06}
                ifOverflow="visible"
              />
            )}
            <Area dataKey="base_band" stroke="none" fill="#a3a3a3" fillOpacity={0.18} isAnimationActive={false} name="No change, 90%" />
            <Area dataKey="scen_band" stroke="none" fill="var(--brand)" fillOpacity={0.2} animationDuration={200} name="With change, 90%" />
            <Line dataKey="observed" stroke="var(--foreground)" strokeWidth={1.5} dot={false} name="Observed" />
            <Line dataKey="base" stroke="#737373" strokeWidth={1.5} strokeDasharray="4 3" dot={false} name="No change, median" />
            <Line dataKey="scen" stroke="var(--brand)" strokeWidth={2} dot={false} animationDuration={200} name="With change, median" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
