"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { percentScale, signedPercent } from "@/lib/launch";
import type { UpdateResult } from "@/lib/types";

export function DensityPlot({ result, showTruth }: { result: UpdateResult; showTruth: boolean }) {
  const u = result.unknown;
  const d = u.density;
  if (!d) return null;
  const data = d.x.map((x, i) => ({ x, prior: d.prior[i], posterior: d.posterior[i] }));
  const pct = u.display === "percent_lift" && u.prior_percent;
  const scale = pct ? percentScale(u.prior, u.prior_percent) : 1;
  const tick = (x: number) => (pct ? signedPercent(Math.exp(x * scale) - 1) : x.toFixed(2));
  const post = pct ? u.posterior_percent! : u.posterior;
  const prior = pct ? u.prior_percent! : u.prior;
  const fmt = (v: number) => (pct ? signedPercent(v, 1) : v.toFixed(2));

  return (
    <div className="rounded-xl border p-4">
      <h2 className="text-base font-medium">{u.label}: prior vs. what the data says</h2>
      <p className="num mt-1 text-sm">
        <span className="text-foreground">{fmt(post.mean)}</span>{" "}
        <span className="text-muted-foreground">
          (90%: {fmt(post.p05)} to {fmt(post.p95)})
        </span>
      </p>
      <p className="num text-xs text-muted-foreground">
        Before any data: {fmt(prior.mean)} (90%: {fmt(prior.p05)} to {fmt(prior.p95)})
      </p>
      <div className="mt-3 h-[170px]" role="img" aria-label="Prior and posterior density of the unknown">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 22, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="x"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={tick}
              tick={{ fontSize: 12, fill: "var(--muted-text)" }}
              tickLine={false}
              axisLine={false}
              tickCount={6}
            />
            <YAxis hide />
            <Area dataKey="prior" stroke="#a3a3a3" fill="#a3a3a3" fillOpacity={0.15} strokeDasharray="4 3" isAnimationActive={false} name="Prior" />
            <Area dataKey="posterior" stroke="var(--brand)" fill="var(--brand)" fillOpacity={0.25} animationDuration={250} name="Posterior" />
            {showTruth && u.truth != null && (
              <ReferenceLine
                x={u.truth}
                stroke="var(--gain)"
                strokeWidth={2}
                label={{ value: "truth", position: "top", fontSize: 12, fill: "var(--gain)" }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
