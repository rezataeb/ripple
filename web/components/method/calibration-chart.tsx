"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CalibrationArm } from "@/lib/method";

const COLORS = ["var(--brand)", "#0ea5e9", "var(--loss)"];

export function CalibrationChart({ arms, target }: { arms: CalibrationArm[]; target: number }) {
  const weeks = Math.max(...arms.map((a) => a.by_week.length));
  const data = Array.from({ length: weeks }, (_, i) => {
    const row: Record<string, number> = { week: i + 1 };
    for (const a of arms) if (a.by_week[i]) row[a.id] = a.by_week[i].coverage_total;
    return row;
  });
  return (
    <div className="h-[320px] w-full" role="img" aria-label="Coverage of the 90% interval by week of real data, per arm">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="week" tickFormatter={(v) => `wk ${v}`} tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 12, fill: "var(--muted-text)" }} tickLine={false} axisLine={false} width={44} />
          <Tooltip
            contentStyle={{ borderRadius: 12, border: "1px solid var(--border)", fontSize: 12 }}
            formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
            labelFormatter={(l) => `After ${l} weeks of real data`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine y={target} stroke="var(--gain)" strokeDasharray="4 3" label={{ value: `target ${Math.round(target * 100)}%`, position: "insideTopRight", fontSize: 12, fill: "var(--gain)" }} />
          {arms.map((a, i) => (
            <Line
              key={a.id}
              dataKey={a.id}
              name={a.label}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              strokeDasharray={a.assumptions_hold ? undefined : "5 4"}
              dot={{ r: 3 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
