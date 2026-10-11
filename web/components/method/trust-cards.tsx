import { Badge } from "@/components/ui/badge";
import type { TrustCheck } from "@/lib/method";
import { cn } from "@/lib/utils";

const fmtThreshold = (c: TrustCheck) => {
  const op = c.comparison === "lte" ? "≤" : c.comparison === "gte" ? "≥" : "";
  return `${op} ${c.threshold}`.trim();
};

export function TrustCards({ checks }: { checks: TrustCheck[] }) {
  return (
    <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {checks.map((c) => (
        <li key={c.name} className="space-y-2 rounded-xl border p-4">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium">{c.name}</span>
            <Badge variant="outline" className={cn(c.passed ? "text-gain" : c.skipped_reason ? "text-muted-foreground" : "text-loss")}>
              {c.passed ? "passed" : c.skipped_reason ? "skipped" : "failed"}
            </Badge>
          </div>
          <p className="num text-2xl font-semibold">{c.value === null ? "n/a" : Number(c.value.toFixed(4))}</p>
          <p className="num text-xs text-muted-foreground">bar: {fmtThreshold(c)}</p>
          <p className="text-sm">{c.plain_english}</p>
          {c.detail && <p className="text-xs text-muted-foreground">{c.detail}</p>}
          {c.skipped_reason && <p className="text-xs text-muted-foreground">{c.skipped_reason}</p>}
        </li>
      ))}
    </ul>
  );
}
