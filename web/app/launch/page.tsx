"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { DensityPlot } from "@/components/launch/density-plot";
import { LaunchChart } from "@/components/launch/launch-chart";
import { Topbar } from "@/components/topbar";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { money, percent, units } from "@/lib/format";
import { loadLaunches } from "@/lib/launch";
import { loadPosterior } from "@/lib/posterior";
import { updateWithLaunch } from "@/lib/update";
import type { Launch, Posterior, UpdateResult } from "@/lib/types";
import { cn } from "@/lib/utils";

// Module-level memo: the 9 week positions per launch are revisited constantly while sliding.
const updateCache = new Map<string, UpdateResult>();
function cachedUpdate(p: Posterior, l: Launch, weeks: number, sigma: number | null): UpdateResult {
  const key = `${p.meta.fitted_at ?? p.meta.seed}|${l.id}|${weeks}|${sigma ?? "default"}`;
  let r = updateCache.get(key);
  if (!r) {
    r = updateWithLaunch(p, l, weeks, { priorSigma: sigma ?? undefined });
    updateCache.set(key, r);
  }
  return r;
}

export default function LaunchPage() {
  const [posterior, setPosterior] = useState<Posterior | null>(null);
  const [launches, setLaunches] = useState<Launch[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [launchId, setLaunchId] = useState<string>("");
  const [weeks, setWeeks] = useState(0);
  const [sigma, setSigma] = useState<number | null>(null);
  const [showTruth, setShowTruth] = useState(false);

  useEffect(() => {
    Promise.all([loadPosterior(), loadLaunches()])
      .then(([p, l]) => {
        setPosterior(p.posterior);
        setLaunches(l.launches);
        setLaunchId(l.launches[0]?.id ?? "");
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const launch = launches.find((l) => l.id === launchId);
  const deferredWeeks = useDeferredValue(weeks);
  const deferredSigma = useDeferredValue(sigma);

  const result = useMemo(
    () => (posterior && launch ? cachedUpdate(posterior, launch, deferredWeeks, deferredSigma) : null),
    [posterior, launch, deferredWeeks, deferredSigma],
  );
  const prior = useMemo(
    () => (posterior && launch ? cachedUpdate(posterior, launch, 0, deferredSigma) : null),
    [posterior, launch, deferredSigma],
  );

  if (error) {
    return (
      <>
        <Topbar datasetName="Launch tracker" demo />
        <p className="mx-auto mt-8 max-w-[1200px] px-6 text-sm">Could not load launch data: {error}</p>
      </>
    );
  }

  const pStart = prior?.forecast.p_profit_positive ?? 0;
  const pNow = result?.forecast.p_profit_positive ?? 0;
  const profit = result?.forecast.incremental_profit;
  const incUnits = result?.forecast.incremental_units;
  const editable = launch?.unknown.prior.editable_sigma;

  return (
    <>
      <Topbar datasetName={posterior?.meta.dataset_name ?? "Loading…"} demo />
      <main className="mx-auto w-full max-w-[1200px] flex-1 space-y-6 px-6 py-8">
        <div className="space-y-3">
          <h1 className="text-xl font-semibold tracking-tight">Launch tracker</h1>
          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Choose a launch">
            {launches.map((l) => (
              <button
                key={l.id}
                role="tab"
                aria-selected={l.id === launchId}
                onClick={() => {
                  setLaunchId(l.id);
                  setWeeks(0);
                  setSigma(null);
                }}
                className={cn(
                  "rounded-lg border px-3 py-1.5 text-sm transition-colors duration-150",
                  l.id === launchId ? "border-brand bg-brand-soft" : "text-muted-foreground",
                )}
              >
                {l.label}
              </button>
            ))}
          </div>
          {launch?.description && <p className="text-sm text-muted-foreground">{launch.description}</p>}
        </div>

        {!posterior || !launch || !result || !prior ? (
          <div className="space-y-4">
            <Skeleton className="h-12 w-64" />
            <Skeleton className="h-[340px] w-full" />
          </div>
        ) : (
          <div className="flex flex-col gap-8 lg:flex-row">
            <aside className="w-full space-y-6 lg:w-[320px] lg:shrink-0">
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <span className="text-sm text-muted-foreground">Real data so far</span>
                  <span className="num text-sm font-medium">
                    week {weeks} of {launch.weeks_observed}
                  </span>
                </div>
                <Slider
                  aria-label="Weeks of real launch data"
                  min={0}
                  max={launch.weeks_observed}
                  step={1}
                  value={[weeks]}
                  onValueChange={(v) => setWeeks(Array.isArray(v) ? v[0] : v)}
                />
              </div>

              {editable && (
                <div className="space-y-3">
                  <div className="flex items-baseline justify-between">
                    <span className="text-sm text-muted-foreground">Skeptical prior scale</span>
                    <span className="num text-sm font-medium">{(sigma ?? launch.unknown.prior.sigma ?? 0.15).toFixed(2)}</span>
                  </div>
                  <Slider
                    aria-label="Prior scale for the new feature's effect"
                    min={0.05}
                    max={0.5}
                    step={0.05}
                    value={[sigma ?? launch.unknown.prior.sigma ?? 0.15]}
                    onValueChange={(v) => setSigma(Array.isArray(v) ? v[0] : v)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Before any data we assume a feature is probably small: effect ~ Normal(0, scale), on the log scale.
                  </p>
                </div>
              )}

              <div className="space-y-2">
                <Badge variant="outline" className={cn(result.gate.ok ? "text-gain" : "text-warning")}>
                  {result.gate.ok ? "Fast update reliable" : "Refit needed"}
                </Badge>
                <p className="num text-xs text-muted-foreground">{result.gate.message}</p>
              </div>

              <div className="space-y-2">
                <Badge variant="outline">{result.causal.label}</Badge>
                <ul className="space-y-1.5 text-xs">
                  {result.causal.assumptions.map((a) => (
                    <li key={a.id} className="flex gap-2">
                      <span aria-hidden className="num text-muted-foreground">
                        {a.holds === true ? "✓" : a.holds === false ? "✗" : "?"}
                      </span>
                      <span>{a.text}</span>
                    </li>
                  ))}
                </ul>
                {result.causal.recommendation && (
                  <p className="text-xs text-muted-foreground">{result.causal.recommendation}</p>
                )}
              </div>

              {launch.truth && (
                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />
                  Show truth (demo only)
                </label>
              )}
            </aside>

            <div className="min-w-0 flex-1 space-y-6">
              <section className="space-y-3" aria-live="polite">
                <p className="text-sm text-muted-foreground">
                  Added profit vs. no change over {result.forecast.horizon} weeks, after {money(launch.fixed_cost, false)} launch cost
                </p>
                <p className={cn("num-hero", (profit?.mean ?? 0) >= 0 ? "text-gain" : "text-loss")}>
                  {money(profit?.mean ?? 0)}
                </p>
                <p className="num text-sm text-muted-foreground">
                  90% interval {money(profit?.p05 ?? 0)} to {money(profit?.p95 ?? 0)} · extra units{" "}
                  {units(incUnits?.mean ?? 0, false)} ({units(incUnits?.p05 ?? 0, false)} to{" "}
                  {units(incUnits?.p95 ?? 0, false)})
                </p>
                <p className="text-sm">
                  Chance it pays:{" "}
                  <span className="num text-muted-foreground">{percent(pStart)}</span>
                  <span className="mx-1">→</span>
                  <span className="num font-semibold">{percent(pNow)}</span>
                </p>
                {showTruth && launch.truth?.total_profit != null && (
                  <p className="num text-xs text-gain">
                    Truth (demo): profit {money(launch.truth.total_profit)}, {units(launch.truth.total_units ?? 0, false)} units
                  </p>
                )}
              </section>
              <LaunchChart result={result} prior={prior} />
              <DensityPlot result={result} showTruth={showTruth} />
            </div>
          </div>
        )}
      </main>
    </>
  );
}
