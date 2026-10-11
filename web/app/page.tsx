"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { ScenarioControls, presetsFor } from "@/components/scenario-controls";
import { Topbar } from "@/components/topbar";
import { Verdict } from "@/components/verdict";
import { FanChart } from "@/components/fan-chart";
import { IncrementalChart } from "@/components/incremental-chart";
import { SegmentBars } from "@/components/segment-bars";
import { Warnings } from "@/components/extrapolation-banner";
import { SCENARIO_EVENT, SCENARIO_KEY } from "@/components/ask/ask-provider";
import { MarketCard } from "@/components/ask/market-card";
import { Skeleton } from "@/components/ui/skeleton";
import { loadPosterior } from "@/lib/posterior";
import { simulate } from "@/lib/simulate";
import type { Posterior, Scenario, ScenarioResult } from "@/lib/types";

/** A scenario handed over by the Ask panel before navigating here ("Open this scenario"). */
function takeHandedOverScenario(): Scenario | null {
  try {
    const raw = sessionStorage.getItem(SCENARIO_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(SCENARIO_KEY);
    return JSON.parse(raw) as Scenario;
  } catch {
    return null; // storage blocked: fall back to the default preset
  }
}

export default function Simulator() {
  const [posterior, setPosterior] = useState<Posterior | null>(null);
  const [source, setSource] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  // "Open this scenario" from the Ask panel arrives as an event, or through sessionStorage after a navigation.
  useEffect(() => {
    const take = (s: Scenario) => setScenario(s);
    const onEvent = (e: Event) => take((e as CustomEvent<Scenario>).detail);
    window.addEventListener(SCENARIO_EVENT, onEvent);
    return () => window.removeEventListener(SCENARIO_EVENT, onEvent);
  }, []);
  // Deferred so slider thumbs stay responsive while 1000 draws are re-simulated.
  const deferred = useDeferredValue(scenario);
  const result: ScenarioResult | null = useMemo(
    () => (posterior && deferred ? simulate(posterior, deferred) : null),
    [posterior, deferred],
  );

  useEffect(() => {
    loadPosterior()
      .then(({ posterior, source }) => {
        setPosterior(posterior);
        setSource(source);
        setScenario(takeHandedOverScenario() ?? presetsFor(posterior)[0]?.scenario ?? { type: "price", price_change: 0.08, horizon: 26 });
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const name = posterior?.meta.dataset_name ?? "Loading…";
  const isDemo = source.includes("fixture") || (posterior?.meta.dataset_name ?? "").startsWith("Demo");

  return (
    <>
      <Topbar datasetName={name} demo={isDemo || !posterior} />
      <main className="mx-auto flex w-full max-w-[1200px] flex-1 flex-col gap-8 px-6 py-8 lg:flex-row">
        {error && (
          <p className="rounded-xl border border-loss/40 bg-loss/5 px-4 py-3 text-sm">
            Could not load the model: {error}. Run the model export or place the fixture in web/public/model/.
          </p>
        )}
        {!posterior || !scenario ? (
          !error && (
            <div className="flex w-full flex-col gap-4">
              <Skeleton className="h-12 w-64" />
              <Skeleton className="h-[320px] w-full" />
            </div>
          )
        ) : (
          <>
            <ScenarioControls posterior={posterior} scenario={scenario} onChange={setScenario} />
            <div className="min-w-0 flex-1 space-y-6">
              <Warnings result={result} />
              <Verdict result={result} />
              <FanChart posterior={posterior} result={result} />
              <IncrementalChart result={result} />
              <SegmentBars result={result} />
              <MarketCard />
              <p className="text-xs text-muted-foreground">
                {posterior.meta.method} · {posterior.meta.n_draws} draws · seed {posterior.meta.seed}
              </p>
            </div>
          </>
        )}
      </main>
    </>
  );
}
