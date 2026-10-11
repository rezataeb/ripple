"use client";

import { useState } from "react";
import { buildLaunchFromCsv, type UploadKind } from "@/lib/launch-upload";
import type { Launch, Posterior } from "@/lib/types";

const KINDS: { id: UploadKind; label: string; hint: string }[] = [
  { id: "new_city", label: "A new city", hint: "learns how big the city really is" },
  { id: "offer", label: "An offer", hint: "learns the lift of a promotion for one group" },
  { id: "feature", label: "A new feature", hint: "learns the lift of something never seen before" },
];

const field = "rounded-lg border bg-background px-2 py-1 text-sm";

export function UploadPanel({ posterior, onLaunch }: { posterior: Posterior; onLaunch: (l: Launch) => void }) {
  const { meta, candidates } = posterior;
  const canNewCity = meta.has_traits && candidates.length > 0;
  const [kind, setKind] = useState<UploadKind>(canNewCity ? "new_city" : "feature");
  const [file, setFile] = useState<File | null>(null);
  const [candidate, setCandidate] = useState(candidates[0]?.name ?? "");
  const [city, setCity] = useState(meta.cities[0] ?? "");
  const [segment, setSegment] = useState(meta.segments[0] ?? "");
  const [horizon, setHorizon] = useState(12);
  const [unitCost, setUnitCost] = useState(meta.unit_cost);
  const [fixedCost, setFixedCost] = useState(0);
  const [errors, setErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const build = async () => {
    if (!file) {
      setErrors(["Choose a CSV file first."]);
      return;
    }
    setBusy(true);
    try {
      const res = buildLaunchFromCsv(posterior, await file.text(), {
        kind,
        candidate,
        city,
        segment,
        horizon,
        unitCost,
        fixedCost,
        fileName: file.name,
      });
      if ("errors" in res) setErrors(res.errors);
      else {
        setErrors([]);
        onLaunch(res.launch);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-labelledby="up-h" className="max-w-[720px] space-y-5">
      <div className="space-y-1">
        <h2 id="up-h" className="text-lg font-medium">Upload launch weeks</h2>
        <p className="text-sm text-muted-foreground">
          Real weeks from a launch you ran, in the same shape as your history: one row per week × city × segment, with
          <span className="num"> week, units, price</span> and optionally <span className="num">city, segment, promo, holdout</span>.
          Ripple keeps the file in your browser, updates the forecast with Bayes&apos; rule, and never sends rows anywhere.{" "}
          <a className="text-brand hover:underline" href="/templates/launch_weeks_template.csv" download>
            Download a template
          </a>
        </p>
      </div>

      <fieldset className="space-y-2">
        <legend className="text-sm text-muted-foreground">What did you launch?</legend>
        <div className="flex flex-wrap gap-2">
          {KINDS.map((k) => {
            const disabled = k.id === "new_city" && !canNewCity;
            return (
              <button
                key={k.id}
                type="button"
                disabled={disabled}
                aria-pressed={kind === k.id}
                title={disabled ? "This model has no city traits" : k.hint}
                onClick={() => setKind(k.id)}
                className={`rounded-lg border px-3 py-1.5 text-sm transition-colors duration-150 disabled:opacity-40 ${kind === k.id ? "border-brand bg-brand-soft" : "text-muted-foreground"}`}
              >
                {k.label}
              </button>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground">{KINDS.find((k) => k.id === kind)?.hint}</p>
      </fieldset>

      <div className="grid gap-4 sm:grid-cols-2">
        {kind === "new_city" && (
          <label className="space-y-1 text-sm">
            <span className="text-muted-foreground">Which candidate city</span>
            <select className={`${field} block w-full`} value={candidate} onChange={(e) => setCandidate(e.target.value)}>
              {candidates.map((c) => (
                <option key={c.name}>{c.name}</option>
              ))}
            </select>
          </label>
        )}
        {kind === "offer" && (
          <>
            <label className="space-y-1 text-sm">
              <span className="text-muted-foreground">City where the offer ran</span>
              <select className={`${field} block w-full`} value={city} onChange={(e) => setCity(e.target.value)}>
                {meta.cities.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-sm">
              <span className="text-muted-foreground">Customer group that got it</span>
              <select className={`${field} block w-full`} value={segment} onChange={(e) => setSegment(e.target.value)}>
                {meta.segments.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
          </>
        )}
        <label className="space-y-1 text-sm">
          <span className="text-muted-foreground">Forecast horizon (weeks)</span>
          <input type="number" min={4} max={52} value={horizon} onChange={(e) => setHorizon(Math.max(4, Number(e.target.value) || 12))} className={`${field} num block w-full`} />
        </label>
        <label className="space-y-1 text-sm">
          <span className="text-muted-foreground">Cost per unit ($)</span>
          <input type="number" min={0} step={0.5} value={unitCost} onChange={(e) => setUnitCost(Number(e.target.value) || 0)} className={`${field} num block w-full`} />
        </label>
        <label className="space-y-1 text-sm">
          <span className="text-muted-foreground">Launch cost, one-off ($)</span>
          <input type="number" min={0} step={500} value={fixedCost} onChange={(e) => setFixedCost(Number(e.target.value) || 0)} className={`${field} num block w-full`} />
        </label>
      </div>

      <label className="flex cursor-pointer flex-col items-center gap-1 rounded-xl border border-dashed bg-surface px-6 py-8 text-center transition-colors duration-150 hover:border-brand">
        <span className="text-sm font-medium">{file ? file.name : "Choose the launch-weeks CSV"}</span>
        <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>

      {errors.length > 0 && (
        <ul role="alert" className="space-y-1 rounded-xl border border-loss/40 bg-loss/5 px-4 py-3 text-sm">
          {errors.map((e) => (
            <li key={e}>✗ {e}</li>
          ))}
        </ul>
      )}

      <button
        type="button"
        onClick={() => void build()}
        disabled={busy}
        className="rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground disabled:opacity-40"
      >
        {busy ? "Reading…" : "Track this launch"}
      </button>
    </section>
  );
}
