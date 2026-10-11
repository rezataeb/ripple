"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { TrustCards } from "@/components/method/trust-cards";
import { Topbar } from "@/components/topbar";
import { Badge } from "@/components/ui/badge";
import {
  API_URL,
  RIPPLE_COLUMNS,
  api,
  parseCsvHead,
  ruleMapping,
  type ApiJob,
  type ApiValidation,
  type MappingResult,
  type RippleKey,
} from "@/lib/data-flow";
import type { TrustReport } from "@/lib/method";
import { saveUserDataset } from "@/lib/posterior";
import type { Posterior } from "@/lib/types";
import { cn } from "@/lib/utils";

type Phase = "choose" | "map" | "checked" | "fitting" | "done";

const FIT_STEPS = ["reading the file", "checking the data", "fitting the model", "running the trust checks", "exporting"];

function Dot({ c }: { c?: number }) {
  const tone = c === undefined ? "bg-border" : c >= 0.8 ? "bg-gain" : c >= 0.5 ? "bg-warning" : "bg-loss";
  const label = c === undefined ? "no match" : c >= 0.8 ? "high confidence" : c >= 0.5 ? "medium confidence" : "low confidence";
  return <span title={label} aria-label={label} className={cn("inline-block size-2.5 rounded-full", tone)} />;
}

export default function DataPage() {
  const [phase, setPhase] = useState<Phase>("choose");
  const [apiUp, setApiUp] = useState<boolean | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [cities, setCities] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [sample, setSample] = useState<Record<string, string>[]>([]);
  const [shareSample, setShareSample] = useState(false);
  const [map, setMap] = useState<MappingResult | null>(null);
  const [mapping, setMapping] = useState<Partial<Record<RippleKey, string>>>({});
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [report, setReport] = useState<ApiValidation | null>(null);
  const [unitCost, setUnitCost] = useState(0);
  const [job, setJob] = useState<ApiJob | null>(null);
  const [trust, setTrust] = useState<TrustReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    api.health().then(() => setApiUp(true)).catch(() => setApiUp(false));
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const runMapping = useCallback(async (cols: string[], rows: Record<string, string>[], share: boolean) => {
    setBusy(true);
    try {
      const res = await fetch("/api/map", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ columns: cols, sample: share ? rows : undefined }),
      });
      const m = res.ok ? ((await res.json()) as MappingResult) : ruleMapping(cols);
      setMap(m);
      setMapping(m.mapping);
    } catch {
      const m = ruleMapping(cols);
      setMap(m);
      setMapping(m.mapping);
    } finally {
      setBusy(false);
    }
  }, []);

  const onFile = useCallback(
    async (f: File) => {
      setError(null);
      setReport(null);
      setDatasetId(null);
      setFile(f);
      const head = parseCsvHead(await f.slice(0, 20000).text());
      if (head.columns.length === 0) {
        setError("That file looks empty. Upload a CSV with a header row.");
        return;
      }
      setColumns(head.columns);
      setSample(head.sample);
      setPhase("map");
      await runMapping(head.columns, head.sample, shareSample);
    },
    [runMapping, shareSample],
  );

  const loadDemo = async () => {
    const [w, c] = await Promise.all([
      fetch("/templates/brightside_demo_weekly.csv").then((r) => r.blob()),
      fetch("/templates/brightside_demo_cities.csv").then((r) => r.blob()),
    ]);
    setCities(new File([c], "cities.csv", { type: "text/csv" }));
    await onFile(new File([w], "brightside_demo_weekly.csv", { type: "text/csv" }));
  };

  const missing = RIPPLE_COLUMNS.filter((c) => c.required && !mapping[c.key]).map((c) => c.key);
  const cleanMapping = () => Object.fromEntries(Object.entries(mapping).filter(([, v]) => v)) as Record<string, string>;

  const check = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      let id = datasetId;
      if (!id) {
        const up = await api.upload(file, cities);
        id = up.dataset_id;
        setDatasetId(id);
      }
      setReport(await api.validate(id, cleanMapping()));
      setPhase("checked");
    } catch (e) {
      setError(apiUp === false ? `The Ripple API is not reachable at ${API_URL}. Start it with "make up".` : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const fit = async () => {
    if (!datasetId) return;
    setBusy(true);
    setError(null);
    try {
      const { job_id } = await api.fit(datasetId, { mapping: cleanMapping(), unit_cost: unitCost });
      setPhase("fitting");
      setJob({ job_id, status: "queued", step: "queued", progress: 0 });
      pollRef.current = setInterval(async () => {
        try {
          const j = await api.job(job_id);
          setJob(j);
          if (j.status === "done" || j.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            if (j.status === "error") {
              setError(j.error ?? "The fit failed.");
              setPhase("checked");
            } else {
              const t = j.trust as TrustReport;
              setTrust(t);
              const ok = saveUserDataset({ name: file?.name ?? "your data", posterior: j.posterior as Posterior, trust: j.trust });
              setSaved(ok);
              setPhase("done");
            }
          }
        } catch (e) {
          if (pollRef.current) clearInterval(pollRef.current);
          setError((e as Error).message);
        }
      }, 1500);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!datasetId) return;
    try {
      await api.remove(datasetId);
      setDatasetId(null);
      setReport(null);
      setPhase("map");
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const stepIdx = job ? Math.max(FIT_STEPS.indexOf(job.step), 0) : 0;

  return (
    <>
      <Topbar datasetName="Your data" demo={false} />
      <main className="mx-auto w-full max-w-[1200px] flex-1 space-y-8 px-4 py-8 sm:px-6">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">Use your data</h1>
          <p className="max-w-[70ch] text-sm text-muted-foreground">
            Upload a weekly CSV. Ripple maps your columns, checks the data, fits the same model as the demo, and opens the simulator on your numbers.
            Rows stay in temporary storage for this session; no rows go to any language model unless you tick the box below.
          </p>
          <p className="text-xs text-muted-foreground">
            Fitting runs on the Ripple API ({API_URL}):{" "}
            <span className={cn(apiUp ? "text-gain" : apiUp === false ? "text-warning" : "")}>
              {apiUp === null ? "checking…" : apiUp ? "online" : "offline, run `make up` to enable validate and fit"}
            </span>
          </p>
        </header>

        {error && (
          <p role="alert" className="rounded-xl border border-loss/40 bg-loss/5 px-4 py-3 text-sm">
            {error}
          </p>
        )}

        <section aria-labelledby="s1" className="space-y-3">
          <h2 id="s1" className="text-lg font-medium">1 · Upload</h2>
          <label
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) void onFile(f);
            }}
            className="flex cursor-pointer flex-col items-center justify-center gap-1 rounded-xl border border-dashed bg-surface px-6 py-10 text-center transition-colors duration-150 hover:border-brand"
          >
            <span className="text-sm font-medium">{file ? file.name : "Drop your weekly CSV here, or click to choose"}</span>
            <span className="text-xs text-muted-foreground">One row per week × city × segment. Required: week, units, price.</span>
            <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => e.target.files?.[0] && void onFile(e.target.files[0])} />
          </label>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
            <label className="flex items-center gap-2">
              <span className="text-muted-foreground">Optional city traits CSV</span>
              <input type="file" accept=".csv,text/csv" onChange={(e) => setCities(e.target.files?.[0] ?? null)} className="text-xs" />
            </label>
            <button type="button" onClick={() => void loadDemo()} className="text-brand underline-offset-2 hover:underline">
              Try it with the demo CSV
            </button>
          </div>
          <p className="text-xs text-muted-foreground">
            Templates:{" "}
            <a className="text-brand hover:underline" href="/templates/weekly_template.csv" download>weekly_template.csv</a> ·{" "}
            <a className="text-brand hover:underline" href="/templates/cities_template.csv" download>cities_template.csv</a> ·{" "}
            <a className="text-brand hover:underline" href="/templates/brightside_demo_weekly.csv" download>demo weekly CSV</a>
          </p>
        </section>

        {phase !== "choose" && (
          <section aria-labelledby="s2" className="space-y-3">
            <h2 id="s2" className="text-lg font-medium">2 · Confirm the mapping</h2>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={shareSample}
                onChange={(e) => {
                  setShareSample(e.target.checked);
                  void runMapping(columns, sample, e.target.checked);
                }}
              />
              Share the first 3 rows with the mapping assistant (off: it sees column names only)
            </label>
            {map && (
              <p className="text-xs text-muted-foreground">
                {map.source === "gemini" ? "Suggested by Gemini" : "Suggested by name matching (no Gemini key)"}. {map.notes}
              </p>
            )}
            <div className="overflow-x-auto rounded-xl border">
              <table className="w-full text-left text-sm">
                <thead className="bg-surface text-xs text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 font-medium">Ripple column</th>
                    <th className="px-4 py-2 font-medium">Your column</th>
                    <th className="px-4 py-2 font-medium">Match</th>
                    <th className="px-4 py-2 font-medium">Example</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {RIPPLE_COLUMNS.map((c) => (
                    <tr key={c.key}>
                      <td className="px-4 py-2">
                        <span className="font-medium">{c.key}</span>
                        {c.required && <span className="ml-1 text-loss">*</span>}
                        <span className="block text-xs text-muted-foreground">{c.hint}</span>
                      </td>
                      <td className="px-4 py-2">
                        <select
                          aria-label={`Column for ${c.key}`}
                          value={mapping[c.key] ?? ""}
                          onChange={(e) => setMapping((m) => ({ ...m, [c.key]: e.target.value || undefined }))}
                          className="rounded-lg border bg-background px-2 py-1"
                        >
                          <option value="">{c.required ? "— choose —" : "— none —"}</option>
                          {columns.map((col) => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </td>
                      <td className="px-4 py-2"><Dot c={mapping[c.key] ? map?.confidence[c.key] ?? 0.6 : undefined} /></td>
                      <td className="num px-4 py-2 text-xs text-muted-foreground">{mapping[c.key] ? sample[0]?.[mapping[c.key]!] : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-sm">
                <span className="text-muted-foreground">Cost per unit ($)</span>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  value={unitCost}
                  onChange={(e) => setUnitCost(Number(e.target.value) || 0)}
                  className="num w-24 rounded-lg border bg-background px-2 py-1"
                />
              </label>
              <button
                type="button"
                disabled={busy || missing.length > 0}
                onClick={() => void check()}
                className="rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground disabled:opacity-40"
              >
                {busy ? "Checking…" : "Check my data"}
              </button>
              {missing.length > 0 && <span className="text-xs text-loss">Map the required columns: {missing.join(", ")}</span>}
            </div>
          </section>
        )}

        {report && (
          <section aria-labelledby="s3" className="space-y-3">
            <h2 id="s3" className="text-lg font-medium">3 · What Ripple found</h2>
            <ul className="space-y-1.5 text-sm">
              {[
                ["Rows", String(report.rows)],
                ["Weeks", report.weeks != null ? `${report.weeks}${report.week_range ? ` (${report.week_range[0]} to ${report.week_range[1]})` : ""}` : "n/a"],
                ["Cities", report.cities != null ? String(report.cities) : "n/a"],
                ["Segments", report.segments != null ? String(report.segments) : "n/a"],
                ["Missing values", report.missing_values != null ? String(report.missing_values) : "n/a"],
                ["Negative units", report.negative_units != null ? String(report.negative_units) : "n/a"],
                ["Price variation (CV)", report.price_cv != null ? report.price_cv.toFixed(3) : "n/a"],
              ].map(([k, v]) => (
                <li key={k} className="flex gap-3">
                  <span className="w-44 text-muted-foreground">{k}</span>
                  <span className="num">{v}</span>
                </li>
              ))}
            </ul>
            {report.errors.map((e) => (
              <p key={e} className="rounded-lg border border-loss/40 bg-loss/5 px-3 py-2 text-sm">✗ {e}</p>
            ))}
            {report.warnings.map((w) => (
              <p key={w} className="rounded-lg border border-warning/40 bg-warning/5 px-3 py-2 text-sm">! {w}</p>
            ))}
            {report.info.map((i) => (
              <p key={i} className="text-xs text-muted-foreground">{i}</p>
            ))}
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                disabled={busy || !report.ok || phase === "fitting"}
                onClick={() => void fit()}
                className="rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground disabled:opacity-40"
              >
                Fit the model
              </button>
              <button type="button" onClick={() => void remove()} className="rounded-lg border px-4 py-2 text-sm text-muted-foreground hover:text-foreground">
                Delete my upload
              </button>
            </div>
          </section>
        )}

        {(phase === "fitting" || phase === "done") && job && (
          <section aria-labelledby="s4" className="space-y-3">
            <h2 id="s4" className="text-lg font-medium">4 · Fitting</h2>
            <ol className="space-y-1.5 text-sm">
              {FIT_STEPS.map((s, i) => {
                const done = phase === "done" || i < stepIdx;
                const active = phase === "fitting" && i === stepIdx;
                return (
                  <li key={s} className={cn("flex items-center gap-2", !done && !active && "text-muted-foreground")}>
                    <span className="num w-4">{done ? "✓" : active ? "…" : "·"}</span>
                    {s}
                  </li>
                );
              })}
            </ol>
            <div className="h-1.5 w-full max-w-md overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-brand transition-[width] duration-200" style={{ width: `${(phase === "done" ? 1 : job.progress) * 100}%` }} />
            </div>
          </section>
        )}

        {phase === "done" && trust && (
          <section aria-labelledby="s5" className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <h2 id="s5" className="text-lg font-medium">5 · Trust checks</h2>
              <Badge variant="outline">{trust.summary.headline ?? `${trust.summary.passed} of ${trust.summary.total} passed`}</Badge>
            </div>
            <TrustCards checks={trust.checks} />
            {trust.checks.some((c) => !c.passed && !c.skipped_reason) && (
              <p className="rounded-lg border border-warning/40 bg-warning/5 px-3 py-2 text-sm">
                Some checks failed. This is a {trust.method.toLowerCase().includes("approx") ? "fast approximation" : "short fit"} on a time budget: use the
                intervals as a rough guide, and refit with more draws before deciding anything large.
              </p>
            )}
            {!saved && (
              <p className="rounded-lg border border-warning/40 bg-warning/5 px-3 py-2 text-sm">
                The fitted model is too large to keep in this browser session, so it cannot be opened in the simulator.
              </p>
            )}
            <Link
              href="/"
              className={cn("inline-block rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground", !saved && "pointer-events-none opacity-40")}
            >
              Open in simulator
            </Link>
          </section>
        )}
      </main>
    </>
  );
}
