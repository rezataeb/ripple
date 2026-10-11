"use client";

import "katex/dist/katex.min.css";
import katex from "katex";
import { useEffect, useMemo, useState } from "react";
import { CalibrationChart } from "@/components/method/calibration-chart";
import { RecoveryChart } from "@/components/method/recovery-chart";
import { TrustCards } from "@/components/method/trust-cards";
import { Topbar } from "@/components/topbar";
import { Skeleton } from "@/components/ui/skeleton";
import { loadMethodData, type CalibrationFile, type TrustReport } from "@/lib/method";

const EQUATIONS = [
  String.raw`y_{c,s,t} \sim \mathrm{NegBin}(\mu_{c,s,t},\, \phi)`,
  String.raw`\log \mu_{c,s,t} = \log \mathrm{share}_{c,s} + a^{\mathrm{city}}_c + a^{\mathrm{seg}}_s + e_s \log\tfrac{p}{p_0} + b^{\mathrm{promo}}_s\,\mathrm{promo} + b^{\mathrm{feat}}\,\mathrm{feature} + b^{\mathrm{dip}}\,\mathrm{promo}_{t-1} + \mathrm{season}(t)`,
  String.raw`a^{\mathrm{city}}_c = g_0 + X_c \cdot g + \tau_{\mathrm{city}}\, z_c, \qquad z_c \sim \mathcal{N}(0,1)`,
  String.raw`e_s \sim \mathcal{N}(\mu_e, \sigma_e), \quad \mu_e \sim \mathcal{N}(-1.5, 0.5), \quad \sigma_e \sim \mathrm{HalfNormal}(0.5)`,
];

const WHY: [string, string, string][] = [
  ["Negative Binomial", "Weekly unit counts are overdispersed: the variance is much larger than the mean.", "A Poisson model calls normal week-to-week variation a surprise, and every interval is far too narrow."],
  ["Log-log price term", "The coefficient on log price is the elasticity: a 1% price rise changes demand by that many percent.", "A linear price term means nothing at a different price level and can predict negative demand."],
  ["Elasticity per segment, partially pooled", "Students and retirees do not respond to price alike. Pooling lets a thin segment borrow strength.", "Pooled: every offer looks the same for everyone. Unpooled: a small segment is driven by noise."],
  ["Promo deal-signal lift", "Being on promotion lifts demand beyond the price cut itself.", "The price term alone understates a promotion, then overstates the discount needed next time."],
  ["Post-promo dip", "A promotion pulls purchases forward from the following week.", "Promotions look better than they are, because borrowed sales are never paid back."],
  ["City effect on traits", "A city with no history still gets a forecast, with honest uncertainty from tau.", "New-city questions become unanswerable."],
  ["Customer-mix offset", "Volume scales with who lives there, so a different mix needs no refit.", "A student-heavy city is forecast as if it had the average mix."],
  ["Fourier seasonality", "Yearly shape with few enough parameters to extrapolate.", "Season is confounded with the promotion calendar and the promo lift absorbs it."],
];

function Tex({ tex }: { tex: string }) {
  const html = useMemo(() => katex.renderToString(tex, { throwOnError: false, displayMode: true }), [tex]);
  return <div className="overflow-x-auto py-1" dangerouslySetInnerHTML={{ __html: html }} />;
}

function Section({ id, title, lead, children }: { id: string; title: string; lead?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 space-y-4">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
        {lead && <p className="max-w-[70ch] text-sm text-muted-foreground">{lead}</p>}
      </div>
      {children}
    </section>
  );
}

export default function MethodPage() {
  const [data, setData] = useState<{ trust: TrustReport; calibration: CalibrationFile | null } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMethodData().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  const trust = data?.trust;
  const cal = data?.calibration;

  return (
    <>
      <Topbar datasetName={trust?.dataset_name ?? "Method"} demo />
      <main className="mx-auto w-full max-w-[1200px] flex-1 space-y-12 px-4 py-8 sm:px-6">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">How Ripple works</h1>
          <p className="max-w-[70ch] text-base text-muted-foreground">
            The model makes the numbers. The agent explains them. The web gives context. This page shows the model, and the evidence that it is right where it claims to be, and wrong where it says so.
          </p>
        </header>

        {error && <p className="rounded-xl border border-loss/40 bg-loss/5 px-4 py-3 text-sm">Could not load the evidence: {error}</p>}

        <Section id="model" title="The model" lead="One row of data is one week × city × customer segment. Everything the simulator shows comes from this equation and its posterior.">
          <div className="rounded-xl border bg-surface px-4 py-3">
            {EQUATIONS.map((t) => (
              <Tex key={t} tex={t} />
            ))}
          </div>
        </Section>

        <Section id="why" title="Why each part is there" lead="Every term earns its place by preventing a specific wrong answer.">
          <div className="overflow-x-auto rounded-xl border">
            <table className="w-full text-left text-sm">
              <thead className="bg-surface text-xs text-muted-foreground">
                <tr>
                  <th className="px-4 py-2 font-medium">Part</th>
                  <th className="px-4 py-2 font-medium">Why it is there</th>
                  <th className="px-4 py-2 font-medium">What breaks without it</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {WHY.map(([a, b, c]) => (
                  <tr key={a} className="align-top">
                    <td className="px-4 py-3 font-medium">{a}</td>
                    <td className="px-4 py-3">{b}</td>
                    <td className="px-4 py-3 text-muted-foreground">{c}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="checks" title="Trust checks" lead={trust ? `${trust.summary.headline ?? `${trust.summary.passed} of ${trust.summary.total} checks passed`}. ${trust.method}.` : undefined}>
          {!trust ? (
            <Skeleton className="h-32 w-full" />
          ) : (
            <TrustCards checks={trust.checks} />
          )}
        </Section>

        {trust?.recovery && trust.recovery.length > 0 && (
          <Section
            id="recovery"
            title="Parameter recovery"
            lead={`The demo company is synthetic, so the true effects are known. Each dot is the planted truth; the bar is the model's 94% interval. ${trust.recovery.filter((r) => r.inside).length} of ${trust.recovery.length} truths sit inside their interval.`}
          >
            <RecoveryChart rows={trust.recovery} />
          </Section>
        )}

        {trust?.holdout && (
          <Section id="holdout" title="Time holdout" lead={`The last ${trust.holdout.weeks_held_out} weeks were hidden from the fit and then predicted.`}>
            <div className="grid gap-4 sm:grid-cols-3">
              {[
                ["90% interval coverage", `${(trust.holdout.coverage_90 * 100).toFixed(1)}%`, "of hidden weeks fell inside the model's 90% range"],
                ["Model error (MAPE)", `${(trust.holdout.mape_model * 100).toFixed(1)}%`, "average miss, as a share of actual units"],
                ["Naive seasonal error", `${(trust.holdout.mape_naive * 100).toFixed(1)}%`, trust.holdout.beats_naive ? "the model beats this baseline" : "the model does not beat this baseline"],
              ].map(([l, v, d]) => (
                <div key={l} className="rounded-xl border p-4">
                  <p className="text-sm text-muted-foreground">{l}</p>
                  <p className="num mt-1 text-2xl font-semibold">{v}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{d}</p>
                </div>
              ))}
            </div>
          </Section>
        )}

        {cal && (
          <Section id="calibration" title="Does the launch tracker tell the truth?" lead={cal.headline}>
            <div className="rounded-xl border p-4">
              <CalibrationChart arms={cal.arms} target={cal.target_coverage} />
              <ul className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
                {cal.arms.map((a) => {
                  const last = a.by_week[a.by_week.length - 1];
                  const first = a.by_week[0];
                  return (
                    <li key={a.id} className="space-y-1">
                      <p className="font-medium">
                        {a.label}
                        {!a.assumptions_hold && <span className="ml-2 text-xs text-warning">assumption broken</span>}
                      </p>
                      <p className="num text-xs text-muted-foreground">
                        coverage {(first.coverage_total * 100).toFixed(0)}% → {(last.coverage_total * 100).toFixed(0)}% · width{" "}
                        {Math.round(first.mean_width_total).toLocaleString("en-US")} → {Math.round(last.mean_width_total).toLocaleString("en-US")} units
                      </p>
                      {a.broken_assumption && <p className="text-xs text-muted-foreground">{a.broken_assumption}</p>}
                      {a.verdict && <p className="text-xs">{a.verdict}</p>}
                    </li>
                  );
                })}
              </ul>
              <p className="mt-4 text-xs text-muted-foreground">
                {cal.n_launches} simulated launches per arm, truth known. Coverage is the share whose true 12-week total fell inside the 90% interval, after each week of real data.
              </p>
            </div>
          </Section>
        )}

        {trust?.limits && (
          <Section id="limits" title="Limits" lead="Where Ripple is wrong, and how you would know.">
            <ul className="max-w-[75ch] list-disc space-y-2 pl-5 text-sm">
              {trust.limits.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </Section>
        )}
      </main>
    </>
  );
}
