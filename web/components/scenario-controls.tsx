"use client";

import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import type { Posterior, Scenario, ScenarioType } from "@/lib/types";

const pct = (x: number) => `${Math.round(x * 100)}%`;

function Field({
  label,
  value,
  children,
}: {
  label: string;
  value: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className="num text-sm font-medium">{value}</span>
      </div>
      {children}
    </div>
  );
}

function Chips({
  options,
  selected,
  onToggle,
  emptyMeansAll = true,
}: {
  options: string[];
  selected: string[];
  onToggle: (name: string) => void;
  emptyMeansAll?: boolean;
}) {
  const allOn = emptyMeansAll && selected.length === 0;
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((name) => {
        const on = allOn || selected.includes(name);
        return (
          <button
            key={name}
            type="button"
            aria-pressed={on}
            onClick={() => onToggle(name)}
            className={cn(
              "rounded-lg border px-2.5 py-1 text-sm transition-colors duration-150",
              on ? "border-brand bg-brand-soft text-foreground" : "bg-background text-muted-foreground",
            )}
          >
            {name}
          </button>
        );
      })}
    </div>
  );
}

function toggle(list: string[], all: string[], name: string): string[] {
  const current = list.length === 0 ? all : list;
  const next = current.includes(name) ? current.filter((n) => n !== name) : [...current, name];
  return next.length === 0 || next.length === all.length ? [] : next;
}

export interface Preset {
  label: string;
  scenario: Scenario;
}

export function presetsFor(p: Posterior): Preset[] {
  const { cities, segments } = p.meta;
  const city = cities.includes("Boston") ? "Boston" : cities[0];
  const seg = segments.includes("students") ? "students" : segments[0];
  const list: Preset[] = [];
  if (p.meta.has_promo) {
    list.push({
      label: `20% ${seg} coupon in ${city}`,
      scenario: { type: "promo", cities: [city], segments: [seg], discount: 0.2, weeks: 4, horizon: 12 },
    });
  }
  list.push({ label: "Raise prices 8%", scenario: { type: "price", price_change: 0.08, horizon: 26 } });
  if (p.meta.has_traits && p.candidates.length > 0) {
    const c = p.candidates.find((x) => x.extrapolation) ?? p.candidates[0];
    list.push({ label: `Launch in ${c.name}`, scenario: { type: "new_city", candidate: c.name, horizon: 26 } });
  }
  return list;
}

export function ScenarioControls({
  posterior,
  scenario,
  onChange,
}: {
  posterior: Posterior;
  scenario: Scenario;
  onChange: (s: Scenario) => void;
}) {
  const { meta, candidates } = posterior;
  const set = (patch: Partial<Scenario>) => onChange({ ...scenario, ...patch });
  const presets = presetsFor(posterior);

  const switchType = (type: ScenarioType) => {
    if (type === "promo") onChange({ type, discount: 0.2, weeks: 4, horizon: 12, cities: [], segments: [] });
    if (type === "price") onChange({ type, price_change: 0.08, horizon: 26, cities: [], segments: [] });
    if (type === "new_city")
      onChange({ type, candidate: candidates[0]?.name, horizon: 26 });
  };

  const disabled: Record<ScenarioType, string | null> = {
    promo: meta.has_promo ? null : "Your data has no promo column",
    price: null,
    new_city: meta.has_traits ? null : "Your data has no city traits (upload cities.csv)",
  };

  return (
    <aside className="w-full space-y-6 lg:w-[320px] lg:shrink-0">
      <Tabs value={scenario.type} onValueChange={(v) => switchType(v as ScenarioType)}>
        <TabsList className="w-full">
          <TabsTrigger value="promo" disabled={!!disabled.promo}>Promotion</TabsTrigger>
          <TabsTrigger value="price">Price</TabsTrigger>
          <TabsTrigger value="new_city" disabled={!!disabled.new_city}>New city</TabsTrigger>
        </TabsList>
      </Tabs>
      {(disabled.promo || disabled.new_city) && (
        <p className="text-xs text-muted-foreground">{disabled.promo ?? disabled.new_city}</p>
      )}

      <div className="flex flex-wrap gap-2">
        {presets.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => onChange(p.scenario)}
            className="rounded-full border bg-surface px-3 py-1 text-xs text-muted-foreground transition-colors duration-150 hover:border-brand hover:text-foreground"
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="space-y-6">
        {scenario.type === "promo" && (
          <>
            <Field label="Discount" value={pct(scenario.discount ?? 0)}>
              <Slider
                aria-label="Discount"
                min={0}
                max={0.5}
                step={0.05}
                value={[scenario.discount ?? 0]}
                onValueChange={(v) => set({ discount: Array.isArray(v) ? v[0] : v })}
              />
            </Field>
            <Field label="Promo length" value={`${scenario.weeks ?? 1} wk`}>
              <Slider
                aria-label="Promo length in weeks"
                min={1}
                max={8}
                step={1}
                value={[scenario.weeks ?? 1]}
                onValueChange={(v) => set({ weeks: Array.isArray(v) ? v[0] : v })}
              />
            </Field>
          </>
        )}

        {scenario.type === "price" && (
          <Field
            label="Price change"
            value={`${(scenario.price_change ?? 0) > 0 ? "+" : ""}${pct(scenario.price_change ?? 0)}`}
          >
            <Slider
              aria-label="Price change"
              min={-0.3}
              max={0.3}
              step={0.01}
              value={[scenario.price_change ?? 0]}
              onValueChange={(v) => set({ price_change: Array.isArray(v) ? v[0] : v })}
            />
          </Field>
        )}

        {scenario.type === "new_city" && (
          <div className="space-y-3">
            <span className="text-sm text-muted-foreground">City</span>
            <div className="flex flex-wrap gap-2">
              {candidates.map((c) => (
                <button
                  key={c.name}
                  type="button"
                  aria-pressed={scenario.candidate === c.name}
                  onClick={() => set({ candidate: c.name })}
                  className={cn(
                    "rounded-lg border px-2.5 py-1 text-sm transition-colors duration-150",
                    scenario.candidate === c.name
                      ? "border-brand bg-brand-soft"
                      : "text-muted-foreground",
                  )}
                >
                  {c.name}
                  {c.extrapolation && <span className="ml-1 text-warning">!</span>}
                </button>
              ))}
            </div>
          </div>
        )}

        {scenario.type !== "new_city" && (
          <>
            <div className="space-y-3">
              <span className="text-sm text-muted-foreground">Cities</span>
              <Chips
                options={meta.cities}
                selected={scenario.cities ?? []}
                onToggle={(n) => set({ cities: toggle(scenario.cities ?? [], meta.cities, n) })}
              />
            </div>
            {meta.has_segments && (
              <div className="space-y-3">
                <span className="text-sm text-muted-foreground">Customer groups</span>
                <Chips
                  options={meta.segments}
                  selected={scenario.segments ?? []}
                  onToggle={(n) => set({ segments: toggle(scenario.segments ?? [], meta.segments, n) })}
                />
              </div>
            )}
          </>
        )}

        <Field label="Horizon" value={`${scenario.horizon} wk`}>
          <Slider
            aria-label="Horizon in weeks"
            min={4}
            max={52}
            step={1}
            value={[scenario.horizon]}
            onValueChange={(v) => set({ horizon: Array.isArray(v) ? v[0] : v })}
          />
        </Field>

        <Field label="Fixed cost" value={`$${(scenario.fixed_cost ?? 0).toLocaleString("en-US")}`}>
          <Slider
            aria-label="Fixed cost"
            min={0}
            max={50000}
            step={500}
            value={[scenario.fixed_cost ?? 0]}
            onValueChange={(v) => set({ fixed_cost: Array.isArray(v) ? v[0] : v })}
          />
        </Field>
      </div>
    </aside>
  );
}
