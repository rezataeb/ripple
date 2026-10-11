// Mirrors contract/*.schema.json. If the contract changes, change this file.

export interface PosteriorMeta {
  dataset_name?: string;
  cities: string[];
  segments: string[];
  weeks: number;
  n_draws: number;
  base_price: number;
  unit_cost: number;
  shares: number[][];
  trait_names: string[];
  trait_mean: number[];
  trait_std: number[];
  trait_min: number[];
  trait_max: number[];
  has_promo: boolean;
  has_feature: boolean;
  has_traits: boolean;
  has_segments: boolean;
  season_period: number;
  n_fourier: number;
  method: string;
  fast_approximation?: boolean;
  seed: number;
  fitted_at?: string;
}

export interface PosteriorDraws {
  g0: number[];
  g: number[][];
  tau_city: number[];
  a_city: number[][];
  a_seg: number[][];
  e: number[][];
  b_promo: number[][];
  b_feature: number[];
  b_dip: number[];
  season: number[][];
  phi: number[];
}

export interface PosteriorHistory {
  week_start: string[];
  t: number[];
  city_idx: number[];
  segment_idx: number[];
  units: number[];
  price: number[];
  promo: number[];
  promo_last_week: number[];
  feature: number[];
}

export interface Candidate {
  name: string;
  traits: number[];
  traits_z: number[];
  shares: number[];
  extrapolation: boolean;
  out_of_range_traits: string[];
  note?: string;
}

export interface Posterior {
  meta: PosteriorMeta;
  draws: PosteriorDraws;
  history: PosteriorHistory;
  candidates: Candidate[];
}

export type ScenarioType = "promo" | "price" | "new_city";

export interface Scenario {
  type: ScenarioType;
  label?: string;
  cities?: string[];
  segments?: string[];
  discount?: number;
  weeks?: number;
  price_change?: number;
  feature?: boolean;
  candidate?: string;
  traits?: number[];
  new_city_shares?: number[];
  new_city_price?: number;
  market_size?: number;
  unit_cost?: number;
  fixed_cost?: number;
  horizon: number;
  seed?: number;
}

export interface Interval {
  mean: number;
  p05: number;
  p50: number;
  p95: number;
}

export interface Band {
  p05: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

export interface Money {
  units: Interval;
  revenue: Interval;
  profit: Interval;
}

export interface WeeklyRow {
  week: number;
  t?: number;
  week_start?: string;
  baseline_units: Band;
  scenario_units: Band;
  incremental_units?: Band;
  promo_active: boolean;
  dip_active?: boolean;
}

export interface ScenarioResult {
  scenario: Scenario;
  horizon: number;
  baseline?: Money;
  with_change?: Money;
  incremental: Money;
  p_profit_positive: number;
  break_even?: { metric: "discount" | "price_change" | "fixed_cost"; value: number | null };
  weekly: WeeklyRow[];
  segments?: { segment: string; incremental: Money; share_of_total?: number }[];
  cities?: { city: string; incremental: Money }[];
  flags: {
    extrapolation: boolean;
    out_of_range_traits?: string[];
    no_price_variation?: boolean;
    promo_disabled?: boolean;
    warnings?: string[];
  };
  meta: {
    n_draws: number;
    seed: number;
    source?: "browser" | "python";
    computed_ms?: number;
    unit_cost?: number;
    fixed_cost?: number;
    net_price?: number;
  };
}
