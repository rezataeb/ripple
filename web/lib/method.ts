export interface TrustCheck {
  name: string;
  group?: "convergence" | "fit" | "recovery" | "holdout";
  passed: boolean;
  value: number | null;
  threshold: number | string | null;
  comparison?: "lte" | "gte" | "eq";
  plain_english: string;
  detail?: string;
  skipped_reason?: string;
}

export interface RecoveryRow {
  param: string;
  label?: string;
  truth: number;
  posterior_mean?: number;
  hdi_low: number;
  hdi_high: number;
  inside: boolean;
}

export interface TrustReport {
  generated_at: string;
  method: string;
  dataset_name?: string;
  summary: { passed: number; total: number; headline?: string };
  checks: TrustCheck[];
  recovery?: RecoveryRow[];
  holdout?: { weeks_held_out: number; coverage_90: number; mape_model: number; mape_naive: number; beats_naive: boolean };
  convergence?: { max_rhat: number; min_ess: number; divergences: number; worst_param?: string; n_chains: number; n_draws_per_chain: number; sampling_seconds?: number };
  limits?: string[];
}

export interface CalibrationWeek {
  week: number;
  coverage_total: number;
  coverage_total_mcse?: number;
  coverage_unknown?: number;
  mean_width_total: number;
  median_width_total?: number;
  bias_total?: number;
  refit_rate: number;
  n: number;
}

export interface CalibrationArm {
  id: string;
  label: string;
  kind: string;
  assumptions_hold: boolean;
  broken_assumption?: string;
  verdict?: string;
  by_week: CalibrationWeek[];
}

export interface CalibrationFile {
  generated_at: string;
  n_launches: number;
  target_coverage: number;
  headline: string;
  arms: CalibrationArm[];
}

export async function loadMethodData(): Promise<{ trust: TrustReport; calibration: CalibrationFile | null }> {
  const [t, c] = await Promise.all([fetch("/model/trust.json"), fetch("/model/calibration.json")]);
  if (!t.ok) throw new Error(`trust.json: ${t.status}`);
  return { trust: (await t.json()) as TrustReport, calibration: c.ok ? ((await c.json()) as CalibrationFile) : null };
}
