import type { LaunchesFile } from "./types";

export async function loadLaunches(): Promise<LaunchesFile> {
  const res = await fetch("/model/launches.json");
  if (!res.ok) throw new Error(`launches.json: ${res.status}`);
  return (await res.json()) as LaunchesFile;
}

/** x-axis helper: percent lift = exp(scale * u) - 1, scale recovered from the prior interval. */
export function percentScale(prior: { p95: number }, priorPercent?: { p95: number }): number {
  if (!priorPercent || !prior.p95) return 1;
  return Math.log(1 + priorPercent.p95) / prior.p95;
}

export const signedPercent = (x: number, digits = 0) =>
  `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(digits)}%`;
