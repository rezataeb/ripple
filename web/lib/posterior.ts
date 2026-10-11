import type { Posterior } from "./types";

// Real export first; the contract fixture has identical shape until it lands.
const SOURCES = ["/model/posterior.json", "/model/posterior.fixture.json"];

export async function loadPosterior(): Promise<{ posterior: Posterior; source: string }> {
  let lastError: unknown;
  for (const url of SOURCES) {
    try {
      const res = await fetch(url);
      if (res.ok) return { posterior: (await res.json()) as Posterior, source: url };
    } catch (err) {
      lastError = err;
    }
  }
  throw new Error(`No posterior found (${String(lastError ?? "404")})`);
}
