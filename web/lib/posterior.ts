import type { Posterior } from "./types";

// Real export first; the contract fixture has identical shape until it lands.
const SOURCES = ["/model/posterior.json", "/model/posterior.fixture.json"];

export const USER_KEY = "ripple.userDataset";
export const USE_USER_KEY = "ripple.useUserDataset";

export interface UserDataset {
  name: string;
  posterior: Posterior;
  trust?: unknown;
}

export function readUserDataset(): UserDataset | null {
  try {
    const raw = sessionStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as UserDataset) : null;
  } catch {
    return null;
  }
}

export function activeUserDataset(): UserDataset | null {
  try {
    return sessionStorage.getItem(USE_USER_KEY) === "1" ? readUserDataset() : null;
  } catch {
    return null;
  }
}

export function saveUserDataset(d: UserDataset): boolean {
  try {
    sessionStorage.setItem(USER_KEY, JSON.stringify(d));
    sessionStorage.setItem(USE_USER_KEY, "1");
    return true;
  } catch {
    return false; // storage full or blocked: the caller tells the user
  }
}

export function setUseUserDataset(on: boolean) {
  try {
    sessionStorage.setItem(USE_USER_KEY, on ? "1" : "0");
  } catch {
    /* ignore */
  }
}

export function clearUserDataset() {
  try {
    sessionStorage.removeItem(USER_KEY);
    sessionStorage.removeItem(USE_USER_KEY);
  } catch {
    /* ignore */
  }
}

export async function loadPosterior(): Promise<{ posterior: Posterior; source: string }> {
  const mine = activeUserDataset();
  if (mine) return { posterior: mine.posterior, source: "user" };
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
