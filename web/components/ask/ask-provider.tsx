"use client";

import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { AgentAnswer, MarketScan, Step } from "@/lib/agent/types";
import type { Scenario } from "@/lib/types";
import { AskPanel } from "./ask-panel";

export interface Message {
  id: number;
  question: string;
  steps: Step[];
  answer?: AgentAnswer;
  error?: string;
  pending: boolean;
  live?: { gemini: boolean; nimble: boolean };
}

interface AskState {
  open: boolean;
  setOpen: (v: boolean) => void;
  messages: Message[];
  ask: (q: string) => Promise<void>;
  openScenario: (s: Scenario) => void;
  lastScan: MarketScan | null;
  busy: boolean;
}

const Ctx = createContext<AskState | null>(null);
export const useAsk = () => {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAsk must be used inside <AskProvider>");
  return v;
};

export const SCENARIO_KEY = "ripple.scenario";
export const SCENARIO_EVENT = "ripple:scenario";

export function AskProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [lastScan, setLastScan] = useState<MarketScan | null>(null);
  const [busy, setBusy] = useState(false);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const openScenario = useCallback(
    (s: Scenario) => {
      if (pathname === "/") {
        window.dispatchEvent(new CustomEvent(SCENARIO_EVENT, { detail: s }));
      } else {
        try {
          sessionStorage.setItem(SCENARIO_KEY, JSON.stringify(s));
        } catch {
          /* storage can be blocked; the simulator then opens with its default preset */
        }
        router.push("/");
      }
      setOpen(false);
    },
    [pathname, router],
  );

  const ask = useCallback(async (question: string) => {
    const id = Date.now();
    setBusy(true);
    setMessages((m) => [...m, { id, question, steps: [], pending: true }]);
    const patch = (f: (m: Message) => Message) => setMessages((all) => all.map((m) => (m.id === id ? f(m) : m)));
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok || !res.body) throw new Error(`The agent is unavailable (HTTP ${res.status}).`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line) as
            | { type: "mode"; live: { gemini: boolean; nimble: boolean } }
            | { type: "step"; step: Step }
            | { type: "answer"; answer: AgentAnswer }
            | { type: "error"; message: string };
          if (ev.type === "mode") patch((m) => ({ ...m, live: ev.live }));
          else if (ev.type === "step")
            patch((m) => ({
              ...m,
              steps: m.steps.some((s) => s.id === ev.step.id)
                ? m.steps.map((s) => (s.id === ev.step.id ? ev.step : s))
                : [...m.steps, ev.step],
            }));
          else if (ev.type === "answer") {
            patch((m) => ({ ...m, answer: ev.answer, pending: false }));
          } else if (ev.type === "error") throw new Error(ev.message);
        }
      }
      patch((m) => ({ ...m, pending: false }));
    } catch (e) {
      patch((m) => ({ ...m, pending: false, error: (e as Error).message }));
    } finally {
      setBusy(false);
    }
  }, []);

  // the market card on the simulator shows the most recent scan the agent made
  useEffect(() => {
    const last = [...messages].reverse().find((m) => m.answer?.sources.some((s) => s.kind === "web"));
    if (last?.answer) {
      const web = last.answer.claims.filter((c) => c.kind === "web");
      if (web.length > 0) {
        setLastScan({
          query: "",
          product: "",
          city: "",
          source: last.answer.text.includes("illustrative sample") ? "demo" : "nimble",
          fetched_at: "",
          competitors: [],
          comparison: null,
          sources: last.answer.sources.filter((s) => s.kind === "web"),
          note: last.answer.text.split("\n").find((l) => l.startsWith("Market context")) ?? "",
        });
      }
    }
  }, [messages]);

  const value = useMemo(
    () => ({ open, setOpen, messages, ask, openScenario, lastScan, busy }),
    [open, messages, ask, openScenario, lastScan, busy],
  );
  return (
    <Ctx.Provider value={value}>
      {children}
      <AskPanel />
    </Ctx.Provider>
  );
}
