"use client";

import { useEffect, useRef, useState } from "react";
import { numbersIn } from "@/lib/agent/guard";
import type { AgentAnswer } from "@/lib/agent/types";
import { cn } from "@/lib/utils";
import { useAsk, type Message } from "./ask-provider";

export const EXAMPLES = [
  "Should we run a 20% student coupon in Boston?",
  "What if we raise prices 8%?",
  "Is Miami a good next city?",
  "Why should I trust this?",
];

function StepChips({ m }: { m: Message }) {
  if (m.steps.length === 0 && !m.pending) return null;
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Steps">
      {m.steps.map((s) => (
        <li
          key={s.id}
          className={cn(
            "rounded-full border px-2.5 py-0.5 text-xs transition-colors duration-150",
            s.status === "running" && "border-brand text-brand",
            s.status === "done" && "text-muted-foreground",
            s.status === "error" && "border-loss text-loss",
          )}
        >
          {s.status === "running" ? "…" : s.status === "done" ? "✓" : "!"} {s.label}
        </li>
      ))}
      {m.pending && m.steps.length === 0 && (
        <li className="rounded-full border px-2.5 py-0.5 text-xs text-muted-foreground">… thinking</li>
      )}
    </ul>
  );
}

function ClaimChips({ a }: { a: AgentAnswer }) {
  const shown = numbersIn(a.text);
  const used = a.claims.filter((c) => numbersIn(c.value).some((n) => shown.some((s) => s.value === n.value && s.unit === n.unit)));
  const seen = new Set<string>();
  const uniq = used.filter((c) => (seen.has(c.value) ? false : (seen.add(c.value), true)));
  if (uniq.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5" aria-label="Numbers in this answer and where they came from">
      {uniq.slice(0, 14).map((c, i) => (
        <span
          key={i}
          title={`${c.label} · ${c.tool_call_id} · from the ${c.kind === "model" ? "model" : c.kind === "web" ? "web" : "docs"}`}
          className={cn(
            "num cursor-help rounded-md border px-1.5 py-0.5 text-xs",
            c.kind === "web" && "border-warning/50 bg-warning/5",
            c.kind === "docs" && "bg-surface",
            c.kind === "model" && "border-brand/30 bg-brand-soft",
          )}
        >
          {c.value}
          <span className="ml-1 text-[10px] uppercase text-muted-foreground">{c.kind}</span>
        </span>
      ))}
    </div>
  );
}

function AnswerView({ m }: { m: Message }) {
  const { openScenario } = useAsk();
  const a = m.answer;
  if (m.error) {
    return <p className="rounded-lg border border-loss/40 bg-loss/5 px-3 py-2 text-sm">{m.error}</p>;
  }
  if (!a) return null;
  return (
    <div className="space-y-3">
      <div className="whitespace-pre-wrap text-sm leading-relaxed">{a.text}</div>
      <ClaimChips a={a} />
      {a.sources.length > 0 && (
        <ul className="space-y-1 text-xs">
          {a.sources.map((s) => (
            <li key={s.url}>
              <a className="text-brand underline-offset-2 hover:underline" href={s.url} target={s.url.startsWith("/") ? undefined : "_blank"} rel="noreferrer">
                {s.title}
              </a>
              <span className="ml-1 text-muted-foreground">{s.kind === "web" ? "web" : "docs"}</span>
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {a.scenario && (
          <button
            type="button"
            onClick={() => openScenario(a.scenario!)}
            className="rounded-lg bg-primary px-3 py-1.5 text-sm text-primary-foreground transition-opacity duration-150 hover:opacity-90"
          >
            Open this scenario
          </button>
        )}
        <span className="text-xs text-muted-foreground">
          {a.mode === "gemini" ? "Gemini wrote the wording" : "Template answer"} ·{" "}
          {a.guard.passed ? "every number is from a tool" : "numbers replaced by the template"}
        </span>
      </div>
    </div>
  );
}

export function AskPanel() {
  const { open, setOpen, messages, ask, busy } = useAsk();
  const [q, setQ] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const end = useRef<HTMLDivElement>(null);
  const live = messages.at(-1)?.live;

  useEffect(() => {
    if (open) setTimeout(() => input.current?.focus(), 50);
  }, [open]);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const submit = (text: string) => {
    const t = text.trim();
    if (!t || busy) return;
    setQ("");
    void ask(t);
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          "fixed bottom-6 right-6 z-30 flex items-center gap-2 rounded-full border bg-background px-4 py-2 text-sm shadow-sm transition-opacity duration-150 hover:border-brand",
          open && "pointer-events-none opacity-0",
        )}
        aria-label="Ask Ripple (command K)"
      >
        <span className="num text-xs text-muted-foreground">⌘K</span> Ask Ripple
      </button>

      <div
        className={cn(
          "fixed inset-y-0 right-0 z-40 flex w-full max-w-[440px] flex-col border-l bg-background transition-transform duration-200 ease-out",
          open ? "translate-x-0" : "translate-x-full",
        )}
        role="dialog"
        aria-label="Ask Ripple"
        aria-hidden={!open}
      >
        <header className="flex items-center justify-between border-b px-4 py-3">
          <div>
            <h2 className="text-base font-medium">Ask Ripple</h2>
            <p className="text-xs text-muted-foreground">
              The model makes the numbers. The agent explains them. The web gives context.
            </p>
          </div>
          <button type="button" onClick={() => setOpen(false)} className="rounded-lg px-2 py-1 text-sm text-muted-foreground hover:text-foreground" aria-label="Close">
            Esc
          </button>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto px-4 py-4">
          {messages.length === 0 && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">Try a question:</p>
              <div className="flex flex-col gap-2">
                {EXAMPLES.map((e) => (
                  <button
                    key={e}
                    type="button"
                    onClick={() => submit(e)}
                    className="rounded-lg border bg-surface px-3 py-2 text-left text-sm transition-colors duration-150 hover:border-brand"
                  >
                    {e}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className="space-y-3">
              <p className="rounded-lg bg-surface px-3 py-2 text-sm">{m.question}</p>
              <StepChips m={m} />
              <AnswerView m={m} />
            </div>
          ))}
          <div ref={end} />
        </div>

        <footer className="space-y-2 border-t px-4 py-3">
          {messages.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLES.map((e) => (
                <button key={e} type="button" onClick={() => submit(e)} className="rounded-full border px-2.5 py-0.5 text-xs text-muted-foreground hover:border-brand hover:text-foreground">
                  {e.length > 30 ? e.slice(0, 28) + "…" : e}
                </button>
              ))}
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit(q);
            }}
            className="flex gap-2"
          >
            <input
              ref={input}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="What happens if…?"
              aria-label="Your question"
              maxLength={500}
              className="min-w-0 flex-1 rounded-lg border bg-background px-3 py-2 text-sm"
            />
            <button type="submit" disabled={busy || !q.trim()} className="rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground disabled:opacity-40">
              Ask
            </button>
          </form>
          <p className="text-[11px] text-muted-foreground">
            {live
              ? `${live.gemini ? "Gemini on" : "Gemini off: template answers"} · ${live.nimble ? "Nimble live" : "market scan from cache"}`
              : "No customer rows are sent to any model."}
          </p>
        </footer>
      </div>
    </>
  );
}
