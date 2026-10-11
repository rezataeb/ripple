"use client";

import { useAsk } from "./ask-provider";

/** Market context from the agent's last market_scan: labelled as web, never a model input. */
export function MarketCard() {
  const { lastScan } = useAsk();
  if (!lastScan) return null;
  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-base font-medium">Market context</h2>
        <span className="text-xs text-muted-foreground">
          {lastScan.source === "demo" ? "illustrative sample, not a live scan" : "web · Nimble"} · not a model input
        </span>
      </div>
      {lastScan.note && <p className="mt-2 text-sm">{lastScan.note.replace(/^Market context[^:]*:\s*/, "")}</p>}
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {lastScan.sources.map((s) => (
          <li key={s.url}>
            <a className="text-brand hover:underline" href={s.url} target="_blank" rel="noreferrer">
              {s.title}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
