"use client";

import { useSyncExternalStore } from "react";
import { readUserDataset, setUseUserDataset, activeUserDataset } from "@/lib/posterior";

const subscribe = () => () => {};
const snapshot = () => {
  const mine = readUserDataset();
  return mine ? `${mine.name}|${activeUserDataset() ? "1" : "0"}` : "";
};

/** Demo ↔ your data. Reloads so every page reads the chosen posterior. */
export function DatasetSwitcher({ datasetName }: { datasetName: string }) {
  const s = useSyncExternalStore(subscribe, snapshot, () => "");
  if (!s) {
    return <span className="rounded-lg border bg-surface px-3 py-1 text-sm text-muted-foreground">{datasetName}</span>;
  }
  const [name, active] = s.split("|");
  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="sr-only">Dataset</span>
      <select
        value={active === "1" ? "mine" : "demo"}
        onChange={(e) => {
          setUseUserDataset(e.target.value === "mine");
          window.location.reload();
        }}
        className="rounded-lg border bg-surface px-3 py-1 text-sm"
      >
        <option value="demo">Demo · Brightside Coffee</option>
        <option value="mine">Your data · {name}</option>
      </select>
    </label>
  );
}
