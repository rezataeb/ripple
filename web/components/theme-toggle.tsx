"use client";

import { useEffect, useState } from "react";

const KEY = "ripple.theme";

/** Light first; dark via the same tokens. The choice is remembered per browser. */
export function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(KEY);
    } catch {
      /* storage blocked: stay light */
    }
    if (saved === "dark") {
      document.documentElement.classList.add("dark");
      // sync React state with what the DOM now shows
      queueMicrotask(() => setDark(true));
    }
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem(KEY, next ? "dark" : "light");
    } catch {
      /* ignore */
    }
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={dark}
      aria-label="Dark mode"
      className="rounded-lg px-2 py-1.5 text-sm text-muted-foreground transition-colors duration-150 hover:text-foreground"
    >
      {dark ? "Light" : "Dark"}
    </button>
  );
}
