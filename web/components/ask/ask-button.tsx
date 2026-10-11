"use client";

import { Button } from "@/components/ui/button";
import { useAsk } from "./ask-provider";

export function AskButton() {
  const { setOpen } = useAsk();
  return (
    <Button variant="outline" size="sm" onClick={() => setOpen(true)} aria-label="Ask Ripple (command K)">
      <span className="num text-xs">⌘K</span> Ask
    </Button>
  );
}
