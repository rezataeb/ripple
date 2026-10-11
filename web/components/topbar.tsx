import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { AskButton } from "@/components/ask/ask-button";

export function Topbar({ datasetName, demo }: { datasetName: string; demo: boolean }) {
  return (
    <header className="border-b bg-background">
      <div className="mx-auto flex h-14 w-full max-w-[1200px] items-center gap-4 px-6">
        <Link href="/" className="text-xl font-semibold tracking-tight">
          ripple
        </Link>
        <span
          className="rounded-lg border bg-surface px-3 py-1 text-sm text-muted-foreground"
          title="Dataset switcher arrives with the /data page"
        >
          {datasetName}
        </span>
        {demo && <Badge variant="outline">demo mode</Badge>}
        <nav className="ml-auto flex items-center gap-1 text-sm">
          <Link className="rounded-lg px-3 py-1.5 text-muted-foreground hover:text-foreground" href="/launch">
            Launch tracker
          </Link>
          <Link className="rounded-lg px-3 py-1.5 text-muted-foreground hover:text-foreground" href="/method">
            Method
          </Link>
          <Link className="rounded-lg px-3 py-1.5 text-muted-foreground hover:text-foreground" href="/data">
            Use your data
          </Link>
          <AskButton />
        </nav>
      </div>
    </header>
  );
}
