export default function Home() {
  return (
    <main className="mx-auto w-full max-w-[1200px] px-6 py-16">
      <p className="text-xs text-muted-foreground">ripple</p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        What happens to sales if…?
      </h1>
      <p className="num-hero mt-8">+$18.4k</p>
      <p className="num mt-2 text-sm text-muted-foreground">
        90% −$3.1k to +$41.0k · 84% chance it pays
      </p>
      <div className="mt-8 flex gap-2">
        <span className="h-8 w-8 rounded-lg bg-brand" />
        <span className="h-8 w-8 rounded-lg bg-brand-soft" />
        <span className="h-8 w-8 rounded-lg bg-gain" />
        <span className="h-8 w-8 rounded-lg bg-loss" />
        <span className="h-8 w-8 rounded-lg bg-warning" />
      </div>
    </main>
  );
}
