# web/

The Ripple front end: Next.js (App Router) + TypeScript + Tailwind + shadcn/ui + Recharts + Framer Motion + KaTeX.

```bash
npm install
npm run dev        # http://localhost:3000
npm test           # vitest: Python <-> TypeScript parity, number guard, agent
npx tsc --noEmit
```

The model ships to the browser as `public/model/posterior.json`, so the simulator and the launch slider run
with no server. Regenerate it from the repo root with `make export`.

| Path | What it is |
|---|---|
| `app/` | Pages: `/` simulator, `/launch`, `/data`, `/method`; routes `api/agent` (Ask Ripple) and `api/map` (CSV column mapping) |
| `lib/simulate.ts`, `lib/update.ts`, `lib/psis.ts` | Ports of the Python reference code, tested against `../contract/*.fixture.json` |
| `lib/agent/` | The Ask Ripple agent: tools, number guard, Gemini loop, Nimble scan, RAG, template fallback |
| `components/` | UI, charts, the Ask panel |

Keys are optional. With none set the agent runs on templates and a cached market scan. See the root
[README](../README.md#services-and-keys).
