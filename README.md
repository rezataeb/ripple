# ripple

**"What happens to sales if…?"** — a Bayesian what-if simulator, a launch tracker that learns from real data, and a market-aware AI agent.

> Built in 3 hours at the Ripple hackathon, 2026-10-11. **Work in progress — this README is written at the end of the build.**

Ripple answers business what-if questions with a distribution, never a single number: incremental units, revenue and profit vs. a no-change baseline, a 90% interval, and the probability the change pays.

- **The model makes the numbers.** A hierarchical Bayesian demand model (PyMC), simulated in the browser so sliders update instantly.
- **The agent explains them.** A Gemini tool-calling agent that never writes a number a tool did not return.
- **The web gives context.** Live competitor prices and offers via Nimble, cited.
- **Reality corrects it.** After a launch, each new week of real data updates the forecast with Bayes' rule, and a calibration study proves the intervals are honest.

## Status

| Part | State |
|---|---|
| `contract/` | schemas + fixture ✅ |
| `model/` | in progress |
| `api/` | in progress |
| `web/` | in progress (Session B) |

## Quickstart

```bash
uv sync                 # Python: model + API
uv run pytest           # model and contract tests
cd web && npm install && npm run dev
```

No API key is needed: the demo runs in demo mode with a cached agent and a cached market scan.

## AI tools used

- **Claude Code** (Claude Opus 5) — wrote the code in two parallel sessions during the event.
- **Google Gemini** — the "Ask Ripple" agent (function calling), the CSV mapping assistant, and embeddings for the method RAG.
- **Nimble** — live web search for competitor prices and current offers.

## License

MIT — see [LICENSE](LICENSE).
