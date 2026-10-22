# CLAUDE.md — Ripple

Bayesian what-if simulator + launch tracker + market-aware agent. Owner: Reza (ML/AI Engineer). This is a
portfolio project, started at a 3-hour hackathon (2026-10-11) and now maintained as a normal repo.

## Layout
- `contract/` schemas + fixtures: the agreement between Python and TypeScript. Change it first, then both sides.
- `model/` PyMC model, synthetic data, checks, calibration (`ripple_model/`). `api/` FastAPI for bring-your-own-data.
- `web/` Next.js app, browser simulator, agent. `docs/spec.md` is the design; `docs/method/` is also the agent's RAG source.
- `DECISIONS.md` records choices the spec left open. Add one line when you make one.

## Commands
- Python: `uv sync`, `make test` (`make test-all` includes the slow BYOD fit), `make lint`, `make check-fixtures`.
- Web: `cd web && npm install && npm run dev`, `npm test`, `npx tsc --noEmit`.
- Regenerate the demo: `make pipeline` (synth, fit, export, calibrate). Fits are ~5-11 min.

## Rules
- Fixed random seeds. Every number shown to a user has an interval.
- The LLM never creates a number. Numbers come from the model, or from cited web results labeled as web.
- No keys in the repo. `.env` is gitignored; keep `.env.example` current.
- The demo works with no API key (demo mode, cached agent answers and market scan). Every Gemini/Nimble feature
  must fall back to a template or cached answer when its key is empty.
- No customer data rows go to the LLM by default.
- Tests pass before each commit: `pytest` (model/api), `vitest` (web, including the Python-TS parity tests).
- Do not cut, skip, simplify or replace a feature, check or test without Reza's approval.
- Do not invent API calls. Check the installed version or the official docs (PyMC, ArviZ, nutpie, FastAPI,
  Next.js, `@google/genai`, Nimble).
- Python changes that touch `simulate.py` or `update.py` need the parity fixtures regenerated (`make fixtures`).

## Keys (in `.env`, never committed)
`GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBED_MODEL`, `NIMBLE_API_KEY`, `DEMO_MODE`.
`DEMO_MODE=1` forces demo mode even when keys are present.
