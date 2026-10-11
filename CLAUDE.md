# CLAUDE.md — Ripple (hackathon project)

## Context
- Owner: Reza (ML/AI Engineer). Project for a portfolio and for a hackathon on 2026-10-11. Hackathon name: **Ripple** (placeholder chosen by Reza). Before you record the demo video or finish the README, ask Reza for the official hackathon name; if he gives one, use it in the video's first line and in the README.
- Hard time limit: 3 hours of build time. Scope is fixed by `RIPPLE_SPEC.md` (v4). Read it fully before you write code.
- Hackathon rules (see `docs/hackathon-rules.md`): all code is written during the event; no code from earlier projects; list all AI tools in the README; demo video < 2 minutes and it says the hackathon name in the first sentence; repo stays public.

## How we work
- Two parallel sessions. **Session A** owns `contract/`, `model/`, `api/`. **Session B** owns `web/`. Do not edit the other session's folders. If you need a change there, write it in `HANDOFF.md`.
- Session A writes `contract/` first (schemas + fixture). Session B builds against the fixture.
- Follow the time table in `RIPPLE_SPEC.md` Section 12. **`MILESTONES.md` is the authority on checkpoints and cuts.** When the user pastes a cut prompt from it, do that cut immediately, record it in `DECISIONS.md`, and continue. Never cut items on the "never cut" list.
- If you are stuck on one error for more than 8 minutes, stop, write it in `HANDOFF.md`, and take the simplest workaround or skip the part.
- At the end of each step, print one status line: `[T+mm] done: … | next: … | risk: …`.
- `timer.sh`, `timer.log`, and `MILESTONES.md` are the user's tools. Do not edit them. Add `timer.log` to `.gitignore`.
- Commit after each step with a clear message. Push often.
- When the spec does not decide something, choose the simplest option and add one line to `DECISIONS.md`.
- Do not invent API calls. Check the installed version or the official docs (PyMC, ArviZ, nutpie, FastAPI, Next.js, `@google/genai`, Nimble).

## Rules
- Fixed random seeds. Every number shown to a user has an interval.
- The LLM never creates a number. Numbers come from the model, or from cited web results labeled as web.
- No keys in the repo. `.env` is in `.gitignore`; keep `.env.example` current.
- The live demo works with no API key (demo mode + cached agent answers and market scan).
- No customer data rows go to the LLM by default.
- Tests must pass before each commit: `pytest` (model/api), `vitest` (web, including the Python–TS parity tests).

## Keys (in `.env`, never committed)
`GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBED_MODEL`, `NIMBLE_API_KEY`, `DEMO_MODE`.
- **At the start, no keys exist.** Build the model and the UI first (they need no key). Build every LLM and Nimble feature so it falls back to cached or template answers when its key is empty, and never block on a missing key. When Reza adds keys to `.env`, the features turn on.
