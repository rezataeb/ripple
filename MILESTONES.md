# Ripple — Milestones and cut decisions

`timer.sh` notifies you at each checkpoint (and 5 minutes before). At each checkpoint, read the row, check "Done when", and decide in 1 minute. **If a row is not done, do the cut and paste the prompt into the late session. Do not "try for 10 more minutes."**

Times are minutes after you start `timer.sh` (T+0).

| T+ | Checkpoint | Done when (both must be true) | If NOT done → cut, and paste this prompt |
|---|---|---|---|
| **0:15** | Setup | A: repo on GitHub + `contract/` with schemas and fixture. B: `web/` runs with fonts and tokens | A: *"Write only `posterior.schema.json` and the fixture now. Other schemas later."* |
| **0:40** | Data | A: demo CSV + `truth.json` + 3 synthetic launches. B: Simulator layout renders the fixture | A: *"Drop the `feature` flag and seasonality from the data and the model. Keep promo, dip, elasticity, city traits."* |
| **1:15** | Model | A: `posterior.json` + `trust.json` exported, recovery passes. B: `simulate.ts` parity passes; verdict + fan chart work | A: *"Use 2 chains × 500 draws. If recovery fails on one parameter, report it honestly in trust.json and continue."* B: *"Skip segment bars. Verdict + fan chart only."* |
| **1:45** | Launch tracker | A: `update.py` + `calibration.json`. B: `/launch` page with week slider works | A: *"Run calibration with 50 launches, new-city only. Skip the stress test."* B: *"Only the Austin launch. Skip the prior-vs-posterior plot."* |
| **2:10** | Agent + API | B: Ask panel answers the 4 example prompts (cached if no key). A: BYOD endpoints run locally | **CUT BYOD fit and API deploy.** A: *"Stop the API. Write README draft and docs/method now."* B: *"Agent tools: build_scenario, run_scenario, update_with_launch only. Market scan from cache."* |
| **2:25** | Pages | B: `/method` with calibration chart. `/data` with mapping + validation | **CUT `/data` page.** B: *"Skip /data. Finish /method, then deploy."* |
| **2:35** | Deploy | Live Vercel link works in demo mode with no key | *"Stop all features. Fix only the deploy. If it fails in 10 min, record the video on localhost."* |
| **2:50** | Video | `assets/demo.mp4` < 2 min, first line says the hackathon name | *"Record only: Simulator preset → Launch slider → /method. No TTS; add text captions with ffmpeg."* |
| **3:00** | **SUBMIT** | Repo public, README with GIF + live link + video + AI tools list; submission sent | Submit what you have now. Fix the README after submission only if the rules permit small fixes. |

## Never cut
Intervals on every number · incremental vs. baseline · recovery + holdout checks · launch tracker with the ESS/PSIS gate · calibration coverage chart · number guard · live link (or local video) · README · video.

## Emergency rules
- **A session is stuck on one error for more than 8 minutes:** paste *"Stop. Write the error and what you tried in HANDOFF.md. Choose the simplest workaround, or skip this part and mark it in DECISIONS.md."*
- **Usage limit reached in one session:** continue the most important work (Session A until 1:45, then Session B) in the other session.
- **Gemini or Nimble key problem:** set `DEMO_MODE=1` and use cached answers. The model part needs no key.
