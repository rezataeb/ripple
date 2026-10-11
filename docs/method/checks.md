# How Ripple is checked

Four groups of checks, each with a threshold set before the fit, and each able to
fail in public. They are written to `trust.json` and rendered as cards on the home
page and on `/method`.

## 1. Convergence — did the sampler do its job?

| Check | Threshold | Why this number |
|---|---|---|
| R-hat | ≤ 1.01 | Independent chains must agree. Above 1.01 they found different answers. |
| Effective sample size | ≥ 400 | Below this the interval edges move when you refit, so the 90% interval is not stable. |
| Divergences | 0 | A divergence means the sampler could not explore part of the posterior, so the tails are understated. |

## 2. Fit — does it describe the data it was given?

**Posterior predictive coverage ≥ 85%.** Simulate new weeks from the fitted model and
check how often the real week falls inside its own 90% range. Far below 90% means
overconfidence; far above means the intervals are padded.

## 3. Parameter recovery — does it find effects that are really there?

Demo only. The synthetic company was generated from known parameter values, so we can
ask the question that matters: **does the 94% interval contain the truth?** 30 of 30
planted values are recovered, including every segment elasticity, every promo lift, the
post-promo dip, the trait coefficients and the overdispersion.

This is the check that cannot be run on real data, and it is the reason the demo company
is synthetic rather than scraped.

## 4. Time holdout — does it predict weeks it has never seen?

The last 12 weeks are removed, the model is **refitted** from scratch on the rest, and
then asked to predict the hidden weeks.

- **Coverage of the 90% interval:** 93%.
- **Error vs. a naive baseline:** 18.4% MAPE against 44.7% for a same-week-last-year
  guess.

Beating a naive baseline is a low bar. That is exactly why failing it would matter.

## 5. Is the launch tracker's updating honest?

A separate question, answered separately — see `limits.md` and the calibration chart on
`/method`. Simulation-based calibration generates launches whose truth is drawn from the
same prior the update uses, so a 90% interval **must** cover the truth 90% of the time
if the computation is right. The stress arm breaks an assumption on purpose and shows
coverage dropping, because knowing where a method fails is part of trusting it.
