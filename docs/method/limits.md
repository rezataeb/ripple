# Limits — where Ripple is wrong, and how you would know

## The launch tracker is an approximation, with a gate

After a launch, Ripple updates the forecast with Bayes' rule: the fitted model is the
prior, the new weeks are the likelihood, importance weighting gives the posterior. The
other parameters stay at their posterior draws; only the one unknown is re-learned.

This is exact Bayes for the joint model **as long as the importance weights behave**.
When the launch weeks are very surprising, a handful of samples carry all the weight and
the answer becomes noise. So Ripple measures that directly:

- **Pareto k-hat > 0.7** (server) or **weight ESS < 300** (browser) → the badge flips to
  **"Refit needed"**, and the full model is refitted with the new rows.

The limit it cannot fix on its own: the existing parameters are *not* re-learned from the
launch weeks. The refit path does that.

## "Lift" is observed vs. counterfactual, not automatically causal

The lift Ripple reports is the observed outcome against what the model predicted with no
change, for the same cells and weeks. That is causal only if:

1. no other change happened in the same weeks;
2. untreated cities and groups were not affected;
3. the demand pattern was stable.

The `/launch` page shows these three as a checklist rather than hiding them, because the
model cannot verify any of them.

**The fix is in your hands:** keep a random holdout group — for example 10% of users get
no offer. If the CSV has a `holdout` column, Ripple uses those cells as the control and
labels the result **"Experiment"** instead of **"Model-based estimate"**. The gift-boxes
demo launch has one, so you can see both labels.

## The known failure mode: a launch that ramps up

The model has no ramp-up term. A launch that builds slowly over its first weeks looks
like a *weak* launch to Ripple, and the calibration study measures exactly how bad that
gets: coverage of the 90% interval falls well below 90% in the stress arm. It is on the
`/method` page next to the arms that pass, on purpose.

## Extrapolation

A new city is a forecast from its traits. When a standardized trait falls outside the
training range, Ripple raises an extrapolation flag, names the trait, and says the
interval is a **lower bound** on the real uncertainty. Miami in the demo is out of range
on income and age — it is there to show the warning, not to be a good recommendation.

## No competitor reaction

The model answers what happens if you change your price. It does not model what a
competitor does next. The Nimble market scan in "Ask Ripple" shows competitor prices and
current offers with sources as **context**, clearly labelled as web data, and it is never
an input to the model.

## Numbers and the language model

The language model never produces a number. Every number in an answer comes from a tool
result — the posterior, or a cited web result labelled as web — and a number guard
rejects any answer containing a figure that no tool returned. If the guard trips, the
template answer is shown instead.

## The demo data is synthetic

"Brightside Coffee Subscriptions" is fictional, with a fixed seed and planted parameter
values. That is a deliberate trade: it makes parameter recovery provable, which no real
dataset allows. City traits are ACS-shaped but illustrative, not pulled live.
