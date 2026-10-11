# Model card — Ripple demand model

## What it does

Ripple answers "what happens to sales if…?" questions — a promotion, a price change,
or a launch in a city with no history — and answers with a **distribution**, not a
number: incremental units, revenue and profit against a no-change baseline, a 90%
interval, and the probability the change pays.

## The model

One row of data is one week × city × customer segment.

```
y[c,s,t] ~ NegativeBinomial(mu, phi)

log mu = log(share[c,s])                 customer mix of the city (a fixed offset)
       + a_city[c]                        how big this city is
       + a_seg[s]                         how much this segment buys per head
       + e[s]      * log(price / base)    price elasticity, per segment
       + b_promo[s] * promo               the deal signal of being on promotion
       + b_feature  * feature             the email campaign flag
       + b_dip      * promo_last_week     pull-forward: last week's promo borrowed from this one
       + season(t)                        2 Fourier pairs, 52-week period

a_city[c] = g0 + X_city[c]·g + tau_city · z_city[c]
e[s] ~ Normal(mu_e, sigma_e),  mu_e ~ Normal(-1.5, 0.5),  sigma_e ~ HalfNormal(0.5)
```

## Why each part

| Part | Why it is there | What breaks without it |
|---|---|---|
| **Negative Binomial** | Weekly unit counts are overdispersed — the variance is much larger than the mean. | Poisson would call normal week-to-week variation a surprise, and every interval would be far too narrow. |
| **log–log price term** | `e[s]` *is* the elasticity: a 1% price rise changes demand by `e[s]`%. | A linear price term gives a number that means nothing at a different price level, and can predict negative demand. |
| **Elasticity per segment, partially pooled** | Students and retirees do not respond to price the same way. Pooling lets a segment with thin data borrow strength from the others. | Pooled: targeted offers look identical for everyone. Unpooled: a small segment's elasticity is driven by noise. |
| **Partially pooled promo lift** | Being on promotion lifts demand beyond the price cut itself — the deal is a signal. | The price term alone understates a promotion, then overstates the discount needed next time. |
| **Post-promo dip (`b_dip`)** | A promotion pulls purchases forward from the following week. | Promotions look better than they are, because the borrowed sales are never paid back in the model. |
| **Hierarchical city effect on traits** | `a_city` is predicted from population, income and age, so a city with **no history at all** still has a forecast — with honest uncertainty from `tau_city`. | New-city questions become unanswerable, which is a third of the product. |
| **Customer-mix offset** | Volume scales with who lives there, so a new city with a different mix is handled without refitting. | A young, student-heavy city gets forecast as if it had the average mix. |
| **Fourier seasonality** | Yearly demand shape, with few enough parameters to extrapolate past the training weeks. | Season is confounded with the promotion calendar, and `b_promo` absorbs it. |

## Fitting

nutpie NUTS, 4 chains × 1000 draws, `target_accept = 0.99`, fixed seed.

`target_accept` is higher than the usual 0.8 on purpose. At the default the fit had
28 divergences out of 4000 draws; `e` and `b_promo` are now non-centred (with only
four segments, sampling them against their own scale makes a funnel) and
`target_accept = 0.99` takes the count to zero.

A customer upload gets 2 chains × 500 draws with a 90-second budget; past the budget
the run falls back to ADVI and is labelled **fast approximation** everywhere it is shown.

## What it is not

- **No competitor reaction.** It answers what happens if *you* change your price, not
  what happens after a competitor answers back. The market scan in "Ask Ripple" shows
  competitor prices as context and is never an input to the model.
- **Effects are assumed stable** across the forecast horizon.
- **A new city is a forecast from its traits.** Outside the training trait range the
  interval is a lower bound on the real uncertainty, and Ripple says so on the chart.
- **The demo company is synthetic.** That is what makes parameter recovery provable;
  on your own data there is no truth to check against, and the holdout does that job.
