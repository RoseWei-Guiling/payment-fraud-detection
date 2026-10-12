# Executive summary: replacing fraud rules with a model

**For:** Head of Fraud Strategy · **Data:** 590,540 real card-not-present payments (IEEE-CIS / Vesta), about six months · **Status:** proof of concept

## The answer

A machine-learning model, used with a blocking rule that takes the payment amount into account, stops **73% of fraud money** on a month it had never seen. The current-style rules stop **8%**. It does this while declining **6.5%** of payments, against 3.8% for the rules.

## What it's worth

Results on the untouched test month, per 100,000 payments, assuming each wrongly blocked genuine payment costs 10 (handling, customer friction, lost sales):

| | Rules today | Model |
|---|---|---|
| Fraud money stopped | 7.6% | **73.3%** |
| Share of blocked payments that are fraud | 19% | **30%** |
| Net benefit | 8,544 | **337,543** |

The model comes out ahead whatever that cost is, from 2 to 50. A simulated trial, sending 10% of traffic to the model, showed a gain well clear of chance.

## Why it works

- **It ranks risk far better.** Gini (0 = guessing, 1 = perfect) is 0.81 for the model against 0.18 for the rules.
- **It weighs money, not just risk.** A risky 900 payment is blocked; an equally risky 5 payment goes through. So it catches the big losses and leaves small genuine payments alone.

## What to be careful about

1. **Some of the gain comes from repeat fraud.** About 60% of test-month fraud is on customers who already had fraud, and the model spots them easily. In practice those cards are often already blocked. On **new customers only**, the model is still strong (Gini 0.71, two-thirds of fraud caught).
2. **First-time fraud on a clean account (likely account takeover) is the weak spot** (Gini 0.25). It needs extra signals: device changes, login behaviour, changed contact details.
3. **This dataset has far more fraud than a typical card book** (3.5% vs about 0.1%). A live version would need a cap on the share of payments declined, and a review queue for borderline cases.
4. **The cost of a false decline is an estimate.** It should be agreed with the fraud strategy and customer teams.

## Is it stable and explainable?

- **Stable:** the model's scores on the new month were distributed almost exactly as before (drift index PSI 0.004; anything under 0.10 counts as stable).
- **Explainable:** every decision comes with reasons in plain words. Reasons shown to customers come from a points-based scorecard.
- **Documented:** a full model development document is in `docs/model_development.md`, ready for model risk review.

## Recommended next steps

1. Run a live **champion/challenger trial** on 5–10% of traffic, with the decline cap in place.
2. Add **account-takeover signals** for customers with a clean history.
3. Agree the **false-decline cost** and the **monitoring triggers** (`reports/monitoring_report.md`) before switching over fully.

*All amounts are in the dataset's currency, which Vesta did not state (most likely US dollars).*
