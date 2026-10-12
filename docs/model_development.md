# Model development document: card payment fraud score

A summary in the shape a bank's model risk team expects: what the model is for, how it was built, how it was tested, where it can fail and how it will be watched. It loosely follows the principles of the PRA's model risk management statement (SS1/23). Detailed numbers are in `reports/`.

| | |
|---|---|
| **Model** | LightGBM fraud score, used with an amount-aware blocking rule |
| **Challenger to** | Hand-written rules (`rules/champion_rules.yaml`) |
| **Benchmark / fallback** | Logistic scorecard, 30 explainable features (`reports/scorecard_points.md`) |
| **Decision** | Block or allow a card-not-present payment in real time |
| **Status** | Portfolio project on public data, not deployed |

## 1. Purpose and scope

The model scores each card payment for the chance that it is fraud, and a rule turns that score into block or allow. The aim is to stop more fraud money than the current rules **without blocking more genuine customers**.

It is in scope for card-not-present payments like those in the IEEE-CIS data (Vesta e-commerce). It is not built for in-store payments, bank transfers (APP scams) or credit decisions.

## 2. Data

- **Source:** IEEE-CIS Fraud Detection (Vesta, via Kaggle). 590,540 payments, with 144,233 of them having device and identity details. About 3.5% are fraud.
- **Masking:** many columns are anonymised (C, D, M, V, id_ families), and there is no customer ID. A customer ID (`uid`) is rebuilt from card, billing area and account start date, following the 1st-place Kaggle solution.
- **Label:** `isFraud`. Vesta marks a payment as fraud after a chargeback, and also marks the same customer's later payments. This matters in section 7.
- **Currency:** not stated by Vesta, most likely US dollars. All money results use the dataset's units.
- **Data quality:** `reports/data_quality.md`.

## 3. Time split

The data covers about six months. It is split by time, never at random, so the model is always tested on the future:

| Split | Days | Used for |
|---|---|---|
| Training | 1–119 | Fitting models. The last 15 days decide when LightGBM stops adding trees |
| Validation | 120–149 | Comparing models and choosing cut-offs |
| Out-of-time test | 150+ | One final check. Nothing is tuned on it |

## 4. Features and leakage controls

There are 285 features (`reports/feature_report.md`): amount and time, customer behaviour (velocity in the last 1 hour, 24 hours and 7 days, built in SQL and checked against Python, in DuckDB and BigQuery), email and device, how common a value is, past fraud rates, and Vesta's own fields after removing near-duplicates.

Leakage controls (each one is tested in `tests/`):

- Every customer feature uses only that customer's **earlier** payments.
- Past-fraud-rate features are computed out-of-fold, grouped by customer, so a customer's own label never feeds their feature.
- How-common counts come from training months only.
- Automated tests change future labels and drop future days, and check that no earlier feature changes.

## 5. Models

| Model | Why it's here |
|---|---|
| Rules champion | Today's approach: chosen from 11 candidate rules by a greedy optimiser, under a precision floor and a decline budget |
| Logistic scorecard | Transparent: binned features, Weight of Evidence, points (600 = 50:1 odds, 20 points doubles the odds) |
| **LightGBM** | Main challenger: gradient-boosted trees on all features |
| LightGBM segmented | Tests whether product W needs its own model. It doesn't (same Gini) |
| Random forest | A second tree method as a sense check |
| Isolation forest | Uses no labels; tests how far "unusual" alone gets you |

## 6. Decision strategy

The cut-off is set **in money**, not by accuracy (`reports/decision_report.md`):

- net benefit = fraud money blocked − cost × genuine payments blocked;
- assumed cost of blocking a genuine payment = 10, with 2 to 50 tested;
- chosen rule: block when **P(fraud) × amount > (1 − P(fraud)) × cost**. Risky large payments are stopped, and equally risky small ones go through.

On the validation month, this rule beat the single best cut-off at every assumed cost. A simulated champion/challenger trial (10% of traffic) gave a 95% interval entirely above zero.

## 7. Validation results

**Ranking (Gini):**

| | Validation | Test month | Change |
|---|---|---|---|
| Rules champion | 0.193 | 0.180 | −0.013 |
| Logistic scorecard | 0.598 | 0.569 | −0.029 |
| **LightGBM** | **0.838** | **0.813** | −0.025 |
| Random forest | 0.785 | 0.765 | −0.019 |

**Money on the test month**, using the cut-offs fixed on the validation month:

| Strategy | Blocks | Fraud caught | Fraud value caught | Precision | Net benefit per 100k payments |
|---|---|---|---|---|---|
| Rules champion | 3.8% | 21.2% | 7.6% | 19.2% | 8,544 |
| **LightGBM, expected-cost rule** | **6.5%** | 55.7% | 73.3% | 29.5% | **337,543** |
| LightGBM, fixed cut-off | 13.4% | 75.1% | 74.4% | 19.5% | 281,746 |

**By customer type (test month):**

| Customers | Share of payments | Fraud rate | LightGBM Gini | Fraud caught |
|---|---|---|---|---|
| New (never seen before) | 46.4% | 2.9% | 0.710 | 65.8% |
| Returning, no earlier fraud | 42.0% | 0.1% | 0.247 | 6.7% |
| Returning, earlier fraud | 11.5% | 17.9% | 0.683 | 82.3% |

**Conclusion:** the model held up out of time. Gini fell by only 0.025, every model kept its rank, and the amount-aware rule kept its lead in money (−12% vs validation). The fair test is new customers, where the Gini is 0.71. The headline 0.81 is helped by returning customers who already had fraud (section 8).

## 8. Limitations and risks

1. **Repeat-fraud customers flatter the results.** About 60% of the test month's fraud comes from customers with an earlier fraud, because Vesta labels their later payments too. A real bank would usually have blocked the card after the first confirmed fraud. Realistic performance is closer to the new-customer figures.
2. **First fraud on a clean account is the weak spot.** Returning customers with no earlier fraud have a Gini of only 0.25 (and very few frauds, about 40, so this is uncertain). This looks like account takeover, and it needs other signals: device change, login behaviour, contact-detail changes.
3. **The decline rates are high for a real bank.** This dataset is about 3.5% fraud, against roughly 0.1% in a typical card portfolio, so blocking more pays off here. Live use would add a cap on the share of payments blocked.
4. **The cost of a false block is an assumption.** The conclusions hold from 2 to 50, but the exact cut-off depends on it.
5. **Masked features.** Some of LightGBM's reasons are anonymised Vesta fields, which can only be described by family. Reasons that customers see should come from the scorecard, which only uses explainable features.
6. **Fairness can't be tested.** The data has no age, sex or other protected characteristics. A live model would need checks for unfair outcomes across customer groups (FCA Consumer Duty).
7. **Fixed cut-offs drift.** On the test month the fixed LightGBM cut-off blocked 13.4% instead of 11.4%. The amount-aware rule moved less (5.9% → 6.5%).
8. **Label delay.** Chargebacks take weeks, so label-based checks always lag.

## 9. Explainability

Each blocked payment gets three reasons in plain words (`src/reasons.py`):

- **Scorecard:** the features where the payment lost the most points, e.g. "past fraud rate of the billing area code: 0.055 (−46 points)". The points add up exactly to the score.
- **LightGBM:** SHAP values, translated into plain words where the feature isn't masked.

Examples are in `reports/oot_report.md`.

## 10. Monitoring

On the test month, score drift is negligible: PSI is 0.004 for both LightGBM and the scorecard. Four of the top 20 features show a major shift, mostly counts that grow over time by design. Since the score didn't move, this isn't a problem yet, but it should be watched. The full plan, with triggers and actions, is in `reports/monitoring_report.md`:

- **Daily:** share of payments blocked.
- **Weekly:** PSI of the score and features, precision of blocked payments.
- **Monthly:** Gini on matured labels, and the champion/challenger comparison.

## 11. Before going live

- Retrain on all six months and confirm the cut-off on the latest month.
- Agree the cost of a false block with the fraud strategy and customer teams.
- Add a cap on the share of payments blocked, and a review queue for borderline scores.
- Run a live champion/challenger trial and account for blocked payments that are never confirmed.
- Independent validation by the model risk team.
