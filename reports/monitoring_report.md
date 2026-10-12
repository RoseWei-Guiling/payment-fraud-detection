# Monitoring: is the data drifting?

**PSI** (Population Stability Index) compares how a variable is spread in a baseline period with a later one. Below 0.10 = stable, 0.10-0.25 = small shift (watch), above 0.25 = major shift (investigate).

## Score drift (validation month -> test month)

| score              |   psi | status   |
|:-------------------|------:|:---------|
| LightGBM           | 0.004 | stable   |
| Logistic scorecard | 0.009 | stable   |

## Feature drift: LightGBM's 20 most important features (training months -> test month)

| variable                 |   psi | status                    |
|:-------------------------|------:|:--------------------------|
| card_type_and_product_te | 2.896 | major shift (investigate) |
| D10_start                | 1.305 | major shift (investigate) |
| D15_start                | 1.013 | major shift (investigate) |
| D4_start                 | 0.991 | major shift (investigate) |
| addr1_te                 | 0.142 | small shift (watch)       |
| D2                       | 0.039 | stable                    |
| V242                     | 0.027 | stable                    |
| C13                      | 0.026 | stable                    |
| V201                     | 0.018 | stable                    |
| V188                     | 0.018 | stable                    |
| C1                       | 0.014 | stable                    |
| C14                      | 0.011 | stable                    |
| TransactionAmt           | 0.01  | stable                    |
| card2                    | 0.006 | stable                    |
| addr1_freq               | 0.005 | stable                    |
| card2_freq               | 0.005 | stable                    |
| card1_freq               | 0.004 | stable                    |
| card1                    | 0.004 | stable                    |
| C4                       | 0.004 | stable                    |
| C8                       | 0.002 | stable                    |

Some drift is expected by design: counts that grow with time (card1_freq, account start dates) naturally move. It matters when it also moves the score or the decline rate.

## Monitoring plan, if this model went live

| What | How often | Trigger | Action |
|---|---|---|---|
| Share of payments blocked at the fixed cut-off | Daily | Moves more than 25% from its usual level | Check for an incident, a new merchant or a data feed fault |
| Score PSI vs the development sample | Weekly | Above 0.10 watch, above 0.25 act | Find which features moved; consider recalibrating |
| Feature CSI, top 20 features | Weekly | Above 0.25 | Check the data feed; retrain if it's a real change in behaviour |
| Precision of blocked payments (from confirmed cases) | Weekly | Falls 25% below validation | Review the cut-off with the fraud strategy team |
| Gini on matured labels | Monthly, after a delay | Drops more than 0.05 | Retrain on recent months |
| Rules vs model (champion / challenger slice) | Monthly | Model's lead falls inside the uncertainty range | Review before the next release |

Fraud labels arrive late (chargebacks can take 30-90 days), so the label-based checks lag. The label-free checks (PSI, decline rate) are the early warning.
