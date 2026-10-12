# Payment Fraud Detection

![tests](https://github.com/YOUR-GITHUB-USERNAME/payment-fraud-detection/actions/workflows/tests.yml/badge.svg)

Can a model beat hand-written fraud rules without blocking more genuine customers? This project compares a rules engine, a logistic scorecard and gradient boosting on about 590,000 real card transactions ([IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection)). It chooses the cut-off in money, tests on a month the models never saw, and documents the model for model risk review.

## Results

On the out-of-time test month, using cut-offs fixed beforehand on the validation month:

| | Rules | Scorecard | **LightGBM** |
|---|---|---|---|
| Gini | 0.18 | 0.57 | **0.81** (0.71 on new customers) |
| Fraud money stopped | 7.6% | 54.1% | **73.3%** |
| Payments declined | 3.8% | 22.8% | **6.5%** |
| Net benefit per 100k payments | 8,544 | 77,253 | **337,543** |

LightGBM uses an amount-aware rule: block when P(fraud) × amount > (1 − P(fraud)) × the cost of a false decline (assumed to be 10; the result holds for any value from 2 to 50). Money is in the dataset's currency, most likely US dollars.

**What to read:**

- [Executive summary](docs/executive_summary.md): one page, for non-specialists
- [Model development document](docs/model_development.md): design, validation, limitations, monitoring
- [Out-of-time test](reports/oot_report.md) · [Drift monitoring](reports/monitoring_report.md) · [Decisions in money](reports/decision_report.md)

![Net benefit by share of payments blocked](reports/cost_curve.png)

**Honest caveats:**

- About 60% of test-month fraud is repeat fraud on known customers, so quote the new-customer Gini (0.71) next to the overall one.
- A first fraud on a clean account (likely account takeover) is the weak spot.

## What's inside

| Step | Work |
| --- | --- |
| 1 | Master dataset, data quality report, rebuilt customer ID |
| 2 | Customer behaviour features in SQL, checked in DuckDB and BigQuery |
| 3 | 285 features with leakage tests; information value |
| 4 | Rules champion chosen by a greedy optimiser |
| 5 | Scorecard, LightGBM, random forest, isolation forest, segmentation |
| 6 | Cut-off chosen in money; champion/challenger simulation |
| 7 | Out-of-time test, new vs returning customers, PSI drift, reason codes |
| 8 | Scoring API, investigator case notes (LLM with a fact check), executive summary |

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # needs a Kaggle API token
python scripts/build_master.py
python scripts/build_velocity.py
python scripts/build_features.py
python scripts/build_rules.py --max-decline-rate 0.10
python scripts/train_models.py
python scripts/choose_cutoff.py
python scripts/validate_oot.py
python scripts/prepare_api.py        # checks the API matches the trained model
```

**Score payments through the API:**

```bash
uvicorn api.app:app --reload         # then open http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" \
  -d '{"payments": [{"TransactionAmt": 250, "ProductCD": "C", "uid_n_24h": 4}]}'
```

**Case notes for investigators:** `python scripts/write_case_notes.py` (add `--llm` with an `ANTHROPIC_API_KEY` to have Claude write them; any note containing a number that isn't in the facts is rejected).

`pytest -q` runs the tests on fake data, with no download needed.

## Data and credits

Data from Vesta and IEEE-CIS via Kaggle; not redistributed here. The customer ID method (`card1` + `addr1` + `day − D1`) is from the [1st place solution](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2).
