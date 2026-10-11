# Payment Fraud Detection

![tests](https://github.com/YOUR-GITHUB-USERNAME/payment-fraud-detection/actions/workflows/tests.yml/badge.svg)

Can a model beat hand-written fraud rules without blocking more genuine customers? This project compares a rules engine, a logistic scorecard and gradient boosting on about 590,000 real card transactions ([IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection)), picks the cut-off by cost in £, and documents the model for model risk review.

## Results

_Coming soon._

## Progress

| Step | Work | Status |
| --- | --- | --- |
| 1 | Master dataset, data quality report, customer ID | In progress |
| 2 | Features in SQL (BigQuery) | |
| 3 | Feature engineering | |
| 4 | Rules champion | |
| 5 | Scorecard, LightGBM, random forest, isolation forest | |
| 6 | Cost-based cut-off, champion vs challenger | |
| 7 | Out-of-time validation, monitoring, model document | |
| 8 | Scoring API, LLM case notes, executive summary | |

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # needs a Kaggle API token
python scripts/build_master.py
```

`pytest -q` runs the tests without the real data.

## Data and credits

Data from Vesta and IEEE-CIS via Kaggle; not redistributed here. The customer ID method (`card1` + `addr1` + `day − D1`) is from the [1st place solution](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2).
