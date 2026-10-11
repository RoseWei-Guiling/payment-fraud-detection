# Payment Fraud Detection

![tests](https://github.com/RoseWei-Guiling/payment-fraud-detection/actions/workflows/tests.yml/badge.svg)

**Rules vs scorecard vs gradient boosting on real card transactions, with a cost-based cut-off and model risk documentation.**

Built on the public [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset: about 590,000 real, anonymised e-commerce transactions from Vesta, with a 3.5% fraud rate.

## Why this project

At a fraud-rules workshop I wrote hand-made rules and watched them start blocking genuine customers as soon as fraud got subtle. This project asks what replaces those rules in practice: a model that catches more fraud at the same customer friction, priced in pounds and documented the way a model risk team would review it.

## Results

Coming soon. Headline metrics: fraud value caught at a fixed false-positive ratio, Gini on the out-of-time month, and expected £ saved per 100,000 transactions.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py    # needs a Kaggle API token, see the script header
python scripts/build_master.py     # master dataset + reports/data_quality.md
jupyter lab notebooks/01_data_audit.ipynb
```

No Kaggle account yet? Test the pipeline on fake data: `pytest -q`

## Method

| Step | What | Status |
| --- | --- | --- |
| 1 | Master dataset, data quality report, rebuilt customer ID | In progress |
| 2 | BigQuery version of the feature pipeline | |
| 3 | Feature engineering: velocity, newness, entity sharing | |
| 4 | Rules champion | |
| 5 | Challengers: WoE scorecard, LightGBM, random forest, isolation forest | |
| 6 | Cost-based cut-off, champion vs challenger simulation | |
| 7 | Out-of-time validation, monitoring, model development document | |
| 8 | Scoring API, LLM investigator notes, executive summary | |

**Time split:** train on days 1-119, tune on days 120-149, test once on days 150+. Fraud drifts, so a random split would overstate performance.

## Project layout

```
scripts/     download_data.py, build_master.py
src/         data.py (load, merge, time split), uid.py (customer ID), quality.py (report)
notebooks/   01_data_audit.ipynb
reports/     data_quality.md and later results
tests/       pipeline tests, run on small fake data with the real column names
```

Folders for later steps (`sql/`, `rules/`, `api/`) are added when those steps are built.

## Credits

- Data: Vesta Corporation and IEEE-CIS, via Kaggle. Not redistributed here.
- Customer ID reconstruction (`card1` + `addr1` + `day - D1`): Chris Deotte and Konstantin Yakovlev, [1st place solution](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2).
- Feature family meanings: [Vesta's data description](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203).
