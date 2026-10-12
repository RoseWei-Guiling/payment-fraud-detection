"""
Step 8: prepare the scoring API and prove it gives the same answers as Step 5.

  python scripts/prepare_api.py

1. Saves data/models/scoring_config.json (feature order, text categories, number types).
2. Scores 2,000 test-month payments through the API code, one dict per payment, as a live
   system would receive them, and checks the probabilities match model_scores.parquet.
3. Calls the API itself (in-process, no server needed) for three example payments.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import all_features  # noqa: E402
from src.scoring import Scorer, export_config  # noqa: E402


def main():
    import lightgbm as lgb
    feats = pd.read_parquet("data/processed/features.parquet")
    names = lgb.Booster(model_file="data/models/lightgbm.txt").feature_name()
    assert set(names) <= set(all_features(feats)), "model features missing from features.parquet"
    config = export_config(feats, names, "data/models")
    print(f"Saved data/models/scoring_config.json ({len(names)} features, "
          f"{len(config['categories'])} text columns)")

    scores = pd.read_parquet("data/processed/model_scores.parquet", columns=["TransactionID", "split", "lightgbm",
                                                                              "TransactionAmt"])
    sample = scores[scores["split"] == "oot"].sample(min(2000, (scores["split"] == "oot").sum()), random_state=0)
    rows = feats.set_index("TransactionID").loc[sample["TransactionID"], names]
    payments = [{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                 for k, v in r.items()} for r in rows.to_dict("records")]  # plain JSON-like values
    scorer = Scorer()
    api_p = np.array([r["probability_of_fraud"] for r in scorer.score(payments)])
    diff = np.abs(api_p - sample["lightgbm"].round(4).to_numpy()).max()
    print(f"API vs Step 5 scores on {len(payments):,} test-month payments: max difference {diff:.4f} "
          f"({'MATCH' if diff <= 1e-4 else 'MISMATCH - check data types'})")

    from fastapi.testclient import TestClient
    from api.app import app
    client = TestClient(app)
    print("\nGET /health ->", client.get("/health").json())
    at_stake = sample["lightgbm"] * sample["TransactionAmt"]  # the money behind each decision
    top = sample.loc[at_stake.sort_values(ascending=False).index, "TransactionID"].head(2).tolist()
    low = sample.sort_values("lightgbm")["TransactionID"].head(1).tolist()
    by_id = dict(zip(sample["TransactionID"], payments))
    example = [{"TransactionID": int(t), **by_id[t]} for t in top + low]
    response = client.post("/score", json={"payments": example}).json()
    print("POST /score (the two payments with most fraud money at stake, and the lowest-risk one) ->")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
