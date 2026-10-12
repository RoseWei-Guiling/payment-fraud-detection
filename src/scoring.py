"""
Step 8: score payments one at a time, the way a live system would.

The Scorer loads the trained LightGBM model, the cut-off settings from Step 6 and (if
available) the scorecard, and returns for each payment:
    probability of fraud, decision (block / allow), the money behind the decision, reasons

Features are assumed to be ready-made (in a bank they come from a feature store, built by
SQL like sql/velocity_features.sql). Computing customer history live is out of scope.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.reasons import plain_reason, scorecard_reasons

CONFIG = "scoring_config.json"


def export_config(feats, feature_names, model_dir):
    """Save what the API needs to rebuild model inputs exactly as in training:
    the feature order, the categories of each text column and which columns were float32."""
    categories, float32 = {}, []
    for c in feature_names:
        if pd.api.types.is_numeric_dtype(feats[c]):
            if feats[c].dtype == "float64" and c != "TransactionAmt":
                float32.append(c)  # train_models.py stored these as float32
        else:
            categories[c] = sorted(feats[c].dropna().astype(str).unique().tolist())
    config = {"features": feature_names, "categories": categories, "float32_columns": float32}
    Path(model_dir, CONFIG).write_text(json.dumps(config))
    return config


class Scorer:
    def __init__(self, model_dir="data/models", cutoffs="reports/cutoffs.json"):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=str(Path(model_dir, "lightgbm.txt")))
        self.config = json.loads(Path(model_dir, CONFIG).read_text())
        self.cost = json.loads(Path(cutoffs).read_text())["cost_per_blocked_genuine"]
        self.scorecard = None
        sc_path = Path(model_dir, "scorecard.pkl")
        if sc_path.exists():
            try:
                with open(sc_path, "rb") as f:
                    self.scorecard = pickle.load(f)
            except ImportError:  # optbinning not installed: LightGBM reasons only
                self.scorecard = None

    def frame(self, payments):
        """Payments (a list of dicts) -> a table with every model feature, typed as in training.
        A missing feature is blank; an unseen category is treated as blank."""
        raw = pd.DataFrame(payments)
        out = {}
        for c in self.config["features"]:
            values = raw[c] if c in raw else pd.Series([np.nan] * len(raw))
            if c in self.config["categories"]:
                known = set(self.config["categories"][c])
                text = [str(v) if v is not None and not pd.isna(v) and str(v) in known else None for v in values]
                out[c] = pd.Categorical(text, categories=self.config["categories"][c])
            else:
                out[c] = pd.to_numeric(values, errors="coerce").astype("float64")
        X = pd.DataFrame(out)
        f32 = [c for c in self.config["float32_columns"] if c in X]
        X[f32] = X[f32].astype("float32")
        return X

    def score(self, payments, n_reasons=3):
        X = self.frame(payments)
        p = self.model.predict(X)
        contrib = self.model.predict(X, pred_contrib=True)[:, :-1]
        names = np.array(self.config["features"])
        sc_reasons = self._scorecard_reasons(X) if self.scorecard else [None] * len(X)
        results = []
        for i in range(len(X)):
            amount = float(X["TransactionAmt"].iloc[i])
            expected_loss = p[i] * amount           # fraud money at stake if we allow it
            cost_if_wrong = (1 - p[i]) * self.cost   # friction cost if we block a genuine payment
            top = names[np.argsort(-contrib[i])[:n_reasons]]
            result = {
                "probability_of_fraud": round(float(p[i]), 4),
                "decision": "block" if expected_loss > cost_if_wrong else "allow",
                "expected_fraud_loss": round(float(expected_loss), 2),
                "expected_cost_of_blocking": round(float(cost_if_wrong), 2),
                "reasons": [plain_reason(f, X[f].iloc[i]) for f in top],
            }
            if sc_reasons[i] is not None:
                result["scorecard_reasons"] = [f"{plain_reason(f, X[f].iloc[i])} (-{pts:.0f} points)"
                                               for f, pts in sc_reasons[i]]
            if "TransactionID" in payments[i]:
                result = {"TransactionID": payments[i]["TransactionID"], **result}
            results.append(result)
        return results

    def _scorecard_reasons(self, X):
        sc = self.scorecard
        data = X[sc["features"]].copy()
        for c in sc["categorical"]:
            data[c] = data[c].astype("object").where(data[c].notna(), np.nan)
        return scorecard_reasons(sc["model"], data, sc["features"])
