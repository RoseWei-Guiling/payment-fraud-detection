"""
Step 5: train the challenger models and compare them with the rules champion.

  python scripts/train_models.py            # full run (roughly 10-20 minutes on a laptop)
  python scripts/train_models.py --quick    # 25% of training rows, to check everything works first

Trained on the training months; compared on the validation month, at the rules champion's
decline rate. The out-of-time test month is scored and saved but NOT evaluated (that's Step 7).

Outputs:
  reports/model_comparison.md        the comparison, segmentation test, top features, example reasons
  reports/scorecard_points.md        the scorecard: points for every bin of every feature
  data/processed/model_scores.parquet scores for validation and test months (used in Steps 6 and 7)
  data/models/                        the trained models
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import (all_features, at_decline_rate, decline_rate_to_match, early_stop_mask,  # noqa: E402
                        interpretable_features, isolation_score, lightgbm_reasons, numeric_features,
                        ranking_metrics, scorecard_predict, train_isolation_forest, train_lightgbm,
                        text_to_categories, train_random_forest, train_scorecard)

MODELS = Path("data/models")


def readable(value):
    """0.123456 -> 0.123, 3.0 -> 3, blanks -> blank."""
    if pd.isna(value):
        return "blank"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def log(message, t0):
    print(f"[{time.time() - t0:5.0f}s] {message}", flush=True)


def evaluate(name, score, valid, decline_rate, champion_caught):
    row = {"model": name}
    row.update(at_decline_rate(score, valid["isFraud"], valid["TransactionAmt"], decline_rate))
    row.update(ranking_metrics(score, valid["isFraud"]))
    row["decline_rate_to_match_champion"] = decline_rate_to_match(score, valid["isFraud"], champion_caught)
    return row


def as_table(rows):
    t = pd.DataFrame(rows)
    for col in ["decline_rate", "fraud_caught", "fraud_value_caught", "precision", "decline_rate_to_match_champion"]:
        t[col] = t[col].map(lambda v: f"{v:.1%}")
    t["false_positive_ratio"] = t["false_positive_ratio"].map(lambda v: f"{v:.1f}")
    for col in ["auc", "gini", "pr_auc"]:
        t[col] = t[col].map(lambda v: f"{v:.3f}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="train on 25%% of training rows")
    ap.add_argument("--skip-rf", action="store_true", help="skip the random forest (the slowest model)")
    args = ap.parse_args()
    t0 = time.time()

    feats = pd.read_parquet("data/processed/features.parquet")
    iv = pd.read_csv("reports/feature_iv.csv")
    groups = iv.groupby("group")["feature"].apply(list).to_dict()
    rules = pd.read_parquet("data/processed/rules_flags.parquet", columns=["TransactionID", "champion_flag"])
    feats = feats.merge(rules, on="TransactionID", how="left")
    everything = all_features(feats.drop(columns="champion_flag"))
    feats = text_to_categories(feats, everything)
    # float32 halves memory; plenty of precision for these features
    wide_floats = [c for c in everything if feats[c].dtype == "float64" and c != "TransactionAmt"]
    feats[wide_floats] = feats[wide_floats].astype("float32")
    numbers = numeric_features(feats, everything)
    log(f"Loaded {len(feats):,} payments, {len(everything)} features", t0)

    train_mask = feats["split"] == "train"
    if args.quick:
        keep = feats.loc[train_mask].sample(frac=0.25, random_state=0).index
        train_mask = feats.index.isin(keep)
    stop_mask = early_stop_mask(feats) & train_mask
    fit_mask = train_mask & ~stop_mask
    valid = feats[feats["split"] == "valid"]
    scored = feats[feats["split"].isin(["valid", "oot"])]

    # The bar: the rules champion on the validation month
    decline_rate = valid["champion_flag"].mean()
    champion = at_decline_rate(valid["champion_flag"], valid["isFraud"], valid["TransactionAmt"], decline_rate)
    rows = [{"model": "Rules champion", **champion, **ranking_metrics(valid["champion_flag"], valid["isFraud"]),
             "decline_rate_to_match_champion": decline_rate}]
    log(f"Champion: {champion['fraud_caught']:.1%} of fraud caught at {decline_rate:.1%} decline rate", t0)

    out = scored[["TransactionID", "split", "isFraud", "TransactionAmt", "ProductCD"]].copy()
    out["rules_champion"] = scored["champion_flag"]
    MODELS.mkdir(parents=True, exist_ok=True)

    # 1. Scorecard on interpretable features
    candidates = interpretable_features(groups, iv)
    scorecard, categorical, sc_features, sign_dropped = train_scorecard(
        feats.loc[train_mask], feats.loc[train_mask, "isFraud"], candidates)
    out["scorecard"], out["scorecard_points"] = scorecard_predict(scorecard, scored, sc_features, categorical)
    rows.append(evaluate("Logistic scorecard (interpretable)", out.loc[valid.index, "scorecard"], valid,
                         decline_rate, champion["fraud_caught"]))
    with open(MODELS / "scorecard.pkl", "wb") as f:
        pickle.dump({"model": scorecard, "features": sc_features, "categorical": categorical}, f)
    log(f"Scorecard trained on {len(sc_features)} interpretable features "
        f"({len(sign_dropped)} dropped for a flipped sign: {', '.join(sign_dropped) or 'none'})", t0)

    # 2. LightGBM on everything
    lgbm = train_lightgbm(feats.loc[fit_mask, everything], feats.loc[fit_mask, "isFraud"],
                          feats.loc[stop_mask, everything], feats.loc[stop_mask, "isFraud"])
    out["lightgbm"] = lgbm.predict(scored[everything], num_iteration=lgbm.best_iteration)
    rows.append(evaluate("LightGBM (all features)", out.loc[valid.index, "lightgbm"], valid,
                         decline_rate, champion["fraud_caught"]))
    lgbm.save_model(str(MODELS / "lightgbm.txt"), num_iteration=lgbm.best_iteration)
    log(f"LightGBM trained ({lgbm.best_iteration} trees)", t0)

    # 5. Segmented LightGBM: product W vs the rest
    out["lightgbm_segmented"] = np.nan
    for segment_name, in_segment in [("W", lambda d: d["ProductCD"].astype("string") == "W"),
                                     ("not W", lambda d: d["ProductCD"].astype("string") != "W")]:
        f_mask, s_mask = fit_mask & in_segment(feats), stop_mask & in_segment(feats)
        model = train_lightgbm(feats.loc[f_mask, everything], feats.loc[f_mask, "isFraud"],
                               feats.loc[s_mask, everything], feats.loc[s_mask, "isFraud"])
        rows_in = in_segment(scored).to_numpy()
        out.loc[out.index[rows_in], "lightgbm_segmented"] = model.predict(
            scored.loc[rows_in, everything], num_iteration=model.best_iteration)
        model.save_model(str(MODELS / f"lightgbm_segment_{segment_name.replace(' ', '_')}.txt"),
                         num_iteration=model.best_iteration)
    rows.append(evaluate("LightGBM segmented (W / not W)", out.loc[valid.index, "lightgbm_segmented"], valid,
                         decline_rate, champion["fraud_caught"]))
    log("Segmented LightGBM trained", t0)

    # 3. Random forest
    if not args.skip_rf:
        forest = train_random_forest(feats.loc[train_mask, numbers], feats.loc[train_mask, "isFraud"])
        out["random_forest"] = forest.predict_proba(scored[numbers].fillna(-999.0))[:, 1]
        rows.append(evaluate("Random forest (numeric features)", out.loc[valid.index, "random_forest"], valid,
                             decline_rate, champion["fraud_caught"]))
        log("Random forest trained", t0)

    # 4. Isolation forest: no labels used at all
    sample = feats.loc[train_mask, numbers].sample(min(100_000, int(train_mask.sum())), random_state=0)
    iso = train_isolation_forest(sample)
    out["isolation_forest"] = isolation_score(iso, scored[numbers])
    rows.append(evaluate("Isolation forest (no labels)", out.loc[valid.index, "isolation_forest"], valid,
                         decline_rate, champion["fraud_caught"]))
    log("Isolation forest trained", t0)

    out.to_parquet("data/processed/model_scores.parquet", index=False)

    # Explanations: top features and example reasons, validation month
    explain = valid.sample(min(20_000, len(valid)), random_state=0)
    _, contrib = lightgbm_reasons(lgbm, explain[everything])
    importance = (pd.DataFrame({"feature": everything, "mean_abs_shap": np.abs(contrib).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False).head(20))
    importance["group"] = importance["feature"].map(iv.set_index("feature")["group"])
    top_scored = valid.loc[out.loc[valid.index, "lightgbm"].sort_values(ascending=False).index[:3]]
    reasons, _ = lightgbm_reasons(lgbm, top_scored[everything])
    examples = []
    for (_, payment), payment_reasons in zip(top_scored.iterrows(), reasons):
        listed = "; ".join(f"{name} = {readable(payment[name])}" for name, _ in payment_reasons)
        examples.append(f"- Payment {payment.TransactionID} (fraud: {'yes' if payment.isFraud else 'no'}): {listed}")

    table = as_table(rows)
    Path("reports/model_comparison.md").write_text("\n".join([
        "# Challenger models vs the rules champion", "",
        "Generated by `scripts/train_models.py`" + (" **(quick mode: 25% of training rows)**" if args.quick else "")
        + ". Trained on training months, compared on the validation month.", "",
        f"Every model blocks the same **{decline_rate:.1%}** of payments as the rules champion, "
        "so the question is simply: blocking the same number of customers, who catches more fraud?", "",
        "- **AUC / Gini / PR-AUC**: how well the score ranks fraud above genuine payments over every cut-off "
        "(Gini 0 = random, 1 = perfect; PR-AUC focuses on the rare fraud class).",
        "- **decline_rate_to_match_champion**: how many payments the model must block to catch as much fraud "
        "as the champion. Lower is better.", "",
        table.to_markdown(index=False), "",
        "## Top 20 features (LightGBM, mean absolute SHAP, validation sample)", "",
        importance.assign(mean_abs_shap=importance["mean_abs_shap"].round(4)).to_markdown(index=False), "",
        "## Example reasons for the three highest-scored validation payments", "",
        "Top 3 features pushing each payment towards fraud (LightGBM SHAP values):", "",
        *examples, "",
        f"## Scorecard features ({len(sc_features)})", "",
        ", ".join(f"`{f}`" for f in sc_features), "",
        "Dropped because the regression gave them the wrong sign (a safer bin would have earned fewer "
        "points, usually because they overlap with other features): "
        + (", ".join(f"`{f}`" for f in sign_dropped) or "none") + ".", "",
        "Points for every bin are in `reports/scorecard_points.md`.", "",
    ]))
    points = scorecard.table(style="summary")
    Path("reports/scorecard_points.md").write_text("\n".join([
        "# Scorecard points", "",
        "600 points = odds of 50 genuine payments to 1 fraud; every 20 points doubles the odds. "
        "Higher total = safer. Each payment's score is the sum of its points across features.", "",
        points.to_markdown(index=False), "",
    ]))
    print(table.to_string(index=False))
    log("Saved reports/model_comparison.md, reports/scorecard_points.md, data/processed/model_scores.parquet", t0)


if __name__ == "__main__":
    main()
