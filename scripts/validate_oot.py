"""
Step 7: the honest test, on the out-of-time (OOT) month that nothing has touched so far.

The cut-offs chosen on the validation month in Step 6 (reports/cutoffs.json) are applied
UNCHANGED. Nothing is re-tuned on this month.

  python scripts/validate_oot.py

Outputs:
  reports/oot_report.md         did the results hold? new vs returning customers, week by week, reasons
  reports/monitoring_report.md  drift (PSI / CSI) and the monitoring plan
  reports/oot_summary.json      headline numbers, used for the model document and README
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.decisions import expected_cost_rule, outcome  # noqa: E402
from src.models import all_features, ranking_metrics, text_to_categories  # noqa: E402
from src.monitoring import psi, psi_label, psi_table  # noqa: E402
from src.reasons import plain_reason, scorecard_reasons  # noqa: E402

MODELS = ["rules_champion", "scorecard", "lightgbm", "lightgbm_segmented", "random_forest", "isolation_forest"]
NAMES = {"rules_champion": "Rules champion", "scorecard": "Logistic scorecard", "lightgbm": "LightGBM",
         "lightgbm_segmented": "LightGBM segmented", "random_forest": "Random forest",
         "isolation_forest": "Isolation forest"}


def pct(v):
    return f"{v:.1%}"


def money(v):
    return f"{v:,.0f}"


def decisions(df, cut):
    """The four Step 6 strategies, with the cut-offs fixed on the validation month."""
    return {
        "Rules champion": df["rules_champion"].astype(bool),
        "Scorecard, fixed cut-off": df["scorecard"] >= cut["scorecard_threshold"],
        "LightGBM, fixed cut-off": df["lightgbm"] >= cut["lightgbm_threshold"],
        "LightGBM, expected-cost rule": expected_cost_rule(df["lightgbm"], df["TransactionAmt"],
                                                           cut["cost_per_blocked_genuine"]),
    }


def decision_row(blocked, df, cost):
    blocked = np.asarray(blocked, dtype=bool)
    o = outcome(blocked, df["isFraud"], df["TransactionAmt"], cost)
    o["precision"] = (blocked & df["isFraud"].astype(bool).to_numpy()).sum() / max(blocked.sum(), 1)
    return o


def customer_groups(feats, oot):
    """New = never seen before the test month. Returning customers are split by whether
    they already had a confirmed fraud: the model may simply remember those customers."""
    history = feats[feats["split"] != "oot"]
    seen = set(history["uid"].dropna())
    had_fraud = set(history.loc[history["isFraud"] == 1, "uid"].dropna())
    group = np.where(~oot["uid"].isin(seen), "New customer",
                     np.where(oot["uid"].isin(had_fraud), "Returning, had fraud before",
                              "Returning, no fraud before"))
    return pd.Series(group, index=oot.index)


def group_table(oot, groups, cut):
    rows = []
    blocked = oot["lightgbm"] >= cut["lightgbm_threshold"]
    for name in ["New customer", "Returning, no fraud before", "Returning, had fraud before"]:
        g = groups == name
        y = oot.loc[g, "isFraud"]
        row = {"customers": name, "payments": int(g.sum()), "share": g.mean(), "fraud_rate": y.mean()}
        row["lightgbm_gini"] = ranking_metrics(oot.loc[g, "lightgbm"], y)["gini"] if y.nunique() == 2 else np.nan
        row["rules_gini"] = ranking_metrics(oot.loc[g, "rules_champion"], y)["gini"] if y.nunique() == 2 else np.nan
        row["lightgbm_fraud_caught"] = (blocked[g] & (y == 1)).sum() / max((y == 1).sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def weekly_table(df, cut):
    df = df.assign(week=(df["day"] - df["day"].min()) // 7 + 1)
    rows = []
    for (split, week), w in df.groupby(["split", "week"]):
        if w["isFraud"].nunique() < 2 or len(w) < 1000:
            continue  # a part-week at the end is too small to judge
        blocked = w["lightgbm"] >= cut["lightgbm_threshold"]
        rows.append({"month": split, "week": int(week), "payments": len(w), "fraud_rate": w["isFraud"].mean(),
                     "lightgbm_gini": ranking_metrics(w["lightgbm"], w["isFraud"])["gini"],
                     "blocked_at_fixed_cutoff": blocked.mean(),
                     "fraud_caught": (blocked & (w["isFraud"] == 1)).sum() / (w["isFraud"] == 1).sum()})
    return pd.DataFrame(rows)


def explain_top(oot_ids, feats, everything, n=5):
    """Plain-word reasons for the n highest-scored test-month payments."""
    import lightgbm as lgb
    booster = lgb.Booster(model_file="data/models/lightgbm.txt")
    with open("data/models/scorecard.pkl", "rb") as f:
        sc = pickle.load(f)
    rows = feats.set_index("TransactionID").loc[oot_ids]
    contrib = booster.predict(rows[booster.feature_name()], pred_contrib=True)[:, :-1]
    names = np.array(booster.feature_name())
    sc_X = rows[sc["features"]].copy()
    for c in sc["categorical"]:
        sc_X[c] = sc_X[c].astype("object").where(sc_X[c].notna(), np.nan)
    sc_reasons = scorecard_reasons(sc["model"], sc_X, sc["features"])
    lines = []
    for i, (tid, payment) in enumerate(rows.iterrows()):
        top = names[np.argsort(-contrib[i])[:3]]
        lines += [f"**Payment {tid}** (amount {payment['TransactionAmt']:,.2f}, "
                  f"actually {'fraud' if payment['isFraud'] else 'genuine'})", "",
                  "- LightGBM: " + "; ".join(plain_reason(f, payment[f]) for f in top),
                  "- Scorecard: " + ("; ".join(f"{plain_reason(f, payment[f])} (-{p:.0f} points)"
                                               for f, p in sc_reasons[i]) or "no points lost"), ""]
    return lines


def main():
    cut = json.loads(Path("reports/cutoffs.json").read_text())
    cost = cut["cost_per_blocked_genuine"]
    scores = pd.read_parquet("data/processed/model_scores.parquet")
    feats = pd.read_parquet("data/processed/features.parquet")
    scores = scores.merge(feats[["TransactionID", "uid", "day"]], on="TransactionID", how="left")
    models = [m for m in MODELS if m in scores.columns]
    valid, oot = scores[scores["split"] == "valid"], scores[scores["split"] == "oot"]

    # 1. Did the ranking hold?
    ranking = []
    for m in models:
        v, o = ranking_metrics(valid[m], valid["isFraud"]), ranking_metrics(oot[m], oot["isFraud"])
        ranking.append({"model": NAMES[m], "valid_gini": v["gini"], "oot_gini": o["gini"],
                        "change": o["gini"] - v["gini"], "valid_pr_auc": v["pr_auc"], "oot_pr_auc": o["pr_auc"]})
    ranking = pd.DataFrame(ranking)

    # 2. Did the money hold, with the cut-offs fixed?
    money_rows = []
    for split_name, df in [("valid", valid), ("oot", oot)]:
        for strategy, blocked in decisions(df, cut).items():
            money_rows.append({"strategy": strategy, "month": split_name, **decision_row(blocked, df, cost)})
    money_table = pd.DataFrame(money_rows)

    # 3. New vs returning customers
    groups = customer_groups(feats, oot)
    by_group = group_table(oot, groups, cut)

    # 4. Week by week
    weekly = weekly_table(pd.concat([valid, oot]), cut)

    # 5. Drift: scores (valid -> oot) and top features (training months -> oot)
    score_psi = pd.DataFrame([{"score": NAMES[m], "psi": psi(valid[m], oot[m])}
                              for m in ["lightgbm", "scorecard"]])
    score_psi["status"] = score_psi["psi"].map(psi_label)
    import lightgbm as lgb
    booster = lgb.Booster(model_file="data/models/lightgbm.txt")
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values(ascending=False)
    top_features = gain.index[:20].tolist()
    feature_psi = psi_table(feats[feats["split"] == "train"], feats[feats["split"] == "oot"], top_features)

    # 6. Reasons for the highest-scored test-month payments
    everything = all_features(feats)
    feats = text_to_categories(feats, everything)
    top_ids = oot.sort_values("lightgbm", ascending=False)["TransactionID"].head(5).tolist()
    reason_lines = explain_top(top_ids, feats, everything)

    # ---- reports ----
    lgb_v = money_table.query("strategy == 'LightGBM, expected-cost rule' and month == 'valid'").iloc[0]
    lgb_o = money_table.query("strategy == 'LightGBM, expected-cost rule' and month == 'oot'").iloc[0]
    rules_o = money_table.query("strategy == 'Rules champion' and month == 'oot'").iloc[0]
    summary = {
        "oot_payments": int(len(oot)), "oot_fraud_rate": float(oot["isFraud"].mean()),
        "lightgbm_gini_valid": float(ranking.loc[ranking.model == "LightGBM", "valid_gini"].iloc[0]),
        "lightgbm_gini_oot": float(ranking.loc[ranking.model == "LightGBM", "oot_gini"].iloc[0]),
        "rules_gini_oot": float(ranking.loc[ranking.model == "Rules champion", "oot_gini"].iloc[0]),
        "expected_cost_rule_oot": {k: float(lgb_o[k]) for k in ["decline_rate", "fraud_caught",
                                                                 "fraud_value_caught", "net_benefit_per_100k"]},
        "rules_oot": {k: float(rules_o[k]) for k in ["decline_rate", "fraud_caught", "fraud_value_caught",
                                                     "net_benefit_per_100k"]},
        "lightgbm_score_psi": float(score_psi.loc[0, "psi"]),
        "new_customer_gini": float(by_group.loc[0, "lightgbm_gini"]),
        "cost_per_blocked_genuine": cost,
    }
    Path("reports/oot_summary.json").write_text(json.dumps(summary, indent=2))

    rk = ranking.copy()
    for c in ["valid_gini", "oot_gini", "valid_pr_auc", "oot_pr_auc"]:
        rk[c] = rk[c].map("{:.3f}".format)
    rk["change"] = ranking["change"].map("{:+.3f}".format)
    mt = money_table.copy()
    for c in ["decline_rate", "fraud_caught", "fraud_value_caught", "precision"]:
        mt[c] = mt[c].map(pct)
    for c in ["genuine_blocked_per_100k", "net_benefit_per_100k"]:
        mt[c] = mt[c].map(money)
    bg = by_group.copy()
    for c in ["share", "fraud_rate", "lightgbm_fraud_caught"]:
        bg[c] = bg[c].map(pct)
    for c in ["lightgbm_gini", "rules_gini"]:
        bg[c] = bg[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    wk = weekly.copy()
    for c in ["fraud_rate", "blocked_at_fixed_cutoff", "fraud_caught"]:
        wk[c] = wk[c].map(pct)
    wk["lightgbm_gini"] = wk["lightgbm_gini"].map("{:.3f}".format)

    Path("reports/oot_report.md").write_text("\n".join([
        "# Out-of-time test", "",
        f"Generated by `scripts/validate_oot.py`. The test month has {len(oot):,} payments "
        f"({oot['isFraud'].mean():.2%} fraud). Nothing was tuned on it: the models were trained on the training "
        "months and the cut-offs were fixed on the validation month (`reports/cutoffs.json`).", "",
        "## 1. Did the ranking hold?", "",
        rk.to_markdown(index=False), "",
        "## 2. Did the money hold? Cut-offs fixed on the validation month", "",
        f"A blocked genuine payment is assumed to cost {cost:g}, as in Step 6.", "",
        mt.to_markdown(index=False), "",
        "## 3. New vs returning customers (test month)", "",
        "A returning customer who already had fraud is easy to spot, and this dataset labels a customer's later "
        "payments as fraud too. If the model only works on those, it is remembering customers, not "
        "detecting fraud. The new-customer row is the fairest test.", "",
        bg.to_markdown(index=False), "",
        "## 4. Week by week", "",
        "If the fixed cut-off starts blocking a very different share of payments, the population has moved.", "",
        wk.to_markdown(index=False), "",
        "## 5. Why were these payments stopped? (highest-scored test-month payments)", "",
        "LightGBM's reasons can include masked Vesta fields, which can only be described by family. The "
        "scorecard's reasons are always plain features, so they are the ones an investigator or customer would see.",
        "", *reason_lines,
    ]))

    fp = feature_psi.copy()
    fp["psi"] = fp["psi"].map("{:.3f}".format)
    sp = score_psi.copy()
    sp["psi"] = sp["psi"].map("{:.3f}".format)
    Path("reports/monitoring_report.md").write_text("\n".join([
        "# Monitoring: is the data drifting?", "",
        "**PSI** (Population Stability Index) compares how a variable is spread in a baseline period with a "
        "later one. Below 0.10 = stable, 0.10-0.25 = small shift (watch), above 0.25 = major shift (investigate).",
        "", "## Score drift (validation month -> test month)", "",
        sp.to_markdown(index=False), "",
        "## Feature drift: LightGBM's 20 most important features (training months -> test month)", "",
        fp.to_markdown(index=False), "",
        "Some drift is expected by design: counts that grow with time (card1_freq, account start dates) "
        "naturally move. It matters when it also moves the score or the decline rate.", "",
        "## Monitoring plan, if this model went live", "",
        "| What | How often | Trigger | Action |",
        "|---|---|---|---|",
        "| Share of payments blocked at the fixed cut-off | Daily | Moves more than 25% from its usual level "
        "| Check for an incident, a new merchant or a data feed fault |",
        "| Score PSI vs the development sample | Weekly | Above 0.10 watch, above 0.25 act | Find which features "
        "moved; consider recalibrating |",
        "| Feature CSI, top 20 features | Weekly | Above 0.25 | Check the data feed; retrain if it's a real "
        "change in behaviour |",
        "| Precision of blocked payments (from confirmed cases) | Weekly | Falls 25% below validation "
        "| Review the cut-off with the fraud strategy team |",
        "| Gini on matured labels | Monthly, after a delay | Drops more than 0.05 | Retrain on recent months |",
        "| Rules vs model (champion / challenger slice) | Monthly | Model's lead falls inside the "
        "uncertainty range | Review before the next release |", "",
        "Fraud labels arrive late (chargebacks can take 30-90 days), so the label-based checks lag. The "
        "label-free checks (PSI, decline rate) are the early warning.", "",
    ]))
    print(rk.to_string(index=False), "\n")
    print(mt.to_string(index=False), "\n")
    print(bg.to_string(index=False), "\n")
    print(sp.to_string(index=False))
    print(f"Top-20 features: {(feature_psi['psi'] >= 0.25).sum()} major shift, "
          f"{((feature_psi['psi'] >= 0.10) & (feature_psi['psi'] < 0.25)).sum()} small shift\n")
    print("Saved reports/oot_report.md, reports/monitoring_report.md, reports/oot_summary.json")


if __name__ == "__main__":
    main()
