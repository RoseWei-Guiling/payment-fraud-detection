"""
Step 5: challenger models.

Every model is trained on the training months and compared on the validation month,
at the SAME decline rate as the rules champion, so the comparison is fair:
"blocking the same number of customers, who catches more fraud?"

  1. Logistic scorecard  - interpretable features only, Weight-of-Evidence bins, points
  2. LightGBM            - gradient-boosted trees on all features
  3. Random forest       - a second tree method, as a sense check
  4. Isolation forest    - unsupervised: finds unusual payments without using labels
  5. Segmented LightGBM  - separate models for product W and the other products
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ID_COLS = ["TransactionID", "isFraud", "split", "day", "uid"]
EARLY_STOP_DAYS = 15  # the last 15 training days are held back to decide when to stop boosting

# Feature groups whose meaning we can explain (no masked C / D / M / V / id_ columns).
INTERPRETABLE_GROUPS = ["missingness", "email", "device", "frequency", "target_encoding",
                        "device_sharing", "velocity"]
INTERPRETABLE_BASE = ["TransactionAmt", "log_amt", "amt_cents", "amt_is_round", "hour", "weekday",
                      "has_identity", "uid_complete", "ProductCD", "card4", "card6", "DeviceType"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def at_decline_rate(score, is_fraud, amount, decline_rate):
    """Block the highest-scoring `decline_rate` share of payments and measure what happens."""
    score = np.asarray(score, dtype="float64")
    is_fraud = np.asarray(is_fraud, dtype=bool)
    amount = np.asarray(amount, dtype="float64")
    n_block = int(round(decline_rate * len(score)))
    blocked = np.zeros(len(score), dtype=bool)
    blocked[np.argsort(-score, kind="stable")[:n_block]] = True
    caught = (blocked & is_fraud).sum()
    return {
        "decline_rate": blocked.mean(),
        "fraud_caught": caught / is_fraud.sum(),
        "fraud_value_caught": amount[blocked & is_fraud].sum() / amount[is_fraud].sum(),
        "precision": caught / max(blocked.sum(), 1),
        "false_positive_ratio": (blocked.sum() - caught) / max(caught, 1),
    }


def ranking_metrics(score, is_fraud):
    """How well the score ranks fraud above genuine payments, over every possible cut-off."""
    auc = roc_auc_score(is_fraud, score)
    return {"auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(is_fraud, score)}


def decline_rate_to_match(score, is_fraud, target_fraud_caught):
    """Smallest decline rate at which this score catches `target_fraud_caught` of the fraud."""
    order = np.argsort(-np.asarray(score, dtype="float64"), kind="stable")
    caught_so_far = np.cumsum(np.asarray(is_fraud)[order]) / np.asarray(is_fraud).sum()
    position = int(np.searchsorted(caught_so_far, target_fraud_caught))
    return (position + 1) / len(order)


# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
def all_features(feats):
    return [c for c in feats.columns if c not in ID_COLS]


def interpretable_features(groups, iv, min_iv=0.02, max_features=30):
    """Named / explainable features with at least weak IV, strongest first."""
    allowed = set(INTERPRETABLE_BASE)
    for g in INTERPRETABLE_GROUPS:
        allowed.update(groups.get(g, []))
    ranked = iv[iv["feature"].isin(allowed) & (iv["iv"] >= min_iv)].sort_values("iv", ascending=False)
    return ranked["feature"].head(max_features).tolist()


def numeric_features(feats, cols):
    return [c for c in cols if pd.api.types.is_numeric_dtype(feats[c])]


def text_to_categories(feats, cols):
    """LightGBM needs text columns as pandas categories. Converting once on the whole table
    keeps the category codes identical across training, validation and test rows."""
    for c in cols:
        if not pd.api.types.is_numeric_dtype(feats[c]) and not isinstance(feats[c].dtype, pd.CategoricalDtype):
            feats[c] = feats[c].astype("category")
    return feats


def early_stop_mask(feats):
    """Training rows from the last EARLY_STOP_DAYS of the training period."""
    train_days = feats.loc[feats["split"] == "train", "day"]
    cut = train_days.max() - EARLY_STOP_DAYS
    return (feats["split"] == "train") & (feats["day"] > cut)


# ---------------------------------------------------------------------------
# 1. Logistic scorecard
# ---------------------------------------------------------------------------
def train_scorecard(X, y, features):
    """Weight-of-Evidence binning + logistic regression, scaled to points.

    Each feature is cut into a few bins (blanks get their own bin), each bin is replaced by
    its Weight of Evidence (how much more or less fraud it holds than average), and a
    logistic regression weights the features. Points: 600 = odds of 50 genuine to 1 fraud,
    and every 20 points doubles those odds (higher points = safer).
    """
    from optbinning import BinningProcess, Scorecard
    from sklearn.linear_model import LogisticRegression

    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
    data = X[features].copy()
    for c in categorical:
        data[c] = data[c].astype("object").where(data[c].notna(), np.nan)
    binning = BinningProcess(features, categorical_variables=categorical, max_n_bins=6, min_bin_size=0.02,
                             split_digits=4)
    scorecard = Scorecard(binning_process=binning, estimator=LogisticRegression(max_iter=1000),
                          scaling_method="pdo_odds",
                          scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
                          rounding=True)  # whole-number points; default direction = higher points, safer
    scorecard.fit(data, y, show_digits=4)
    return scorecard, categorical


def scorecard_predict(scorecard, X, features, categorical):
    data = X[features].copy()
    for c in categorical:
        data[c] = data[c].astype("object").where(data[c].notna(), np.nan)
    return scorecard.predict_proba(data)[:, 1], scorecard.score(data)


# ---------------------------------------------------------------------------
# 2 and 5. LightGBM (whole population or one segment)
# ---------------------------------------------------------------------------
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=0)


def train_lightgbm(X_fit, y_fit, X_stop, y_stop, max_rounds=3000):
    import lightgbm as lgb
    fit = lgb.Dataset(X_fit, y_fit)
    stop = lgb.Dataset(X_stop, y_stop, reference=fit)
    return lgb.train(LGB_PARAMS, fit, num_boost_round=max_rounds, valid_sets=[stop],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def lightgbm_reasons(model, X, top=3):
    """Top reasons per payment from LightGBM's built-in SHAP values (pred_contrib).
    A positive value pushes the payment towards fraud."""
    contrib = model.predict(X, pred_contrib=True)[:, :-1]  # last column is the baseline
    names = np.array(X.columns)
    order = np.argsort(-contrib, axis=1)[:, :top]
    return [[(names[j], float(contrib[i, j])) for j in order[i]] for i in range(len(X))], contrib


# ---------------------------------------------------------------------------
# 3. Random forest and 4. isolation forest (numbers only, blanks filled)
# ---------------------------------------------------------------------------
BLANK = -999.0  # tree models just need a value far outside the real range


def train_random_forest(X, y, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=50,
                                   max_samples=0.3, n_jobs=-1, random_state=seed)
    return model.fit(X.fillna(BLANK), y)


def train_isolation_forest(X, seed=0):
    """Learns what 'normal' looks like without labels; higher score = more unusual."""
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(n_estimators=200, max_samples=256, n_jobs=-1, random_state=seed)
    return model.fit(X.fillna(BLANK))


def isolation_score(model, X):
    return -model.score_samples(X.fillna(BLANK))
