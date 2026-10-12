"""
Step 7: reason codes in plain words.

An investigator (or a customer complaint) needs "why was this payment stopped?" in words,
not "V188 = 4". Two sources of reasons:

  1. LightGBM: the features that pushed this payment's score up most (SHAP values).
     Some are masked Vesta columns (C, D, M, V, id_), so those can only be described by
     their family, e.g. "a Vesta count field (C1)". That's an honest limit of this dataset.
  2. Scorecard: the features where this payment lost the most points compared with the
     best possible bin. Every scorecard feature has a plain meaning, so these reasons are
     always readable. This is how credit scorecards produce reason codes.
"""
import re

import numpy as np
import pandas as pd

PLAIN = {
    "TransactionAmt": "payment amount",
    "log_amt": "payment amount (log scale)",
    "amt_cents": "cents part of the amount",
    "amt_is_round": "amount is a round number",
    "hour": "hour of day",
    "weekday": "day of week",
    "has_identity": "device/identity details present",
    "uid_complete": "customer could be identified",
    "account_age_days": "days since the account started",
    "ProductCD": "product type",
    "card1": "card group (issuer-like code)",
    "card2": "card group 2 (issuer-like code)",
    "card3": "card country-like code",
    "card4": "card network",
    "card5": "card category code",
    "card6": "debit or credit card",
    "addr1": "billing area code",
    "addr2": "billing country code",
    "dist1": "distance between addresses",
    "dist2": "distance between addresses (2)",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "DeviceType": "desktop or mobile",
    "n_missing_v": "number of blank Vesta fields",
    "n_missing_d": "number of blank time-gap fields",
    "n_missing_m": "number of blank match checks",
    "n_missing_id": "number of blank device fields",
    "m_true_count": "number of match checks passed",
    "P_email_provider": "purchaser email provider",
    "P_email_suffix": "purchaser email country suffix",
    "R_email_provider": "recipient email provider",
    "email_P_equals_R": "purchaser and recipient email match",
    "P_email_missing": "purchaser email missing",
    "device_family": "device make",
    "os_family": "operating system",
    "browser_family": "browser",
    "uids_per_fingerprint": "customers seen on this device",
    "new_fingerprint_for_uid": "first time this customer uses this device",
    "uid_txn_number": "customer's payment number (1 = first)",
    "uid_n_1h": "customer's payments in the last hour",
    "uid_n_24h": "customer's payments in the last 24 hours",
    "uid_n_7d": "customer's payments in the last 7 days",
    "uid_amt_24h": "customer's spend in the last 24 hours",
    "uid_amt_mean_prior": "customer's average earlier payment",
    "amt_to_uid_mean": "amount vs customer's usual amount (ratio)",
    "uid_secs_since_prev": "seconds since customer's previous payment",
    "new_device_for_uid": "first time this customer uses this device type",
    "new_email_for_uid": "first time this customer uses this email",
}

FAMILIES = [
    (r"^C\d+$", "a Vesta count field ({f})"),
    (r"^D\d+$", "a Vesta time-gap field, in days ({f})"),
    (r"^M\d$", "a Vesta match check, e.g. name vs card ({f})"),
    (r"^V\d+$", "a Vesta engineered risk signal ({f})"),
    (r"^id_\d+$", "a device or connection detail ({f})"),
]


def plain_name(feature):
    if feature in PLAIN:
        return PLAIN[feature]
    for suffix, template in [("_freq", "how common the {x} is"), ("_te", "past fraud rate of the {x}"),
                             ("_start", "account start date implied by {x}")]:
        if feature.endswith(suffix):
            base = feature[: -len(suffix)]
            words = {"card_type_and_product": "card type and product", "fingerprint": "device",
                     "DeviceInfo": "device model", "id_31": "browser version", "id_33": "screen size"}
            return template.format(x=words.get(base, plain_name(base) if base in PLAIN else base))
    if feature.startswith("uid_prior_mean_"):
        return f"customer's earlier average of {feature.removeprefix('uid_prior_mean_')}"
    for pattern, template in FAMILIES:
        if re.match(pattern, feature):
            return template.format(f=feature)
    return feature


def show_value(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "blank"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}"
    return str(value)


def plain_reason(feature, value):
    return f"{plain_name(feature)}: {show_value(value)}"


def points_from_woe(table, feature, woe):
    """Points a payment earns on one feature, given its Weight of Evidence (WoE).

    Inside one feature, points are a straight line in WoE (points = a + b x WoE, then
    rounded), because every bin shares the feature's coefficient. So:
      - a WoE listed in the scorecard table gets that bin's exact points;
      - any other WoE (e.g. a category first seen after training) gets the line's value.
    """
    rows = table[table["Variable"] == feature]
    known_woe, known_points = rows["WoE"].to_numpy(float), rows["Points"].to_numpy(float)
    woe = np.asarray(woe, dtype=float)
    nearest = np.abs(woe[:, None] - known_woe[None, :]).argmin(axis=1)
    exact = np.isclose(woe, known_woe[nearest], rtol=0, atol=1e-6)
    if np.unique(known_woe.round(6)).size >= 2:
        slope, intercept = np.polyfit(known_woe, known_points, 1)
        on_line = np.round(intercept + slope * woe)
    else:
        on_line = np.full(len(woe), known_points[0])
    return np.where(exact, known_points[nearest], on_line)


def scorecard_reasons(scorecard, X, features, top=3):
    """For each payment, the features where it lost most points vs that feature's best bin."""
    table = scorecard.table(style="detailed")
    best = table.groupby("Variable")["Points"].max()
    woe = scorecard.binning_process_.transform(X[features], metric="woe")
    lost = pd.DataFrame({f: best[f] - points_from_woe(table, f, woe[f]) for f in features}, index=X.index)
    out = []
    for i in range(len(lost)):
        row = lost.iloc[i].sort_values(ascending=False).head(top)
        out.append([(f, float(p)) for f, p in row.items() if p > 0])
    return out
