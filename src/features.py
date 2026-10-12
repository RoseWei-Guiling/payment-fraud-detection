"""
Step 3: feature engineering.

Every feature must be computable at the moment the transaction happens. Three rules:
  1. Customer history features only look at the customer's EARLIER transactions.
  2. Anything learned from labels (target encoding) is fitted on training months only.
     Training rows get out-of-fold values, so a row never sees its own label.
  3. Counts and frequencies are measured on training months only, then applied forward.

tests/test_features.py checks these rules automatically.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ID_COLS = ["TransactionID", "isFraud", "split", "day", "uid"]

V_COLS_PREFIX = "V"
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
ID_FIELD_COLS = [f"id_{i:02d}" for i in range(1, 39)]
HIGH_CARDINALITY_ID_COLS = ["id_30", "id_31", "id_33"]  # OS, browser, screen: grouped instead (see device_features)

FREQ_COLS = ["card1", "card2", "addr1", "P_emaildomain", "DeviceInfo", "id_31", "id_33", "fingerprint"]
TARGET_ENCODE_COLS = ["P_emaildomain", "R_emaildomain", "card_type_and_product", "device_family", "addr1"]
CUSTOMER_HISTORY_COLS = ["C1", "C13", "C14", "D15", "dist1"]


def v_columns(df):
    return [c for c in df.columns if c.startswith(V_COLS_PREFIX) and c[1:].isdigit()]


# ---------------------------------------------------------------------------
# 1. Features read straight off the transaction
# ---------------------------------------------------------------------------
def base_features(df):
    amount = df["TransactionAmt"].astype("float64")
    cents = np.round(amount - np.floor(amount), 3)
    derived = pd.DataFrame({
        "TransactionAmt": amount,
        "log_amt": np.log1p(amount),
        "amt_cents": cents,  # non-round cents often mean a currency conversion, i.e. a foreign card
        "amt_is_round": (cents == 0).astype("int8"),
        "hour": df["hour"],
        "weekday": df["weekday"],
        "has_identity": df["has_identity"],
        "uid_complete": df["uid_complete"],
        "account_age_days": df["account_age_days"],
    }, index=df.index)

    transaction_fields = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
                          "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain", "DeviceType"]
    identity_fields = [c for c in ID_FIELD_COLS if c not in HIGH_CARDINALITY_ID_COLS]
    raw = transaction_fields + C_COLS + D_COLS + M_COLS + identity_fields
    raw = [c for c in raw if c in df.columns]
    return pd.concat([derived, df[raw]], axis=1)


# ---------------------------------------------------------------------------
# 2. Missingness, email, device and time-normalised features
# ---------------------------------------------------------------------------
def first_word(values):
    """'Windows 10' -> 'windows', 'iOS 11.2' -> 'ios', 'SM-G930V Build/NRD90M' -> 'sm'."""
    return values.astype("string").str.lower().str.extract(r"^([a-z]+)", expand=False)


def missing_features(df):
    column_families = {
        "v": v_columns(df),
        "d": [c for c in D_COLS if c in df.columns],
        "m": [c for c in M_COLS if c in df.columns],
        "id": [c for c in ID_FIELD_COLS if c in df.columns],
    }
    out = {}
    for family, cols in column_families.items():
        out[f"n_missing_{family}"] = df[cols].isna().sum(axis=1).astype("int16")

    # M columns are True/False match flags (M4 holds codes instead, so it is left out).
    match_flags = [c for c in column_families["m"] if c != "M4"]
    out["m_true_count"] = (df[match_flags].astype("string") == "T").sum(axis=1).astype("int8")
    return pd.DataFrame(out, index=df.index)


def email_features(df):
    purchaser = df["P_emaildomain"].astype("string")
    recipient = df["R_emaildomain"].astype("string")
    both_known = purchaser.notna() & recipient.notna()
    return pd.DataFrame({
        "P_email_provider": purchaser.str.split(".").str[0].astype("category"),          # 'yahoo.co.uk' -> 'yahoo'
        "P_email_suffix": purchaser.str.split(".", n=1).str[1].astype("category"),       # 'yahoo.co.uk' -> 'co.uk'
        "R_email_provider": recipient.str.split(".").str[0].astype("category"),
        "email_P_equals_R": (purchaser == recipient).astype("float64").where(both_known),
        "P_email_missing": purchaser.isna().astype("int8"),
    }, index=df.index)


def device_fingerprint(df):
    """Device + OS + browser + screen resolution joined into one string.

    DeviceInfo on its own is coarse (e.g. just 'Windows'), so combining the four fields is
    much closer to identifying an actual device. Missing parts become '?'. A transaction
    with all four parts missing gets no fingerprint at all.
    """
    parts = [c for c in ["DeviceInfo", "id_30", "id_31", "id_33"] if c in df.columns]
    as_text = df[parts].astype("string")
    fingerprint = as_text.fillna("?").agg("|".join, axis=1)
    all_missing = as_text.isna().all(axis=1)
    return fingerprint.where(~all_missing)


def device_features(df):
    def family(col):
        return first_word(df[col]).astype("category") if col in df.columns else pd.NA

    return pd.DataFrame({
        "device_family": family("DeviceInfo"),
        "os_family": family("id_30"),
        "browser_family": family("id_31"),
    }, index=df.index)


def d_normalised(df):
    """D columns count 'days since' something, so they grow over time for the same customer.
    day - D turns them into a fixed start date, which stays stable across months."""
    out = {}
    for i in (4, 10, 11, 15):
        if f"D{i}" in df.columns:
            out[f"D{i}_start"] = (df["day"] - df[f"D{i}"]).astype("float32")
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# 3. Encodings learned on training months only
# ---------------------------------------------------------------------------
def frequency_encode(df, cols, train_mask):
    """How often each value appears in the training months (0 if never seen there)."""
    out = {}
    for col in cols:
        values = df[col].astype("string")
        counts_in_training = values[train_mask].value_counts()
        out[f"{col}_freq"] = values.map(counts_in_training).fillna(0).astype("float32")
    return pd.DataFrame(out, index=df.index)


def smoothed_fraud_rates(categories, labels, overall_rate, smoothing):
    """Fraud rate per category, pulled towards the overall rate when a category is small.

    rate = (frauds + overall_rate x smoothing) / (transactions + smoothing)
    A category with 5 transactions stays close to the overall rate; one with 50,000
    transactions gets (almost) its own observed rate.
    """
    stats = labels.groupby(categories.to_numpy()).agg(["sum", "count"])
    return (stats["sum"] + overall_rate * smoothing) / (stats["count"] + smoothing)


def customer_groups(df, rows):
    """Customer ID for each row; rows without a reliable ID become their own group."""
    groups = df["uid"].astype("string").to_numpy()[rows].copy()
    no_id = df["uid_complete"].to_numpy()[rows] == 0
    groups[no_id] = [f"row{i}" for i in rows[no_id]]
    return groups


def target_encode(df, cols, y, train_mask, smoothing=100, n_splits=5):
    """Replace a category with its smoothed fraud rate.

    - Validation and test months: rates learned from ALL training rows.
    - Training rows: out-of-fold. Training is split into 5 folds, and each fold gets rates
      learned from the other 4, so no row ever contributes to its own encoding.
    - Folds keep each customer's transactions together (GroupKFold). Fraud labels are shared
      across a customer's transactions, so with ordinary folds a row could still 'see' its
      own customer's labels through another fold.
    """
    overall_rate = y[train_mask].mean()
    train_rows = np.flatnonzero(train_mask.to_numpy())
    later_rows = np.flatnonzero(~train_mask.to_numpy())
    folds = list(GroupKFold(n_splits).split(train_rows, groups=customer_groups(df, train_rows)))

    out = {}
    for col in cols:
        category = df[col].astype("string").fillna("__missing__")
        encoded = pd.Series(np.nan, index=df.index, dtype="float64")

        for fit_positions, apply_positions in folds:
            fit_rows, apply_rows = train_rows[fit_positions], train_rows[apply_positions]
            rates = smoothed_fraud_rates(category.iloc[fit_rows], y.iloc[fit_rows], overall_rate, smoothing)
            encoded.iloc[apply_rows] = category.iloc[apply_rows].map(rates).fillna(overall_rate).to_numpy()

        rates = smoothed_fraud_rates(category.iloc[train_rows], y.iloc[train_rows], overall_rate, smoothing)
        encoded.iloc[later_rows] = category.iloc[later_rows].map(rates).fillna(overall_rate).to_numpy()
        out[f"{col}_te"] = encoded.astype("float32")
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# 4. Customer history (earlier transactions only)
# ---------------------------------------------------------------------------
def in_time_order_per_customer(df):
    return df.sort_values(["uid", "TransactionDT", "TransactionID"]).index


def customer_prior_means(df, cols):
    """Customer's average of each column over their EARLIER transactions (blanks ignored)."""
    order = in_time_order_per_customer(df)
    customer = df.loc[order, "uid"].astype("string").to_numpy()
    out = {}
    for col in cols:
        values = df.loc[order, col].astype("float64")
        value_or_zero = values.fillna(0.0)
        is_present = values.notna().astype("float64")
        # running total up to and including this row, minus this row = total of earlier rows
        earlier_sum = value_or_zero.groupby(customer).cumsum() - value_or_zero
        earlier_count = is_present.groupby(customer).cumsum() - is_present
        out[f"uid_prior_mean_{col}"] = (earlier_sum / earlier_count).where(earlier_count > 0)

    result = pd.DataFrame(out, index=order).reindex(df.index)
    result.loc[df["uid_complete"] == 0] = np.nan
    return result.astype("float32")


def fingerprint_sharing(df, fingerprint, train_mask):
    """Two device features:
    - uids_per_fingerprint: how many different customers used this fingerprint in the
      training months (0 = never seen). Many customers on one device can mean a fraud ring.
    - new_fingerprint_for_uid: 1 if a returning customer uses this fingerprint for the first time.
    """
    usable = train_mask & (df["uid_complete"] == 1) & fingerprint.notna()
    pairs = pd.DataFrame({"fingerprint": fingerprint[usable], "uid": df.loc[usable, "uid"].astype("string")})
    customers_per_fingerprint = pairs.drop_duplicates().groupby("fingerprint").size()
    shared = fingerprint.map(customers_per_fingerprint).astype("float32")
    shared[fingerprint.notna() & shared.isna()] = 0

    order = in_time_order_per_customer(df)
    history = pd.DataFrame({"uid": df.loc[order, "uid"].astype("string"), "fingerprint": fingerprint.loc[order]},
                           index=order)
    n_earlier = history.groupby("uid").cumcount()
    first_time_seen = history.groupby(["uid", "fingerprint"], dropna=False).cumcount() == 0
    has_history_and_value = history["fingerprint"].notna() & (n_earlier > 0)
    new_fingerprint = first_time_seen.astype("float32").where(has_history_and_value).reindex(df.index)
    new_fingerprint[df["uid_complete"] == 0] = np.nan

    return pd.DataFrame({"uids_per_fingerprint": shared, "new_fingerprint_for_uid": new_fingerprint},
                        index=df.index)


# ---------------------------------------------------------------------------
# 5. V-column reduction
# ---------------------------------------------------------------------------
def reduce_v(df, train_mask, threshold=0.75, sample_size=200_000, seed=0):
    """Remove near-duplicate V columns.

    1. V columns come in blocks that are missing on exactly the same rows, so group them
       by their missing-value count in the training months.
    2. Within each block, walk through the columns in order and keep a column only if it
       is not highly correlated (|r| >= threshold) with a column already kept.
    Returns (kept columns, total V columns, number of blocks).
    """
    all_v = v_columns(df)
    train = df.loc[train_mask, all_v]
    if len(train) > sample_size:
        train = train.sample(sample_size, random_state=seed)
    varying = [c for c in all_v if train[c].nunique(dropna=True) > 1]

    blocks = {}
    for col in varying:
        blocks.setdefault(int(train[col].isna().sum()), []).append(col)

    kept = []
    for cols in blocks.values():
        corr = train[cols].astype("float64").corr().abs()
        chosen = []
        for col in cols:
            too_similar = any(corr.loc[col, k] >= threshold for k in chosen)
            if not too_similar:
                chosen.append(col)
        kept += chosen
    return sorted(kept, key=lambda c: int(c[1:])), len(all_v), len(blocks)


# ---------------------------------------------------------------------------
# 6. Information value (training months only)
# ---------------------------------------------------------------------------
def bin_codes(x, bins=10, min_share=0.01):
    """Turn a column into bin numbers: deciles for numbers, levels for categories.
    Rare levels (< 1%) are grouped together, and missing values get their own bin."""
    is_categorical = (isinstance(x.dtype, pd.CategoricalDtype) or x.dtype == object
                      or pd.api.types.is_string_dtype(x))
    if is_categorical:
        values = x.astype("string")
        share = values.value_counts(normalize=True)
        common = values.isin(share[share >= min_share].index) | values.isna()
        values = values.where(common, "rare").fillna("missing")
        return pd.factorize(values)[0]

    values = x.to_numpy(dtype="float64", na_value=np.nan)
    missing = np.isnan(values)
    if missing.all():
        return np.zeros(len(values), dtype="int64")
    edges = np.unique(np.quantile(values[~missing], np.linspace(0, 1, bins + 1)[1:-1]))
    codes = np.searchsorted(edges, values, side="right")
    codes[missing] = len(edges) + 1
    return codes


def information_value(x, y, bins=10, min_share=0.01):
    """IV = sum over bins of (share of frauds - share of genuine) x ln(share of frauds / share of genuine).
    0.5 is added to every bin's count so empty bins don't divide by zero."""
    y = np.asarray(y, dtype="int64")
    codes = bin_codes(x, bins, min_share)
    transactions = np.bincount(codes)
    frauds = np.bincount(codes, weights=y, minlength=len(transactions))
    genuine = transactions - frauds
    if frauds.sum() == 0 or genuine.sum() == 0:
        return 0.0
    used = transactions > 0
    fraud_share = (frauds[used] + 0.5) / (frauds.sum() + 0.5)
    genuine_share = (genuine[used] + 0.5) / (genuine.sum() + 0.5)
    return float(((fraud_share - genuine_share) * np.log(fraud_share / genuine_share)).sum())


def iv_table(feats, groups):
    train = feats[feats["split"] == "train"]
    y = train["isFraud"].astype(int)
    rows = []
    for group, cols in groups.items():
        for feature in cols:
            rows.append({"feature": feature, "group": group, "iv": information_value(train[feature], y)})
    table = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    table["strength"] = pd.cut(table["iv"], [-1, 0.02, 0.1, 0.3, 0.5, np.inf],
                               labels=["useless", "weak", "medium", "strong", "check for leakage"])
    return table


# ---------------------------------------------------------------------------
# Build everything
# ---------------------------------------------------------------------------
def card_type_and_product(df):
    """e.g. 'visa|debit|W': card network, debit/credit and product code in one category."""
    network = df["card4"].astype("string").fillna("?")
    debit_or_credit = df["card6"].astype("string").fillna("?")
    product = df["ProductCD"].astype("string")
    return network + "|" + debit_or_credit + "|" + product


def build_features(master, velocity):
    df = master.reset_index(drop=True)
    train_mask = df["split"] == "train"
    y = df["isFraud"].astype("float64")

    device = device_features(df)
    df["fingerprint"] = device_fingerprint(df)
    df["card_type_and_product"] = card_type_and_product(df)
    df["device_family"] = device["device_family"]
    v_kept, v_total, v_blocks = reduce_v(df, train_mask)

    velocity_aligned = df[["TransactionID"]].merge(velocity, on="TransactionID", how="left")
    velocity_aligned = velocity_aligned.drop(columns="TransactionID").set_index(df.index)

    parts = {
        "base": base_features(df),
        "missingness": missing_features(df),
        "email": email_features(df),
        "device": device,
        "d_normalised": d_normalised(df),
        "frequency": frequency_encode(df, FREQ_COLS, train_mask),
        "target_encoding": target_encode(df, TARGET_ENCODE_COLS, y, train_mask),
        "customer_history": customer_prior_means(df, CUSTOMER_HISTORY_COLS),
        "device_sharing": fingerprint_sharing(df, df["fingerprint"], train_mask),
        "vesta_v_reduced": df[v_kept],
        "velocity": velocity_aligned,
    }
    features = pd.concat([df[ID_COLS]] + list(parts.values()), axis=1)
    meta = {
        "groups": {name: list(part.columns) for name, part in parts.items()},
        "v_total": v_total,
        "v_groups": v_blocks,
        "v_kept": len(v_kept),
    }
    return features, meta
