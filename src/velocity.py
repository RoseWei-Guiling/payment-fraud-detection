"""
Customer velocity features in pandas: the twin of sql/velocity_features.sql.

Both versions must give identical numbers. tests/test_velocity.py checks this on fake
data, and scripts/build_velocity.py checks it on the real data. For every transaction,
only the customer's EARLIER transactions are used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "velocity_features.sql"

COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "uid", "uid_complete",
           "DeviceInfo", "P_emaildomain"]

WINDOW_SECONDS = {"1h": 3600, "24h": 86400, "7d": 604800}

FEATURES = ["uid_txn_number", "uid_n_1h", "uid_n_24h", "uid_n_7d", "uid_amt_24h",
            "uid_amt_mean_prior", "amt_to_uid_mean", "uid_secs_since_prev",
            "new_device_for_uid", "new_email_for_uid"]


def slim(df):
    """Keep only the columns the features need, as plain types (this is what goes to BigQuery)."""
    out = df[COLUMNS].copy()
    for col in ["uid", "DeviceInfo", "P_emaildomain"]:
        out[col] = out[col].astype("string")
    out["TransactionDT"] = out["TransactionDT"].astype("int64")
    out["TransactionAmt"] = out["TransactionAmt"].astype("float64")
    out["uid_complete"] = out["uid_complete"].astype("int64")
    return out


def count_and_sum_in_window(customer, seconds, amount, window):
    """For each transaction: how many of the SAME customer's transactions happened in the
    `window` seconds before it, and how much they added up to.

    The window is [t - window, t): earlier transactions only, matching the SQL
    `RANGE BETWEEN window PRECEDING AND 1 PRECEDING`. Transactions at exactly the same
    second as the current one are not counted.

    How it works: rows are sorted by customer then time. Within one customer, the rows in
    the window form one continuous block, so we only need the block's first and last row.
    np.searchsorted finds both by binary search instead of looping over every pair.
    To stop a window spilling into the previous customer, we search on one combined key
    (customer code x 10^9 + seconds): the data spans about 16 million seconds, far less
    than 10^9, so two customers' keys never overlap.

    Example for one customer with transactions at t = 0, 100, 5000 and a 1-hour window:
        t = 5000 -> window [1400, 5000) -> no earlier rows inside -> count 0
        t = 100  -> window [-3500, 100) -> the row at t = 0      -> count 1
    """
    key = customer * 10**9 + seconds
    window_start = np.searchsorted(key, key - window, side="left")  # first row inside the window
    window_end = np.searchsorted(key, key, side="left")             # first row at the current second
    count = window_end - window_start

    running_total = np.concatenate([[0.0], np.cumsum(amount)])
    total = running_total[window_end] - running_total[window_start]
    return count, total


def velocity_features(df):
    d = slim(df).sort_values(["uid", "TransactionDT", "TransactionID"]).reset_index(drop=True)
    customer = pd.factorize(d["uid"])[0].astype("int64")
    seconds = d["TransactionDT"].to_numpy()
    amount = d["TransactionAmt"].to_numpy()
    out = pd.DataFrame({"TransactionID": d["TransactionID"]})

    # How busy was the customer just before this transaction?
    for name, window in WINDOW_SECONDS.items():
        count, total = count_and_sum_in_window(customer, seconds, amount, window)
        out[f"uid_n_{name}"] = count
        if name == "24h":
            out["uid_amt_24h"] = total

    # How does this transaction compare with the customer's history?
    by_customer = d.groupby("uid", sort=False)
    n_earlier = by_customer.cumcount()  # 0 for the customer's first transaction
    out["uid_txn_number"] = n_earlier + 1

    earlier_total = by_customer["TransactionAmt"].cumsum() - d["TransactionAmt"]
    out["uid_amt_mean_prior"] = (earlier_total / n_earlier).where(n_earlier > 0)
    out["amt_to_uid_mean"] = d["TransactionAmt"] / out["uid_amt_mean_prior"].replace(0, np.nan)
    out["uid_secs_since_prev"] = by_customer["TransactionDT"].diff()

    # Is the device / email new for a customer we have seen before?
    for col, feature in [("DeviceInfo", "new_device_for_uid"), ("P_emaildomain", "new_email_for_uid")]:
        first_time_seen = d.groupby(["uid", col], sort=False).cumcount() == 0
        has_history_and_value = d[col].notna() & (n_earlier > 0)
        out[feature] = first_time_seen.astype("float64").where(has_history_and_value)

    # Without a reliable customer ID these features would be meaningless.
    out.loc[d["uid_complete"].to_numpy() == 0, FEATURES] = np.nan
    return out[["TransactionID"] + FEATURES].sort_values("TransactionID").reset_index(drop=True)


def sql_text(table):
    return SQL_PATH.read_text().replace("{table}", table)


def velocity_features_duckdb(df):
    """Run the SQL file locally with DuckDB (used by the tests and build_velocity.py)."""
    import duckdb
    con = duckdb.connect()
    con.register("transactions", slim(df))
    return con.execute(sql_text("transactions")).df()


def compare(a, b, tol=1e-6):
    """Largest absolute difference per feature between two feature tables (0 means identical)."""
    merged = a.merge(b, on="TransactionID", suffixes=("_a", "_b"), validate="one_to_one")
    rows = []
    for feature in FEATURES:
        x = merged[f"{feature}_a"].astype("float64")
        y = merged[f"{feature}_b"].astype("float64")
        both_missing = x.isna() & y.isna()
        difference = (x - y).abs().where(~both_missing, 0.0)
        missing_in_one_only = int((x.isna() != y.isna()).sum())
        rows.append({
            "feature": feature,
            "max_abs_diff": float(difference.max()),
            "nan_mismatches": missing_in_one_only,
            "match": bool(difference.max() <= tol and missing_in_one_only == 0),
        })
    return pd.DataFrame(rows)
