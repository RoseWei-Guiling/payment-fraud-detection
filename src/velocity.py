"""
Customer velocity features in pandas: the twin of sql/velocity_features.sql.

Both versions must give identical numbers (checked by tests/test_velocity.py and
scripts/check_sql_vs_pandas.py). Only earlier transactions are used for each row.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "velocity_features.sql"
COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "uid", "uid_complete", "DeviceInfo", "P_emaildomain"]
WINDOWS = {"1h": 3600, "24h": 86400, "7d": 604800}
FEATURES = ["uid_txn_number", "uid_n_1h", "uid_n_24h", "uid_n_7d", "uid_amt_24h", "uid_amt_mean_prior",
            "amt_to_uid_mean", "uid_secs_since_prev", "new_device_for_uid", "new_email_for_uid"]


def slim(df):
    """The columns the features need, as plain types (what gets uploaded to BigQuery)."""
    out = df[COLUMNS].copy()
    for c in ["uid", "DeviceInfo", "P_emaildomain"]:
        out[c] = out[c].astype("string")
    out["TransactionDT"] = out["TransactionDT"].astype("int64")
    out["TransactionAmt"] = out["TransactionAmt"].astype("float64")
    out["uid_complete"] = out["uid_complete"].astype("int64")
    return out


def velocity_features(df):
    d = slim(df).sort_values(["uid", "TransactionDT", "TransactionID"]).reset_index(drop=True)
    t = d["TransactionDT"].to_numpy()
    amt = d["TransactionAmt"].to_numpy()
    uid_code = pd.factorize(d["uid"])[0].astype("int64")

    # Rows are sorted by (customer, time), so one combined key lets us find every window
    # with a binary search: count = rows in [t - window, t) for the same customer.
    key = uid_code * 10**9 + t
    start = np.searchsorted(key, key, side="left")  # first row at this exact time
    cum_amt = np.concatenate([[0.0], np.cumsum(amt)])
    out = pd.DataFrame({"TransactionID": d["TransactionID"]})
    for name, secs in WINDOWS.items():
        lo = np.searchsorted(key, key - secs, side="left")
        out[f"uid_n_{name}"] = start - lo
        if name == "24h":
            out["uid_amt_24h"] = cum_amt[start] - cum_amt[lo]

    g = d.groupby("uid", sort=False)
    n_prior = g.cumcount()
    out["uid_txn_number"] = n_prior + 1
    prior_sum = g["TransactionAmt"].cumsum() - d["TransactionAmt"]
    out["uid_amt_mean_prior"] = (prior_sum / n_prior).where(n_prior > 0)
    out["amt_to_uid_mean"] = d["TransactionAmt"] / out["uid_amt_mean_prior"].replace(0, np.nan)
    out["uid_secs_since_prev"] = g["TransactionDT"].diff()

    for col, feat in [("DeviceInfo", "new_device_for_uid"), ("P_emaildomain", "new_email_for_uid")]:
        first_seen = (d.groupby(["uid", col], sort=False).cumcount() == 0).astype("float64")
        out[feat] = first_seen.where(d[col].notna() & (n_prior > 0))

    unknown = d["uid_complete"].to_numpy() == 0
    out.loc[unknown, FEATURES] = np.nan
    return out[["TransactionID"] + FEATURES].sort_values("TransactionID").reset_index(drop=True)


def sql_text(table):
    return SQL_PATH.read_text().replace("{table}", table)


def velocity_features_duckdb(df):
    """Run the same SQL locally with DuckDB (used by the tests, no cloud needed)."""
    import duckdb
    con = duckdb.connect()
    con.register("transactions", slim(df))
    return con.execute(sql_text("transactions")).df()


def compare(a, b, tol=1e-6):
    """Max absolute difference per feature between two feature tables (0 = identical)."""
    m = a.merge(b, on="TransactionID", suffixes=("_a", "_b"), validate="one_to_one")
    rows = []
    for f in FEATURES:
        x, y = m[f"{f}_a"].astype("float64"), m[f"{f}_b"].astype("float64")
        both_nan = x.isna() & y.isna()
        diff = (x - y).abs().where(~both_nan, 0.0)
        mismatched_nans = int((x.isna() != y.isna()).sum())
        rows.append({"feature": f, "max_abs_diff": float(diff.max()), "nan_mismatches": mismatched_nans,
                     "match": bool(diff.max() <= tol and mismatched_nans == 0)})
    return pd.DataFrame(rows)
