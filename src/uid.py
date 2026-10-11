"""
Rebuild a customer identifier.

IEEE-CIS has no customer ID. The competition winners showed that card1 + addr1 +
(day - D1) picks out individual customers, because D1 behaves like "days since the
card relationship started", so day - D1 is roughly the start date, which stays
constant for one customer. Credit: Chris Deotte and Konstantin Yakovlev, 1st place,
https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
"""
import numpy as np
import pandas as pd


def add_uid(df):
    start_day = np.floor(df["day"] - df["D1"])
    parts = [df["card1"].astype("string"), df["addr1"].astype("Int64").astype("string"),
             start_day.astype("Int64").astype("string")]
    uid = parts[0] + "_" + parts[1] + "_" + parts[2]
    new = pd.DataFrame({
        "account_start_day": start_day.astype("float32"),
        "account_age_days": df["D1"].astype("float32"),
        # Missing addr1 or D1 means we can't identify the customer reliably.
        "uid_complete": (df["addr1"].notna() & df["D1"].notna()).astype("int8"),
        "uid": uid.fillna("unknown").astype("category"),
    }, index=df.index)
    return pd.concat([df, new], axis=1)


def uid_check(df):
    """Sanity checks that the rebuilt ID behaves like a real customer ID."""
    known = df[df["uid_complete"] == 1]
    per_uid = known.groupby("uid", observed=True).agg(n=("isFraud", "size"), fraud=("isFraud", "mean"))
    multi = per_uid[per_uid["n"] > 1]
    any_fraud = multi[multi["fraud"] > 0]
    return {
        "transactions_with_complete_uid": f"{len(known) / len(df):.1%}",
        "distinct_customers": len(per_uid),
        "median_transactions_per_customer": float(per_uid["n"].median()),
        "customers_with_2plus_transactions": f"{len(multi) / len(per_uid):.1%}",
        # If the ID is real, fraud should cluster: customers with any fraud are mostly all-fraud.
        "fraud_customers_that_are_all_fraud": f"{(any_fraud['fraud'] == 1).mean():.1%}" if len(any_fraud) else "n/a",
    }
