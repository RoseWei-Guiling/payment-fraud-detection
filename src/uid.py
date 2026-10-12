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
    """Adds uid = 'card1_addr1_startday', e.g. '13926_315_-12', plus account age columns."""
    start_day = np.floor(df["day"] - df["D1"])
    card = df["card1"].astype("string")
    address = df["addr1"].astype("Int64").astype("string")
    start = start_day.astype("Int64").astype("string")
    uid = card + "_" + address + "_" + start  # missing if address or start day is missing

    new = pd.DataFrame({
        "account_start_day": start_day.astype("float32"),
        "account_age_days": df["D1"].astype("float32"),
        # Missing addr1 or D1 means we can't identify the customer reliably.
        "uid_complete": (df["addr1"].notna() & df["D1"].notna()).astype("int8"),
        "uid": uid.fillna("unknown").astype("category"),
    }, index=df.index)
    return pd.concat([df, new], axis=1)


def pct(numerator, denominator):
    return f"{numerator / denominator:.1%}" if denominator else "n/a"


def uid_check(df):
    """Sanity checks that the rebuilt ID behaves like a real customer ID.

    Vesta labelled a transaction as fraud after a chargeback, and then also labelled
    later transactions linked to that customer. So if uid really identifies customers,
    transactions AFTER a customer's first fraud should be fraud far more often than
    the 3.5% baseline.
    """
    known = df.loc[df["uid_complete"] == 1, ["uid", "TransactionDT", "isFraud"]].copy()
    known["uid"] = known["uid"].astype(str)

    per_customer = known.groupby("uid").agg(n=("isFraud", "size"), fraud_share=("isFraud", "mean"))
    repeat_customers = per_customer[per_customer["n"] > 1]
    repeat_fraud_customers = repeat_customers[repeat_customers["fraud_share"] > 0]

    first_fraud_time = known[known["isFraud"] == 1].groupby("uid")["TransactionDT"].min()
    fraud_customer_rows = known.join(first_fraud_time.rename("first_fraud_dt"), on="uid", how="inner")
    later = fraud_customer_rows["TransactionDT"] > fraud_customer_rows["first_fraud_dt"]
    earlier = fraud_customer_rows["TransactionDT"] < fraud_customer_rows["first_fraud_dt"]
    after_first_fraud = fraud_customer_rows[later]
    had_genuine_history = earlier.groupby(fraud_customer_rows["uid"]).any()

    return {
        "transactions_with_complete_uid": pct(len(known), len(df)),
        "distinct_customers": len(per_customer),
        "median_transactions_per_customer": float(per_customer["n"].median()),
        "customers_with_2plus_transactions": pct(len(repeat_customers), len(per_customer)),
        "fraud_customers_that_are_all_fraud": pct((repeat_fraud_customers["fraud_share"] == 1).sum(),
                                                  len(repeat_fraud_customers)),
        "baseline_fraud_rate": f"{df['isFraud'].mean():.1%}",
        # The key check: should be far above the baseline if uid works.
        "fraud_rate_after_customers_first_fraud": pct(after_first_fraud["isFraud"].sum(), len(after_first_fraud)),
        # Fraud customers who looked genuine first: these are the hard cases for rules.
        "fraud_customers_with_genuine_history": pct(had_genuine_history.sum(), len(had_genuine_history)),
    }
