"""Load the IEEE-CIS files, merge them into one master table, and add time columns."""
from pathlib import Path

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 60 * 60

# Time split by day number. TransactionDT is seconds from an unpublished reference
# point; the training data covers about 182 days. Never split fraud data at random:
# patterns drift over time, so a random split overstates performance.
TRAIN_END_DAY = 120   # days   1-119: development (about 4 months)
VALID_END_DAY = 150   # days 120-149: validation / tuning (month 5)
#                       days 150+   : out-of-time test (month 6), touched once at the end


def load_raw(raw_dir="data/raw"):
    raw = Path(raw_dir)
    tx = pd.read_csv(raw / "train_transaction.csv")
    ident = pd.read_csv(raw / "train_identity.csv")
    return tx, ident


def merge(tx, ident):
    """Left join: every transaction kept, identity fields where they exist."""
    df = tx.merge(ident, on="TransactionID", how="left", validate="one_to_one")
    df.insert(1, "has_identity", df["TransactionID"].isin(ident["TransactionID"]).astype("int8"))
    return df


def reduce_memory(df):
    """Downcast numbers and turn text columns into categories (cuts memory by ~70%)."""
    def shrink(s):
        if pd.api.types.is_float_dtype(s):
            return pd.to_numeric(s, downcast="float")
        if pd.api.types.is_integer_dtype(s):
            return pd.to_numeric(s, downcast="integer")
        if s.dtype == object:
            return s.astype("category")
        return s

    return pd.DataFrame({col: shrink(df[col]) for col in df.columns}, index=df.index)


def add_time_columns(df):
    day = (df["TransactionDT"] // SECONDS_PER_DAY).astype("int16")
    new = pd.DataFrame({
        "day": day,
        "hour": ((df["TransactionDT"] // 3600) % 24).astype("int8"),
        "weekday": (day % 7).astype("int8"),  # relative weekday: the real start date is unknown
        "month": np.minimum(day // 30, 5).astype("int8"),  # 0-5, the last block is slightly longer
        "split": pd.Categorical(
            np.select([day < TRAIN_END_DAY, day < VALID_END_DAY], ["train", "valid"], default="oot"),
            categories=["train", "valid", "oot"],
        ),
    }, index=df.index)
    return pd.concat([df, new], axis=1)


def build_master(raw_dir="data/raw"):
    tx, ident = load_raw(raw_dir)
    # Shrink each table before merging, to keep peak memory down (the real files are ~1.3 GB as CSV).
    tx, ident = reduce_memory(tx), reduce_memory(ident)
    df = merge(tx, ident)
    return add_time_columns(df)
