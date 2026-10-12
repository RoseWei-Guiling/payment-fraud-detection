"""Leakage tests: features must not change when later labels or later transactions change."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_master  # noqa: E402
from src.features import ID_COLS, build_features, information_value  # noqa: E402
from src.uid import add_uid  # noqa: E402
from src.velocity import velocity_features  # noqa: E402
from tests.fake_data import make  # noqa: E402


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    tx, ident = make(4000)
    tx.to_csv(raw / "train_transaction.csv", index=False)
    ident.to_csv(raw / "train_identity.csv", index=False)
    return add_uid(build_master(raw))


def _build(m):
    feats, meta = build_features(m, velocity_features(m))
    return feats.sort_values("TransactionID").reset_index(drop=True), meta


def _same(a, b):
    cols = [c for c in a.columns if c not in ID_COLS]
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x):
            assert np.allclose(x.astype(float), y.astype(float), equal_nan=True), c
        else:
            assert x.astype("string").fillna("NA").equals(y.astype("string").fillna("NA")), c


def test_later_labels_do_not_change_features(master):
    """Flip every label outside the training months: no feature may move."""
    before, _ = _build(master)
    flipped = master.copy()
    later = flipped["split"] != "train"
    flipped.loc[later, "isFraud"] = 1 - flipped.loc[later, "isFraud"]
    after, _ = _build(flipped)
    _same(before, after)


def test_customer_history_ignores_the_future(master):
    """Drop the last 30 days: features of earlier transactions must be unchanged."""
    full, _ = _build(master)
    cut = master["day"].max() - 30
    early_master = master[master["day"] <= cut]
    early, _ = _build(early_master)
    hist = [c for c in full.columns if c.startswith(("uid_prior_mean_", "uid_n_", "uid_amt_", "uid_txn"))]
    ids = early["TransactionID"]
    a = full.set_index("TransactionID").loc[ids, hist].reset_index(drop=True)
    b = early[hist].reset_index(drop=True)
    assert np.allclose(a.astype(float), b.astype(float), equal_nan=True)


def test_one_row_per_transaction(master):
    feats, meta = _build(master)
    assert len(feats) == len(master) and feats["TransactionID"].is_unique
    assert meta["v_kept"] <= meta["v_total"]


def test_information_value_finds_signal():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.random(5000) < 0.1).astype(int)
    strong = pd.Series(y * 2 + rng.normal(0, 1, 5000))
    noise = pd.Series(rng.normal(0, 1, 5000))
    assert information_value(strong, y) > 0.3 > information_value(noise, y)


def test_target_encoding_keeps_customers_in_one_fold():
    """Worst case: every customer has its own category. With customer-grouped folds a row can
    never see its own customer's labels, so the out-of-fold encoding carries no signal at all."""
    from src.features import target_encode
    rng = np.random.default_rng(0)
    n_cust = 2000
    cust = np.repeat(np.arange(n_cust), 5)
    fraud = (rng.random(n_cust) < 0.05)[cust].astype(int)
    df = pd.DataFrame({"uid": [f"u{c}" for c in cust], "uid_complete": 1, "key": [f"k{c}" for c in cust]})
    train = pd.Series(True, index=df.index)
    enc = target_encode(df, ["key"], pd.Series(fraud, dtype="float64"), train)["key_te"]
    assert enc.nunique() == 1  # every row falls back to the overall fraud rate
    assert information_value(enc, pd.Series(fraud)) < 0.01
