"""Run with:  pytest -q   (uses fake data, no download needed)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_master  # noqa: E402
from src.uid import add_uid, uid_check  # noqa: E402
from tests.fake_data import make  # noqa: E402


def _master(tmp_path):
    tx, ident = make(3000)
    tx.to_csv(tmp_path / "train_transaction.csv", index=False)
    ident.to_csv(tmp_path / "train_identity.csv", index=False)
    return add_uid(build_master(tmp_path)), len(tx)


def test_merge_keeps_every_transaction(tmp_path):
    df, n = _master(tmp_path)
    assert len(df) == n
    assert df["TransactionID"].is_unique


def test_time_split_is_ordered(tmp_path):
    df, _ = _master(tmp_path)
    days = df.groupby("split", observed=True)["day"].agg(["min", "max"])
    assert days.loc["train", "max"] < days.loc["valid", "min"]
    assert days.loc["valid", "max"] < days.loc["oot", "min"]


def test_uid_groups_customers(tmp_path):
    df, _ = _master(tmp_path)
    stats = uid_check(df)
    assert stats["median_transactions_per_customer"] >= 2
    assert (df.loc[df["uid_complete"] == 0, "uid"] == "unknown").all()
