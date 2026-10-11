"""The SQL (run in DuckDB) and the pandas version must give identical features."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_master  # noqa: E402
from src.uid import add_uid  # noqa: E402
from src.velocity import compare, velocity_features, velocity_features_duckdb  # noqa: E402
from tests.fake_data import make  # noqa: E402

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    tx, ident = make(4000)
    tx.to_csv(raw / "train_transaction.csv", index=False)
    ident.to_csv(raw / "train_identity.csv", index=False)
    return add_uid(build_master(raw))


def test_sql_matches_pandas(master):
    result = compare(velocity_features(master), velocity_features_duckdb(master))
    assert result["match"].all(), result.to_string()


def test_features_only_use_the_past(master):
    feats = velocity_features(master).merge(master[["TransactionID", "uid_complete"]], on="TransactionID")
    known = feats[feats.uid_complete == 1]
    # A customer's first transaction has no history at all.
    first = known[known.uid_txn_number == 1]
    assert (first.uid_n_7d == 0).all()
    assert first.uid_amt_mean_prior.isna().all()
