"""The comparison maths every Step 5 result depends on (no model training needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import at_decline_rate, decline_rate_to_match, interpretable_features, text_to_categories  # noqa: E402


def test_at_decline_rate_blocks_the_highest_scores():
    score = [0.9, 0.8, 0.1, 0.2, 0.3, 0.05, 0.6, 0.4, 0.7, 0.0]
    fraud = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    amount = [50, 1, 1, 1, 1, 1, 150, 1, 1, 1]
    m = at_decline_rate(score, fraud, amount, decline_rate=0.2)  # blocks the top 2: 0.9 and 0.8
    assert m["decline_rate"] == 0.2
    assert m["fraud_caught"] == 0.5        # caught the 0.9 fraud, missed the 0.6 one
    assert m["fraud_value_caught"] == 0.25  # 50 of 200
    assert m["precision"] == 0.5


def test_decline_rate_to_match():
    score = np.arange(10)[::-1]           # payment 0 scores highest
    fraud = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    assert decline_rate_to_match(score, fraud, 2 / 3) == 0.3  # top 3 contain 2 of the 3 frauds


def test_scorecard_only_gets_interpretable_features():
    iv = pd.DataFrame({"feature": ["uid_n_24h", "V218", "C5", "TransactionAmt", "P_emaildomain_te", "weak_one"],
                       "iv": [0.5, 0.6, 0.5, 0.1, 0.3, 0.001],
                       "group": ["velocity", "vesta_v_reduced", "base", "base", "target_encoding", "velocity"]})
    groups = iv.groupby("group")["feature"].apply(list).to_dict()
    chosen = interpretable_features(groups, iv)
    assert chosen == ["uid_n_24h", "P_emaildomain_te", "TransactionAmt"]  # no masked V / C, no useless IV


def test_text_becomes_one_shared_category():
    df = pd.DataFrame({"email": ["a", "b", None, "a"], "amt": [1.0, 2.0, 3.0, 4.0]})
    df = text_to_categories(df, ["email", "amt"])
    assert isinstance(df["email"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_float_dtype(df["amt"])


def test_scorecard_drops_a_feature_with_a_flipped_sign():
    pytest = __import__("pytest")
    pytest.importorskip("optbinning")
    from src.models import train_scorecard
    rng = np.random.default_rng(0)
    n = 30_000
    z = rng.normal(size=n)
    a = z + rng.normal(scale=0.3, size=n)
    b = z + rng.normal(scale=0.3, size=n)  # on its own, high b looks risky (it moves with z)...
    logit = -3 + 2.0 * a - 1.5 * b          # ...but once a is known, high b is SAFER: a flipped sign
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    sc, _, kept, dropped = train_scorecard(pd.DataFrame({"a": a, "b": b}), y, ["a", "b"])
    assert dropped == ["b"] and kept == ["a"]
    assert all(c < 0 for c in sc.estimator_.coef_[0])  # every kept feature points the right way
