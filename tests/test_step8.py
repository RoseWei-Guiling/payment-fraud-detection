"""Step 8: the scoring API and the case-note guard."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case_notes import is_grounded, template_note, write_note  # noqa: E402

FACTS = {"payment_id": 3500001, "amount": 250.0, "decision": "block", "probability_of_fraud_pct": 41.2,
         "customer_payments_last_24h": 4, "model_reasons": ["customer's payments in the last 24 hours: 4"]}


def test_note_with_invented_number_is_rejected():
    assert is_grounded("Payment 3500001 of 250.0 was blocked at 41.2% risk after 4 payments.", FACTS)
    assert not is_grounded("Payment 3500001 of 250.0 was blocked; the customer has 7 cards.", FACTS)


def test_llm_note_falls_back_to_template_when_it_invents():
    class FakeClient:
        class messages:
            @staticmethod
            def create(**_):
                class R:
                    content = [type("T", (), {"text": "Blocked: the customer made 12 payments today."})]
                return R
    note, source = write_note(FACTS, FakeClient(), "fake-model")
    assert source.startswith("template (LLM note rejected")
    assert note == template_note(FACTS)


def test_api_scores_match_the_model(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("fastapi")
    from src.scoring import Scorer, export_config

    rng = np.random.default_rng(0)
    n = 3000
    feats = pd.DataFrame({"TransactionAmt": rng.lognormal(4, 1, n), "uid_n_24h": rng.integers(0, 6, n).astype(float),
                          "ProductCD": pd.Categorical(rng.choice(["C", "H", "W"], n))})
    feats["uid_n_24h"] = feats["uid_n_24h"].astype("float32")
    y = (rng.random(n) < 0.05 + 0.1 * (feats["uid_n_24h"] > 3)).astype(int)
    model = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(feats, y), num_boost_round=20)
    model.save_model(str(tmp_path / "lightgbm.txt"))
    export_config(feats, list(feats.columns), tmp_path)
    (tmp_path / "cutoffs.json").write_text(json.dumps({"cost_per_blocked_genuine": 10}))

    scorer = Scorer(model_dir=tmp_path, cutoffs=tmp_path / "cutoffs.json")
    payments = [{"TransactionAmt": float(a), "uid_n_24h": float(u), "ProductCD": str(p)}
                for a, u, p in feats.head(50).itertuples(index=False)]
    results = scorer.score(payments)
    expected = model.predict(feats.head(50))
    assert np.allclose([r["probability_of_fraud"] for r in results], expected.round(4), atol=1e-4)
    for r in results:  # the decision is the Step 6 expected-cost rule
        assert r["decision"] == ("block" if r["expected_fraud_loss"] > r["expected_cost_of_blocking"] else "allow")

    blank = scorer.score([{"TransactionAmt": 100.0, "ProductCD": "never seen"}])[0]  # unknown + missing -> blank
    assert 0 <= blank["probability_of_fraud"] <= 1
