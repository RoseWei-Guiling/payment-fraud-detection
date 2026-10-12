"""Rules: conditions evaluate safely, metrics are right, the optimiser respects its limits."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rules import fire, fire_all, load_rules, metrics, rule_columns, select_rules  # noqa: E402


def test_blank_values_never_fire():
    df = pd.DataFrame({"uid_n_24h": [5, np.nan, 1], "email_P_equals_R": [0, np.nan, 1]})
    assert fire(df, "uid_n_24h >= 3").tolist() == [True, False, False]
    assert fire(df, "email_P_equals_R == 0").tolist() == [True, False, False]


def test_metrics_by_hand():
    flagged = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    fraud = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    amount = [100, 1, 1, 300, 1, 1, 1, 1, 1, 1]
    m = metrics(flagged, fraud, amount)
    assert m["decline_rate"] == 0.3
    assert m["fraud_caught"] == 0.5          # 1 of 2 frauds
    assert m["fraud_value_caught"] == 0.25   # 100 of 400
    assert abs(m["precision"] - 1 / 3) < 1e-9
    assert m["false_positive_ratio"] == 2.0  # 2 genuine blocked per fraud caught


def test_optimiser_respects_precision_and_budget():
    rng = np.random.default_rng(0)
    n = 10_000
    fraud = rng.random(n) < 0.03
    fired = pd.DataFrame({
        "good": fraud & (rng.random(n) < 0.5) | (rng.random(n) < 0.005),  # mostly fraud
        "useless": rng.random(n) < 0.2,                                    # random, huge
    })
    chosen, log = select_rules(fired, fraud, min_precision=0.10, max_decline_rate=0.05)
    assert chosen == ["good"]
    assert fired[chosen].any(axis=1).mean() <= 0.05


def test_all_project_rules_parse_on_the_columns_they_name():
    rules = load_rules()
    cols = rule_columns(rules)
    df = pd.DataFrame({c: [0.0, 1.0, np.nan] for c in cols})
    fired = fire_all(df, rules)
    assert list(fired.columns) == [r["name"] for r in rules]


def test_optimiser_skips_a_rule_that_is_too_big_and_keeps_going():
    rng = np.random.default_rng(1)
    n = 10_000
    fraud = rng.random(n) < 0.03
    fired = pd.DataFrame({
        "huge_but_precise": fraud | (rng.random(n) < 0.10),            # flags ~13%: over a 5% budget
        "small_and_precise": fraud & (rng.random(n) < 0.3),            # flags ~1%
    })
    chosen, log = select_rules(fired, fraud, min_precision=0.10, max_decline_rate=0.05)
    assert chosen == ["small_and_precise"]
    assert "over budget" in log.loc[log.rule == "huge_but_precise", "decision"].iloc[0]
