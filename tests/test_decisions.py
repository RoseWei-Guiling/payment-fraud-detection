"""Step 6 maths: net benefit, best cut-off, expected-cost rule, A/B simulation."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decisions import ab_test, best_cutoff, expected_cost_rule, net_benefit_curve, outcome  # noqa: E402


def test_net_benefit_by_hand():
    score = [0.9, 0.8, 0.1]
    fraud = [1, 0, 0]
    amount = [100, 50, 50]
    curve = net_benefit_curve(score, fraud, amount, cost_per_false_block=10)
    # block 0: 0 | block 1: +100 | block 2: 100 - 10 | block 3: 100 - 20   (per 3 payments)
    expected = np.array([0, 100, 90, 80]) / 3 * 100_000
    assert np.allclose(curve["net_benefit_per_100k"], expected)
    assert best_cutoff(curve)["blocked"] == 1


def test_outcome_matches_curve():
    rng = np.random.default_rng(0)
    score = rng.random(1000)
    fraud = rng.random(1000) < 0.05
    amount = rng.lognormal(4, 1, 1000)
    curve = net_benefit_curve(score, fraud, amount, 10)
    best = best_cutoff(curve)
    blocked = score >= best["threshold"]
    assert np.isclose(outcome(blocked, fraud, amount, 10)["net_benefit_per_100k"], best["net_benefit_per_100k"])


def test_expected_cost_rule_uses_the_amount():
    p = np.array([0.3, 0.3])
    amount = np.array([900, 5])
    assert expected_cost_rule(p, amount, cost_per_false_block=10).tolist() == [True, False]


def test_ab_test_finds_a_real_difference():
    rng = np.random.default_rng(1)
    n = 50_000
    fraud = rng.random(n) < 0.04
    amount = np.full(n, 100.0)
    good = fraud.copy()                  # blocks exactly the fraud
    bad = rng.random(n) < 0.04           # blocks at random
    result = ab_test(bad, good, fraud, amount, 10)
    assert result["ci_low"] > 0          # challenger clearly better
