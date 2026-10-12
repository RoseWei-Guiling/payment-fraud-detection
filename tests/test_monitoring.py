"""Step 7: drift (PSI) maths and plain-word reasons."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring import psi, psi_label  # noqa: E402
from src.reasons import plain_name, plain_reason  # noqa: E402


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=50_000), rng.normal(size=50_000)) < 0.01


def test_psi_flags_a_real_shift():
    rng = np.random.default_rng(0)
    value = psi(rng.normal(size=50_000), rng.normal(1.0, 1.0, size=50_000))
    assert psi_label(value) == "major shift (investigate)"


def test_psi_by_hand_for_categories():
    base = pd.Series(["a"] * 50 + ["b"] * 50)
    now = pd.Series(["a"] * 80 + ["b"] * 20)
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert np.isclose(psi(base, now), expected)


def test_psi_treats_blanks_as_their_own_bin():
    base = pd.Series([1.0, 2.0, 3.0, 4.0] * 100)
    now = pd.Series([1.0, 2.0, np.nan, np.nan] * 100)  # half the values went blank
    assert psi(base, now) > 0.25


def test_plain_words():
    assert plain_name("uid_n_24h") == "customer's payments in the last 24 hours"
    assert plain_name("P_emaildomain_te") == "past fraud rate of the purchaser email domain"
    assert plain_name("V188") == "a Vesta engineered risk signal (V188)"
    assert plain_reason("TransactionAmt", 1234.5) == "payment amount: 1,234"
    assert plain_reason("dist1", np.nan) == "distance between addresses: blank"
