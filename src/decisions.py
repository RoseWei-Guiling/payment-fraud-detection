"""
Step 6: turn risk scores into decisions, priced in money.

Blocking a payment has two possible outcomes:
  - it was fraud     -> the bank saves the payment's amount
  - it was genuine   -> the bank pays a friction cost (call-centre time, an annoyed or
                        lost customer, lost sales). This cost is an ASSUMPTION, so every
                        result is shown for several values of it.

  net benefit = fraud money blocked - cost per blocked genuine payment x genuine payments blocked

Amounts are in the dataset's currency, which Vesta didn't state (most likely US dollars).
"""
import numpy as np
import pandas as pd

PER = 100_000  # results are reported per 100,000 payments, so months of different size compare


def net_benefit_curve(score, is_fraud, amount, cost_per_false_block):
    """Net benefit of blocking the top k payments by score, for every k from 0 to all.
    Returns a table with one row per k (decline rate, fraud caught, net benefit...)."""
    score = np.asarray(score, dtype="float64")
    is_fraud = np.asarray(is_fraud, dtype=bool)
    amount = np.asarray(amount, dtype="float64")
    order = np.argsort(-score, kind="stable")
    fraud_value_blocked = np.concatenate([[0.0], np.cumsum(np.where(is_fraud[order], amount[order], 0.0))])
    genuine_blocked = np.concatenate([[0], np.cumsum(~is_fraud[order])])
    frauds_blocked = np.concatenate([[0], np.cumsum(is_fraud[order])])
    n = len(score)
    k = np.arange(n + 1)
    benefit = fraud_value_blocked - cost_per_false_block * genuine_blocked
    return pd.DataFrame({
        "blocked": k,
        "decline_rate": k / n,
        "fraud_caught": frauds_blocked / max(is_fraud.sum(), 1),
        "fraud_value_caught": fraud_value_blocked / max(amount[is_fraud].sum(), 1e-9),
        "genuine_blocked": genuine_blocked,
        "net_benefit_per_100k": benefit / n * PER,
        "threshold": np.concatenate([[np.inf], score[order]]),  # block if score >= threshold
    })


def best_cutoff(curve):
    """The row of the curve with the highest net benefit."""
    return curve.loc[curve["net_benefit_per_100k"].idxmax()]


def outcome(blocked, is_fraud, amount, cost_per_false_block):
    """Same measures for any yes/no blocking decision (rules, or the expected-cost rule)."""
    blocked = np.asarray(blocked, dtype=bool)
    is_fraud = np.asarray(is_fraud, dtype=bool)
    amount = np.asarray(amount, dtype="float64")
    fraud_value = amount[blocked & is_fraud].sum()
    genuine = int((blocked & ~is_fraud).sum())
    n = len(blocked)
    return {
        "decline_rate": blocked.mean(),
        "fraud_caught": (blocked & is_fraud).sum() / max(is_fraud.sum(), 1),
        "fraud_value_caught": fraud_value / max(amount[is_fraud].sum(), 1e-9),
        "genuine_blocked_per_100k": genuine / n * PER,
        "net_benefit_per_100k": (fraud_value - cost_per_false_block * genuine) / n * PER,
    }


def expected_cost_rule(probability, amount, cost_per_false_block):
    """Block a payment when blocking is expected to save more than it costs:

        P(fraud) x amount  >  (1 - P(fraud)) x cost per blocked genuine payment

    Unlike a single score cut-off, this blocks a 30% risk on a 900 payment but lets a 30%
    risk on a 5 payment through. It needs probabilities that mean what they say
    (calibration), which calibration_table checks.
    """
    p = np.asarray(probability, dtype="float64")
    return p * np.asarray(amount, dtype="float64") > (1 - p) * cost_per_false_block


def calibration_table(probability, is_fraud, bins=10):
    """Predicted vs actual fraud rate in 10 equal-sized groups of payments."""
    df = pd.DataFrame({"p": probability, "y": np.asarray(is_fraud, dtype=float)})
    df["group"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    t = df.groupby("group").agg(predicted=("p", "mean"), actual=("y", "mean"), payments=("y", "size"))
    return t.reset_index()


def ab_test(champion_blocked, challenger_blocked, is_fraud, amount, cost_per_false_block,
            challenger_share=0.10, n_boot=1000, seed=0):
    """Simulate a champion/challenger trial: route a random share of payments to the
    challenger and the rest to the champion, then compare net benefit per 100,000 payments.
    A bootstrap (re-running the comparison on resampled payments) gives a 95% interval.

    Caveat for real life: once a payment is blocked you rarely learn whether it was fraud,
    so a live trial has to account for that. Here we know every label, so it's a simulation.
    """
    rng = np.random.default_rng(seed)
    n = len(is_fraud)
    is_fraud = np.asarray(is_fraud, dtype=bool)
    amount = np.asarray(amount, dtype="float64")
    gain = np.where(is_fraud, amount, -cost_per_false_block)  # value of blocking each payment

    def arm_benefit(blocked, rows):
        return (gain[rows] * blocked[rows]).sum() / len(rows) * PER

    champion_blocked = np.asarray(champion_blocked, dtype=bool)
    challenger_blocked = np.asarray(challenger_blocked, dtype=bool)
    to_challenger = rng.random(n) < challenger_share
    champ_rows, chall_rows = np.flatnonzero(~to_challenger), np.flatnonzero(to_challenger)
    observed = arm_benefit(challenger_blocked, chall_rows) - arm_benefit(champion_blocked, champ_rows)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        c = rng.choice(champ_rows, len(champ_rows))
        h = rng.choice(chall_rows, len(chall_rows))
        diffs[b] = arm_benefit(challenger_blocked, h) - arm_benefit(champion_blocked, c)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"payments_to_challenger": len(chall_rows), "payments_to_champion": len(champ_rows),
            "difference_per_100k": observed, "ci_low": low, "ci_high": high}
