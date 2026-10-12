"""
Step 4: the rules champion.

A rule flags a payment when its condition is true. The rule set flags a payment when
ANY selected rule fires. Rules are judged the way a fraud team judges them:

  - decline rate         share of all payments flagged (how many customers we interrupt)
  - fraud caught         share of fraud payments flagged
  - fraud value caught   share of fraud money flagged (what the bank actually saves)
  - precision            share of flagged payments that really are fraud
  - false-positive ratio genuine payments flagged per fraud caught (customer friction)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RULES_PATH = Path(__file__).resolve().parents[1] / "rules" / "champion_rules.yaml"


def load_rules(path=RULES_PATH):
    return yaml.safe_load(Path(path).read_text())["rules"]


def rule_columns(rules):
    """Feature columns the rule conditions mention (so we only load what we need)."""
    words = set()
    for rule in rules:
        for token in rule["condition"].replace("(", " ").replace(")", " ").split():
            if token.isidentifier() and token not in {"and", "or", "not", "in"}:
                words.add(token)
    return sorted(words)


def fire(df, condition):
    """True where the condition holds. Blank values make the comparison False, so they never fire."""
    result = df.eval(condition)
    return pd.Series(result, index=df.index).fillna(False).astype(bool)


def fire_all(df, rules):
    return pd.DataFrame({rule["name"]: fire(df, rule["condition"]) for rule in rules}, index=df.index)


def metrics(flagged, is_fraud, amount):
    flagged = np.asarray(flagged, dtype=bool)
    is_fraud = np.asarray(is_fraud, dtype=bool)
    amount = np.asarray(amount, dtype="float64")
    n_flagged = int(flagged.sum())
    caught = int((flagged & is_fraud).sum())
    false_positives = n_flagged - caught
    return {
        "flagged": n_flagged,
        "decline_rate": n_flagged / len(flagged),
        "fraud_caught": caught / max(int(is_fraud.sum()), 1),
        "fraud_value_caught": amount[flagged & is_fraud].sum() / max(amount[is_fraud].sum(), 1e-9),
        "precision": caught / n_flagged if n_flagged else np.nan,
        "false_positive_ratio": false_positives / caught if caught else np.inf,
    }


def rule_table(fired, is_fraud, amount):
    """One row per rule: how it performs on its own."""
    rows = []
    for name in fired.columns:
        rows.append({"rule": name, **metrics(fired[name], is_fraud, amount)})
    return pd.DataFrame(rows).sort_values("precision", ascending=False).reset_index(drop=True)


def select_rules(fired, is_fraud, min_precision=0.10, max_decline_rate=0.05):
    """Greedy rule-set optimisation, on TRAINING months only.

    Start with no rules. At each step, add the rule whose NEW flags (payments not already
    flagged) are most often fraud. Stop when that best addition is fraud less than
    `min_precision` of the time (10% = accept at most 9 genuine payments blocked per extra
    fraud caught), or when the whole set would flag more than `max_decline_rate` of payments.
    Returns the chosen rule names in the order they were added, with the reason each was taken.
    """
    is_fraud = np.asarray(is_fraud, dtype=bool)
    flagged = np.zeros(len(fired), dtype=bool)
    chosen, log = [], []
    remaining = list(fired.columns)
    while remaining:
        best, best_precision, best_new = None, -1.0, 0
        for name in remaining:
            new = fired[name].to_numpy() & ~flagged
            if new.sum() == 0:
                continue
            precision = is_fraud[new].mean()
            if precision > best_precision:
                best, best_precision, best_new = name, precision, int(new.sum())
        if best is None or best_precision < min_precision:
            break
        if (flagged | fired[best].to_numpy()).mean() > max_decline_rate:
            # Too big to fit the budget: skip it and try the next-best rule.
            log.append({"rule": best, "decision": "not added: would push the decline rate over budget",
                        "precision_of_new_flags": best_precision, "new_flags": best_new})
            remaining.remove(best)
            continue
        flagged |= fired[best].to_numpy()
        chosen.append(best)
        remaining.remove(best)
        log.append({"rule": best, "decision": "added", "precision_of_new_flags": best_precision,
                    "new_flags": best_new})
    # Explain every rule that was not chosen, measured against what the chosen set already flags.
    for name in remaining:
        if any(entry["rule"] == name for entry in log):
            continue
        new = fired[name].to_numpy() & ~flagged
        precision = is_fraud[new].mean() if new.sum() else np.nan
        reason = ("not added: never fires on anything new" if new.sum() == 0
                  else f"not added: only {precision:.1%} of its new flags are fraud")
        log.append({"rule": name, "decision": reason, "precision_of_new_flags": precision,
                    "new_flags": int(new.sum())})
    return chosen, pd.DataFrame(log)


def rules_score(fired, chosen):
    """How many chosen rules fired (0, 1, 2...). A crude score that lets rules be ranked like a model."""
    return fired[chosen].sum(axis=1) if chosen else pd.Series(0, index=fired.index)
