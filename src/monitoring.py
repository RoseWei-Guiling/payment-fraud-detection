"""
Step 7: is the data drifting? Population Stability Index (PSI).

PSI compares how a variable is spread across bins in a baseline period with a later period:

    PSI = sum over bins of (current share - baseline share) x ln(current share / baseline share)

Industry rule of thumb:  < 0.10 stable  |  0.10 - 0.25 small shift, watch  |  > 0.25 major shift, investigate.
On a model's score it's called PSI; on an input feature it's often called CSI
(characteristic stability index). Same formula.
"""
import numpy as np
import pandas as pd

FLOOR = 1e-4  # an empty bin would make ln() blow up, so shares never go below this


def bin_shares(baseline, current, bins=10, max_categories=20):
    """Shares of baseline and current in the same bins. Numbers: 10 equal-sized bins of the
    baseline. Text: the baseline's most common categories plus 'other'. Blanks get their own bin."""
    baseline, current = pd.Series(baseline), pd.Series(current)
    if pd.api.types.is_numeric_dtype(baseline) and not pd.api.types.is_bool_dtype(baseline):
        edges = np.unique(np.nanquantile(baseline.astype("float64"), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])

        def label(s):
            codes = pd.cut(s.astype("float64"), edges, labels=False, include_lowest=True)
            return codes.astype("float64").fillna(-1)  # -1 = blank
    else:
        keep = set(baseline.astype("string").value_counts().index[:max_categories])

        def label(s):
            s = s.astype("string")
            return s.where(s.isin(keep), "other").fillna("blank")

    b = label(baseline).value_counts(normalize=True)
    c = label(current).value_counts(normalize=True)
    shares = pd.concat([b.rename("baseline"), c.rename("current")], axis=1).fillna(0.0)
    return shares.clip(lower=FLOOR)


def psi(baseline, current, bins=10):
    s = bin_shares(baseline, current, bins)
    return float(((s["current"] - s["baseline"]) * np.log(s["current"] / s["baseline"])).sum())


def psi_label(value):
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "small shift (watch)"
    return "major shift (investigate)"


def psi_table(baseline_df, current_df, columns):
    rows = [{"variable": c, "psi": psi(baseline_df[c], current_df[c])} for c in columns]
    t = pd.DataFrame(rows).sort_values("psi", ascending=False)
    t["status"] = t["psi"].map(psi_label)
    return t.reset_index(drop=True)
