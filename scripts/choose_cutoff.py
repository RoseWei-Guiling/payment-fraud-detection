"""
Step 6: choose where to draw the line, in money, on the validation month.

  python scripts/choose_cutoff.py                 # default: a blocked genuine payment costs 10
  python scripts/choose_cutoff.py --cost 25       # try a different assumption

Outputs:
  reports/decision_report.md   strategies compared in money, cost sensitivity, A/B simulation, calibration
  reports/cost_curve.png       net benefit vs share of payments blocked
  reports/cutoffs.json         the chosen cut-offs, applied unchanged to the test month in Step 7
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.decisions import (ab_test, best_cutoff, calibration_table, expected_cost_rule,  # noqa: E402
                           net_benefit_curve, outcome)

COSTS = [2, 5, 10, 25, 50]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def fmt(table):
    t = table.copy()
    for col in t.columns:
        if col in {"decline_rate", "fraud_caught", "fraud_value_caught"}:
            t[col] = t[col].map(lambda v: f"{v:.1%}")
        elif col in {"net_benefit_per_100k", "genuine_blocked_per_100k", "difference_per_100k", "ci_low", "ci_high"}:
            t[col] = t[col].map(lambda v: f"{v:,.0f}")
        elif col == "vs_rules_per_100k":
            t[col] = t[col].map(lambda v: f"{v:+,.0f}")
    return t


def plot(curves, champion, cost, path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x_max = max(0.12, champion["decline_rate"] + 0.02)
    for (name, curve, colour), y_offset in zip(curves, [8, -16]):
        c = curve[curve["decline_rate"] <= x_max]
        ax.plot(c["decline_rate"] * 100, c["net_benefit_per_100k"], color=colour, lw=2, label=name)
        best = best_cutoff(curve)
        ax.scatter([best["decline_rate"] * 100], [best["net_benefit_per_100k"]], color=colour, s=40, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{name}: best at {best['decline_rate']:.1%} blocked",
                    (best["decline_rate"] * 100, best["net_benefit_per_100k"]),
                    xytext=(8, y_offset), textcoords="offset points", fontsize=8.5, color=INK)
    ax.scatter([champion["decline_rate"] * 100], [champion["net_benefit_per_100k"]], color=MUTED, marker="D", s=36,
               zorder=3, edgecolor="white", linewidth=1.5, label="Rules champion")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Payments blocked (%)", color=MUTED)
    ax.set_ylabel("Net benefit per 100,000 payments", color=MUTED)
    ax.set_title(f"Net benefit by share of payments blocked (blocked genuine payment costs {cost})",
                 loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=10, help="cost of blocking one genuine payment (assumption)")
    args = ap.parse_args()

    scores = pd.read_parquet("data/processed/model_scores.parquet")
    valid = scores[scores["split"] == "valid"].reset_index(drop=True)
    y, amount = valid["isFraud"], valid["TransactionAmt"]
    champ_blocked = valid["rules_champion"].astype(bool)

    def strategies(cost):
        lgb_curve = net_benefit_curve(valid["lightgbm"], y, amount, cost)
        sc_curve = net_benefit_curve(valid["scorecard"], y, amount, cost)
        lgb_best, sc_best = best_cutoff(lgb_curve), best_cutoff(sc_curve)
        lgb_cut = valid["lightgbm"] >= lgb_best["threshold"]
        sc_cut = valid["scorecard"] >= sc_best["threshold"]
        ev_rule = expected_cost_rule(valid["lightgbm"], amount, cost)
        rows = [
            {"strategy": "Rules champion", **outcome(champ_blocked, y, amount, cost)},
            {"strategy": "Scorecard, best cut-off", **outcome(sc_cut, y, amount, cost)},
            {"strategy": "LightGBM, best cut-off", **outcome(lgb_cut, y, amount, cost)},
            {"strategy": "LightGBM, expected-cost rule (amount-aware)", **outcome(ev_rule, y, amount, cost)},
        ]
        return pd.DataFrame(rows), lgb_curve, sc_curve, lgb_best, sc_best, ev_rule

    table, lgb_curve, sc_curve, lgb_best, sc_best, ev_rule = strategies(args.cost)
    rules_benefit = table.loc[0, "net_benefit_per_100k"]
    table["vs_rules_per_100k"] = table["net_benefit_per_100k"] - rules_benefit

    sensitivity = []
    for cost in COSTS:
        t, *_ = strategies(cost)
        best = t.set_index("strategy")
        sensitivity.append({
            "cost_per_blocked_genuine": cost,
            "rules_net_benefit_per_100k": best.loc["Rules champion", "net_benefit_per_100k"],
            "lightgbm_best_cutoff_blocks": best.loc["LightGBM, best cut-off", "decline_rate"],
            "lightgbm_best_cutoff_net_benefit_per_100k": best.loc["LightGBM, best cut-off", "net_benefit_per_100k"],
            "expected_cost_rule_blocks": best.loc["LightGBM, expected-cost rule (amount-aware)", "decline_rate"],
            "expected_cost_rule_net_benefit_per_100k":
                best.loc["LightGBM, expected-cost rule (amount-aware)", "net_benefit_per_100k"],
        })
    sens = pd.DataFrame(sensitivity)
    for col in sens.columns:
        if "blocks" in col:
            sens[col] = sens[col].map(lambda v: f"{v:.1%}")
        elif "benefit" in col:
            sens[col] = sens[col].map(lambda v: f"{v:,.0f}")

    lgb_blocked = valid["lightgbm"] >= lgb_best["threshold"]
    ab = ab_test(champ_blocked, lgb_blocked, y, amount, args.cost)
    calib = calibration_table(valid["lightgbm"], y)
    plot([("LightGBM", lgb_curve, BLUE), ("Scorecard", sc_curve, ORANGE)],
         table.iloc[0], args.cost, "reports/cost_curve.png")

    cutoffs = {"cost_per_blocked_genuine": args.cost,
               "lightgbm_threshold": float(lgb_best["threshold"]),
               "scorecard_threshold": float(sc_best["threshold"]),
               "chosen_on": "validation month"}
    Path("reports/cutoffs.json").write_text(json.dumps(cutoffs, indent=2))

    calib_fmt = calib.assign(predicted=calib["predicted"].map("{:.2%}".format),
                             actual=calib["actual"].map("{:.2%}".format))
    Path("reports/decision_report.md").write_text("\n".join([
        "# Decisions in money", "",
        "Generated by `scripts/choose_cutoff.py`, on the validation month. Amounts are in the dataset's "
        "currency (not stated by Vesta; most likely US dollars).", "",
        f"**Assumption:** blocking a genuine payment costs **{args.cost:g}** (handling, an annoyed or lost "
        "customer, lost sales). A blocked fraud saves its amount. "
        "Net benefit = fraud money blocked - cost x genuine payments blocked, per 100,000 payments.", "",
        "## Strategies compared", "",
        fmt(table).to_markdown(index=False), "",
        "- **Best cut-off**: one score threshold for every payment, chosen where net benefit peaks.",
        "- **Expected-cost rule**: block when P(fraud) x amount > (1 - P(fraud)) x cost. It uses the amount, "
        "so a risky large payment is blocked while an equally risky tiny one goes through.", "",
        "![Net benefit curve](cost_curve.png)", "",
        "## How sensitive is this to the cost assumption?", "",
        sens.to_markdown(index=False), "",
        "## Champion / challenger simulation", "",
        f"{ab['payments_to_challenger']:,} randomly chosen payments (10%) go to LightGBM at its best cut-off, the "
        f"other {ab['payments_to_champion']:,} to the rules. Difference in net benefit per 100,000 payments "
        f"(LightGBM minus rules): **{ab['difference_per_100k']:,.0f}**, 95% interval "
        f"{ab['ci_low']:,.0f} to {ab['ci_high']:,.0f} (bootstrap, 1,000 resamples).", "",
        "In a live trial, blocked payments rarely get confirmed as fraud or genuine, so the measurement must "
        "allow for that; here every label is known.", "",
        "## Calibration: do LightGBM's probabilities mean what they say?", "",
        "The expected-cost rule needs this. Payments split into 10 equal groups by predicted risk:", "",
        calib_fmt.to_markdown(index=False), "",
        "## Saved for Step 7", "",
        f"`reports/cutoffs.json`: LightGBM threshold {cutoffs['lightgbm_threshold']:.4f}, scorecard threshold "
        f"{cutoffs['scorecard_threshold']:.4f}. Step 7 applies these unchanged to the untouched test month.", "",
    ]))
    print(fmt(table).to_string(index=False), "\n")
    print(sens.to_string(index=False), "\n")
    print(f"A/B: LightGBM - rules = {ab['difference_per_100k']:,.0f} per 100k (95% CI {ab['ci_low']:,.0f} to "
          f"{ab['ci_high']:,.0f})")
    print("Saved reports/decision_report.md, reports/cost_curve.png, reports/cutoffs.json")


if __name__ == "__main__":
    main()
