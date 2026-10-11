"""
Make a small FAKE dataset with the same columns as IEEE-CIS, so the pipeline can be
tested without downloading anything (e.g. in GitHub Actions). The values are random
and mean nothing: never use this for modelling or results.

  python tests/fake_data.py --out tests/fake_raw --n 5000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n, seed=0):
    rng = np.random.default_rng(seed)
    n_cust = n // 5
    cust_card1 = rng.integers(1000, 18000, n_cust)
    cust_addr1 = rng.choice([np.nan, *range(100, 540)], n_cust)
    cust_start = rng.integers(-300, 150, n_cust)
    cust_fraud = rng.random(n_cust) < 0.035
    who = rng.integers(0, n_cust, n)

    dt = np.sort(rng.integers(86400, 86400 * 183, n))
    day = dt // 86400
    d1 = np.clip(day - cust_start[who], 0, None).astype(float)
    d1[rng.random(n) < 0.02] = np.nan

    tx = pd.DataFrame({
        "TransactionID": np.arange(2987000, 2987000 + n),
        "isFraud": (cust_fraud[who] & (rng.random(n) < 0.9)).astype(int),
        "TransactionDT": dt,
        "TransactionAmt": np.round(rng.lognormal(4, 1, n), 3),
        "ProductCD": rng.choice(list("WHCSR"), n, p=[0.74, 0.06, 0.12, 0.02, 0.06]),
        "card1": cust_card1[who],
        "card2": rng.choice([np.nan, *range(100, 600)], n),
        "card3": rng.choice([150.0, 185.0], n),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n),
        "card5": rng.choice([226.0, 224.0, 166.0], n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": cust_addr1[who],
        "addr2": rng.choice([87.0, np.nan], n, p=[0.9, 0.1]),
        "dist1": np.where(rng.random(n) < 0.6, np.nan, rng.exponential(100, n)),
        "dist2": np.where(rng.random(n) < 0.9, np.nan, rng.exponential(200, n)),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], n),
        "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None], n, p=[0.1, 0.05, 0.85]),
    })
    for i in range(1, 15):
        tx[f"C{i}"] = rng.poisson(1.5, n).astype(float)
    tx["D1"] = d1
    for i in range(2, 16):
        tx[f"D{i}"] = np.where(rng.random(n) < 0.5, np.nan, rng.integers(0, 600, n))
    for i in range(1, 10):
        tx[f"M{i}"] = rng.choice(["T", "F", None], n)
    v = rng.random((n, 339)).astype("float32")
    v[rng.random((n, 339)) < 0.4] = np.nan
    tx = pd.concat([tx, pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 340)])], axis=1)

    has_id = rng.random(n) < 0.25
    ident = pd.DataFrame({"TransactionID": tx.loc[has_id, "TransactionID"].to_numpy()})
    m = len(ident)
    for i in range(1, 39):
        ident[f"id_{i:02d}"] = rng.choice(["a", "b", "c"], m) if i >= 12 else rng.normal(0, 1, m)
    ident["DeviceType"] = rng.choice(["desktop", "mobile"], m)
    ident["DeviceInfo"] = rng.choice(["Windows", "iOS Device", "MacOS", "Trident/7.0"], m)
    return tx, ident


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fake_raw")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tx, ident = make(args.n)
    tx.to_csv(out / "train_transaction.csv", index=False)
    ident.to_csv(out / "train_identity.csv", index=False)
    print(f"Fake data: {len(tx)} transactions, {len(ident)} identity rows -> {out}")
