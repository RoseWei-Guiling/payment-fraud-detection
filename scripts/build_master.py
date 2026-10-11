"""
Step 1: build the master dataset and the data quality report.

Run from the project root, after scripts/download_data.py:
  python scripts/build_master.py

Outputs:
  data/processed/master.parquet   one row per transaction, identity merged, time + customer ID columns
  reports/data_quality.md         data quality report (commit this one)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_master  # noqa: E402
from src.quality import write_report  # noqa: E402
from src.uid import add_uid, uid_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/master.parquet")
    ap.add_argument("--report", default="reports/data_quality.md")
    args = ap.parse_args()

    t0 = time.time()
    df = build_master(args.raw)
    df = add_uid(df)
    stats = uid_check(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_report(df, stats, Path(args.report))

    print(f"Master dataset: {len(df):,} rows x {df.shape[1]} columns -> {out}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    print(df["split"].value_counts().reindex(["train", "valid", "oot"]).to_string())
    print("Customer ID check:", stats)
    print(f"Report -> {args.report}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
