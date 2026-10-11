"""
Step 2b: run the velocity SQL in Google BigQuery and check it against pandas.

One-time setup (free, no credit card):
  1. Go to https://console.cloud.google.com/bigquery and sign in with a Google account.
  2. Accept the terms. You get the BigQuery sandbox automatically.
  3. Create a project (top bar > project picker > New project), e.g. "payment-fraud".
     Copy its Project ID (shown under the name, e.g. payment-fraud-123456).

Run from the project root, after scripts/build_velocity.py:
  python scripts/bigquery_features.py --project YOUR_PROJECT_ID

A browser window opens once to sign in to Google. The script then:
  - uploads the 7 columns the SQL needs (about 590k rows) to dataset `fraud`, table `transactions`
  - runs sql/velocity_features.sql in BigQuery
  - compares BigQuery's results with the pandas features, and writes reports/bigquery_check.md

Sandbox notes: tables expire after 60 days (rerun this script to recreate them); the
free tier covers far more queries than this project needs.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.velocity import COLUMNS, compare, slim, sql_text  # noqa: E402

SCOPES = ["https://www.googleapis.com/auth/bigquery"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="your Google Cloud project ID")
    ap.add_argument("--dataset", default="fraud")
    ap.add_argument("--location", default="EU", help="where the data is stored (EU keeps it in Europe)")
    args = ap.parse_args()

    import pydata_google_auth
    from google.cloud import bigquery

    creds = pydata_google_auth.get_user_credentials(SCOPES)
    client = bigquery.Client(project=args.project, credentials=creds, location=args.location)

    dataset = bigquery.Dataset(f"{args.project}.{args.dataset}")
    dataset.location = args.location
    client.create_dataset(dataset, exists_ok=True)
    table_id = f"{args.project}.{args.dataset}.transactions"

    t0 = time.time()
    master = pd.read_parquet("data/processed/master.parquet", columns=COLUMNS)
    upload = slim(master)
    for c in ["uid", "DeviceInfo", "P_emaildomain"]:  # plain Python strings upload most reliably
        upload[c] = upload[c].astype(object).where(upload[c].notna(), None)
    job = client.load_table_from_dataframe(
        upload, table_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
    job.result()
    print(f"Uploaded {len(upload):,} rows to {table_id} ({time.time() - t0:.0f}s)")

    query = client.query(sql_text(f"`{table_id}`"))
    bq = query.to_dataframe()
    gb = (query.total_bytes_processed or 0) / 1e9
    print(f"BigQuery returned {len(bq):,} rows, processed {gb:.2f} GB ({time.time() - t0:.0f}s)")

    pandas_feats = pd.read_parquet("data/processed/velocity.parquet")
    check = compare(pandas_feats, bq)
    print(check.to_string(index=False))

    Path("reports/bigquery_check.md").write_text("\n".join([
        "# Velocity features: BigQuery vs pandas", "",
        f"Table `{args.dataset}.transactions` in location {args.location}, {len(bq):,} rows, "
        f"{gb:.2f} GB processed.", "",
        f"**All features match: {bool(check['match'].all())}**", "",
        check.to_markdown(index=False), "",
    ]))
    print("Saved reports/bigquery_check.md")


if __name__ == "__main__":
    main()
