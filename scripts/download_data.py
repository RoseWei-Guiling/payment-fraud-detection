"""
Download the IEEE-CIS Fraud Detection training data from Kaggle into data/raw/.

Before running, do these once:
  1. Create a free Kaggle account.
  2. Accept the competition rules: https://www.kaggle.com/c/ieee-fraud-detection/rules
  3. Kaggle > Settings > API > "Create New Token". Move the downloaded kaggle.json to
     ~/.kaggle/kaggle.json, then run:  chmod 600 ~/.kaggle/kaggle.json

Run from the project root:
  python scripts/download_data.py

The data can't be redistributed, so never commit data/ to GitHub (.gitignore already excludes it).
"""
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
WANTED = ["train_transaction.csv", "train_identity.csv"]  # test labels were never published
RAW = Path("data/raw")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in WANTED):
        print("Already downloaded:", ", ".join(WANTED))
        return

    zip_path = RAW / f"{COMPETITION}.zip"
    if not zip_path.exists():
        print("Downloading from Kaggle (about 120 MB)...")
        result = subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW)])
        if result.returncode != 0:
            sys.exit("Download failed. Check your kaggle.json token and that you accepted the competition rules.")

    with zipfile.ZipFile(zip_path) as z:
        for name in WANTED:
            print("Extracting", name)
            z.extract(name, RAW)
    print("Done. Files are in", RAW.resolve())


if __name__ == "__main__":
    main()
