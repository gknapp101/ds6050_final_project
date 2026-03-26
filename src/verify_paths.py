"""
Verifies that every path in the label CSVs resolves to an actual file on disk.
Run this after download_data.py to confirm the extraction paths match the labels.

Usage:
    python src/verify_paths.py
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "classification"

LABEL_FILES = [
    DATA_DIR / "opensar1_labels.csv",
    DATA_DIR / "opensar2_labels.csv",
]


def verify(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    missing = [r["path"] for r in rows if not (PROJECT_ROOT / r["path"]).exists()]

    print(f"{csv_path.name}: {len(rows)} rows — ", end="")
    if missing:
        print(f"{len(missing)} MISSING paths")
        for p in missing[:5]:
            print(f"  {p}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    else:
        print("all paths OK")


def main():
    for csv_path in LABEL_FILES:
        if not csv_path.exists():
            print(f"{csv_path.name}: not found, skipping")
            continue
        verify(csv_path)


if __name__ == "__main__":
    main()
