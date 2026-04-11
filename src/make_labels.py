"""
Generates label CSVs for all three datasets, copies images to flat output
folders, and adds a google_path column for Google Colab.

Outputs:
  data/classification/opensar1_labels.csv  + os1/ images
  data/classification/opensar2_labels.csv  + os2/ images
  data/classification/fusar_labels.csv     + fs/  images

Usage:
    python src/make_labels.py
    python src/make_labels.py --data-dir /path/to/data/classification
    python src/make_labels.py --drive-root /content/drive/MyDrive
    python src/make_labels.py --skip-copy   # labels only, no image copying
"""

import argparse
import csv
import io
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR  = Path(__file__).parent.parent / "data" / "classification"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive"
LOCAL_PREFIX       = "data/"

# ---------------------------------------------------------------------------
# Shared AIS label logic (OpenSARShip 1 & 2)
# ---------------------------------------------------------------------------

PATCH_RE = re.compile(r"^Visual_(.+)_x(\d+)_y(\d+)(?:_(vh|vv|hh|hv))?\.tif$", re.IGNORECASE)


def ais_label(ship_type_str: str) -> str:
    """Map a numeric AIS ship_type string to a simplified 6-class label."""
    if not ship_type_str:
        return "unknown"
    try:
        code = int(ship_type_str)
    except ValueError:
        return "unknown"
    if 70 <= code <= 79:
        return "cargo"
    if 80 <= code <= 89:
        return "tanker"
    if 60 <= code <= 69:
        return "passenger"
    if code in {31, 32, 33, 34, 52, 53, 54, 58}:
        return "engineering"
    if (20 <= code <= 29) or code in {30, 36, 37} or (40 <= code <= 57) or code in {59} or (90 <= code <= 99):
        return "other"
    return "unknown"


def parse_xml_ship_types(raw: bytes) -> dict[tuple[int, int], str]:
    """Return {(center_x, center_y): ship_type_str} from Ship.xml bytes."""
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    tree = ET.fromstring(content)
    result = {}
    for ship in tree.findall("ship"):
        sar = ship.find("SARShipInformation")
        cx = int(sar.findtext("Center_x"))
        cy = int(sar.findtext("Center_y"))
        ship_type = ship.findtext("AISShipInformation/Ship_Type", default="")
        result[(cx, cy)] = ship_type
    return result

# ---------------------------------------------------------------------------
# OpenSARShip 1
# ---------------------------------------------------------------------------

def _os1_from_zip(data_dir: Path) -> tuple[list[dict], list[Path], int]:
    src_zip = data_dir / "OpenSARShip_1.zip"
    outer = zipfile.ZipFile(src_zip, "r")
    inner_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
    rows, src_paths, missing = [], [], 0

    for zip_path in inner_zips:
        scene = zip_path.split("/")[-1]
        inner = zipfile.ZipFile(io.BytesIO(outer.read(zip_path)))
        all_files = inner.namelist()

        xml_path = next((f for f in all_files if f.endswith("/Ship.xml") or f == "Ship.xml"), None)
        if not xml_path:
            print(f"  [os1] WARNING: no Ship.xml in {scene}, skipping")
            continue
        coord_to_type = parse_xml_ship_types(inner.read(xml_path))

        patch_files = sorted(f for f in all_files if "/Patch_Uint8/" in f and not f.endswith("/"))

        for fpath in patch_files:
            fname = fpath.split("/")[-1]
            m = PATCH_RE.match(fname)
            if not m:
                continue
            class_label, x_str, y_str, pol = m.groups()
            ship_type = coord_to_type.get((int(x_str), int(y_str)), "")
            if not ship_type:
                missing += 1
            scene_name = scene.replace(".zip", "")
            os1_id = f"os1_{len(rows) + 1:05d}"
            lbl = ais_label(ship_type)
            rows.append({
                "os1_id": os1_id,
                "path": f"data/classification/os1/{os1_id}.tif",
                "class_label": class_label,
                "ship_type": ship_type,
                "polarization": pol.lower() if pol else "",
                "label": lbl,
                "is_cargo": int(lbl == "cargo"),
            })
            src_paths.append(data_dir / "OpenSARShip_1" / scene_name / "Patch_Uint8" / fname)

        inner.close()

    outer.close()
    return rows, src_paths, missing


def _os1_from_csv(out_csv: Path, data_dir: Path) -> tuple[list[dict], list[Path]]:
    fieldnames = ["os1_id", "path", "class_label", "ship_type", "polarization", "label", "is_cargo"]
    rows, src_paths = [], []
    with open(out_csv, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            os1_id = f"os1_{i:05d}"
            row["os1_id"] = os1_id
            row["path"] = f"data/classification/os1/{os1_id}.tif"
            row["is_cargo"] = int(row.get("label", "") == "cargo")
            src_paths.append(data_dir / "os1" / f"{os1_id}.tif")
            rows.append({k: row[k] for k in fieldnames})
    return rows, src_paths


def make_opensar1(data_dir: Path, skip_copy: bool) -> list[dict]:
    out_csv = data_dir / "opensar1_labels.csv"
    src_zip = data_dir / "OpenSARShip_1.zip"
    fieldnames = ["os1_id", "path", "class_label", "ship_type", "polarization", "label", "is_cargo"]

    if src_zip.exists():
        print(f"\n[OpenSARShip 1] Generating labels from zip...")
        rows, src_paths, missing = _os1_from_zip(data_dir)
        with open(out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
        print(f"  Wrote {len(rows)} rows")
        if missing:
            print(f"  WARNING: {missing} files had no matching XML entry")
    else:
        print(f"\n[OpenSARShip 1] Zip not found, reading existing CSV...")
        rows, src_paths = _os1_from_csv(out_csv, data_dir)
        tmp = out_csv.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out_csv)
        print(f"  Updated {out_csv.name} ({len(rows)} rows)")

    if not skip_copy:
        out_dir = data_dir / "os1"
        out_dir.mkdir(exist_ok=True)
        copied = sum(
            1 for row, src in zip(rows, src_paths)
            if src.exists() and src != (dst := out_dir / f"{row['os1_id']}.tif")
            and not shutil.copy2(src, dst) or True
        )
        print(f"  Copied {len(rows)} images to {out_dir}")

    return rows

# ---------------------------------------------------------------------------
# OpenSARShip 2
# ---------------------------------------------------------------------------

def _os2_from_zip(data_dir: Path) -> tuple[list[dict], list[Path], int]:
    src_zip = data_dir / "OpenSARShip_2.zip"
    outer = zipfile.ZipFile(src_zip, "r")
    inner_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
    rows, src_paths, missing = [], [], 0

    for zip_path in inner_zips:
        scene = zip_path.split("/")[-1]
        inner = zipfile.ZipFile(io.BytesIO(outer.read(zip_path)))

        if "Ship.xml" not in inner.namelist():
            print(f"  [os2] WARNING: no Ship.xml in {scene}, skipping")
            continue
        coord_to_type = parse_xml_ship_types(inner.read("Ship.xml"))

        patch_files = sorted(
            f for f in inner.namelist()
            if f.startswith("Patch_Uint8/") and not f.endswith("/")
        )

        for fpath in patch_files:
            fname = fpath.split("/")[-1]
            m = PATCH_RE.match(fname)
            if not m:
                continue
            class_label, x_str, y_str, pol = m.groups()
            ship_type = coord_to_type.get((int(x_str), int(y_str)), "")
            if not ship_type:
                missing += 1
            scene_name = scene.replace(".zip", "")
            os2_id = f"os2_{len(rows) + 1:05d}"
            lbl = ais_label(ship_type)
            rows.append({
                "os2_id": os2_id,
                "path": f"data/classification/os2/{os2_id}.tif",
                "class_label": class_label,
                "ship_type": ship_type,
                "polarization": pol.lower() if pol else "",
                "label": lbl,
                "is_cargo": int(lbl == "cargo"),
            })
            src_paths.append(data_dir / "OpenSARShip_2" / scene_name / "Patch_Uint8" / fname)

        inner.close()

    outer.close()
    return rows, src_paths, missing


def _os2_from_csv(out_csv: Path, data_dir: Path) -> tuple[list[dict], list[Path]]:
    fieldnames = ["os2_id", "path", "class_label", "ship_type", "polarization", "label", "is_cargo"]
    rows, src_paths = [], []
    with open(out_csv, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            os2_id = f"os2_{i:05d}"
            row["os2_id"] = os2_id
            row["path"] = f"data/classification/os2/{os2_id}.tif"
            row["is_cargo"] = int(row.get("label", "") == "cargo")
            src_paths.append(data_dir / "os2" / f"{os2_id}.tif")
            rows.append({k: row[k] for k in fieldnames})
    return rows, src_paths


def make_opensar2(data_dir: Path, skip_copy: bool) -> list[dict]:
    out_csv = data_dir / "opensar2_labels.csv"
    src_zip = data_dir / "OpenSARShip_2.zip"
    fieldnames = ["os2_id", "path", "class_label", "ship_type", "polarization", "label", "is_cargo"]

    if src_zip.exists():
        print(f"\n[OpenSARShip 2] Generating labels from zip...")
        rows, src_paths, missing = _os2_from_zip(data_dir)
        with open(out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
        print(f"  Wrote {len(rows)} rows")
        if missing:
            print(f"  WARNING: {missing} files had no matching XML entry")
    else:
        print(f"\n[OpenSARShip 2] Zip not found, reading existing CSV...")
        rows, src_paths = _os2_from_csv(out_csv, data_dir)
        tmp = out_csv.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out_csv)
        print(f"  Updated {out_csv.name} ({len(rows)} rows)")

    if not skip_copy:
        out_dir = data_dir / "os2"
        out_dir.mkdir(exist_ok=True)
        for row, src in zip(rows, src_paths):
            dst = out_dir / f"{row['os2_id']}.tif"
            if src.exists() and src != dst:
                shutil.copy2(src, dst)
        print(f"  Copied {len(rows)} images to {out_dir}")

    return rows

# ---------------------------------------------------------------------------
# FuSARShip
# ---------------------------------------------------------------------------

ALL_LABELS = {"cargo", "tanker", "passenger", "engineering", "other", "unknown"}

FUSAR_CLASS_TO_LABEL: dict[str, str] = {
    "Cargo":          "cargo",
    "Tanker":         "tanker",
    "Passenger":      "passenger",
    "Dredger":        "engineering",
    "Tug":            "engineering",
    "DiveVessel":     "engineering",
    "SAR":            "engineering",
    "PortTender":     "engineering",
    "Fishing":        "other",
    "HighSpeedCraft": "other",
    "LawEnforce":     "other",
    "Reserved":       "other",
    "WingInGrnd":     "other",
    "Other":          "other",
    "Unspecified":    "unknown",
}

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def make_fusar(data_dir: Path, skip_copy: bool, keep_labels: set[str]) -> list[dict]:
    fusar_dir = data_dir / "FuSARShip"
    out_csv   = data_dir / "fusar_labels.csv"
    fieldnames = ["fs_id", "path", "class_label", "ship_type", "polarization", "label", "is_cargo"]

    if not fusar_dir.exists():
        print(f"\n[FuSARShip] ERROR: directory not found at {fusar_dir}, skipping.")
        return []

    print(f"\n[FuSARShip] Building labels from {fusar_dir}...")
    rows, src_paths, skipped = [], [], 0

    for class_dir in sorted(fusar_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        label = FUSAR_CLASS_TO_LABEL.get(class_name)
        if label is None:
            print(f"  WARNING: unknown class folder '{class_name}', skipping")
            continue

        images = sorted(f for f in class_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)

        if label not in keep_labels:
            skipped += len(images)
            continue

        for img in images:
            subtype = img.parent.name
            fs_id = f"fs_{len(rows) + 1:04d}"
            rows.append({
                "fs_id": fs_id,
                "path": f"data/classification/fs/{fs_id}.tiff",
                "class_label": f"{class_name}/{subtype}",
                "ship_type": "",
                "polarization": "",
                "label": label,
                "is_cargo": int(label == "cargo"),
            })
            src_paths.append(img)

        print(f"  {class_name}: {len(images)} images -> '{label}'")

    if skipped:
        print(f"  Skipped {skipped} images with excluded labels")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {out_csv.name}")

    if not skip_copy:
        out_dir = data_dir / "fs"
        out_dir.mkdir(exist_ok=True)
        for row, src in zip(rows, src_paths):
            shutil.copy2(src, out_dir / f"{row['fs_id']}.tiff")
        print(f"  Copied {len(rows)} images to {out_dir}")

    return rows

# ---------------------------------------------------------------------------
# Path verification
# ---------------------------------------------------------------------------

def verify_paths(data_dir: Path) -> bool:
    """Returns True if all paths in all CSVs resolve to existing files."""
    project_root = data_dir.parent.parent
    csv_configs = [
        (data_dir / "opensar1_labels.csv", "path"),
        (data_dir / "opensar2_labels.csv", "path"),
        (data_dir / "fusar_labels.csv",    "path"),
    ]
    all_ok = True
    for csv_path, col in csv_configs:
        if not csv_path.exists():
            print(f"  {csv_path.name}: not found, skipping")
            continue
        rows = list(csv.DictReader(open(csv_path, newline="")))
        missing = [r[col] for r in rows if not (project_root / r[col]).exists()]
        if missing:
            print(f"  {csv_path.name}: {len(missing)}/{len(rows)} paths MISSING")
            for p in missing[:3]:
                print(f"    {p}")
            if len(missing) > 3:
                print(f"    ... and {len(missing) - 3} more")
            all_ok = False
        else:
            print(f"  {csv_path.name}: all {len(rows)} paths OK")
    return all_ok


def cleanup_source_dirs(data_dir: Path) -> None:
    """Delete the original extracted source directories now that images are copied to flat folders."""
    source_dirs = ["OpenSARShip_1", "OpenSARShip_2", "FuSARShip"]
    for name in source_dirs:
        d = data_dir / name
        if d.exists():
            shutil.rmtree(d)
            print(f"  Deleted {d}")
        else:
            print(f"  {name}: not found, skipping")


# ---------------------------------------------------------------------------
# Google paths
# ---------------------------------------------------------------------------

def add_google_paths(csv_path: Path, drive_root: str) -> None:
    drive_root = drive_root.rstrip("/")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "google_path" not in fieldnames:
        fieldnames.append("google_path")

    for row in rows:
        path = row["path"]
        suffix = path[len(LOCAL_PREFIX):] if path.startswith(LOCAL_PREFIX) else path
        row["google_path"] = f"{drive_root}/{suffix}"

    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)
    print(f"  google_path added to {csv_path.name} (example: {rows[0]['google_path']})")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate label CSVs for OpenSARShip 1, OpenSARShip 2, and FuSARShip."
    )
    parser.add_argument("--data-dir",   type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--drive-root", type=str,  default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--skip-copy",    action="store_true", help="Skip copying images to flat folders")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip deleting source directories after verification")
    parser.add_argument(
        "--fusar-exclude",
        nargs="+",
        default=["unknown"],
        metavar="LABEL",
        help="FuSARShip labels to exclude (default: unknown)",
    )
    args = parser.parse_args()

    data_dir    = args.data_dir
    keep_labels = ALL_LABELS - set(args.fusar_exclude)

    make_opensar1(data_dir, args.skip_copy)
    make_opensar2(data_dir, args.skip_copy)
    make_fusar(data_dir, args.skip_copy, keep_labels)

    print(f"\n[Google Paths] Adding google_path column (drive root: {args.drive_root})...")
    for csv_name in ("opensar1_labels.csv", "opensar2_labels.csv", "fusar_labels.csv"):
        add_google_paths(data_dir / csv_name, args.drive_root)

    print(f"\n[Verify Paths] Checking that all image paths resolve...")
    all_ok = verify_paths(data_dir)

    if args.skip_cleanup:
        print(f"\n[Cleanup] Skipped — --skip-cleanup was set.")
    elif not all_ok:
        print(f"\n[Cleanup] Skipped — missing paths detected, source directories kept.")
    elif args.skip_copy:
        print(f"\n[Cleanup] Skipped — --skip-copy was set, source directories kept.")
    else:
        print(f"\n[Cleanup] Verification passed — deleting source directories...")
        cleanup_source_dirs(data_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
