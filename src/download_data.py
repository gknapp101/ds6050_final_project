"""
Downloads OpenSARShip 1 & 2 and FuSARShip datasets from Google Drive and
extracts them into the data/classification directory.

OpenSARShip: downloaded as a folder, only Patch_Uint8 images extracted from
the nested zip-in-zip structure.

FuSARShip: downloaded as a single zip file, all files extracted preserving
the folder structure (top-level root folder stripped if present).

Usage:
    pip install gdown
    python src/download_data.py                        # download all
    python src/download_data.py --skip-opensar         # FuSARShip only
    python src/download_data.py --skip-fusar           # OpenSARShip only
    python src/download_data.py --data-dir /content/drive/MyDrive/ds6050/data/classification
    python src/download_data.py --zip-dir /path/to/zips --data-dir /path/to/output
"""

import argparse
import io
import re
import tempfile
import zipfile
from pathlib import Path

import gdown

OPENSAR_FOLDER_URL = "https://drive.google.com/drive/folders/19jLMSzHChVLk-vVAg2muNN2OALzksWob"
FUSAR_FILE_URL = "https://drive.google.com/file/d/1hXglX6kKMbNZkJEF9T8V3JNRWU9C3r8H/view?usp=drive_link"

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "classification"

# Maps zip filename -> dataset name for OpenSARShip
OPENSAR_ZIPS = {
    "OpenSARShip_1.zip": "OpenSARShip_1",
    "OpenSARShip_2.zip": "OpenSARShip_2",
}


# ---------------------------------------------------------------------------
# OpenSARShip 2 XML fix
# ---------------------------------------------------------------------------

_BACKSLASH_CLOSE = ("<" + chr(92) + "ShipList>").encode()
_FORWARD_CLOSE   = b"</ShipList>"
_BARE_LT_BEFORE_CLOSE = re.compile(rb"([^<])\s*<(</\w+>)")
_INVALID_BANG         = re.compile(rb"<!(?!--|" + rb"\[CDATA\[)")


def _fix_ship_xml(content: bytes) -> bytes:
    content = content.replace(_BACKSLASH_CLOSE, _FORWARD_CLOSE)
    content = _BARE_LT_BEFORE_CLOSE.sub(lambda m: m.group(1) + b"&lt;" + m.group(2), content)
    content = _INVALID_BANG.sub(b"&lt;!", content)
    return content


def _patch_inner_zip(inner_bytes: bytes) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(inner_bytes), "r")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "Ship.xml":
                data = _fix_ship_xml(data)
            dst.writestr(item, data)
    src.close()
    return buf.getvalue()


def fix_opensar2_xml(zip_path: Path) -> None:
    """Patch malformed Ship.xml files inside OpenSARShip_2.zip in-place."""
    print(f"[OpenSARShip 2] Fixing Ship.xml files in {zip_path.name}...")
    outer_src = zipfile.ZipFile(zip_path, "r")
    buf = io.BytesIO()
    fixed_count = 0

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as outer_dst:
        for item in outer_src.infolist():
            data = outer_src.read(item.filename)
            if item.filename.endswith(".zip"):
                try:
                    inner = zipfile.ZipFile(io.BytesIO(data))
                    needs_fix = "Ship.xml" in inner.namelist()
                    inner.close()
                except Exception:
                    needs_fix = False
                if needs_fix:
                    data = _patch_inner_zip(data)
                    fixed_count += 1
            outer_dst.writestr(item, data)

    outer_src.close()
    tmp = zip_path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    zip_path.unlink()
    tmp.rename(zip_path)
    print(f"  Fixed Ship.xml in {fixed_count} inner zip(s).")


# ---------------------------------------------------------------------------
# OpenSARShip extraction
# ---------------------------------------------------------------------------

def extract_patch_uint8(zip_path: Path, dataset_name: str, data_dir: Path) -> int:
    """
    Walk the nested zip structure and extract only Patch_Uint8 files.

    Both datasets share the same inner structure:
        {scene}/Patch_Uint8/{file}  (OpenSARShip_1)
        Patch_Uint8/{file}          (OpenSARShip_2)

    Extracts to: data/classification/{dataset_name}/{scene_name}/Patch_Uint8/{file}
    """
    out_base = data_dir / dataset_name
    outer = zipfile.ZipFile(zip_path)
    inner_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
    total_files = 0

    for zip_entry in inner_zips:
        scene_name = zip_entry.split("/")[-1].replace(".zip", "")
        scene_out = out_base / scene_name / "Patch_Uint8"

        inner = zipfile.ZipFile(io.BytesIO(outer.read(zip_entry)))
        patch_files = [
            f for f in inner.namelist()
            if "Patch_Uint8/" in f and not f.endswith("/")
        ]

        if not patch_files:
            print(f"  WARNING: no Patch_Uint8 files in {zip_entry.split('/')[-1]}")
            inner.close()
            continue

        scene_out.mkdir(parents=True, exist_ok=True)
        for fpath in patch_files:
            fname = fpath.split("/")[-1]
            (scene_out / fname).write_bytes(inner.read(fpath))
            total_files += 1

        inner.close()
        print(f"  {scene_name}: {len(patch_files)} files")

    outer.close()
    return total_files


def process_opensar_zips(zip_dir: Path, data_dir: Path):
    for zip_file in zip_dir.glob("**/*.zip"):
        fname = zip_file.name
        if fname not in OPENSAR_ZIPS:
            continue

        dataset_name = OPENSAR_ZIPS[fname]
        out_dir = data_dir / dataset_name

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[{dataset_name}] Already extracted, skipping.")
            continue

        if dataset_name == "OpenSARShip_2":
            fix_opensar2_xml(zip_file)
        print(f"\n[{dataset_name}] Extracting Patch_Uint8 files...")
        total = extract_patch_uint8(zip_file, dataset_name, data_dir)
        print(f"[{dataset_name}] Done — {total} files extracted to {out_dir}")


# ---------------------------------------------------------------------------
# FuSARShip extraction
# ---------------------------------------------------------------------------

def extract_fusar(zip_path: Path, data_dir: Path) -> int:
    """
    Extract all FuSARShip files to data_dir/FuSARShip/, stripping any single
    top-level root folder that the zip may contain (e.g. 'FuSARShip-Dataset 1.0/').
    """
    out_base = data_dir / "FuSARShip"
    total = 0

    with zipfile.ZipFile(zip_path) as zf:
        all_entries = [e for e in zf.namelist() if not e.endswith("/")]

        # Detect and strip a common top-level root folder if present
        top_dirs = {e.split("/")[0] for e in zf.namelist() if "/" in e}
        strip_prefix = top_dirs.pop() + "/" if len(top_dirs) == 1 else ""

        for entry in all_entries:
            rel = entry[len(strip_prefix):] if entry.startswith(strip_prefix) else entry
            if not rel:
                continue
            out_path = out_base / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(entry))
            total += 1

    print(f"  Extracted {total} files to {out_base}")
    return total


def process_fusar_zips(zip_dir: Path, data_dir: Path):
    zips = list(zip_dir.glob("**/*.zip"))
    if not zips:
        print(f"No zip files found in {zip_dir}")
        return
    for zip_file in zips:
        print(f"\n[FuSARShip] Extracting from {zip_file.name}...")
        total = extract_fusar(zip_file, data_dir)
        print(f"[FuSARShip] Done — {total} files extracted.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to extract datasets into (default: repo data/classification)",
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=None,
        help="Directory containing zip files to extract (skips Google Drive download for both datasets)",
    )
    parser.add_argument(
        "--fusar-url",
        type=str,
        default=FUSAR_FILE_URL,
        help="Google Drive URL for the FuSARShip dataset zip",
    )
    parser.add_argument(
        "--skip-opensar",
        action="store_true",
        help="Skip OpenSARShip 1 & 2 download/extraction",
    )
    parser.add_argument(
        "--skip-fusar",
        action="store_true",
        help="Skip FuSARShip download/extraction",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_dir:
        # Use pre-downloaded zips for both datasets
        print(f"Using zip files from {args.zip_dir}")
        if not args.skip_opensar:
            process_opensar_zips(args.zip_dir, data_dir)
        if not args.skip_fusar:
            fusar_out = data_dir / "FuSARShip"
            if fusar_out.exists() and any(fusar_out.iterdir()):
                print("[FuSARShip] Already extracted, skipping.")
            else:
                process_fusar_zips(args.zip_dir, data_dir)
    else:
        # Download from Google Drive
        if not args.skip_opensar:
            with tempfile.TemporaryDirectory() as tmp_dir:
                print("Downloading OpenSARShip files from Google Drive folder...")
                gdown.download_folder(OPENSAR_FOLDER_URL, output=tmp_dir, quiet=False, use_cookies=False)
                process_opensar_zips(Path(tmp_dir), data_dir)

        if not args.skip_fusar:
            fusar_out = data_dir / "FuSARShip"
            if fusar_out.exists() and any(fusar_out.iterdir()):
                print("[FuSARShip] Already extracted, skipping.")
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    print("\nDownloading FuSARShip from Google Drive...")
                    zip_path = Path(tmp_dir) / "FuSARShip.zip"
                    gdown.download(args.fusar_url, str(zip_path), quiet=False, fuzzy=True)
                    process_fusar_zips(Path(tmp_dir), data_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
