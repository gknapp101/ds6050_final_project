"""
Downloads OpenSARShip datasets from a shared Google Drive folder and extracts
only Patch_Uint8 images to save disk space. Raw zip files are deleted after
extraction.

Usage:
    pip install gdown
    python src/download_data.py
"""

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/19jLMSzHChVLk-vVAg2muNN2OALzksWob"

DATA_DIR = Path(__file__).parent.parent / "data" / "classification"

# Maps downloaded zip filename -> dataset extraction config
DATASET_CONFIG = {
    "OpenSARShip_1.zip": {
        "name": "OpenSARShip_1",
    },
    "OpenSARShip_2.zip": {
        "name": "OpenSARShip_2",
    },
}


def extract_patch_uint8(zip_path: Path, dataset_name: str) -> int:
    """
    Walk the nested zip structure and extract only Patch_Uint8 files.

    Both datasets share the same inner structure after fix:
        Patch_Uint8/{file}   (OpenSARShip_2)
        {scene}/Patch_Uint8/{file}  (OpenSARShip_1)

    Extracts to: data/classification/{dataset_name}/{scene_name}/Patch_Uint8/{file}
    """
    out_base = DATA_DIR / dataset_name
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


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download all files from the shared folder into a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Downloading files from Google Drive folder...")
        gdown.download_folder(FOLDER_URL, output=tmp_dir, quiet=False, use_cookies=False)

        for zip_file in Path(tmp_dir).glob("*.zip"):
            fname = zip_file.name
            if fname not in DATASET_CONFIG:
                print(f"Skipping unrecognised file: {fname}")
                continue

            cfg = DATASET_CONFIG[fname]
            out_dir = DATA_DIR / cfg["name"]

            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"[{cfg['name']}] Already extracted, skipping.")
                continue

            print(f"\n[{cfg['name']}] Extracting Patch_Uint8 files...")
            total = extract_patch_uint8(zip_file, cfg["name"])
            print(f"[{cfg['name']}] Done — {total} files extracted to {out_dir}")

    print("\nAll done. Temp files cleaned up.")


if __name__ == "__main__":
    main()
