# DS6050 Final Project — SAR Ship Classification

Deep learning classification of ships from Sentinel-1 SAR imagery using OpenSARShip and FuSARShip datasets.

---

## Setup

```bash
pip install gdown pandas torch torchvision pillow scikit-learn wandb
```

---

## Downloading the Data

Raw dataset archives are stored on Google Drive and are **not** included in the repository. Run the download script once to pull and extract all datasets:

```bash
python src/download_data.py
```

This will:
1. Download `OpenSARShip_1.zip` and `OpenSARShip_2.zip` from a Google Drive folder
2. Patch malformed `Ship.xml` files in OpenSARShip 2 before extraction
3. Extract only the `Patch_Uint8` image files for OpenSARShip
4. Download and extract the full FuSARShip dataset

To download only one dataset:
```bash
python src/download_data.py --skip-fusar      # OpenSARShip only
python src/download_data.py --skip-opensar    # FuSARShip only
```

To use pre-downloaded zips instead of re-downloading:
```bash
python src/download_data.py --zip-dir /path/to/zips
```

---

## Generating Labels

After downloading, run the label script to generate CSVs and copy images into flat output folders:

```bash
python src/make_labels.py
```

This will:
1. Parse `Ship.xml` metadata for OpenSARShip 1 & 2 and assign AIS-based labels
2. Map FuSARShip folder names to labels
3. Assign unique IDs (`os1_XXXXX`, `os2_XXXXX`, `fs_XXXX`) and copy images to flat folders
4. Add a `google_path` column for Google Colab compatibility
5. Verify all output paths resolve to real files
6. Delete the original extracted source directories to free disk space

To skip the final cleanup:
```bash
python src/make_labels.py --skip-cleanup
```

To generate labels only without copying images:
```bash
python src/make_labels.py --skip-copy
```

---

## Label CSVs

| File | Dataset | Rows |
|------|---------|------|
| `data/classification/opensar1_labels.csv` | OpenSARShip 1 | 11,346 |
| `data/classification/opensar2_labels.csv` | OpenSARShip 2 | 18,864 |
| `data/classification/fusar_labels.csv` | FuSARShip | 5,101 |

All three CSVs share the following columns:

| Column | Description |
|--------|-------------|
| `*_id` | Unique image ID (`os1_XXXXX`, `os2_XXXXX`, or `fs_XXXX`) |
| `path` | Local relative path to the image from the project root |
| `google_path` | Absolute path for Google Colab (`/content/drive/MyDrive/...`) |
| `class_label` | Original class name from source data |
| `ship_type` | Numeric AIS ship type code (OpenSARShip only; empty for FuSARShip) |
| `polarization` | Polarization channel: `vh`, `vv`, `hh`, `hv`, or empty |
| `label` | Simplified 6-class label for training (see below) |
| `is_cargo` | Binary label: `1` if cargo, `0` otherwise |

### Label Classes

| Label | Source | AIS Codes / FuSARShip Folders |
|-------|--------|-------------------------------|
| `cargo` | Both | AIS 70–79 / `Cargo` |
| `tanker` | Both | AIS 80–89 / `Tanker` |
| `passenger` | Both | AIS 60–69 / `Passenger` |
| `engineering` | Both | AIS 31–34, 52–54, 58 / `Dredger`, `Tug`, `DiveVessel`, `SAR`, `PortTender` |
| `other` | Both | AIS 20–29, 30, 36–37, 40–57, 59, 90–99 / `Fishing`, `HighSpeedCraft`, `LawEnforce`, `Reserved`, `WingInGrnd`, `Other` |
| `unknown` | OpenSARShip | AIS codes not matching any category (excluded from FuSARShip by default) |

---

## Repository Structure

```
├── data/
│   └── classification/
│       ├── opensar1_labels.csv
│       ├── opensar2_labels.csv
│       ├── fusar_labels.csv
│       ├── os1/                  # Flat folder of OpenSARShip 1 images (os1_XXXXX.tif)
│       ├── os2/                  # Flat folder of OpenSARShip 2 images (os2_XXXXX.tif)
│       └── fs/                   # Flat folder of FuSARShip images (fs_XXXX.tiff)
└── src/
    ├── download_data.py          # Download and extract all datasets from Google Drive
    ├── make_labels.py            # Generate label CSVs, copy images, verify paths, cleanup
    ├── MultiModelTrain.ipynb     # Training notebook with stratified splits and wandb logging
    └── model_SAR.ipynb           # Google Colab inference/exploration notebook
```
