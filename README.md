# DS6050 Final Project — SAR Ship Classification

Deep learning classification of ships from Sentinel-1 SAR imagery using the OpenSARShip dataset.

---

## Setup

```bash
pip install gdown pandas torch torchvision pillow
```

---

## Downloading the Data

Raw dataset archives are stored on Google Drive and are **not** included in the repository. Run the download script once to pull and extract the image patches:

```bash
python src/download_data.py
```

This will:
1. Download `OpenSARShip_1.zip` and `OpenSARShip_2.zip` from Google Drive
2. Extract only the `Patch_Uint8` image files into `data/classification/`
3. Delete the raw zip files automatically to save disk space

Extracted images land at:
```
data/classification/
├── OpenSARShip_1/{scene_name}/Patch_Uint8/Visual_*.tif
└── OpenSARShip_2/{scene_name}/Patch_Uint8/Visual_*.tif
```

To verify all images downloaded correctly:
```bash
python src/verify_paths.py
```

---

## Training Data Files

| File | Description |
|------|-------------|
| `data/classification/opensar1_labels.csv` | Labels for all OpenSARShip 1 patches |
| `data/classification/opensar2_labels.csv` | Labels for all OpenSARShip 2 patches |

Both CSVs are included in the repository. Combined they cover **30,210 image patches** across 85 Sentinel-1 scenes.

### CSV Columns

| Column | Description |
|--------|-------------|
| `path` | Relative path to the image file from the project root |
| `class_label` | Ship class name from the filename (e.g. `Cargo`, `Tanker`) |
| `ship_type` | Numeric AIS ship type code from the scene's `Ship.xml` |
| `polarization` | Polarization channel: `vh`, `vv`, `hh`, `hv`, or empty for single-pol scenes |
| `label` | Simplified 6-class label for training (see below) |

### Label Classes

| Label | AIS Codes | Count |
|-------|-----------|-------|
| `cargo` | 70–79 | 19,133 |
| `tanker` | 80–89 | 5,575 |
| `other` | 20–29, 30, 36–37, 40–57, 59, 90–99 | 1,260 |
| `unknown` | 0–19, 38–39, >99 | 3,328 |
| `engineering` | 31–34, 52–54, 58 | 737 |
| `passenger` | 60–69 | 177 |

> **Note:** `unknown` indicates the AIS transponder did not broadcast a valid ship type. These rows should be excluded from training.

### Loading the Data

```python
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent  # adjust as needed

df = pd.concat([
    pd.read_csv("data/classification/opensar1_labels.csv"),
    pd.read_csv("data/classification/opensar2_labels.csv"),
], ignore_index=True)

# Drop unknown labels before training
df = df[df["label"] != "unknown"]

# Resolve full path to an image
img_path = PROJECT_ROOT / df.iloc[0]["path"]
```

---

## Repository Structure

```
├── data/
│   └── classification/
│       ├── opensar1_labels.csv
│       └── opensar2_labels.csv
└── src/
    ├── download_data.py        # Download and extract images from Google Drive
    ├── verify_paths.py         # Check all CSV paths resolve to real files
    ├── make_opensar1_labels.py # Regenerate opensar1_labels.csv from source zip
    ├── make_opensar2_labels.py # Regenerate opensar2_labels.csv from source zip
    └── fix_opensar2_xml.py     # One-time fix for malformed Ship.xml files in OpenSARShip 2
```
