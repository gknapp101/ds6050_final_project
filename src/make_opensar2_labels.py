"""
Creates a CSV of ship type labels for every patch file in Patch_Uint8 across all
inner zips of OpenSARShip_2.zip.

Row order matches the sorted Patch_Uint8 file order within each scene, with scenes
processed in sorted zip-name order (same order the files appear in the outer zip).

Output columns:
  path        - relative path to the image from the project root
  class_label - ship class parsed from filename (e.g. Cargo)
  ship_type   - numeric AIS Ship_Type code from Ship.xml
  polarization- vh, vv, hh, hv, or empty for single-pol scenes
  label       - simplified 6-class label (cargo/tanker/passenger/engineering/other/unknown)
"""

import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "classification"
SRC_ZIP = DATA_DIR / "OpenSARShip_2.zip"
OUT_CSV = DATA_DIR / "opensar2_labels.csv"

# Polarization suffix (vh/vv/hh/hv) is optional — single-pol scenes omit it
PATCH_RE = re.compile(r"^Visual_(.+)_x(\d+)_y(\d+)(?:_(vh|vv|hh|hv))?\.tif$", re.IGNORECASE)

# AIS ship type codes per ITU-R M.1371-5 / IALA guidelines
AIS_SHIP_TYPES: dict[int, str] = {
    0: "Not available",
    1: "Reserved",
    2: "Reserved",
    3: "Reserved",
    4: "Reserved",
    5: "Reserved",
    6: "Reserved",
    7: "Reserved",
    8: "Reserved",
    9: "Reserved",
    10: "Reserved",
    11: "Reserved",
    12: "Reserved",
    13: "Reserved",
    14: "Reserved",
    15: "Reserved",
    16: "Reserved",
    17: "Reserved",
    18: "Reserved",
    19: "Reserved",
    20: "Wing in Ground",
    21: "Wing in Ground - Hazardous category A",
    22: "Wing in Ground - Hazardous category B",
    23: "Wing in Ground - Hazardous category C",
    24: "Wing in Ground - Hazardous category D",
    25: "Wing in Ground - Reserved",
    26: "Wing in Ground - Reserved",
    27: "Wing in Ground - Reserved",
    28: "Wing in Ground - Reserved",
    29: "Wing in Ground - No additional information",
    30: "Fishing",
    31: "Towing",
    32: "Towing - length >200m or breadth >25m",
    33: "Dredging or underwater ops",
    34: "Diving ops",
    35: "Military ops",
    36: "Sailing",
    37: "Pleasure Craft",
    38: "Reserved",
    39: "Reserved",
    40: "High speed craft",
    41: "High speed craft - Hazardous category A",
    42: "High speed craft - Hazardous category B",
    43: "High speed craft - Hazardous category C",
    44: "High speed craft - Hazardous category D",
    45: "High speed craft - Reserved",
    46: "High speed craft - Reserved",
    47: "High speed craft - Reserved",
    48: "High speed craft - Reserved",
    49: "High speed craft - No additional information",
    50: "Pilot Vessel",
    51: "Search and Rescue vessel",
    52: "Tug",
    53: "Port Tender",
    54: "Anti-pollution equipment",
    55: "Law Enforcement",
    56: "Spare - Local Vessel",
    57: "Spare - Local Vessel",
    58: "Medical Transport",
    59: "Noncombatant ship",
    60: "Passenger",
    61: "Passenger - Hazardous category A",
    62: "Passenger - Hazardous category B",
    63: "Passenger - Hazardous category C",
    64: "Passenger - Hazardous category D",
    65: "Passenger - Reserved",
    66: "Passenger - Reserved",
    67: "Passenger - Reserved",
    68: "Passenger - Reserved",
    69: "Passenger - No additional information",
    70: "Cargo",
    71: "Cargo - Hazardous category A",
    72: "Cargo - Hazardous category B",
    73: "Cargo - Hazardous category C",
    74: "Cargo - Hazardous category D",
    75: "Cargo - Reserved",
    76: "Cargo - Reserved",
    77: "Cargo - Reserved",
    78: "Cargo - Reserved",
    79: "Cargo - No additional information",
    80: "Tanker",
    81: "Tanker - Hazardous category A",
    82: "Tanker - Hazardous category B",
    83: "Tanker - Hazardous category C",
    84: "Tanker - Hazardous category D",
    85: "Tanker - Reserved",
    86: "Tanker - Reserved",
    87: "Tanker - Reserved",
    88: "Tanker - Reserved",
    89: "Tanker - No additional information",
    90: "Other Type",
    91: "Other Type - Hazardous category A",
    92: "Other Type - Hazardous category B",
    93: "Other Type - Hazardous category C",
    94: "Other Type - Hazardous category D",
    95: "Other Type - Reserved",
    96: "Other Type - Reserved",
    97: "Other Type - Reserved",
    98: "Other Type - Reserved",
    99: "Other Type - No additional information",
}


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


def main():
    outer = zipfile.ZipFile(SRC_ZIP, "r")
    inner_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))

    rows = []
    missing = 0

    for zip_path in inner_zips:
        scene = zip_path.split("/")[-1]
        inner = zipfile.ZipFile(io.BytesIO(outer.read(zip_path)))

        # Build coordinate -> ship_type lookup from XML
        if "Ship.xml" not in inner.namelist():
            print(f"  WARNING: no Ship.xml in {scene}, skipping")
            continue
        coord_to_type = parse_xml_ship_types(inner.read("Ship.xml"))

        # Collect and sort Patch_Uint8 files
        patch_files = sorted(
            f for f in inner.namelist()
            if f.startswith("Patch_Uint8/") and not f.endswith("/")
        )

        for fpath in patch_files:
            fname = fpath.split("/")[-1]
            m = PATCH_RE.match(fname)
            if not m:
                print(f"  WARNING: unexpected filename format: {fname}")
                continue

            class_label, x_str, y_str, pol = m.groups()
            key = (int(x_str), int(y_str))
            ship_type = coord_to_type.get(key, "")
            if not ship_type:
                missing += 1

            scene_name = scene.replace(".zip", "")
            path = f"data/classification/OpenSARShip_2/{scene_name}/Patch_Uint8/{fname}"
            rows.append({
                "path": path,
                "class_label": class_label,
                "ship_type": ship_type,
                "polarization": pol.lower() if pol else "",
                "label": ais_label(ship_type),
            })

        inner.close()

    outer.close()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class_label", "ship_type", "polarization", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV.name}")
    if missing:
        print(f"WARNING: {missing} files had no matching XML entry")


if __name__ == "__main__":
    main()
