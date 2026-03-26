"""
Fixes three bugs in Ship.xml files inside OpenSARShip_2.zip:

1. <\ShipList> closing tag uses backslash instead of forward slash
   -> replace with </ShipList>

2. Bare < immediately before a closing tag in any element
   e.g. "JIA GANG TUO 8    <</Name>"  or  "AR][D_<</Callsign>"
   -> escape trailing < as &lt;

3. <! inside element text that is NOT a comment (<!--) or CDATA (<![CDATA[)
   e.g. "<!VI5" inside <Callsign>
   -> escape < as &lt;
"""

import io
import re
import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "classification"
SRC_ZIP = DATA_DIR / "OpenSARShip_2.zip"
TMP_ZIP = DATA_DIR / "OpenSARShip_2_fixed.zip"

BACKSLASH_CLOSE = ("<" + chr(92) + "ShipList>").encode()
FORWARD_CLOSE = b"</ShipList>"

# Matches a bare < immediately before any closing tag
# e.g. "JIA GANG TUO 8    <</Name>"  or  "AR][D_<</Callsign>"
BARE_LT_BEFORE_CLOSE = re.compile(rb"([^<])\s*<(</\w+>)")

# Matches <! that is NOT a valid XML comment (<!--) or CDATA (<![CDATA[)
INVALID_BANG = re.compile(rb"<!(?!--|" + rb"\[CDATA\[)")


def fix_ship_xml(content: bytes) -> tuple[bytes, list[str]]:
    """Apply all fixes to raw Ship.xml bytes. Returns (fixed_bytes, list_of_changes)."""
    changes = []

    # Fix 1: backslash closing tag  <\ShipList> -> </ShipList>
    count = content.count(BACKSLASH_CLOSE)
    if count:
        content = content.replace(BACKSLASH_CLOSE, FORWARD_CLOSE)
        changes.append(f"replaced {count} occurrence(s) of <\\ShipList> -> </ShipList>")

    # Fix 2: bare < immediately before a closing tag in any element
    def escape_bare_lt(m: re.Match) -> bytes:
        return m.group(1) + b"&lt;" + m.group(2)

    fixed, n = BARE_LT_BEFORE_CLOSE.subn(escape_bare_lt, content)
    if n:
        content = fixed
        changes.append(f"escaped {n} bare '<' before closing tag(s)")

    # Fix 3: <! that is not <!-- or <![CDATA[  (e.g. "<!VI5" in callsign text)
    fixed, n = INVALID_BANG.subn(b"&lt;!", content)
    if n:
        content = fixed
        changes.append(f"escaped {n} invalid '<!' sequence(s) in text content")

    return content, changes


def patch_inner_zip(inner_bytes: bytes) -> tuple[bytes, list[str]]:
    """Re-package an inner zip with a fixed Ship.xml. Returns (new_zip_bytes, changes)."""
    src = zipfile.ZipFile(io.BytesIO(inner_bytes), "r")
    buf = io.BytesIO()
    all_changes = []

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "Ship.xml":
                data, changes = fix_ship_xml(data)
                all_changes.extend(changes)
            dst.writestr(item, data)

    src.close()
    return buf.getvalue(), all_changes


def main():
    print(f"Reading {SRC_ZIP.name} ...")

    outer_src = zipfile.ZipFile(SRC_ZIP, "r")
    buf = io.BytesIO()
    total_changes = 0

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as outer_dst:
        for item in outer_src.infolist():
            data = outer_src.read(item.filename)

            if item.filename.endswith(".zip"):
                # Check if Ship.xml is inside and needs fixing
                try:
                    inner = zipfile.ZipFile(io.BytesIO(data))
                    needs_fix = "Ship.xml" in inner.namelist()
                    inner.close()
                except Exception:
                    needs_fix = False

                if needs_fix:
                    data, changes = patch_inner_zip(data)
                    if changes:
                        total_changes += 1
                        name = item.filename.split("/")[-1]
                        print(f"  Fixed {name}:")
                        for c in changes:
                            print(f"    - {c}")

            outer_dst.writestr(item, data)

    outer_src.close()

    # Write to temp then replace original
    TMP_ZIP.write_bytes(buf.getvalue())
    SRC_ZIP.unlink()
    TMP_ZIP.rename(SRC_ZIP)

    print(f"\nDone. Fixed {total_changes} inner zip(s). Output: {SRC_ZIP.name}")


if __name__ == "__main__":
    main()
