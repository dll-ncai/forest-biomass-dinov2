#!/usr/bin/env python3
"""
Download the large aerial source GeoTIFFs that are not tracked in git.

The bundle contains the two high-resolution aerial RGB rasters (Balakot, Pakistan
and Karlsruhe, Germany). They are hosted on Google Drive and are needed only for:
  * rebuilding the German dataset from scratch
    (german_data_results/build_complete_pipeline.py, which reads german data/karlsruhe.tif)
  * regenerating the aerial panels of Figure 4
    (paper_writeup/create_biomass_raster_figure.py)

All committed CSVs/JSONs are already sufficient to reproduce every numeric result
and every figure that is built from the CSVs. The predicted-biomass output rasters
are not distributed (regenerable outputs); the rendered Figure 4 is committed under
paper_writeup/, so you only need this bundle for the two scripts above.

Usage
-----
    pip install gdown            # one-time
    python download_data.py

The archive is downloaded and extracted in place, so each raster lands at the
exact relative path the scripts expect. Re-running skips files that already exist
and match the expected size (use --force to re-download).
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile

# ── Google Drive file id for the data bundle (a single .zip) ──────────────────
# Set after uploading forest-biomass-dinov2-data.zip to Drive (share: "Anyone
# with the link"). Paste ONLY the id portion of the share URL here, e.g. for
#   https://drive.google.com/file/d/1AbC.../view?usp=sharing
# the id is  1AbC...
DRIVE_FILE_ID = "PASTE_DRIVE_FILE_ID_HERE"

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "data_manifest.json")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def already_present(files):
    """Return True if every expected file exists with the right size."""
    for rel, meta in files.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or os.path.getsize(p) != meta["size"]:
            return False
    return True


def verify(files, check_hash=False):
    ok = True
    for rel, meta in files.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"  MISSING  {rel}")
            ok = False
            continue
        size = os.path.getsize(p)
        if size != meta["size"]:
            print(f"  BAD SIZE {rel}  ({human(size)} != {human(meta['size'])})")
            ok = False
            continue
        if check_hash and meta.get("sha256"):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != meta["sha256"]:
                print(f"  BAD HASH {rel}")
                ok = False
                continue
        print(f"  OK       {rel}  ({human(size)})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if files already exist")
    ap.add_argument("--verify", action="store_true",
                    help="only verify existing files (checks sha256)")
    ap.add_argument("--keep-zip", action="store_true",
                    help="do not delete the downloaded archive after extraction")
    args = ap.parse_args()

    manifest = load_manifest()
    files = manifest["files"]

    if args.verify:
        print("Verifying data files (sha256):")
        sys.exit(0 if verify(files, check_hash=True) else 1)

    if not args.force and already_present(files):
        print("All data files already present. Nothing to do (use --force to re-download).")
        verify(files)
        return

    file_id = manifest.get("drive_file_id") or DRIVE_FILE_ID
    if not file_id or file_id == "PASTE_DRIVE_FILE_ID_HERE":
        sys.exit("ERROR: Google Drive file id is not set. Edit DRIVE_FILE_ID in "
                 "download_data.py (or data_manifest.json).")

    try:
        import gdown
    except ImportError:
        sys.exit("ERROR: gdown is required. Install it with:  pip install gdown")

    zip_path = os.path.join(ROOT, manifest["archive_name"])
    print(f"Downloading {manifest['archive_name']} (~{manifest['archive_size_human']}) "
          f"from Google Drive ...")
    gdown.download(id=file_id, output=zip_path, quiet=False)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ROOT)

    if not args.keep_zip:
        os.remove(zip_path)

    print("\nVerifying:")
    ok = verify(files)
    print("\nDone." if ok else "\nWARNING: verification found problems (see above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
