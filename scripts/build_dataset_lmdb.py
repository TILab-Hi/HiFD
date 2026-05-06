"""Pack a folder of images into a balanced.lmdb-compatible LMDB.

Walks --input-dir recursively, re-encodes every image as JPEG, and writes
to LMDB with the schema expected by all downstream extractors:

    per-key value (pickled):
        {"image": jpeg_bytes}

Keys are the relative path from --input-dir with the extension replaced by
'.jpg', e.g. 'sub0008_01/00000.jpg'.

Usage:
    python build_dataset_lmdb.py \\
        --input-dir /path/to/images \\
        --output /path/to/out.lmdb \\
        --map-size-gb 100
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import lmdb
import numpy as np


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(input_dir: str):
    """Collect all image paths under input_dir, sorted."""
    paths = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        for fname in sorted(files):
            if Path(fname).suffix.lower() in _IMAGE_EXTS:
                paths.append(os.path.join(root, fname))
    return paths


def main():
    ap = argparse.ArgumentParser(
        description="Pack an image folder into LMDB with {image: jpeg_bytes} schema."
    )
    ap.add_argument("--input-dir", required=True,
                    help="Root folder containing images (searched recursively).")
    ap.add_argument("--output", required=True,
                    help="Output LMDB directory path.")
    ap.add_argument("--map-size-gb", type=int, default=100,
                    help="LMDB map size in GB. Default: 100.")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="JPEG re-encode quality (1-100). Default: 95.")
    ap.add_argument("--commit-every", type=int, default=500,
                    help="Commit transaction every N writes. Default: 500.")
    ap.add_argument("--log-every", type=int, default=2000,
                    help="Log progress every N images. Default: 2000.")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        log(f"ERROR: --input-dir not found: {args.input_dir}")
        sys.exit(1)

    if os.path.exists(args.output):
        log(f"Output already exists: {args.output} — refusing to overwrite.")
        sys.exit(2)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)

    log(f"Scanning {args.input_dir} …")
    paths = collect_images(args.input_dir)
    total = len(paths)
    if total == 0:
        log("ERROR: No images found.")
        sys.exit(3)
    log(f"Found {total:,} images.")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    input_root = str(Path(args.input_dir).resolve())

    env = lmdb.open(
        args.output,
        map_size=args.map_size_gb * (1024 ** 3),
        subdir=True,
        meminit=False,
        map_async=True,
        writemap=True,
    )

    t0 = time.time()
    written = 0
    failed = 0
    txn = env.begin(write=True)
    try:
        for img_path in paths:
            # Derive key: relative path with .jpg extension
            rel = os.path.relpath(img_path, input_root)
            stem = os.path.splitext(rel)[0]
            key = stem + ".jpg"

            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                failed += 1
                continue
            ok, enc = cv2.imencode(".jpg", img_bgr, encode_params)
            if not ok:
                failed += 1
                continue
            jpeg_bytes = enc.tobytes()
            entry = {"image": jpeg_bytes}
            txn.put(key.encode("utf-8"), pickle.dumps(entry))
            written += 1

            if written % args.commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)

            if written % args.log_every == 0:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                eta = (total - written) / max(rate, 1e-6)
                log(
                    f"  written={written:,}/{total:,}  failed={failed}  "
                    f"rate={rate:.1f} img/s  ETA={eta/60:.1f} min"
                )
        txn.commit()
    except Exception:
        txn.abort()
        raise

    env.sync()
    env.close()

    elapsed = time.time() - t0
    log(
        f"Done. written={written:,}  failed={failed}  "
        f"elapsed={elapsed/60:.1f} min  → {args.output}"
    )


if __name__ == "__main__":
    main()
