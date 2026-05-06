"""Cache RetinaFace detections from an LMDB dataset to JSON.

This artifact (detections.json) is reused by all crop-dependent extractors
(age/gender, ethnicity, face-embedding) so cropping is identical across all
sources — including de-identified images that the detector would otherwise
fail on.

Output: {sample_id: {"bbox":[x1,y1,x2,y2], "landmarks":[5,2], "confidence":f}}
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import lmdb
import numpy as np

PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_DIR))

from predictors import FaceDetector  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb-path", required=True)
    ap.add_argument("--retinaface-weights", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--log-interval", type=int, default=5000)
    args = ap.parse_args()

    detector = FaceDetector(
        model_path=args.retinaface_weights,
        network="resnet50",
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False,
                    map_size=700 * 1024 ** 3)
    total = env.stat()["entries"]
    log(f"Detecting on {total:,} entries")

    detections = {}
    n_det, n_fail = 0, 0
    t0 = time.time()

    with env.begin() as txn:
        for i, (k, v) in enumerate(txn.cursor()):
            ks = k.decode("utf-8")
            rec = pickle.loads(v)
            img = cv2.imdecode(np.frombuffer(rec["image"], dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                n_fail += 1
                continue
            res = detector.detect(img)
            if res:
                d = res[0]
                detections[ks] = {
                    "bbox": d.bbox.tolist(),
                    "landmarks": d.landmarks.tolist(),
                    "confidence": float(d.confidence),
                }
                n_det += 1
            else:
                n_fail += 1
            if (i + 1) % args.log_interval == 0:
                rate = (i + 1) / max(time.time() - t0, 1)
                log(f"  {i+1:,}/{total:,}  det={n_det:,}  fail={n_fail:,}  "
                    f"{rate:.1f} img/s")

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(detections, f)
    os.replace(tmp, args.output)
    log(f"Wrote {n_det:,} detections to {args.output} "
        f"({n_fail:,} failures)")


if __name__ == "__main__":
    main()
