"""Gaze estimation with L2CS-Net, writing per-key predictions to LMDB.

Reuses the cached RetinaFace detections at detections.json for bbox cropping
(de-id counterparts use the original image's bbox — consistent alignment
across methods).

Per-key value (pickled):
    {
      "yaw":        float,   # radians
      "pitch":      float,   # radians
      "yaw_conf":   float,   # max softmax bin probability
      "pitch_conf": float,   # max softmax bin probability
    }
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_DIR))
from predictors.l2cs import L2CSPredictor, DEFAULT_WEIGHT_PATH  # noqa: E402


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def iter_lmdb(lmdb_paths):
    for path in lmdb_paths:
        env = lmdb.open(path, readonly=True, lock=False,
                        readahead=False, meminit=False,
                        map_size=700 * 1024 ** 3)
        with env.begin() as txn:
            for key, val in txn.cursor():
                yield key.decode('utf-8'), pickle.loads(val)
        env.close()


def count_entries(lmdb_paths):
    n = 0
    for p in lmdb_paths:
        env = lmdb.open(p, readonly=True, lock=False,
                        readahead=False, meminit=False)
        n += env.stat()['entries']
        env.close()
    return n


def load_done_keys(out_path):
    if not os.path.exists(out_path):
        return set()
    env = lmdb.open(out_path, readonly=True, lock=False, readahead=False)
    keys = set()
    with env.begin() as txn:
        for k, _ in txn.cursor():
            keys.add(k.decode('utf-8'))
    env.close()
    return keys


def crop_with_bbox(img_bgr, bbox):
    """bbox = [x1, y1, x2, y2] (RetinaFace)."""
    H, W = img_bgr.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2])); y2 = min(H, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return img_bgr
    return img_bgr[y1:y2, x1:x2]


def _batched_gaze(predictor, batch_bgr):
    """Returns (yaws_rad, pitches_rad, yaw_conf, pitch_conf) — numpy arrays."""
    tensors = [predictor.preprocess(c) for c in batch_bgr]
    batch = torch.stack(tensors).to(predictor.device)

    with torch.no_grad():
        yaw_logits, pitch_logits = predictor.model(batch)
        yaw_probs = F.softmax(yaw_logits, dim=1)
        pitch_probs = F.softmax(pitch_logits, dim=1)
        idx = predictor.idx_tensor
        yaw_deg = torch.sum(yaw_probs * idx, dim=1) * 4 - 180
        pitch_deg = torch.sum(pitch_probs * idx, dim=1) * 4 - 180
        yaw_rad = yaw_deg * math.pi / 180.0
        pitch_rad = pitch_deg * math.pi / 180.0
        yaw_conf = torch.max(yaw_probs, dim=1).values
        pitch_conf = torch.max(pitch_probs, dim=1).values

    return (yaw_rad.cpu().numpy(), pitch_rad.cpu().numpy(),
            yaw_conf.cpu().numpy(), pitch_conf.cpu().numpy())


def run(args):
    lmdb_paths = args.lmdb_paths
    batch_size = args.batch_size
    log_interval = args.log_interval
    commit_every = args.commit_every
    map_size_gb = args.map_size_gb

    log("=" * 60)
    log("L2CS-Net Gaze Extraction → LMDB")
    log(f"  Inputs  : {len(lmdb_paths)} LMDB(s)")
    for p in lmdb_paths:
        log(f"    - {p}")
    log(f"  Detect  : {args.detections}")
    log(f"  Weights : {args.weights}")
    log(f"  Output  : {args.output}")
    log(f"  Device  : {args.device}  batch={batch_size}")
    log("=" * 60)

    log("Loading detections…")
    with open(args.detections, 'r') as f:
        detections = json.load(f)
    log(f"  {len(detections):,} detections loaded")

    total = count_entries(lmdb_paths)
    log(f"Total input entries: {total:,}")

    done_keys = load_done_keys(args.output) if args.resume else set()
    if done_keys:
        log(f"Resume: {len(done_keys):,} keys already present, skipping")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_env = lmdb.open(args.output, map_size=map_size_gb * 1024 ** 3,
                        subdir=True)

    log("Loading L2CS-Net…")
    predictor = L2CSPredictor(weight_path=args.weights, device=args.device)
    log(f"  L2CS ready on {args.device}")

    n_processed = len(done_keys)
    n_ok_run, n_skip_run = 0, 0
    pending = []
    batch_keys, batch_crops = [], []
    t_start = time.time()

    def flush_pending():
        if not pending:
            return
        with out_env.begin(write=True) as txn:
            for k, v in pending:
                txn.put(k.encode('utf-8'),
                        pickle.dumps(v, pickle.HIGHEST_PROTOCOL))
        pending.clear()

    def flush_batch():
        nonlocal n_ok_run
        if not batch_keys:
            return
        yaws, pitches, yconf, pconf = _batched_gaze(predictor, batch_crops)
        for i, k in enumerate(batch_keys):
            pending.append((k, {
                "yaw": float(yaws[i]),
                "pitch": float(pitches[i]),
                "yaw_conf": float(yconf[i]),
                "pitch_conf": float(pconf[i]),
            }))
            n_ok_run += 1
        batch_keys.clear()
        batch_crops.clear()
        if len(pending) >= commit_every:
            flush_pending()

    for key_str, rec in iter_lmdb(lmdb_paths):
        if key_str in done_keys:
            continue
        n_processed += 1

        det = detections.get(key_str)
        if det is None:
            n_skip_run += 1
        else:
            img_bgr = cv2.imdecode(
                np.frombuffer(rec['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                n_skip_run += 1
            else:
                crop = crop_with_bbox(img_bgr, det['bbox'])
                if crop.size == 0:
                    n_skip_run += 1
                else:
                    batch_keys.append(key_str)
                    batch_crops.append(crop)
                    if len(batch_keys) >= batch_size:
                        flush_batch()

        if n_processed % log_interval == 0:
            elapsed = time.time() - t_start
            done_this = n_ok_run + n_skip_run
            rate = done_this / max(elapsed, 1)
            remaining = max(total - n_processed, 0)
            eta_min = remaining / max(rate, 1) / 60
            pct = 100.0 * n_processed / max(total, 1)
            log(f"  {n_processed:>7,}/{total:,} ({pct:5.1f}%)  "
                f"ok+={n_ok_run:,}  skip+={n_skip_run:,}  "
                f"{rate:.1f} img/s  ETA {eta_min:.1f} min")

    flush_batch()
    flush_pending()
    out_env.close()

    elapsed = time.time() - t_start
    log("-" * 60)
    log(f"Complete: processed={n_processed:,}, "
        f"new_ok={n_ok_run:,}, new_skip={n_skip_run:,}")
    log(f"  Time: {elapsed/60:.1f} min")
    log(f"Done. Output: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lmdb-paths', nargs='+', required=True)
    p.add_argument('--detections', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--weights', default=DEFAULT_WEIGHT_PATH)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--log-interval', type=int, default=5000)
    p.add_argument('--commit-every', type=int, default=500)
    p.add_argument('--map-size-gb', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
