"""Facial (macro) expression recognition with POSTER, writing per-key
predictions to a single LMDB per source.

Reuses the cached RetinaFace detections at detections.json for bbox cropping
(de-id counterparts use the original image's bbox — consistent alignment
across methods).

Per-key value (pickled):
    {
      "logits":     np.ndarray(C,) float32,   # raw model output
      "probs":      np.ndarray(C,) float32,   # softmax
      "pred_class": int,                      # argmax
      "class_name": str,                      # e.g. "Happy"
    }

Keys with no cached detection or failed decode are omitted (sparse output).
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
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_DIR))
from predictors.poster import load_poster_model, EXPRESSION_LABELS_7CLASS  # noqa: E402


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


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


def run(args):
    lmdb_paths = args.lmdb_paths
    batch_size = args.batch_size
    log_interval = args.log_interval
    commit_every = args.commit_every
    map_size_gb = args.map_size_gb
    num_classes = args.num_classes

    log("=" * 60)
    log("POSTER Macro-Expression Recognition → LMDB")
    log(f"  Inputs    : {len(lmdb_paths)} LMDB(s)")
    for p in lmdb_paths:
        log(f"    - {p}")
    log(f"  Detect    : {args.detections}")
    log(f"  Weights   : {args.weights}")
    log(f"  Output    : {args.output}")
    log(f"  Device    : {args.device}  batch={batch_size}")
    log(f"  Classes   : {num_classes}")
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

    device = torch.device(args.device)
    log("Loading POSTER…")
    model = load_poster_model(
        num_classes=num_classes,
        model_type=args.model_type,
        device=device,
        checkpoint_path=args.weights,
    )
    log(f"  POSTER ready on {device}")

    # Class names (default to 7-class RAF-DB labels; otherwise numeric fallback)
    if num_classes == len(EXPRESSION_LABELS_7CLASS):
        class_names = list(EXPRESSION_LABELS_7CLASS)
    else:
        class_names = [str(i) for i in range(num_classes)]

    n_processed = len(done_keys)
    n_ok_run, n_skip_run = 0, 0
    pending = []
    batch_keys, batch_tensors = [], []
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
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits, _ = model(x)
            probs = F.softmax(logits, dim=1)
        logits_np = logits.float().cpu().numpy()
        probs_np = probs.float().cpu().numpy()
        preds = probs_np.argmax(axis=1)
        for i, k in enumerate(batch_keys):
            pending.append((k, {
                "logits": logits_np[i].astype(np.float32),
                "probs": probs_np[i].astype(np.float32),
                "pred_class": int(preds[i]),
                "class_name": class_names[int(preds[i])],
            }))
            n_ok_run += 1
        batch_keys.clear()
        batch_tensors.clear()
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
                try:
                    crop = crop_with_bbox(img_bgr, det['bbox'])
                    if crop.size == 0:
                        n_skip_run += 1
                    else:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        t = _preprocess(crop_rgb)
                        batch_keys.append(key_str)
                        batch_tensors.append(t)
                        if len(batch_keys) >= batch_size:
                            flush_batch()
                except Exception:
                    n_skip_run += 1

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
    p.add_argument('--weights', required=True)
    p.add_argument('--num-classes', type=int, default=7)
    p.add_argument('--model-type', default='large',
                   choices=['small', 'base', 'large'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--log-interval', type=int, default=5000)
    p.add_argument('--commit-every', type=int, default=500)
    p.add_argument('--map-size-gb', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
