"""Full-reference LPIPS-VGG distance between de-id images and the original
balanced images, written to LMDB (one LMDB per de-id method).

Per-key value (pickled):
    {"lpips": float}

Iterates the de-id LMDB(s). For each key, looks up the matching key in
balanced.lmdb and computes LPIPS(adv, balanced).

Shapes are assumed to match per key (adv images are produced from balanced
via per-key perturbation). Batches are flushed whenever the shape changes
to avoid stack-shape errors.
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
import torch
import pyiqa

PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_DIR))


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


def decode_to_tensor(jpeg_bytes):
    img_bgr = cv2.imdecode(
        np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t


def run(args):
    lmdb_paths = args.lmdb_paths
    batch_size = args.batch_size
    log_interval = args.log_interval
    commit_every = args.commit_every
    map_size_gb = args.map_size_gb

    log("=" * 60)
    log("LPIPS-VGG Quality Metric (vs balanced) → LMDB")
    log(f"  Balanced: {args.balanced_lmdb}")
    log(f"  Adv paths: {len(lmdb_paths)} LMDB(s)")
    for p in lmdb_paths:
        log(f"    - {p}")
    log(f"  Output  : {args.output}")
    log(f"  Device  : {args.device}  batch={batch_size}")
    log("=" * 60)

    total = count_entries(lmdb_paths)
    log(f"Total adv entries: {total:,}")

    done_keys = load_done_keys(args.output) if args.resume else set()
    if done_keys:
        log(f"Resume: {len(done_keys):,} keys present, skipping")

    # Open balanced once (random access lookups)
    bal_env = lmdb.open(args.balanced_lmdb, readonly=True, lock=False,
                        readahead=False, meminit=False,
                        map_size=700 * 1024 ** 3)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_env = lmdb.open(args.output, map_size=map_size_gb * 1024 ** 3,
                        subdir=True)

    log("Loading LPIPS-VGG…")
    device = torch.device(args.device)
    metric = pyiqa.create_metric('lpips-vgg', device=device)
    log(f"  LPIPS-VGG ready on {device}")

    n_processed = len(done_keys)
    n_ok_run = 0
    n_skip_run = 0
    pending = []
    batch_keys = []
    batch_bal = []
    batch_adv = []
    batch_shape = None
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
        nonlocal n_ok_run, batch_shape
        if not batch_keys:
            return
        a = torch.stack(batch_bal).to(device)
        b = torch.stack(batch_adv).to(device)
        with torch.no_grad():
            scores = metric(a, b)
        scores = scores.view(-1).float().cpu().numpy()
        for k, s in zip(batch_keys, scores):
            pending.append((k, {"lpips": float(s)}))
            n_ok_run += 1
        batch_keys.clear()
        batch_bal.clear()
        batch_adv.clear()
        batch_shape = None
        if len(pending) >= commit_every:
            flush_pending()

    for key_str, rec in iter_lmdb(lmdb_paths):
        if key_str in done_keys:
            continue
        n_processed += 1

        # Lookup balanced counterpart
        with bal_env.begin() as txn:
            bal_val = txn.get(key_str.encode())
        if bal_val is None:
            n_skip_run += 1
        else:
            bal_rec = pickle.loads(bal_val)
            t_bal = decode_to_tensor(bal_rec['image'])
            t_adv = decode_to_tensor(rec['image'])
            if (t_bal is None) or (t_adv is None) \
                    or (t_bal.shape != t_adv.shape):
                n_skip_run += 1
            else:
                shp = tuple(t_bal.shape)
                if (batch_shape is not None and shp != batch_shape) \
                        or len(batch_keys) >= batch_size:
                    flush_batch()
                if batch_shape is None:
                    batch_shape = shp
                batch_keys.append(key_str)
                batch_bal.append(t_bal)
                batch_adv.append(t_adv)

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
    bal_env.close()

    elapsed = time.time() - t_start
    log("-" * 60)
    log(f"Complete: processed={n_processed:,}, "
        f"new_ok={n_ok_run:,}, new_skip={n_skip_run:,}")
    log(f"  Time: {elapsed/60:.1f} min")
    log(f"Done. Output: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--balanced-lmdb', required=True)
    p.add_argument('--lmdb-paths', nargs='+', required=True,
                   help='De-id LMDB path(s).')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--log-interval', type=int, default=5000)
    p.add_argument('--commit-every', type=int, default=500)
    p.add_argument('--map-size-gb', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
