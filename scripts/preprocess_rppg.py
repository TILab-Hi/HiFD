"""Preprocess per-frame PURE-style LMDB(s) into the rPPG-ready LMDB schema
expected by LMDBDataset.

Two subcommands:
    cache_bboxes  — run RetinaFace on first frame of each subject in the
                    original pure.lmdb and save enlarged bboxes to JSON.
    build         — for a given source (1 LMDB for original, 8 shards for
                    de-id), read frames, crop with cached bbox, resize to
                    72×72, pull BVP labels from original PURE JSON, and
                    write rPPG-ready LMDB.

Output LMDB schema (one LMDB per source):
    __subjects__  : JSON list of subject IDs
    __nframes__   : JSON dict {sid: T}
    __dataset__   : "PURE"
    {sid}_frames  : np.save(uint8 (T, 72, 72, 3))
    {sid}_labels  : np.save(float64 (T,))   # BVP from original PURE JSON
    {sid}_nframes : str(T)

External dependency:
    Set RPPG_REPO env var to your cloned rPPG-Toolbox checkout.
    (Only required at runtime, not for --help.)
"""

import argparse
import io
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


LARGER_COEF = 1.5          # matches pure.yaml LARGE_BOX_COEF
RESIZE_H, RESIZE_W = 72, 72


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def subj_from_key(key_str: str) -> str:
    """'01-01/ImageNNN.jpg' → '0101'."""
    return key_str.split('/', 1)[0].replace('-', '')


def trial_from_sid(sid: str) -> str:
    """'0101' → '01-01'."""
    return f"{sid[:2]}-{sid[2:]}"


def numpy_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def enlarge_bbox(x, y, w, h):
    new_x = max(0, int(x - (LARGER_COEF - 1.0) / 2 * w))
    new_y = max(0, int(y - (LARGER_COEF - 1.0) / 2 * h))
    new_w = int(LARGER_COEF * w)
    new_h = int(LARGER_COEF * h)
    return [new_x, new_y, new_w, new_h]


def build_key_index(lmdb_paths):
    """Return {sid: [(key_str, shard_idx), ...]} sorted by key_str."""
    per_subj = {}
    for shard_idx, path in enumerate(lmdb_paths):
        env = lmdb.open(path, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn:
            for k, _ in txn.cursor():
                key_str = k.decode('utf-8')
                sid = subj_from_key(key_str)
                per_subj.setdefault(sid, []).append((key_str, shard_idx))
        env.close()
    for sid in per_subj:
        per_subj[sid].sort(key=lambda x: x[0])
    return per_subj


def load_bvp_for_trial(pure_data_path, trial_name, n_frames):
    """Read BVP JSON and resample to n_frames."""
    json_path = os.path.join(pure_data_path, f"{trial_name}.json")
    if not os.path.exists(json_path):
        alt = os.path.join(pure_data_path, trial_name, f"{trial_name}.json")
        if os.path.exists(alt):
            json_path = alt
        else:
            raise FileNotFoundError(
                f"No BVP JSON for {trial_name} in {pure_data_path}")
    with open(json_path, 'r') as f:
        labels = json.load(f)
    bvps = np.array([l["Value"]["waveform"]
                     for l in labels["/FullPackage"]], dtype=np.float64)
    if len(bvps) != n_frames:
        bvps = np.interp(
            np.linspace(1, len(bvps), n_frames),
            np.linspace(1, len(bvps), len(bvps)), bvps,
        )
    return bvps.astype(np.float64)


def crop_with_box(frame, bbox_xywh):
    x, y, w, h = bbox_xywh
    cropped = frame[max(y, 0):min(y + h, frame.shape[0]),
                    max(x, 0):min(x + w, frame.shape[1])]
    if cropped.size == 0:
        cropped = frame
    return cv2.resize(cropped, (RESIZE_W, RESIZE_H),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Subcommand: cache_bboxes
# ---------------------------------------------------------------------------

def cmd_cache_bboxes(args):
    PROJ_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJ_DIR))
    from predictors import FaceDetector  # noqa: E402

    log("=" * 60)
    log("Caching bboxes on original pure.lmdb")
    log(f"  pure_lmdb: {args.pure_lmdb}")
    log(f"  output   : {args.output}")
    log("=" * 60)

    detector = FaceDetector(
        model_path=args.retinaface_weights,
        network='resnet50',
        confidence_threshold=0.5,
        device='cuda',
    )

    log("Indexing pure.lmdb subjects…")
    per_subj = build_key_index([args.pure_lmdb])
    log(f"  {len(per_subj)} subjects")

    env = lmdb.open(args.pure_lmdb, readonly=True, lock=False,
                    readahead=False, meminit=False)
    bboxes = {}
    for sid in sorted(per_subj.keys()):
        first_key = per_subj[sid][0][0]
        with env.begin() as txn:
            val = txn.get(first_key.encode())
        rec = pickle.loads(val)
        img_bgr = cv2.imdecode(
            np.frombuffer(rec['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        dets = detector.detect(img_bgr)
        if not dets:
            h, w = img_bgr.shape[:2]
            log(f"  {sid}: NO face detected, falling back to full frame")
            bboxes[sid] = [0, 0, w, h]
            continue
        best = max(dets, key=lambda d: d.confidence)
        x1, y1, x2, y2 = best.bbox
        raw = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        bboxes[sid] = enlarge_bbox(*raw)
        log(f"  {sid}: raw={raw} → enlarged={bboxes[sid]}")
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(bboxes, f, indent=2, sort_keys=True)
    os.replace(tmp, args.output)
    log(f"Saved {len(bboxes)} bboxes → {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: build
# ---------------------------------------------------------------------------

def cmd_build(args):
    log("=" * 60)
    log(f"Building rPPG-ready LMDB: source={args.source_name}")
    log(f"  inputs  : {len(args.lmdb_paths)} LMDB(s)")
    for p in args.lmdb_paths:
        log(f"    - {p}")
    log(f"  bboxes  : {args.bbox_cache}")
    log(f"  bvp src : {args.pure_data_path}")
    log(f"  output  : {args.output}")
    log("=" * 60)

    with open(args.bbox_cache, 'r') as f:
        bboxes = json.load(f)

    log("Indexing inputs…")
    t0 = time.time()
    per_subj = build_key_index(args.lmdb_paths)
    log(f"  {len(per_subj)} subjects in {time.time()-t0:.1f}s")

    envs = [lmdb.open(p, readonly=True, lock=False,
                      readahead=False, meminit=False)
            for p in args.lmdb_paths]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_env = lmdb.open(args.output, map_size=args.map_size_gb * 1024 ** 3)

    subject_ids = []
    subject_nframes = {}
    t_start = time.time()
    for sid in sorted(per_subj.keys()):
        bbox = bboxes.get(sid)
        if bbox is None:
            log(f"  {sid}: SKIP (no bbox cached)")
            continue

        keys = per_subj[sid]
        frames_list = []
        for key_str, shard_idx in keys:
            with envs[shard_idx].begin() as txn:
                val = txn.get(key_str.encode())
            rec = pickle.loads(val)
            img_bgr = cv2.imdecode(
                np.frombuffer(rec['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames_list.append(crop_with_box(img_rgb, bbox))
        frames = np.stack(frames_list, axis=0).astype(np.uint8)
        n = frames.shape[0]

        trial = trial_from_sid(sid)
        bvps = load_bvp_for_trial(args.pure_data_path, trial, n)

        with out_env.begin(write=True) as txn:
            txn.put(f"{sid}_frames".encode(), numpy_to_bytes(frames))
            txn.put(f"{sid}_labels".encode(), numpy_to_bytes(bvps))
            txn.put(f"{sid}_nframes".encode(), str(n).encode())

        subject_ids.append(sid)
        subject_nframes[sid] = n
        elapsed = time.time() - t_start
        log(f"  {sid}: {n} frames → LMDB  "
            f"[{len(subject_ids)}/{len(per_subj)}  "
            f"{elapsed/60:.1f} min elapsed]")

    for env in envs:
        env.close()

    with out_env.begin(write=True) as txn:
        txn.put(b"__subjects__", json.dumps(subject_ids).encode())
        txn.put(b"__dataset__", b"PURE")
        txn.put(b"__nframes__", json.dumps(subject_nframes).encode())
    out_env.close()

    log(f"Done: {len(subject_ids)} subjects in "
        f"{(time.time()-t_start)/60:.1f} min → {args.output}")


def main():
    p = argparse.ArgumentParser(
        description="Preprocess PURE LMDB for rPPG extraction.",
    )
    sub = p.add_subparsers(dest='command', required=True)

    pc = sub.add_parser('cache_bboxes',
                        help='Cache per-subject RetinaFace bboxes to JSON.')
    pc.add_argument('--pure-lmdb', required=True)
    pc.add_argument('--retinaface-weights', required=True)
    pc.add_argument('--output', required=True)

    pb = sub.add_parser('build',
                        help='Build rPPG-ready LMDB from frame LMDB(s).')
    pb.add_argument('--source-name', required=True)
    pb.add_argument('--lmdb-paths', nargs='+', required=True)
    pb.add_argument('--bbox-cache', required=True)
    pb.add_argument('--pure-data-path', required=True)
    pb.add_argument('--output', required=True)
    pb.add_argument('--map-size-gb', type=int, default=4)

    args = p.parse_args()
    if args.command == 'cache_bboxes':
        cmd_cache_bboxes(args)
    elif args.command == 'build':
        cmd_build(args)


if __name__ == '__main__':
    main()
