"""Micro-expression recognition on DFME-style per-frame LMDB(s) using SAMER
(from 'Mimicking the Annotation Process for Recognizing the Micro Expressions').

Pipeline per clip (subNNNN_MM/):
    1. Group frames by clip key prefix, sort by filename.
    2. Pick Onset=first, Apex=middle, Offset=last frame (heuristic — DFME
       does not ship per-clip annotations).
    3. Face-crop all three frames with a cached RetinaFace bbox (from apex of
       original DFME).
    4. Compute TVL1 optical flow magnitudes: onset→apex and offset→apex.
    5. Assemble 3-channel 'optical' image = [gray(apex), on_mag, off_mag].
    6. Resize to 224×224, ImageNet-normalize, split into eyes half (top 112)
       and mouth half (bottom 112).
    7. SAMER.predict(eyes, mouth) → logits (num_classes=5 for CASME II five-class).
       Ensemble across all 26 LOSO-fold checkpoints by averaging logits.

Per-clip output LMDB value (pickled):
    {
      "logits": np.ndarray(5,) float32,   # averaged across LOSO folds
      "pred_class": int,                  # argmax over logits
      "class_name": str,                  # CASME II five-class label
    }

CASME II five-class mapping:
    0=negative, 1=positive, 2=surprise, 3=repression, 4=others

External dependency:
    Set SAMER_REPO env var to your cloned SAMER repo
    ('Mimicking the Annotation Process for Recognizing the Micro Expressions').
    (Only required at runtime, not for --help.)
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# CASME II five-class label order
CASME_FIVE_CLASSES = [
    "negative", "positive", "surprise", "repression", "others",
]


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def clip_id_from_key(key_str: str) -> str:
    """'sub0008_01/00000.jpg' → 'sub0008_01'."""
    return key_str.split('/', 1)[0]


def build_clip_index(lmdb_paths):
    """{clip_id: [(key_str, shard_idx), ...]} sorted by key_str."""
    per_clip = {}
    for shard_idx, path in enumerate(lmdb_paths):
        env = lmdb.open(path, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn:
            for k, _ in txn.cursor():
                key_str = k.decode('utf-8')
                cid = clip_id_from_key(key_str)
                per_clip.setdefault(cid, []).append((key_str, shard_idx))
        env.close()
    for cid in per_clip:
        per_clip[cid].sort(key=lambda x: x[0])
    return per_clip


def load_frame_from_shards(envs, shard_idx, key_str):
    with envs[shard_idx].begin() as txn:
        val = txn.get(key_str.encode())
    rec = pickle.loads(val)
    img_bgr = cv2.imdecode(
        np.frombuffer(rec['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
    return img_bgr


def pick_onset_apex_offset(keys):
    n = len(keys)
    return keys[0], keys[n // 2], keys[-1]


def crop_with_box(img, bbox_xywh):
    x, y, w, h = bbox_xywh
    H, W = img.shape[:2]
    y1 = max(0, int(y)); y2 = min(H, int(y + h))
    x1 = max(0, int(x)); x2 = min(W, int(x + w))
    if y2 <= y1 or x2 <= x1:
        return img
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Bbox cache subcommand
# ---------------------------------------------------------------------------

LARGER_COEF = 1.3  # mild enlargement to include full face context


def enlarge_bbox(x1, y1, x2, y2, W, H, coef=LARGER_COEF):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    nw = w * coef
    nh = h * coef
    nx = max(0, int(cx - nw / 2))
    ny = max(0, int(cy - nh / 2))
    nw = int(min(W - nx, nw))
    nh = int(min(H - ny, nh))
    return [nx, ny, nw, nh]


def cmd_cache_bboxes(args):
    PROJ_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJ_DIR))
    from predictors import FaceDetector  # noqa: E402

    log("=" * 60)
    log("Caching RetinaFace bboxes for DFME clips")
    log(f"  dfme_lmdb: {args.dfme_lmdb}")
    log(f"  output   : {args.output}")
    log("=" * 60)

    detector = FaceDetector(
        model_path=args.retinaface_weights,
        network='resnet50', confidence_threshold=0.5,
        device='cuda',
    )

    log("Indexing clips…")
    per_clip = build_clip_index([args.dfme_lmdb])
    log(f"  {len(per_clip)} clips")

    env = lmdb.open(args.dfme_lmdb, readonly=True, lock=False,
                    readahead=False, meminit=False)
    bboxes = {}
    for cid in sorted(per_clip.keys()):
        keys = per_clip[cid]
        _, apex_key, _ = pick_onset_apex_offset(keys)
        with env.begin() as txn:
            val = txn.get(apex_key[0].encode())
        rec = pickle.loads(val)
        img_bgr = cv2.imdecode(
            np.frombuffer(rec['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        H, W = img_bgr.shape[:2]

        dets = detector.detect(img_bgr)
        if not dets:
            log(f"  {cid}: NO face, using full frame")
            bboxes[cid] = [0, 0, W, H]
            continue
        best = max(dets, key=lambda d: d.confidence)
        x1, y1, x2, y2 = best.bbox
        bboxes[cid] = enlarge_bbox(x1, y1, x2, y2, W, H)
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(bboxes, f, indent=2, sort_keys=True)
    os.replace(tmp, args.output)
    log(f"Saved {len(bboxes)} bboxes → {args.output}")


# ---------------------------------------------------------------------------
# Extract subcommand
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_samer_imports():
    """Deferred import of SAMER (requires SAMER_REPO env var)."""
    SAMER_REPO = os.environ.get("SAMER_REPO")
    if SAMER_REPO is None:
        raise RuntimeError(
            "Set SAMER_REPO to your cloned SAMER repo "
            "('Mimicking the Annotation Process for Recognizing the Micro Expressions')"
        )
    if SAMER_REPO not in sys.path:
        sys.path.insert(0, SAMER_REPO)
    from model import SAMER  # noqa: E402

    # Import optical_related.py directly (bypass utils/__init__.py, which pulls
    # in a dlib shape predictor file that isn't downloaded).
    import importlib.util as _iu
    _opt_spec = _iu.spec_from_file_location(
        "samer_optical_related",
        os.path.join(SAMER_REPO, "utils", "optical_related.py"),
    )
    _opt_mod = _iu.module_from_spec(_opt_spec)
    _opt_spec.loader.exec_module(_opt_mod)
    TVL1_optical_flow = _opt_mod.TVL1_optical_flow
    TVL1_magnitude = _opt_mod.TVL1_magnitude

    return SAMER, TVL1_optical_flow, TVL1_magnitude


def strip_module(sd):
    if isinstance(sd, dict) and any(k.startswith('module.') for k in sd.keys()):
        return {k[len('module.'):]: v for k, v in sd.items()}
    return sd


def load_fold_models(SAMER, weight_root: str, num_classes: int, device):
    pts = sorted(Path(weight_root).glob("model_best_*.pt"))
    log(f"Loading {len(pts)} SAMER fold checkpoints from {weight_root}")
    models = []
    for p in pts:
        m = SAMER(num_classes=num_classes, pretrained=False)
        sd = torch.load(str(p), map_location='cpu', weights_only=False)
        sd = strip_module(sd)
        m.load_state_dict(sd, strict=True)
        m.to(device).eval()
        models.append(m)
    return models


def build_optical_image(TVL1_optical_flow, TVL1_magnitude,
                        onset_bgr, apex_bgr, offset_bgr):
    """Return HxWx3 uint8 [gray(apex), on_mag, off_mag]."""
    # Size all to apex's shape (flow requires same shape)
    H, W = apex_bgr.shape[:2]
    onset = cv2.resize(onset_bgr, (W, H), interpolation=cv2.INTER_AREA) \
            if onset_bgr.shape[:2] != (H, W) else onset_bgr
    offset = cv2.resize(offset_bgr, (W, H), interpolation=cv2.INTER_AREA) \
            if offset_bgr.shape[:2] != (H, W) else offset_bgr

    onset_rgb = cv2.cvtColor(onset, cv2.COLOR_BGR2RGB)
    apex_rgb = cv2.cvtColor(apex_bgr, cv2.COLOR_BGR2RGB)
    offset_rgb = cv2.cvtColor(offset, cv2.COLOR_BGR2RGB)

    on_flow = TVL1_optical_flow(onset_rgb, apex_rgb)
    off_flow = TVL1_optical_flow(offset_rgb, apex_rgb)

    on_mag = TVL1_magnitude(on_flow)    # uint8, already normalized
    off_mag = TVL1_magnitude(off_flow)

    # Threshold noise (match reference crop_face_optical.py)
    on_thresh = np.mean(on_mag) * 1.5
    on_mag = on_mag.copy()
    on_mag[on_mag < on_thresh] = 0
    off_thresh = np.mean(off_mag) * 1.5
    off_mag = off_mag.copy()
    off_mag[off_mag < off_thresh] = 0

    gray_apex = cv2.cvtColor(apex_rgb, cv2.COLOR_RGB2GRAY)

    # Stack: (H, W, 3) uint8
    combined = np.stack([gray_apex, on_mag, off_mag], axis=-1).astype(np.uint8)
    return combined


_resize_224 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])
_normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def optical_to_eyes_mouth_tensors(opt_img):
    """Return (eyes_tensor, mouth_tensor), both shape (3, 112, 224) after
    split (top half = eyes, bottom half = mouth)."""
    t = _resize_224(opt_img)           # (3, 224, 224) float [0,1]
    t = _normalize(t)
    eyes = t[:, :112, :]
    mouth = t[:, 112:, :]
    return eyes, mouth


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


def cmd_extract(args):
    # Deferred SAMER import (requires SAMER_REPO env var)
    SAMER, TVL1_optical_flow, TVL1_magnitude = _load_samer_imports()

    lmdb_paths = args.lmdb_paths
    log_interval = args.log_interval
    commit_every = args.commit_every
    map_size_gb = args.map_size_gb
    num_classes = args.num_classes

    log("=" * 60)
    log("SAMER Micro-Expression Recognition → LMDB")
    log(f"  Source  : {args.source_name}")
    log(f"  Inputs  : {len(lmdb_paths)} LMDB(s)")
    for p in lmdb_paths:
        log(f"    - {p}")
    log(f"  BBoxes  : {args.bbox_cache}")
    log(f"  Weights : {args.weight_root}")
    log(f"  Output  : {args.output}")
    log(f"  Classes : {num_classes}")
    log("=" * 60)

    device = torch.device(args.device)
    with open(args.bbox_cache, 'r') as f:
        bboxes = json.load(f)

    log("Indexing input LMDB(s) by clip…")
    per_clip = build_clip_index(lmdb_paths)
    log(f"  {len(per_clip)} clips")

    envs = [lmdb.open(p, readonly=True, lock=False,
                      readahead=False, meminit=False)
            for p in lmdb_paths]

    done_keys = load_done_keys(args.output) if args.resume else set()
    if done_keys:
        log(f"Resume: {len(done_keys)} clips already present, skipping")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_env = lmdb.open(args.output, map_size=map_size_gb * 1024 ** 3,
                        subdir=True)

    models = load_fold_models(SAMER, args.weight_root, num_classes, device)

    class_names = (CASME_FIVE_CLASSES
                   if num_classes == 5
                   else [str(i) for i in range(num_classes)])

    n_processed = 0
    n_ok = 0
    n_skip = 0
    pending = []
    t_start = time.time()

    def flush_pending():
        if not pending:
            return
        with out_env.begin(write=True) as txn:
            for k, v in pending:
                txn.put(k.encode('utf-8'),
                        pickle.dumps(v, pickle.HIGHEST_PROTOCOL))
        pending.clear()

    for cid in sorted(per_clip.keys()):
        n_processed += 1
        if cid in done_keys:
            continue
        bbox = bboxes.get(cid)
        if bbox is None:
            n_skip += 1
            continue
        keys = per_clip[cid]
        if len(keys) < 3:
            n_skip += 1
            continue
        onset_key, apex_key, offset_key = pick_onset_apex_offset(keys)
        try:
            onset = load_frame_from_shards(envs, onset_key[1], onset_key[0])
            apex = load_frame_from_shards(envs, apex_key[1], apex_key[0])
            offset = load_frame_from_shards(envs, offset_key[1], offset_key[0])
        except Exception:
            n_skip += 1
            continue
        if onset is None or apex is None or offset is None:
            n_skip += 1
            continue

        onset = crop_with_box(onset, bbox)
        apex = crop_with_box(apex, bbox)
        offset = crop_with_box(offset, bbox)

        try:
            opt_img = build_optical_image(
                TVL1_optical_flow, TVL1_magnitude, onset, apex, offset)
            eyes_t, mouth_t = optical_to_eyes_mouth_tensors(opt_img)
        except Exception:
            n_skip += 1
            continue

        eyes_batch = eyes_t.unsqueeze(0).to(device)
        mouth_batch = mouth_t.unsqueeze(0).to(device)

        with torch.no_grad():
            logits_sum = None
            for m in models:
                out = m.predict(eyes_batch, mouth_batch)  # (1, C)
                logits_sum = out if logits_sum is None else logits_sum + out
            logits_avg = (logits_sum / len(models)).squeeze(0).float().cpu().numpy()

        pred_class = int(np.argmax(logits_avg))
        pending.append((cid, {
            "logits": logits_avg.astype(np.float32),
            "pred_class": pred_class,
            "class_name": class_names[pred_class],
        }))
        n_ok += 1

        if len(pending) >= commit_every:
            flush_pending()

        if n_processed % log_interval == 0:
            elapsed = time.time() - t_start
            rate = (n_ok + n_skip) / max(elapsed, 1)
            log(f"  {n_processed}/{len(per_clip)} clips  "
                f"ok={n_ok} skip={n_skip}  {rate:.2f} clips/s")

    flush_pending()
    out_env.close()
    for env in envs:
        env.close()

    elapsed = time.time() - t_start
    log("-" * 60)
    log(f"Complete: processed={n_processed}, ok={n_ok}, skip={n_skip}")
    log(f"  Time: {elapsed/60:.1f} min")
    log(f"Done. Output: {args.output}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Micro-expression extraction with SAMER."
    )
    sub = p.add_subparsers(dest='command', required=True)

    pc = sub.add_parser('cache_bboxes',
                        help='Cache RetinaFace bboxes for DFME clips.')
    pc.add_argument('--dfme-lmdb', required=True)
    pc.add_argument('--retinaface-weights', required=True)
    pc.add_argument('--output', required=True)

    pe = sub.add_parser('extract',
                        help='Run SAMER on DFME clips and write to LMDB.')
    pe.add_argument('--source-name', required=True)
    pe.add_argument('--lmdb-paths', nargs='+', required=True)
    pe.add_argument('--bbox-cache', required=True)
    pe.add_argument('--weight-root', required=True)
    pe.add_argument('--output', required=True)
    pe.add_argument('--num-classes', type=int, default=5)
    pe.add_argument('--device', default='cuda')
    pe.add_argument('--log-interval', type=int, default=25)
    pe.add_argument('--commit-every', type=int, default=25)
    pe.add_argument('--map-size-gb', type=int, default=1)
    pe.add_argument('--resume', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.command == 'cache_bboxes':
        cmd_cache_bboxes(args)
    else:
        cmd_extract(args)
