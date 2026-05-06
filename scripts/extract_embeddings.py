"""
Extract face embeddings (ArcFace + CosFace + AdaFace) from LMDB datasets and
write to a single output LMDB per source.

Requires a pre-computed detections.json (run extract_face_detections.py first).
The 5-point landmarks in detections.json are reused across all sources — de-id
images often can't be detected, so we align them with the original image's
landmarks.

Per-key value (pickled):
    {
      "arcface": np.ndarray(512,) float32,   # L2-normalized
      "cosface": np.ndarray(512,) float32,   # L2-normalized
      "adaface": np.ndarray(512,) float32,   # L2-normalized
    }

Keys for which no landmarks exist (detection failed) or alignment failed are
omitted (sparse output).

Usage:
    python extract_embeddings.py \
        --lmdb-paths /path/to/balanced.lmdb \
        --detections /path/to/detections.json \
        --weights-root /path/to/weights \
        --output /path/to/balanced_embeddings.lmdb \
        --device cuda --resume
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_DIR))

from predictors import (  # noqa: E402
    FaceDetector, ArcFace, CosFace, AdaFace,
    get_reference_facial_points, warp_and_crop_face,
)


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Embedding extraction → LMDB
# ---------------------------------------------------------------------------

_BACKBONES = ("arcface", "cosface", "adaface")


def _build_backbone(name: str, weights_root: str, device: str):
    if name == "arcface":
        return ArcFace(
            model_path=os.path.join(
                weights_root, "ms1mv3_arcface_r100_fp16/backbone.pth"),
            num_layers=100, device=device,
        )
    if name == "cosface":
        return CosFace(
            model_path=os.path.join(
                weights_root, "glint360k_cosface_r50_fp16_0.1/backbone.pth"),
            num_layers=50, device=device,
        )
    if name == "adaface":
        return AdaFace(
            model_path=os.path.join(
                weights_root, "adaface_pre_trained/adaface_ir50_ms1mv2.ckpt"),
            architecture="ir_50", device=device,
        )
    raise ValueError(f"Unknown backbone: {name}")


def _forward(model, name: str, batch_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        if name == "adaface":
            out = model.model(batch_tensor)
            emb = out[0] if isinstance(out, tuple) else out
        else:
            emb = model.model(batch_tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.float().cpu().numpy()


def _iter_lmdb(lmdb_paths: List[str]):
    for path in lmdb_paths:
        env = lmdb.open(
            path, readonly=True, lock=False,
            readahead=False, meminit=False,
            map_size=700 * 1024 ** 3,
        )
        with env.begin() as txn:
            for key, val in txn.cursor():
                yield key.decode("utf-8"), pickle.loads(val)
        env.close()


def _count(lmdb_paths: List[str]) -> int:
    n = 0
    for p in lmdb_paths:
        env = lmdb.open(p, readonly=True, lock=False,
                        readahead=False, meminit=False)
        n += env.stat()["entries"]
        env.close()
    return n


def _load_done_keys(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    env = lmdb.open(out_path, readonly=True, lock=False, readahead=False)
    keys = set()
    with env.begin() as txn:
        for key, _ in txn.cursor():
            keys.add(key.decode("utf-8"))
    env.close()
    return keys


def run_extraction(lmdb_paths: List[str], detections_path: str,
                   weights_root: str, output_path: str,
                   device: str = "cuda", batch_size: int = 64,
                   log_interval: int = 5000, commit_every: int = 500,
                   map_size_gb: int = 4, resume: bool = False) -> None:
    log("=" * 60)
    log("Embedding Extraction → LMDB")
    log(f"  Inputs  : {len(lmdb_paths)} LMDB(s)")
    for p in lmdb_paths:
        log(f"    - {p}")
    log(f"  Detect  : {detections_path}")
    log(f"  Output  : {output_path}")
    log(f"  Device  : {device}, batch={batch_size}")
    log("=" * 60)

    log("Loading detections…")
    with open(detections_path, "r") as f:
        detections: Dict[str, dict] = json.load(f)
    log(f"  {len(detections):,} detections loaded")

    ref_pts = get_reference_facial_points(
        output_size=(112, 112), default_square=True)

    total = _count(lmdb_paths)
    log(f"Total input entries: {total:,}")

    done_keys = _load_done_keys(output_path) if resume else set()
    if done_keys:
        log(f"Resume: {len(done_keys):,} keys already present, skipping")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_env = lmdb.open(output_path, map_size=map_size_gb * 1024 ** 3,
                        subdir=True)

    log("Loading backbones…")
    models = {name: _build_backbone(name, weights_root, device)
              for name in _BACKBONES}
    log(f"  Loaded: {list(models.keys())}")

    batch_keys, batch_aligned = [], []
    n_processed = len(done_keys)
    n_ok_run, n_skip_run = 0, 0
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

    def flush_batch():
        nonlocal n_ok_run
        if not batch_keys:
            return
        tensor = torch.stack(batch_aligned).to(device)
        embs = {name: _forward(models[name], name, tensor)
                for name in _BACKBONES}
        for i, key_str in enumerate(batch_keys):
            pending.append((key_str, {
                name: embs[name][i].astype(np.float32)
                for name in _BACKBONES
            }))
            n_ok_run += 1
        batch_keys.clear()
        batch_aligned.clear()
        if len(pending) >= commit_every:
            flush_pending()

    for key_str, rec in _iter_lmdb(lmdb_paths):
        if key_str in done_keys:
            continue
        n_processed += 1

        det = detections.get(key_str)
        if det is None:
            n_skip_run += 1
        else:
            img_bgr = cv2.imdecode(
                np.frombuffer(rec["image"], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img_bgr is None:
                n_skip_run += 1
            else:
                try:
                    lm5 = np.asarray(det["landmarks"], dtype=np.float32)
                    aligned = warp_and_crop_face(
                        src_img=img_bgr,
                        facial_pts=lm5,
                        reference_pts=ref_pts,
                        crop_size=(112, 112),
                    )
                    bgr_norm = ((aligned / 255.0) - 0.5) / 0.5
                    t = torch.tensor(
                        bgr_norm.transpose(2, 0, 1), dtype=torch.float32)
                    batch_keys.append(key_str)
                    batch_aligned.append(t)
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
    log(f"Complete: processed={n_processed:,}, new_ok={n_ok_run:,}, "
        f"new_skip={n_skip_run:,}")
    log(f"  Time: {elapsed/60:.1f} min")
    log(f"Done. Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract ArcFace/CosFace/AdaFace embeddings from an LMDB."
    )
    p.add_argument("--lmdb-paths", nargs="+", required=True)
    p.add_argument("--detections", required=True,
                   help="Path to detections.json (from extract_face_detections.py)")
    p.add_argument("--weights-root", required=True,
                   help="Root directory of pretrained weight files")
    p.add_argument("--output", required=True,
                   help="Output LMDB path (directory)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--log-interval", type=int, default=5000)
    p.add_argument("--commit-every", type=int, default=500)
    p.add_argument("--map-size-gb", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_extraction(
        lmdb_paths=args.lmdb_paths,
        detections_path=args.detections,
        weights_root=args.weights_root,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        commit_every=args.commit_every,
        map_size_gb=args.map_size_gb,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
