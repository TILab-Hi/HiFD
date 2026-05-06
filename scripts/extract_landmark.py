"""
Extract MediaPipe Face Landmarker landmarks from LMDB datasets and write to a
single output LMDB.

Per-image value (pickled):
    {
      "points": [[x1, y1], ..., [x68, y68]],  # iBUG-68 mapping, pixel coords (2dp)
      "left_eye_center":  [lx, ly],           # iris center (MP idx 468), pixel coords
      "right_eye_center": [rx, ry],           # iris center (MP idx 473), pixel coords
    }

Keys missing from the output LMDB correspond to frames where MediaPipe could
not detect a face (sparse output).

Resume: --resume reads existing keys from the output LMDB and skips them.
"""

import argparse
import multiprocessing as mp_proc
import os
import pickle
import time
from datetime import datetime

import cv2
import lmdb
import numpy as np


# Same iBUG-68 mapping as extract_landmarks.py
LANDMARK_68_FROM_MP = [
    127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 197, 5, 4,
    75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 81, 13, 311, 308, 402, 14, 178,
]
assert len(LANDMARK_68_FROM_MP) == 68

LEFT_EYE_MP_IDX = 468
RIGHT_EYE_MP_IDX = 473


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


_W = {}


def _init_worker(fl_path, min_conf):
    import mediapipe as mp_lib
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    _W['mp_lib'] = mp_lib
    fl_opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=fl_path),
        num_faces=1,
        min_face_detection_confidence=min_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _W['fl'] = vision.FaceLandmarker.create_from_options(fl_opts)


def _process_one(item):
    key_str, img_bytes = item
    mp_lib = _W['mp_lib']
    fl = _W['fl']

    try:
        img_bgr = cv2.imdecode(
            np.frombuffer(img_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img_bgr is None:
            return key_str, None
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(
            image_format=mp_lib.ImageFormat.SRGB, data=img_rgb,
        )
    except Exception:
        return key_str, None

    lm_res = fl.detect(mp_image)
    if not lm_res.face_landmarks:
        return key_str, None

    face_lms = lm_res.face_landmarks[0]

    points_68 = [
        [round(face_lms[i].x * w, 2), round(face_lms[i].y * h, 2)]
        for i in LANDMARK_68_FROM_MP
    ]
    left_eye = [
        round(face_lms[LEFT_EYE_MP_IDX].x * w, 2),
        round(face_lms[LEFT_EYE_MP_IDX].y * h, 2),
    ]
    right_eye = [
        round(face_lms[RIGHT_EYE_MP_IDX].x * w, 2),
        round(face_lms[RIGHT_EYE_MP_IDX].y * h, 2),
    ]

    return key_str, {
        "points": points_68,
        "left_eye_center": left_eye,
        "right_eye_center": right_eye,
    }


def iter_items(lmdb_paths, skip_keys):
    for lmdb_path in lmdb_paths:
        env = lmdb.open(
            lmdb_path, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                key_str = key.decode('utf-8')
                if key_str in skip_keys:
                    continue
                rec = pickle.loads(val)
                img_bytes = rec['image']
                yield (key_str, img_bytes)
        env.close()


def load_existing_keys(out_path):
    if not os.path.exists(out_path):
        return set()
    env = lmdb.open(out_path, readonly=True, lock=False, readahead=False)
    keys = set()
    with env.begin() as txn:
        for key, _ in txn.cursor():
            keys.add(key.decode('utf-8'))
    env.close()
    return keys


def count_entries(lmdb_paths):
    total = 0
    for p in lmdb_paths:
        env = lmdb.open(
            p, readonly=True, lock=False, readahead=False, meminit=False,
        )
        total += env.stat()['entries']
        env.close()
    return total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lmdb-paths', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--face-landmarker', required=True)
    p.add_argument('--min-det-conf', type=float, default=0.5)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--log-interval', type=int, default=5000)
    p.add_argument('--commit-every', type=int, default=500)
    p.add_argument('--map-size-gb', type=int, default=10)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    lmdb_paths = args.lmdb_paths
    face_landmarker = args.face_landmarker
    min_det_conf = args.min_det_conf
    log_interval = args.log_interval
    commit_every = args.commit_every
    map_size_gb = args.map_size_gb

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    log("=" * 60)
    log("MediaPipe Landmark Extraction → LMDB")
    log(f"  Inputs : {len(lmdb_paths)} LMDB(s)")
    for p in lmdb_paths:
        log(f"    - {p}")
    log(f"  Output : {args.output}")
    log(f"  Workers: {args.workers}")
    log("=" * 60)

    total = count_entries(lmdb_paths)
    log(f"Total input entries: {total:,}")

    done_keys = set()
    if args.resume:
        done_keys = load_existing_keys(args.output)
        if done_keys:
            log(f"Resume: {len(done_keys):,} keys already present, skipping")

    map_size = map_size_gb * 1024 ** 3
    out_env = lmdb.open(args.output, map_size=map_size, subdir=True)

    n_processed = len(done_keys)
    n_ok_run = 0
    n_fail_run = 0
    start = time.time()

    ctx = mp_proc.get_context('spawn')
    pending = []

    def flush():
        if not pending:
            return
        with out_env.begin(write=True) as txn:
            for k, v in pending:
                txn.put(k.encode('utf-8'),
                        pickle.dumps(v, pickle.HIGHEST_PROTOCOL))
        pending.clear()

    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(face_landmarker, min_det_conf)) as pool:
        for key_str, result in pool.imap_unordered(
            _process_one,
            iter_items(lmdb_paths, done_keys),
            chunksize=args.chunk_size,
        ):
            n_processed += 1
            if result is not None:
                pending.append((key_str, result))
                n_ok_run += 1
                if len(pending) >= commit_every:
                    flush()
            else:
                n_fail_run += 1

            if n_processed % log_interval == 0:
                elapsed = time.time() - start
                done_this_run = n_ok_run + n_fail_run
                rate = done_this_run / max(elapsed, 1)
                remaining = total - n_processed
                eta_min = remaining / max(rate, 1) / 60
                pct = 100.0 * n_processed / max(total, 1)
                log(f"  {n_processed:>7,}/{total:,} ({pct:5.1f}%)  "
                    f"ok+={n_ok_run:,}  fail+={n_fail_run:,}  "
                    f"{rate:5.1f} img/s  ETA {eta_min:5.1f} min")

    flush()
    out_env.close()

    elapsed = time.time() - start
    log("-" * 60)
    log(f"Complete: processed={n_processed:,}, "
        f"new_ok={n_ok_run:,}, new_fail={n_fail_run:,}")
    log(f"  Time: {elapsed/60:.1f} min")
    log(f"Done. Output: {args.output}")


if __name__ == '__main__':
    main()
