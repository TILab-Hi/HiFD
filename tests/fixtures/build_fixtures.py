"""Build small synthetic LMDB fixtures for end-to-end testing.

Run once: `python tests/fixtures/build_fixtures.py`
Outputs LMDBs under tests/fixtures/predictions/<task>/<source>.lmdb that
mirror the production schema with 10 random sample IDs.
"""

import pickle
from pathlib import Path

import lmdb
import numpy as np

OUT = Path(__file__).resolve().parent / "predictions"
SOURCES = ["balanced", "MockMethod"]
SAMPLES = [f"img_{i:04d}" for i in range(10)]
RNG = np.random.default_rng(0)


def write(path: Path, items: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        for k, v in items.items():
            txn.put(k.encode(), pickle.dumps(v))
    env.close()


def main() -> None:
    for src in SOURCES:
        # face_embedding
        emb_items = {
            sid: {bb: RNG.standard_normal(512).astype(np.float32)
                  for bb in ("arcface", "cosface", "adaface")}
            for sid in SAMPLES
        }
        write(OUT / "face_embedding" / f"{src}.lmdb", emb_items)

        # age_gender
        age_items = {
            sid: {"age": float(RNG.uniform(20, 60)),
                  "gender": "male" if RNG.uniform() < 0.5 else "female",
                  "gender_score": float(RNG.uniform(0.5, 1.0))}
            for sid in SAMPLES
        }
        write(OUT / "age_gender" / f"{src}.lmdb", age_items)

        # ethnicity
        eth_items = {
            sid: {"race_prob": (lambda p: (p / p.sum()).tolist())(RNG.uniform(size=7))}
            for sid in SAMPLES
        }
        write(OUT / "ethnicity" / f"{src}.lmdb", eth_items)

        # macro_exp
        mx_items = {
            sid: {"probs": (lambda p: (p / p.sum()).tolist())(RNG.uniform(size=7))}
            for sid in SAMPLES
        }
        write(OUT / "macro_exp" / f"{src}.lmdb", mx_items)

        # landmark
        lm_items = {
            sid: {"points": RNG.uniform(0, 256, size=(68, 2)).round(2).tolist(),
                  "left_eye_center": [100.0, 100.0],
                  "right_eye_center": [156.0, 100.0]}
            for sid in SAMPLES
        }
        write(OUT / "landmark" / f"{src}.lmdb", lm_items)

        # gaze
        gz_items = {
            sid: {"yaw": float(RNG.uniform(-0.5, 0.5)),
                  "pitch": float(RNG.uniform(-0.5, 0.5))}
            for sid in SAMPLES
        }
        write(OUT / "gaze" / f"{src}.lmdb", gz_items)

    # LPIPS / NIQE only for the de-id source
    write(OUT / "lpips" / "MockMethod.lmdb",
          {sid: {"lpips": float(RNG.uniform(0.05, 0.4))} for sid in SAMPLES})
    write(OUT / "niqe" / "MockMethod.lmdb",
          {sid: {"niqe": float(RNG.uniform(2.0, 4.5))} for sid in SAMPLES})

    print(f"Wrote synthetic LMDBs under {OUT}")


if __name__ == "__main__":
    main()
