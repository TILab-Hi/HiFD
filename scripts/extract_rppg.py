"""Extract rPPG predictions on the PURE LMDB with a single neural method.

Uses rPPG-Toolbox model classes and weights (UBFC-rPPG cross-dataset),
and reuses LMDBDataset for on-the-fly transforms.

External dependency:
    Set RPPG_REPO env var to your cloned rPPG-Toolbox checkout.
    (Only required at runtime, not for --help.)

Output (per method):
    {output_dir}/{method}_orig.npy    — dict {sid: {chunk_id: np.float32[T]}}
    {output_dir}/{method}_labels.npy  — dict {sid: {chunk_id: np.float32[T]}}
    {output_dir}/meta.json            — appended record for this method

Usage:
    python extract_rppg.py --method physnet       --weights ...pth --lmdb-path ... --output-dir ...
    python extract_rppg.py --method efficientphys --weights ...pth --lmdb-path ... --output-dir ...
    python extract_rppg.py --method factorizephys --weights ...pth --lmdb-path ... --output-dir ...
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Per-method configuration (defined before runtime imports)
# ---------------------------------------------------------------------------

FACTORIZEPHYS_MD_CONFIG = {
    "MD_FSAM": False,
    "MD_TYPE": "NMF",
    "MD_TRANSFORM": "T_KAB",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": True,
    "MD_RESIDUAL": True,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "in_channels": 3,
    "data_channels": 4,
    "align_channels": 8,
    "height": 72,
    "weight": 72,
    "batch_size": 4,
    "frames": 160,
    "debug": False,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "",
    "data_path": "",
    "label_path": "",
}

METHOD_CONFIGS = {
    "physnet": {
        "data_type": "DiffNormalized",
        "label_type": "DiffNormalized",
        "chunk_length": 128,
        "data_format": "NCDHW",
    },
    "efficientphys": {
        "data_type": "Standardized",
        "label_type": "DiffNormalized",
        "chunk_length": 180,
        "data_format": "NDCHW",
    },
    "factorizephys": {
        "data_type": "Raw",
        "label_type": "Standardized",
        "chunk_length": 160,
        "data_format": "NCDHW",
    },
}


# ---------------------------------------------------------------------------
# Dataset config shim for LMDBDataset
# ---------------------------------------------------------------------------

def make_dataset_config(data_type, label_type, chunk_length, data_format):
    """Build the SimpleNamespace config that LMDBDataset expects."""
    return SimpleNamespace(
        DATA_FORMAT=data_format,
        BEGIN=0.0,
        END=1.0,
        PREPROCESS=SimpleNamespace(
            DATA_TYPE=[data_type],
            LABEL_TYPE=label_type,
            CHUNK_LENGTH=chunk_length,
        ),
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_predictions(predictions, path):
    """Save a nested dict {sid: {chunk_id: np.float32[T]}} as .npy."""
    np.save(path, predictions, allow_pickle=True)


def update_meta(meta_path, record):
    """Append/overwrite this (source_tag, method) record in meta.json."""
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    key = f"{record.get('source_tag', 'orig')}_{record['method']}"
    meta[key] = record
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Runtime imports (deferred so --help works without RPPG_REPO set)
# ---------------------------------------------------------------------------

def _load_rppg_imports():
    RPPG_REPO = os.environ.get("RPPG_REPO")
    if RPPG_REPO is None:
        raise RuntimeError(
            "Set RPPG_REPO env var to your cloned rPPG-Toolbox checkout "
            "(https://github.com/ubicomplab/rPPG-Toolbox)"
        )
    if RPPG_REPO not in sys.path:
        sys.path.insert(0, RPPG_REPO)

    from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
    from neural_methods.model.EfficientPhys import EfficientPhys
    from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys

    PROJ_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJ_DIR))
    from predictors.lmdb_loader import LMDBDataset

    return (PhysNet_padding_Encoder_Decoder_MAX, EfficientPhys,
            FactorizePhys, LMDBDataset)


def _build_physnet(PhysNet, device):
    return PhysNet(frames=128).to(device)


def _build_efficientphys(EfficientPhys, device):
    return EfficientPhys(
        in_channels=3, frame_depth=10, img_size=72,
        dropout_rate1=0.2, dropout_rate2=0.5,
    ).to(device)


def _build_factorizephys(FactorizePhys, device):
    return FactorizePhys(
        frames=160,
        md_config=FACTORIZEPHYS_MD_CONFIG,
        device=device,
        in_channels=3,
        dropout=0.1,
    ).to(device)


def _infer_physnet(model, data, cfg):
    # data: (B, C, T, H, W)
    out = model(data)
    preds = out[0] if isinstance(out, tuple) else out
    return preds  # (B, T)


def _infer_efficientphys(model, data, cfg):
    # data: (B, D, C, H, W)
    N, D, C, H, W = data.shape
    flat = data.reshape(N * D, C, H, W)
    # EfficientPhys applies torch.diff(dim=0), so pad one extra frame
    flat = torch.cat([flat, flat[-1:].clone()], dim=0)
    out = model(flat)                  # (N*D, 1)
    preds = out[: N * D].squeeze(-1)
    return preds.reshape(N, D)         # (B, D)


def _infer_factorizephys(model, data, cfg):
    # data: (B, C, T, H, W); model needs T+1 frames for 3D temporal diff
    extra = data[:, :, -1:, :, :]
    data_ext = torch.cat([data, extra], dim=2)
    out = model(data_ext)
    preds = out[0] if isinstance(out, tuple) else out
    # preds is typically (B, T) or (B, 1, T, 1, 1) — squeeze if needed
    if preds.ndim > 2:
        preds = preds.squeeze()
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract rPPG predictions using a neural method."
    )
    parser.add_argument("--method", required=True,
                        choices=list(METHOD_CONFIGS.keys()))
    parser.add_argument("--weights", required=True,
                        help="Path to .pth weights file")
    parser.add_argument("--lmdb-path", required=True,
                        help="Path to PURE LMDB directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-tag", default="orig",
                        help="Tag to use in output filenames "
                             "({source_tag}_{method}.npy). Default: 'orig'.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Deferred runtime imports (requires RPPG_REPO env var)
    (PhysNet, EfficientPhys,
     FactorizePhys, LMDBDataset) = _load_rppg_imports()

    builders = {
        "physnet": lambda d: _build_physnet(PhysNet, d),
        "efficientphys": lambda d: _build_efficientphys(EfficientPhys, d),
        "factorizephys": lambda d: _build_factorizephys(FactorizePhys, d),
    }
    infer_fns = {
        "physnet": _infer_physnet,
        "efficientphys": _infer_efficientphys,
        "factorizephys": _infer_factorizephys,
    }

    cfg = METHOD_CONFIGS[args.method]
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.isdir(args.lmdb_path):
        raise FileNotFoundError(f"LMDB dir not found: {args.lmdb_path}")

    log("=" * 60)
    log(f"rPPG extraction: {args.method}")
    log(f"  weights:     {args.weights}")
    log(f"  lmdb:        {args.lmdb_path}")
    log(f"  output:      {args.output_dir}")
    log(f"  data_type:   {cfg['data_type']}")
    log(f"  label_type:  {cfg['label_type']}")
    log(f"  chunk_len:   {cfg['chunk_length']}")
    log(f"  data_format: {cfg['data_format']}")
    log("=" * 60)

    device = torch.device(args.device)

    # Build model + load weights
    log("Building model...")
    model = builders[args.method](device)
    log(f"Loading weights from {args.weights}")
    state = torch.load(args.weights, map_location=device, weights_only=True)
    # rPPG-Toolbox .pth files are often saved from DataParallel, strip "module."
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log(f"  strict load failed; missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            log(f"  missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            log(f"  unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    else:
        log("  load_state_dict: strict match OK")
    model.eval()

    # Build dataset
    log("Building dataset...")
    ds_cfg = make_dataset_config(
        cfg["data_type"], cfg["label_type"],
        cfg["chunk_length"], cfg["data_format"],
    )
    dataset = LMDBDataset("PURE", args.lmdb_path, ds_cfg)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    log(f"Dataset chunks: {len(dataset)}")

    # Inference loop
    predictions = {}
    labels = {}
    infer_fn = infer_fns[args.method]
    start = time.time()

    log("Running inference...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data = batch[0].to(torch.float32).to(device)
            label = batch[1].to(torch.float32)
            sids = batch[2]
            chunk_ids = batch[3]

            preds = infer_fn(model, data, cfg).detach().cpu()

            B = preds.shape[0]
            # EfficientPhys may trim the batch — trust preds.shape, not len(sids)
            for j in range(B):
                sid = str(sids[j])
                cid = int(chunk_ids[j])
                predictions.setdefault(sid, {})[cid] = preds[j].numpy().astype(np.float32)
                labels.setdefault(sid, {})[cid] = label[j].numpy().astype(np.float32)

            if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                done = sum(len(v) for v in predictions.values())
                log(f"  batch {i+1}/{len(loader)}  chunks={done}")

    elapsed = time.time() - start
    n_subjects = len(predictions)
    n_chunks = sum(len(v) for v in predictions.values())
    log(f"Inference done: {n_subjects} subjects, {n_chunks} chunks in {elapsed:.1f}s")

    # Save outputs
    preds_path = os.path.join(
        args.output_dir, f"{args.source_tag}_{args.method}.npy")
    labels_path = os.path.join(
        args.output_dir, f"{args.source_tag}_{args.method}_labels.npy")
    meta_path = os.path.join(args.output_dir, "meta.json")
    save_predictions(predictions, preds_path)
    save_predictions(labels, labels_path)
    log(f"Saved predictions: {preds_path}")
    log(f"Saved labels:      {labels_path}")

    update_meta(meta_path, {
        "method": args.method,
        "source_tag": args.source_tag,
        "weights_file": args.weights,
        "lmdb_path": args.lmdb_path,
        "data_type": cfg["data_type"],
        "label_type": cfg["label_type"],
        "chunk_length": cfg["chunk_length"],
        "data_format": cfg["data_format"],
        "fs": 30,
        "n_subjects": n_subjects,
        "n_chunks": n_chunks,
        "runtime_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    log(f"Updated meta:      {meta_path}")
    log("Done.")


if __name__ == "__main__":
    main()
