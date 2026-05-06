"""Minimal Dataset wrapper around an LMDB of frame tensors for rPPG batch
inference. The rPPG-Toolbox training pipeline expects a torch DataLoader.
"""

import pickle

import lmdb
import torch
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    """Read pre-decoded float-tensor frame chunks from an LMDB."""

    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(
            lmdb_path, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        with self.env.begin() as txn:
            self.keys = [k.decode("utf-8") for k, _ in txn.cursor()]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        with self.env.begin() as txn:
            payload = pickle.loads(txn.get(key.encode("utf-8")))
        # Expected payload: {"frames": np.ndarray, "fps": float, ...}
        frames = torch.from_numpy(payload["frames"]).float()
        return key, frames, payload.get("fps", 30.0)
