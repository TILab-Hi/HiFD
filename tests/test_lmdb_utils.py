# $DST/tests/test_lmdb_utils.py
import os
import pickle
import lmdb
import pytest
import yaml
from src.lmdb_utils import (
    lmdb_to_dict, count_entries, iter_lmdb, load_paths_config,
)


def _make_lmdb(path, items):
    env = lmdb.open(path, map_size=10 * 1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        for k, v in items.items():
            txn.put(k.encode(), pickle.dumps(v))
    env.close()


def test_lmdb_to_dict_roundtrip(tmp_path):
    p = str(tmp_path / "a.lmdb")
    _make_lmdb(p, {"x": 1, "y": 2})
    out = lmdb_to_dict(p)
    assert out == {"x": 1, "y": 2}


def test_count_entries(tmp_path):
    p1 = str(tmp_path / "a.lmdb")
    p2 = str(tmp_path / "b.lmdb")
    _make_lmdb(p1, {"a": 1})
    _make_lmdb(p2, {"b": 2, "c": 3})
    assert count_entries([p1, p2]) == 3


def test_iter_lmdb_skip_keys(tmp_path):
    p = str(tmp_path / "a.lmdb")
    _make_lmdb(p, {"x": 1, "y": 2, "z": 3})
    seen = {k: v for k, v in iter_lmdb([p], skip_keys={"y"})}
    assert seen == {"x": 1, "z": 3}


def test_load_paths_config_env_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/data")
    monkeypatch.delenv("PRED_DIR", raising=False)
    cfg_path = tmp_path / "paths.yaml"
    cfg_path.write_text(
        "data_dir: ${DATA_DIR}\n"
        "predictions_dir: ${PRED_DIR:-./predictions}\n"
    )
    cfg = load_paths_config(str(cfg_path))
    assert cfg["data_dir"] == "/data"
    assert cfg["predictions_dir"] == "./predictions"
