"""LMDB utility helpers used across extraction and aggregation scripts.

Provides:
- lmdb_to_dict / iter_lmdb: read helpers
- count_entries: total entries across multiple LMDBs
- load_paths_config: YAML loader with ${ENV_VAR} and ${ENV_VAR:-default} interp
- LMDBWriter: batched write context manager
"""

from __future__ import annotations

import os
import pickle
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import lmdb
import yaml

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _interp(value: str) -> str:
    def sub(m: "re.Match[str]") -> str:
        var, default = m.group(1), m.group(2)
        env = os.environ.get(var)
        if env is not None:
            return env
        if default is not None:
            return default
        return m.group(0)
    return _ENV_RE.sub(sub, value)


def load_paths_config(yaml_path: str) -> Dict[str, Any]:
    """Load YAML and substitute ${ENV} and ${ENV:-default} in string values."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    def walk(x: Any) -> Any:
        if isinstance(x, str):
            return _interp(x)
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(raw)


def lmdb_to_dict(path: str) -> Dict[str, Any]:
    env = lmdb.open(path, readonly=True, lock=False, readahead=False)
    out: Dict[str, Any] = {}
    with env.begin() as txn:
        for k, v in txn.cursor():
            out[k.decode("utf-8")] = pickle.loads(v)
    env.close()
    return out


def count_entries(lmdb_paths: List[str]) -> int:
    n = 0
    for p in lmdb_paths:
        env = lmdb.open(p, readonly=True, lock=False,
                        readahead=False, meminit=False)
        n += env.stat()["entries"]
        env.close()
    return n


def iter_lmdb(
    lmdb_paths: List[str],
    skip_keys: Optional[Set[str]] = None,
    map_size_gb: int = 50,
) -> Iterator[Tuple[str, Any]]:
    skip = skip_keys or set()
    for path in lmdb_paths:
        env = lmdb.open(path, readonly=True, lock=False,
                        readahead=False, meminit=False,
                        map_size=map_size_gb * 1024 ** 3)
        with env.begin() as txn:
            for k, v in txn.cursor():
                ks = k.decode("utf-8")
                if ks in skip:
                    continue
                yield ks, pickle.loads(v)
        env.close()


def load_existing_keys(out_path: str) -> Set[str]:
    if not os.path.exists(out_path):
        return set()
    env = lmdb.open(out_path, readonly=True, lock=False, readahead=False)
    keys: Set[str] = set()
    with env.begin() as txn:
        for k, _ in txn.cursor():
            keys.add(k.decode("utf-8"))
    env.close()
    return keys


@contextmanager
def lmdb_writer(out_path: str, map_size_gb: int = 4, commit_every: int = 500):
    """Context manager that yields a `put(key, value)` callable batching writes."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    env = lmdb.open(out_path, map_size=map_size_gb * 1024 ** 3, subdir=True)
    pending: List[Tuple[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        with env.begin(write=True) as txn:
            for k, v in pending:
                txn.put(k.encode("utf-8"),
                        pickle.dumps(v, pickle.HIGHEST_PROTOCOL))
        pending.clear()

    def put(k: str, v: Any) -> None:
        pending.append((k, v))
        if len(pending) >= commit_every:
            flush()

    try:
        yield put
    finally:
        flush()
        env.close()
