"""Small helpers shared by the pipeline: seeding, hashing, timing, JSON output."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def config_hash(obj: dict, length: int = 12) -> str:
    """Stable short hash of a JSON-serialisable dict (used for run and cache keys)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:length]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(path: str | os.PathLike, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
    os.replace(tmp, path)


def load_json(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


@contextmanager
def timer(store: dict, key: str):
    """Accumulate wall-clock seconds of a block into ``store[key]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0
