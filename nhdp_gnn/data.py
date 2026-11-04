"""Dataset loading: Cora, Citeseer, PubMed, Amazon Photo, ogbn-arxiv/products/proteins, synthetic.

Everything is returned as numpy arrays in a ``GraphData``; edges are undirected
(both directions stored), without self-loops and without duplicates.
Bag-of-words features (Planetoid, Amazon) are row-normalised.
ogbn-proteins has no node features; they are the mean of the 8-dimensional
features of incident edges (the usual OGB practice).
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
AMAZON = {"photo": "Photo", "computers": "Computers"}
OGB = ("ogbn-arxiv", "ogbn-products", "ogbn-proteins")


@dataclass
class GraphData:
    name: str
    x: np.ndarray                         # (N, F) float32
    y: np.ndarray                         # (N,) int64, or (N, L) float32 for multi-label
    edge_index: np.ndarray                # (2, E) int64, undirected, no self-loops
    num_classes: int
    multilabel: bool = False
    official_split: tuple | None = None   # OGB (train, val, test) masks
    public_split: tuple | None = None     # Planetoid public split masks
    community: np.ndarray | None = None   # planted communities (synthetic only)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def metric(self) -> str:
        return "rocauc" if self.multilabel else "acc"


def undirected(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    src, dst = np.asarray(edge_index, dtype=np.int64)
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    keep = s != d
    key = np.unique(s[keep] * np.int64(num_nodes) + d[keep])
    return np.stack([key // num_nodes, key % num_nodes])


def _row_normalise(x: np.ndarray) -> np.ndarray:
    x = x - min(0.0, float(x.min()))
    s = x.sum(1, keepdims=True)
    return (x / np.maximum(s, 1.0)).astype(np.float32)


def _masks(n: int, idx_dict) -> tuple:
    out = []
    for key in ("train", "valid", "test"):
        m = np.zeros(n, dtype=bool)
        m[np.asarray(idx_dict[key]).reshape(-1)] = True
        out.append(m)
    return tuple(out)


def _interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (ValueError, OSError):
        return False


@contextmanager
def _ogb_compat():
    """Work around two behaviours of ogb 1.3.6, the latest release.

    * It calls ``torch.load`` on its own processed files without ``weights_only``.
      From torch 2.6 the default is ``weights_only=True``, which refuses the pickled
      PyG objects.  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD applies only to calls that do
      not pass ``weights_only``, so PyG's own loads are unaffected.
    * It asks on stdin before downloading more than 1 GB (ogbn-products), which
      raises EOFError in batch and torchrun jobs.  Without a terminal the download
      goes ahead; interactive sessions keep the prompt.
    """
    try:
        import ogb.nodeproppred.dataset_pyg as ogb_pyg
    except ImportError:
        ogb_pyg = None
    env_key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    old_env = os.environ.get(env_key)
    old_decide = getattr(ogb_pyg, "decide_download", None)
    os.environ[env_key] = "1"
    if old_decide is not None and not _interactive():
        def _decide(url):
            print(f"downloading {url} (no terminal to confirm)", flush=True)
            return True
        ogb_pyg.decide_download = _decide
    try:
        yield
    finally:
        if old_decide is not None:
            ogb_pyg.decide_download = old_decide
        if old_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_env


def load_dataset(name: str, root: str = "data", synthetic: dict | None = None) -> GraphData:
    name = name.lower()
    if name == "synthetic":
        from .synthetic import hierarchical_sbm

        g = hierarchical_sbm(**(synthetic or {}))
        return GraphData(name, g["x"], g["y"], undirected(g["edge_index"], g["x"].shape[0]),
                         int(g["num_classes"]), community=g["community"])

    if name in PLANETOID or name in AMAZON:
        from torch_geometric.datasets import Amazon, Planetoid

        if name in PLANETOID:
            data = Planetoid(f"{root}/planetoid", PLANETOID[name])[0]
            public = tuple(m.numpy().astype(bool) for m in (data.train_mask, data.val_mask, data.test_mask))
        else:
            data = Amazon(f"{root}/amazon", AMAZON[name])[0]
            public = None
        x = _row_normalise(data.x.numpy())
        y = data.y.numpy().astype(np.int64)
        return GraphData(name, x, y, undirected(data.edge_index.numpy(), x.shape[0]),
                         int(y.max()) + 1, public_split=public)

    if name in OGB:
        from ogb.nodeproppred import PygNodePropPredDataset

        with _ogb_compat():
            ds = PygNodePropPredDataset(name=name, root=f"{root}/ogb")
            data = ds[0]
            idx = ds.get_idx_split()
        n = int(data.num_nodes)
        split = _masks(n, {k: v.numpy() for k, v in idx.items()})
        ei = undirected(data.edge_index.numpy(), n)
        if name == "ogbn-proteins":
            ea = data.edge_attr.numpy().astype(np.float64)
            dst = data.edge_index.numpy()[1]
            x = np.stack([np.bincount(dst, weights=ea[:, c], minlength=n) for c in range(ea.shape[1])], axis=1)
            x /= np.maximum(np.bincount(dst, minlength=n), 1)[:, None]
            y = data.y.numpy().astype(np.float32)
            return GraphData(name, x.astype(np.float32), y, ei, y.shape[1], multilabel=True, official_split=split)
        x = data.x.numpy().astype(np.float32)
        y = data.y.numpy().reshape(-1).astype(np.int64)
        return GraphData(name, x, y, ei, int(y.max()) + 1, official_split=split)

    raise ValueError(f"unknown dataset {name!r}")
