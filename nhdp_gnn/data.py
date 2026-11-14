"""Dataset loading: Cora, Citeseer, PubMed, Amazon Photo/Computers, ogbn-arxiv/products/proteins, synthetic.

Everything is returned as numpy arrays in a ``GraphData``; edges are undirected
(both directions stored), without self-loops and without duplicates.
Bag-of-words features (Planetoid, Amazon) are row-normalised.
ogbn-proteins has no node features; they are the mean of the 8-dimensional
features of incident edges (the usual OGB practice).

Planetoid and Amazon graphs are read from the original raw files (the files
PyTorch Geometric downloads, stored in the same directory layout), with the same
processing as PyG's ``Planetoid`` and ``Amazon`` datasets; tests/test_torch.py
checks the equality.  They therefore load without PyTorch, so stages A and B run
on real graphs in a numpy-only environment.  The pickled Planetoid files are
read with an unpickler that only admits the numpy/scipy classes they contain.
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import shutil
import sys
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
AMAZON = {"photo": "Photo", "computers": "Computers"}
OGB = ("ogbn-arxiv", "ogbn-products", "ogbn-proteins")
PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"      # as in PyG
AMAZON_URL = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz"   # as in PyG


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


def _fetch(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` unless it exists (atomic rename, so no partial files remain)."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    os.replace(tmp, path)
    return path


# (module, name) pairs that occur in the Planetoid pickles, in Python 2 and 3 spellings
_PICKLE_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("scipy.sparse.csr", "csr_matrix"), ("scipy.sparse", "csr_matrix"),
    ("collections", "defaultdict"),
    ("__builtin__", "object"), ("builtins", "object"), ("__builtin__", "list"), ("builtins", "list"),
    ("copy_reg", "_reconstructor"), ("copyreg", "_reconstructor"),
    ("_codecs", "encode"),     # bytes in protocol-2 pickles written by Python 3 (no code execution)
}


class _DatasetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _PICKLE_ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a dataset file")
        if module.startswith("numpy.core") and importlib.util.find_spec("numpy._core") is not None:
            module = "numpy._core" + module[len("numpy.core"):]      # numpy >= 2
        if module == "scipy.sparse.csr":
            module = "scipy.sparse"
        return super().find_class(module, name)


def _dense32(m) -> np.ndarray:
    return np.asarray(m.todense() if hasattr(m, "todense") else m, dtype=np.float32)


def read_planetoid(root: str | Path, name: str):
    """Features, labels, raw edge list and public split, processed as PyG's ``read_planetoid_data``."""
    raw = Path(root) / "planetoid" / PLANETOID[name] / "raw"
    items = {}
    for part in ("x", "tx", "allx", "y", "ty", "ally", "graph"):
        with open(_fetch(f"{PLANETOID_URL}/ind.{name}.{part}", raw / f"ind.{name}.{part}"), "rb") as fh:
            items[part] = _DatasetUnpickler(fh, encoding="latin1").load()
    test_index = np.loadtxt(_fetch(f"{PLANETOID_URL}/ind.{name}.test.index", raw / f"ind.{name}.test.index"),
                            dtype=np.int64).reshape(-1)
    tx, allx = _dense32(items["tx"]), _dense32(items["allx"])
    y, ty, ally = _dense32(items["y"]), _dense32(items["ty"]), _dense32(items["ally"])
    sorted_test = np.sort(test_index)
    if name == "citeseer":       # isolated test nodes: test indices are not consecutive
        n_ext = int(test_index.max() - test_index.min()) + 1
        tx_ext = np.zeros((n_ext, tx.shape[1]), dtype=np.float32)
        tx_ext[sorted_test - test_index.min()] = tx
        ty_ext = np.zeros((n_ext, ty.shape[1]), dtype=np.float32)
        ty_ext[sorted_test - test_index.min()] = ty
        tx, ty = tx_ext, ty_ext
    x = np.concatenate([allx, tx])
    x[test_index] = x[sorted_test]
    labels = np.concatenate([ally, ty]).argmax(1).astype(np.int64)
    labels[test_index] = labels[sorted_test]
    n = labels.shape[0]
    graph = items["graph"]
    rows = np.fromiter((u for u, nbrs in graph.items() for _ in nbrs), dtype=np.int64)
    cols = np.fromiter((v for nbrs in graph.values() for v in nbrs), dtype=np.int64)
    if rows.size and max(rows.max(), cols.max()) >= n:
        raise ValueError(f"{name}: edge endpoint outside the {n} nodes")
    masks = []
    for idx in (np.arange(y.shape[0]), np.arange(y.shape[0], min(y.shape[0] + 500, n)), test_index):
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        masks.append(m)
    return x, labels, np.stack([rows, cols]), tuple(masks)


def read_amazon(root: str | Path, name: str):
    """Binary features, labels and raw edge list, processed as PyG's ``read_npz``."""
    import scipy.sparse as sp

    file = f"amazon_electronics_{name}.npz"
    path = _fetch(f"{AMAZON_URL}/{file}", Path(root) / "amazon" / AMAZON[name] / "raw" / file)
    with np.load(path, allow_pickle=False) as f:
        x = sp.csr_matrix((f["attr_data"], f["attr_indices"], f["attr_indptr"]),
                          shape=tuple(f["attr_shape"])).toarray().astype(np.float32)
        adj = sp.csr_matrix((f["adj_data"], f["adj_indices"], f["adj_indptr"]), shape=tuple(f["adj_shape"])).tocoo()
        labels = np.asarray(f["labels"]).astype(np.int64)
    x[x > 0] = 1.0
    return x, labels, np.stack([adj.row, adj.col]).astype(np.int64)


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
        if name in PLANETOID:
            x, y, ei, public = read_planetoid(root, name)
        else:
            (x, y, ei), public = read_amazon(root, name), None
        x = _row_normalise(x)
        return GraphData(name, x, y, undirected(ei, x.shape[0]), int(y.max()) + 1, public_split=public)

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
