"""Raw-file readers for Planetoid and Amazon (numpy only).

Synthetic raw files exercise the processing rules; the real datasets are checked
against PyTorch Geometric's published statistics when they can be read from
``$NHDP_DATA_ROOT`` (default ``data``) or downloaded.  tests/test_torch.py checks
equality with PyG's own processing.
"""
import os
import pickle
from collections import defaultdict

import numpy as np
import pytest

from nhdp_gnn.data import load_dataset, read_amazon, read_planetoid


def _write_planetoid(raw, name, x, tx, allx, y, ty, ally, graph, test_index):
    raw.mkdir(parents=True, exist_ok=True)
    for part, obj in dict(x=x, tx=tx, allx=allx, y=y, ty=ty, ally=ally, graph=graph).items():
        (raw / f"ind.{name}.{part}").write_bytes(pickle.dumps(obj, protocol=2))
    (raw / f"ind.{name}.test.index").write_text("\n".join(str(i) for i in test_index))


def _onehot(labels, k):
    return np.eye(k, dtype=np.float64)[labels]


def test_planetoid_reader_reorders_test_nodes_and_builds_public_split(tmp_path):
    rng = np.random.default_rng(0)
    # 2 training nodes, 1 extra unlabelled node in allx, 3 test nodes stored in shuffled order
    allx, tx = rng.random((3, 4)), rng.random((3, 4))
    ally, ty = _onehot([0, 1, 1], 2), _onehot([1, 0, 0], 2)
    test_index = [5, 3, 4]
    graph = defaultdict(list, {0: [1, 3], 1: [0], 2: [2, 4], 3: [0], 4: [2, 5], 5: [4]})
    _write_planetoid(tmp_path / "planetoid" / "Cora" / "raw", "cora", allx[:2], tx, allx,
                     ally[:2], ty, ally, graph, test_index)
    x, y, ei, (tr, va, te) = read_planetoid(tmp_path, "cora")
    # PyG: x = [allx; tx], then x[test_index] = x[sorted(test_index)]
    expect = np.concatenate([allx, tx]).astype(np.float32)
    expect[test_index] = expect[np.sort(test_index)]
    assert np.array_equal(x, expect) and x.dtype == np.float32
    labels = np.concatenate([ally, ty]).argmax(1)
    labels[test_index] = labels[np.sort(test_index)]
    assert np.array_equal(y, labels)
    assert tr.tolist() == [True, True, False, False, False, False]
    assert va.sum() == 4 and va[2:].all()                     # the next 500 indices, clipped at N
    assert te.tolist() == [False, False, False, True, True, True]
    assert ei.shape == (2, sum(len(v) for v in graph.values()))

    data = load_dataset("cora", tmp_path)                     # undirected, no self-loops, row-normalised
    src, dst = data.edge_index
    assert not np.any(src == dst)
    assert set(zip(src.tolist(), dst.tolist())) == set(zip(dst.tolist(), src.tolist()))
    sums = x.sum(1)                                           # divided by the row sum clamped at 1, as in PyG
    assert np.allclose(data.x, x / np.maximum(sums, 1.0)[:, None], atol=1e-7)


def test_citeseer_isolated_test_nodes_become_zero_rows(tmp_path):
    rng = np.random.default_rng(1)
    allx, tx = rng.random((2, 3)), rng.random((2, 3))
    ally, ty = _onehot([0, 1], 2), _onehot([1, 1], 2)
    test_index = [2, 4]                                       # index 3 is missing from the test file
    graph = defaultdict(list, {0: [1], 1: [0], 2: [4], 3: [], 4: [2]})
    _write_planetoid(tmp_path / "planetoid" / "CiteSeer" / "raw", "citeseer", allx[:1], tx, allx,
                     ally[:1], ty, ally, graph, test_index)
    x, y, _, (_, _, te) = read_planetoid(tmp_path, "citeseer")
    assert x.shape == (5, 3) and np.all(x[3] == 0) and y[3] == 0
    assert te.tolist() == [False, False, True, False, True]


class _Evil:
    def __reduce__(self):
        return (os.system, ("true",))


def test_planetoid_reader_refuses_foreign_pickles(tmp_path):
    raw = tmp_path / "planetoid" / "Cora" / "raw"
    _write_planetoid(raw, "cora", np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((2, 2)), _onehot([0], 2),
                     _onehot([1], 2), _onehot([0, 1], 2), defaultdict(list), [2])
    (raw / "ind.cora.x").write_bytes(pickle.dumps(_Evil(), protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        read_planetoid(tmp_path, "cora")


def test_amazon_reader_binarises_features_and_keeps_edges(tmp_path):
    import scipy.sparse as sp

    attr = sp.csr_matrix(np.array([[0, 2.0, 0], [3.0, 0, 0], [0, 0, 0.5]]))
    adj = sp.csr_matrix(np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32))   # self-loop at 0
    raw = tmp_path / "amazon" / "Photo" / "raw"
    raw.mkdir(parents=True)
    np.savez(raw / "amazon_electronics_photo.npz", attr_data=attr.data, attr_indices=attr.indices,
             attr_indptr=attr.indptr, attr_shape=attr.shape, adj_data=adj.data, adj_indices=adj.indices,
             adj_indptr=adj.indptr, adj_shape=adj.shape, labels=np.array([0, 1, 1]))
    x, y, ei = read_amazon(tmp_path, "photo")
    assert np.array_equal(x, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32))
    assert y.tolist() == [0, 1, 1]
    data = load_dataset("photo", tmp_path)
    assert sorted(zip(*data.edge_index.tolist())) == [(0, 1), (1, 0), (1, 2), (2, 1)]


# PyTorch Geometric documentation: nodes, directed edges, features, classes; public split sizes
STATS = {
    "cora": (2708, 10556, 1433, 7, (140, 500, 1000)),
    "citeseer": (3327, 9104, 3703, 6, (120, 500, 1000)),
    "pubmed": (19717, 88648, 500, 3, (60, 500, 1000)),
    "photo": (7650, 238162, 745, 8, None),
    "computers": (13752, 491722, 767, 10, None),
}


@pytest.mark.parametrize("name", sorted(STATS))
def test_real_datasets_match_published_statistics(name):
    try:
        d = load_dataset(name, os.environ.get("NHDP_DATA_ROOT", "data"))
    except OSError as exc:                                    # offline and not cached
        pytest.skip(f"{name} not available: {exc}")
    n, e, f, c, split = STATS[name]
    assert (d.num_nodes, d.edge_index.shape[1], d.x.shape[1], d.num_classes) == (n, e, f, c)
    if split is not None:
        assert tuple(int(m.sum()) for m in d.public_split) == split
