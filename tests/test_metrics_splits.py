import numpy as np
import pytest

from nhdp_gnn.metrics import (centroid_gap, edge_homophily, kmeans_nmi_ari, madgap, mean_ci,
                              paired_wilcoxon, partition_purity)
from nhdp_gnn.splits import regime_split, stratified_split
from nhdp_gnn.synthetic import hierarchical_sbm


def test_centroid_gap_separated_vs_collapsed():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 50)
    sep = np.eye(3)[y] * 5 + 0.1 * rng.normal(size=(150, 3))
    collapsed = np.ones((150, 3)) + 1e-9 * rng.normal(size=(150, 3))
    assert centroid_gap(sep, y) > 0.5
    assert abs(centroid_gap(collapsed, y)) < 1e-6


def test_centroid_gap_multilabel_runs():
    rng = np.random.default_rng(1)
    Y = (rng.random((120, 5)) < 0.3).astype(int)
    H = Y @ rng.normal(size=(5, 8)) + 0.1 * rng.normal(size=(120, 8))
    assert np.isfinite(centroid_gap(H, Y))


def test_madgap_zero_for_constant_and_positive_for_spread_embeddings():
    N = 40
    ei = np.stack([np.arange(N - 1), np.arange(1, N)])
    ei = np.concatenate([ei, ei[::-1]], 1)                   # path graph
    assert madgap(np.ones((N, 4)), ei, N) == 0.0
    t = np.linspace(0, np.pi, N)
    H = np.stack([np.cos(t), np.sin(t)], 1)                  # smooth along the path
    assert madgap(H, ei, N) > 0.1


def test_kmeans_nmi_ari_perfect_and_multilabel():
    y = np.repeat(np.arange(4), 25)
    H = np.eye(4)[y] * 10
    out = kmeans_nmi_ari(H, y)
    assert out["nmi"] == pytest.approx(1.0) and out["ari"] == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    Y = (rng.random((100, 4)) < 0.4).astype(int)
    out = kmeans_nmi_ari(rng.normal(size=(100, 6)), Y)
    assert {"nmi", "ari", "nmi_dominant", "ari_dominant"} <= out.keys()


def test_partition_purity_and_homophily():
    y = np.array([0, 0, 1, 1, 1, 2])
    assert partition_purity(np.array([0, 0, 1, 1, 1, 2]), y) == 1.0
    assert partition_purity(np.zeros(6), y) == pytest.approx(3 / 6)
    ei = np.array([[0, 1, 2], [1, 2, 3]])
    assert edge_homophily(ei, y) == pytest.approx(2 / 3)


def test_mean_ci_and_wilcoxon():
    m, h = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and h == pytest.approx(2.4841, rel=1e-3)
    assert np.isnan(paired_wilcoxon([1, 2, 3], [1, 2, 3]))
    assert 0 <= paired_wilcoxon(np.arange(10) + 1.0, np.arange(10) + 0.5) <= 1


def test_splits_disjoint_stratified_and_regimes_share_val_test():
    y = np.repeat(np.arange(5), 100)
    tr, va, te = stratified_split(y, seed=0)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    assert np.all(np.bincount(y[tr]) == 60)
    semi = regime_split(y, "semi", seed=0)
    assert np.all(np.bincount(y[semi[0]]) == 20) and np.array_equal(semi[1], va)
    scarce = regime_split(y, "scarce", seed=0)
    assert scarce[0].sum() == 25 and np.array_equal(scarce[2], te)
    assert not (scarce[0] & ~tr).any()                      # drawn from the full training pool
    again = regime_split(y, "scarce", seed=0)
    assert np.array_equal(again[0], scarce[0])


def test_synthetic_alignment_controls_purity_and_homophily():
    hi = hierarchical_sbm(class_alignment=1.0, seed=0)
    lo = hierarchical_sbm(class_alignment=0.5, seed=0)
    assert partition_purity(hi["community"], hi["y"]) == 1.0
    assert 0.4 < partition_purity(lo["community"], lo["y"]) < 0.65
    assert edge_homophily(hi["edge_index"], hi["y"]) > edge_homophily(lo["edge_index"], lo["y"])
    src, dst = hi["edge_index"]
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert all((b, a) in pairs for a, b in pairs)              # undirected: both directions stored
