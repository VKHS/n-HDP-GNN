import numpy as np
from scipy.sparse.linalg import eigsh

from nhdp_gnn.communities import louvain_levels, nested_communities
from nhdp_gnn.gating import (cross_partition_ratio, edge_reduction, gate_weights, gated_operator,
                             num_directed_edges, super_graph)
from nhdp_gnn.partition import partition_nodes
from nhdp_gnn.synthetic import hierarchical_sbm


def graph(seed=0, **kw):
    return hierarchical_sbm(num_super=3, comm_per_super=3, nodes_per_comm=30, seed=seed, **kw)


def test_hard_responsibilities_gate_exactly_intra_community_edges():
    g = graph()
    R = np.eye(g["community"].max() + 1)[g["community"]]
    w = gate_weights(g["edge_index"], R)
    src, dst = g["edge_index"]
    assert np.array_equal(w > 0.5, g["community"][src] == g["community"][dst])
    P = gated_operator(g["edge_index"], R.shape[0], R).tocoo()
    assert np.all(g["community"][P.row] == g["community"][P.col])      # block diagonal


def test_gated_operator_is_non_expansive():
    g = graph(seed=1)
    rng = np.random.default_rng(0)
    R = rng.dirichlet(np.ones(5) * 0.3, size=g["x"].shape[0])
    P = gated_operator(g["edge_index"], R.shape[0], R)
    lam = eigsh(P, k=1, which="LM", return_eigenvectors=False)[0]
    assert abs(lam) <= 1.0 + 1e-9


def test_super_graph_counts_inter_community_edges():
    g = graph(seed=2)
    C = g["community"].max() + 1
    R = np.eye(C)[g["community"]]
    AS, es = super_graph(g["edge_index"], R.shape[0], R)
    src, dst = g["edge_index"]
    expect = np.zeros((C, C))
    np.add.at(expect, (g["community"][src], g["community"][dst]), 1.0)
    assert np.allclose(AS, expect)
    r_e = edge_reduction(num_directed_edges(g["edge_index"]), es)
    assert 0.0 < r_e < 1.0


def test_cross_partition_ratio_and_gated_mass():
    ei = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    parts = np.array([0, 0, 1, 1])
    assert np.isclose(cross_partition_ratio(ei, parts), 2 / 6)
    w = np.array([1, 1, 0.1, 0.1, 1, 1])
    assert np.isclose(cross_partition_ratio(ei, parts, w), 0.2 / 4.2)


def test_louvain_dendrogram_is_nested_and_keeps_isolated_nodes():
    g = graph(seed=3)
    N = g["x"].shape[0]
    ei = g["edge_index"][:, g["edge_index"][0] != N - 1]
    ei = ei[:, ei[1] != N - 1]                          # isolate the last node
    levels = louvain_levels(ei, N, seed=0)
    for fine, coarse in zip(levels, levels[1:]):
        for c in np.unique(fine):
            assert np.unique(coarse[fine == c]).size == 1
    nc = nested_communities(ei, N, seed=0)
    assert (nc.community == nc.community[N - 1]).sum() == 1
    assert nc.super_of.shape[0] == nc.community.max() + 1


def test_per_partition_communities_do_not_cross_parts():
    g = graph(seed=4)
    N = g["x"].shape[0]
    parts = partition_nodes(g["edge_index"], N, 3, method="bfs")
    nc = nested_communities(g["edge_index"], N, seed=0, parts=parts)
    for c in np.unique(nc.community):
        assert np.unique(parts[nc.community == c]).size == 1


def test_partitioners_cover_all_nodes_and_balance():
    g = graph(seed=5)
    N = g["x"].shape[0]
    for method in ("bfs", "random"):
        parts = partition_nodes(g["edge_index"], N, 4, method=method)
        sizes = np.bincount(parts, minlength=4)
        assert sizes.sum() == N and sizes.max() - sizes.min() <= 1
    parts = partition_nodes(g["edge_index"], N, 3, method="community", communities=g["community"])
    sizes = np.bincount(parts, minlength=3)
    assert sizes.sum() == N and sizes.max() <= N / 3 + 30
    for c in np.unique(g["community"]):                 # communities below the cap stay whole
        assert np.unique(parts[g["community"] == c]).size == 1
    bfs = cross_partition_ratio(g["edge_index"], partition_nodes(g["edge_index"], N, 3, method="community",
                                                                 communities=g["community"]))
    rnd = cross_partition_ratio(g["edge_index"], partition_nodes(g["edge_index"], N, 3, method="random"))
    assert bfs < rnd
