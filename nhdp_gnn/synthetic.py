"""Synthetic hierarchical graphs with controllable structure (paper Sect. 10, last paragraph).

A hierarchical stochastic block model generated with NetworkX: communities are
nested in super-communities, and the generator exposes the properties the
paper lists as controllable -- connectivity (``p_in``, ``p_super``, ``p_out``),
community-class alignment / homophily (``class_alignment``) and the node
feature distribution (``feature_snr``, ``feature_dim``).

All communities of a super-community share a dominant class; each node keeps
that class with probability ``class_alignment`` and otherwise takes one of the
other classes uniformly.  ``class_alignment`` therefore sets the purity of the
true communities, which is the quantity that bounds what community-gated
propagation can achieve at large depth.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def hierarchical_sbm(num_super: int = 4, comm_per_super: int = 4, nodes_per_comm: int = 60,
                     p_in: float = 0.2, p_super: float = 0.03, p_out: float = 0.002,
                     num_classes: int = 4, class_alignment: float = 0.9,
                     feature_dim: int = 16, feature_snr: float = 1.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    J = num_super * comm_per_super
    N = J * nodes_per_comm
    super_of = np.repeat(np.arange(num_super), comm_per_super)
    P = np.full((J, J), p_out)
    P[super_of[:, None] == super_of[None, :]] = p_super
    np.fill_diagonal(P, p_in)
    G = nx.stochastic_block_model([nodes_per_comm] * J, P.tolist(), seed=int(seed))
    community = np.repeat(np.arange(J), nodes_per_comm)

    y = (super_of % num_classes)[community]
    flip = rng.random(N) >= class_alignment
    if num_classes > 1 and flip.any():
        y[flip] = (y[flip] + rng.integers(1, num_classes, size=int(flip.sum()))) % num_classes

    means = rng.normal(size=(num_classes, feature_dim)) * feature_snr
    x = (means[y] + rng.normal(size=(N, feature_dim))).astype(np.float32)

    e = np.asarray(list(G.edges()), dtype=np.int64).reshape(-1, 2).T
    edge_index = np.concatenate([e, e[::-1]], axis=1)
    return {"x": x, "y": y.astype(np.int64), "edge_index": edge_index,
            "community": community, "super_of": super_of, "num_classes": num_classes}
