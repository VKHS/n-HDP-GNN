"""Stage A: nested Louvain communities (paper Sects. 3, 5 and 10).

The Louvain dendrogram supplies the nesting used by the n-HDP: the finest
level gives communities ``j`` and the coarsest level groups them into
super-communities ``s``.  When a node-to-partition map is given, Louvain runs
inside each partition separately, so no community crosses a partition
boundary (pipeline stage A of Sect. 10).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx


@dataclass
class NestedCommunities:
    community: np.ndarray        # (N,) fine community id, 0..J-1
    super_of: np.ndarray         # (J,) super-community id of each community, 0..S-1
    levels: int                  # number of Louvain levels found (max over partitions)

    @property
    def super_community(self) -> np.ndarray:
        return self.super_of[self.community]

    @property
    def num_communities(self) -> int:
        return int(self.super_of.shape[0])

    @property
    def num_super(self) -> int:
        return int(self.super_of.max()) + 1 if self.super_of.size else 0


def _canonical(labels: np.ndarray) -> np.ndarray:
    """Relabel to 0..C-1 in order of first appearance (deterministic)."""
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty_like(first)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    return rank[inv]


def louvain_levels(edge_index: np.ndarray, num_nodes: int, seed: int = 0,
                   resolution: float = 1.0, backend: str = "auto") -> list[np.ndarray]:
    """Louvain dendrogram as label arrays, finest level first.

    ``backend``: "networkx", "igraph" (C implementation, much faster on large
    graphs, used when installed and backend="auto"), or "auto".
    Isolated nodes form singleton communities at every level.
    """
    src, dst = np.asarray(edge_index)
    keep = src != dst
    src, dst = src[keep], dst[keep]

    if backend in ("auto", "igraph"):
        try:
            import igraph as ig
            import random

            random.seed(seed)
            ig.set_random_number_generator(random)
            g = ig.Graph(n=num_nodes, edges=np.stack([src, dst], 1).tolist(), directed=False)
            g.simplify()
            if g.ecount() == 0:
                return [np.arange(num_nodes)]
            levels = g.community_multilevel(return_levels=True, resolution=resolution)
            return [_canonical(np.asarray(lv.membership)) for lv in levels]
        except ImportError:
            if backend == "igraph":
                raise

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    out = []
    for partition in nx.community.louvain_partitions(G, resolution=resolution, seed=seed):
        lab = np.empty(num_nodes, dtype=np.int64)
        for c, members in enumerate(partition):
            lab[list(members)] = c
        out.append(_canonical(lab))
    return out or [np.arange(num_nodes)]


def _fine_and_super(levels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    fine = _canonical(levels[0])
    coarse = levels[-1]
    J = int(fine.max()) + 1
    super_raw = np.empty(J, dtype=np.int64)
    super_raw[fine] = coarse                     # nesting: each fine community lies in one coarse one
    return fine, _canonical(super_raw)


def nested_communities(edge_index: np.ndarray, num_nodes: int, seed: int = 0,
                       resolution: float = 1.0, backend: str = "auto",
                       parts: np.ndarray | None = None) -> NestedCommunities:
    """Fine Louvain communities nested in super-communities.

    If ``parts`` (node -> partition id) is given, Louvain runs on each
    partition's induced subgraph and ids are made globally unique.
    """
    edge_index = np.asarray(edge_index)
    if parts is None:
        levels = louvain_levels(edge_index, num_nodes, seed, resolution, backend)
        fine, sup = _fine_and_super(levels)
        return NestedCommunities(fine, sup, len(levels))

    parts = np.asarray(parts)
    community = np.full(num_nodes, -1, dtype=np.int64)  # parts with id < 0 are skipped (left at -1)
    super_list, j_off, s_off, max_levels = [], 0, 0, 1
    src, dst = edge_index
    same = parts[src] == parts[dst]
    for p in np.unique(parts[parts >= 0]):
        nodes = np.flatnonzero(parts == p)
        local = -np.ones(num_nodes, dtype=np.int64)
        local[nodes] = np.arange(nodes.size)
        m = same & (parts[src] == p)
        e_local = np.stack([local[src[m]], local[dst[m]]])
        levels = louvain_levels(e_local, nodes.size, seed, resolution, backend)
        fine, sup = _fine_and_super(levels)
        community[nodes] = fine + j_off
        super_list.append(sup + s_off)
        j_off += int(fine.max()) + 1
        s_off += int(sup.max()) + 1
        max_levels = max(max_levels, len(levels))
    return NestedCommunities(community, np.concatenate(super_list), max_levels)
