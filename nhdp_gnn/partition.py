"""One-time edge-cut partitioning of the graph (pipeline stage A, paper Sect. 10).

Methods
-------
metis      : METIS via ``pymetis`` (if installed).
community  : greedy longest-processing-time packing of Louvain communities into
             balanced parts; communities larger than the per-part cap are split
             along a BFS order ("capping/resharding oversized communities").
bfs        : contiguous chunks of a BFS ordering (locality without METIS).
random     : random equal-size chunks (no locality; reference point for rho).
auto       : metis if available, else bfs.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


def _csr(edge_index: np.ndarray, num_nodes: int) -> sparse.csr_matrix:
    src, dst = np.asarray(edge_index)
    keep = src != dst
    A = sparse.csr_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(num_nodes, num_nodes))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    return A


def bfs_order(edge_index: np.ndarray, num_nodes: int, seed: int = 0) -> np.ndarray:
    """BFS visiting order over all connected components (components in random order)."""
    A = _csr(edge_index, num_nodes)
    ncomp, comp = csgraph.connected_components(A, directed=False)
    rng = np.random.default_rng(seed)
    order = []
    for c in rng.permutation(ncomp):
        members = np.flatnonzero(comp == c)
        start = members[rng.integers(members.size)]
        order.append(csgraph.breadth_first_order(A, start, directed=False, return_predecessors=False))
    return np.concatenate(order)


def _chunks(order: np.ndarray, num_parts: int, num_nodes: int) -> np.ndarray:
    parts = np.empty(num_nodes, dtype=np.int64)
    for p, block in enumerate(np.array_split(order, num_parts)):
        parts[block] = p
    return parts


def _metis(edge_index: np.ndarray, num_nodes: int, num_parts: int, seed: int) -> np.ndarray:
    import pymetis

    A = _csr(edge_index, num_nodes)
    kwargs = {"xadj": A.indptr, "adjncy": A.indices}
    try:
        _, membership = pymetis.part_graph(num_parts, options=pymetis.Options(seed=seed), **kwargs)
    except (AttributeError, TypeError):          # older pymetis without seeded options
        _, membership = pymetis.part_graph(num_parts, **kwargs)
    return np.asarray(membership, dtype=np.int64)


def community_packing(communities: np.ndarray, num_parts: int, order: np.ndarray,
                      imbalance: float = 0.05) -> np.ndarray:
    """Pack communities into balanced parts; split communities above the cap along ``order``."""
    N = communities.shape[0]
    cap = int(np.ceil(N / num_parts * (1.0 + imbalance)))
    pos = np.empty(N, dtype=np.int64)
    pos[order] = np.arange(N)
    idx = np.lexsort((pos, communities))                 # by community, then BFS position
    bounds = np.flatnonzero(np.diff(communities[idx])) + 1
    pieces = []
    for members in np.split(idx, bounds):
        for k in range(0, members.size, cap):
            pieces.append(members[k:k + cap])
    pieces.sort(key=lambda m: (-m.size, int(m[0])))
    heap = [(0, p) for p in range(num_parts)]
    parts = np.empty(N, dtype=np.int64)
    for members in pieces:
        load, p = heapq.heappop(heap)
        parts[members] = p
        heapq.heappush(heap, (load + members.size, p))
    return parts


def partition_nodes(edge_index: np.ndarray, num_nodes: int, num_parts: int, method: str = "auto",
                    communities: np.ndarray | None = None, seed: int = 0, imbalance: float = 0.05) -> np.ndarray:
    """Node -> part id in 0..num_parts-1."""
    if num_parts <= 1:
        return np.zeros(num_nodes, dtype=np.int64)
    if method in ("auto", "metis"):
        try:
            return _metis(edge_index, num_nodes, num_parts, seed)
        except ImportError:
            if method == "metis":
                raise
            method = "bfs"
    if method == "bfs":
        return _chunks(bfs_order(edge_index, num_nodes, seed), num_parts, num_nodes)
    if method == "random":
        return _chunks(np.random.default_rng(seed).permutation(num_nodes), num_parts, num_nodes)
    if method == "community":
        if communities is None:
            raise ValueError("method='community' needs the Louvain communities")
        return community_packing(np.asarray(communities), num_parts,
                                 bfs_order(edge_index, num_nodes, seed), imbalance)
    raise ValueError(f"unknown partition method {method!r}")
