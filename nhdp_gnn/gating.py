"""Responsibility-gated propagation and locality metrics (paper Sect. 10).

The paper's operator ``P = sum_c Diag(pi_c) A_c``, with ``A_c`` the normalised
adjacency whose columns are masked by ``pi_c`` (``A_c = A_hat Diag(pi_c)``), is

    P_ij = A_hat_ij * <pi_i, pi_j>.

With one-hot responsibilities it keeps exactly the intra-community edges; with
soft responsibilities a cross-community edge keeps weight ``<pi_i, pi_j> < 1``.
Because ``0 <= <pi_i, pi_j> <= 1``, ``|P| <= A_hat`` entrywise, so the spectral
radius of ``P`` is at most that of ``A_hat`` (= 1).

The responsibilities are inferred once before training (Stage B), so the
operator is the same at every layer.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def gate_weights(edge_index: np.ndarray, R: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """g_ij = <pi_i, pi_j> for every edge (self-loops included if present)."""
    src, dst = np.asarray(edge_index)
    g = np.empty(src.shape[0], dtype=np.float64)
    for a in range(0, src.shape[0], chunk):
        b = min(a + chunk, src.shape[0])
        g[a:b] = np.einsum("ek,ek->e", R[src[a:b]], R[dst[a:b]])
    return np.clip(g, 0.0, 1.0)


def _binary_adjacency(edge_index: np.ndarray, num_nodes: int) -> sparse.csr_matrix:
    src, dst = np.asarray(edge_index)
    keep = src != dst
    A = sparse.csr_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(num_nodes, num_nodes))
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


def normalized_adjacency(edge_index: np.ndarray, num_nodes: int, self_loops: bool = True) -> sparse.csr_matrix:
    """A_hat = D^-1/2 (A + I) D^-1/2 (GCN normalisation)."""
    A = _binary_adjacency(edge_index, num_nodes)
    if self_loops:
        A = A + sparse.eye(num_nodes, format="csr")
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    D = sparse.diags(dinv)
    return (D @ A @ D).tocsr()


def gated_operator(edge_index: np.ndarray, num_nodes: int, R: np.ndarray,
                   self_loops: bool = True) -> sparse.csr_matrix:
    """P = sum_c Diag(pi_c) A_hat Diag(pi_c), i.e. P_ij = A_hat_ij <pi_i, pi_j>."""
    A = normalized_adjacency(edge_index, num_nodes, self_loops).tocoo()
    g = np.einsum("ek,ek->e", R[A.row], R[A.col])
    P = sparse.csr_matrix((A.data * g, (A.row, A.col)), shape=A.shape)
    P.eliminate_zeros()
    return P


def super_graph(edge_index: np.ndarray, num_nodes: int, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Super-node graph A_S = R^T A R and its off-diagonal edge list (2, |E_s|)."""
    A = _binary_adjacency(edge_index, num_nodes)
    AS = np.asarray(R.T @ (A @ R))
    off = AS.copy()
    np.fill_diagonal(off, 0.0)
    rows, cols = np.nonzero(off > 0)
    return AS, np.stack([rows, cols])


def num_directed_edges(edge_index: np.ndarray) -> int:
    src, dst = np.asarray(edge_index)
    return int((src != dst).sum())


def edge_reduction(num_edges: int, super_edge_index: np.ndarray) -> float:
    """r_E = 1 - |E_s| / |E|, both counted as directed edge entries without self-loops."""
    return 1.0 - super_edge_index.shape[1] / max(num_edges, 1)


def cross_partition_ratio(edge_index: np.ndarray, parts: np.ndarray, weights: np.ndarray | None = None) -> float:
    """rho = |cut edges| / |E|.

    With ``weights`` (gate values per edge) it returns rho_g, the share of the
    gated propagation mass that crosses partition boundaries; this is the
    quantity the gate can actually change, since rho itself depends only on
    the partition.
    """
    src, dst = np.asarray(edge_index)
    keep = src != dst
    cut = parts[src[keep]] != parts[dst[keep]]
    if cut.size == 0:
        return 0.0
    if weights is None:
        return float(cut.mean())
    w = np.asarray(weights)[keep]
    total = w.sum()
    return float(w[cut].sum() / total) if total > 0 else 0.0
