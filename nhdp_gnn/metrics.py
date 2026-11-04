"""Evaluation metrics: over-smoothing diagnostics, clustering quality, statistics.

madgap           MADGap of Chen et al. (AAAI 2020): mean cosine distance over remote
                 pairs (>= remote_hops) minus over neighbour pairs (1..neighbor_hops).
centroid_gap     The class-separation statistic defined in the paper's Eq. (28):
                 inter-class centroid cosine distance minus intra-class spread on
                 L2-normalised embeddings.  Multi-label data: per label (positive vs
                 negative centroid), micro-averaged by positive count.
kmeans_nmi_ari   k-means on L2-normalised embeddings, k = #classes; NMI and ARI
                 against the labels (protocol of the paper's Table 5).
partition_purity Accuracy of the best predictor that is constant within each cell of a
                 partition.  For propagation confined to communities this bounds the
                 accuracy of the infinite-depth limit (see docs/METHOD.md).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse, stats
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _l2n(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)


def _sym_adjacency(edge_index: np.ndarray, num_nodes: int) -> sparse.csr_matrix:
    src, dst = np.asarray(edge_index)
    keep = src != dst
    A = sparse.csr_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(num_nodes, num_nodes))
    return ((A + A.T) > 0).astype(np.float64).tocsr()


def madgap(H: np.ndarray, edge_index: np.ndarray, num_nodes: int, neighbor_hops: int = 3,
           remote_hops: int = 8, max_sources: int = 1000, max_targets: int = 20000,
           seed: int = 0, budget: float = 5e7) -> float:
    """Topological MADGap (Chen et al. 2020) on a seeded sample of source/target nodes.

    Unreachable pairs count as remote, as in the original mask construction.
    """
    rng = np.random.default_rng(seed)
    A = _sym_adjacency(edge_index, num_nodes)
    n_src = int(min(max_sources, num_nodes, max(16, budget // num_nodes)))
    sources = np.sort(rng.choice(num_nodes, size=n_src, replace=False))
    targets = (np.arange(num_nodes) if num_nodes <= max_targets
               else np.sort(rng.choice(num_nodes, size=max_targets, replace=False)))
    hops = csgraph.shortest_path(A, directed=False, unweighted=True, indices=sources)[:, targets]
    Hn = _l2n(H)
    D = np.clip(1.0 - Hn[sources] @ Hn[targets].T, 0.0, 2.0)

    def mad(mask: np.ndarray) -> float:
        Dm = D * mask
        cnt = (Dm > 0).sum(1)
        node = np.where(cnt > 0, Dm.sum(1) / np.maximum(cnt, 1), 0.0)
        return float(node[node > 0].mean()) if (node > 0).any() else 0.0

    return mad(hops >= remote_hops) - mad((hops >= 1) & (hops <= neighbor_hops))


def centroid_gap(H: np.ndarray, y: np.ndarray) -> float:
    """Paper Eq. (28): inter-class centroid distance minus intra-class spread (cosine)."""
    Hn = _l2n(H)
    y = np.asarray(y)
    if y.ndim == 1:
        classes, yi = np.unique(y, return_inverse=True)
        if classes.size < 2:
            return 0.0
        mu = _l2n(np.stack([Hn[yi == c].mean(0) for c in range(classes.size)]))
        cos = mu @ mu.T
        iu = np.triu_indices(classes.size, 1)
        inter = float(np.mean(1.0 - cos[iu]))
        intra = float(np.mean(1.0 - np.einsum("nd,nd->n", Hn, mu[yi])))
        return inter - intra
    gaps, weights = [], []
    for l in range(y.shape[1]):
        pos = y[:, l] > 0
        n_pos = int(pos.sum())
        if n_pos == 0 or n_pos == y.shape[0]:
            continue
        mu = _l2n(np.stack([Hn[pos].mean(0), Hn[~pos].mean(0)]))
        inter = 1.0 - float(mu[0] @ mu[1])
        own = np.where(pos[:, None], mu[0][None, :], mu[1][None, :])
        intra = float(np.mean(1.0 - np.einsum("nd,nd->n", Hn, own)))
        gaps.append(inter - intra)
        weights.append(n_pos)
    return float(np.average(gaps, weights=weights)) if gaps else 0.0


def kmeans_nmi_ari(H: np.ndarray, y: np.ndarray, seed: int = 0, n_init: int = 10) -> dict:
    """k-means (k = #classes) on L2-normalised embeddings; NMI/ARI vs labels.

    Multi-label: ``micro`` uses one k-means with k = #labels and averages per-label
    NMI/ARI weighted by positives; ``dominant`` assigns each node its globally most
    frequent positive label (nodes without labels form one extra class).
    """
    Hn = _l2n(H)
    y = np.asarray(y)
    if y.ndim == 1:
        k = np.unique(y).size
        pred = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(Hn)
        return {"nmi": float(normalized_mutual_info_score(y, pred)), "ari": float(adjusted_rand_score(y, pred))}
    yb = (y > 0).astype(np.int64)
    pos = yb.sum(0)
    valid = np.flatnonzero((pos > 0) & (pos < yb.shape[0]))
    pred = KMeans(n_clusters=max(2, valid.size), n_init=n_init, random_state=seed).fit_predict(Hn)
    nmi = [normalized_mutual_info_score(yb[:, l], pred) for l in valid]
    ari = [adjusted_rand_score(yb[:, l], pred) for l in valid]
    w = pos[valid]
    rank = np.argsort(np.argsort(-pos))                   # 0 = most frequent label
    masked = np.where(yb > 0, rank[None, :], yb.shape[1])
    dom = masked.min(1)
    k_dom = np.unique(dom).size
    pred_dom = KMeans(n_clusters=max(2, k_dom), n_init=n_init, random_state=seed).fit_predict(Hn)
    return {"nmi": float(np.average(nmi, weights=w)), "ari": float(np.average(ari, weights=w)),
            "nmi_dominant": float(normalized_mutual_info_score(dom, pred_dom)),
            "ari_dominant": float(adjusted_rand_score(dom, pred_dom))}


def partition_purity(cells: np.ndarray, y: np.ndarray) -> float:
    """sum over cells of the majority-class count, divided by N."""
    cells, y = np.asarray(cells), np.asarray(y)
    _, ci = np.unique(cells, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    table = sparse.csr_matrix((np.ones(y.size), (ci, yi))).toarray()
    return float(table.max(1).sum() / y.size)


def edge_homophily(edge_index: np.ndarray, y: np.ndarray) -> float:
    src, dst = np.asarray(edge_index)
    keep = src != dst
    return float(np.mean(np.asarray(y)[src[keep]] == np.asarray(y)[dst[keep]])) if keep.any() else 0.0


def mean_ci(values, conf: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the t-based confidence interval."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return (float(v.mean()) if v.size else float("nan")), float("nan")
    h = stats.t.ppf(0.5 + conf / 2.0, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(h)


def paired_wilcoxon(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired runs (same seeds)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2 or np.allclose(a, b):
        return float("nan")
    return float(stats.wilcoxon(a, b).pvalue)
