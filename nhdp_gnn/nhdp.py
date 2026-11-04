"""Stage B: nested hierarchical Dirichlet process over Louvain communities.

Generative model (paper Fig. 4 and Eqs. 5-7, 11-14).  Groups are the nested
Louvain communities of ``communities.py``: community ``j`` lies in
super-community ``s(j)``.

    global      beta'_k ~ Beta(1, gamma),   theta_k = (mu_k, lambda_k) ~ NormalGamma(m0, kappa0, a0, b0)
    super s     pi'_st  ~ Beta(1, alpha1),  c_st ~ Cat(beta)          (t = 1..T1, G_0k in the paper)
    community j pi'_ju  ~ Beta(1, alpha0),  d_ju ~ Cat(pi_s(j))       (u = 1..T2, G_j / G_jk)
    node i in j z_i ~ Cat(pi_j),            x_i ~ N(mu_k, diag(1/lambda_k)) with k = c_{s, d_{j, z_i}}

``levels=2`` removes the super-community level (a two-level HDP: each
community draws its atoms directly from beta), which is the HDP baseline of
the paper's Table 1 at matched truncation.

Inference is mean-field coordinate ascent with closed-form updates (the
three-level extension of Wang, Paisley & Blei, AISTATS 2011).  Every update
maximises the evidence lower bound in its block, so the ELBO is monotone; it
is computed exactly and returned as ``elbo_trace``.  The node
responsibilities ``r_ik = q(global atom of node i = k)`` are the paper's
``pi_ic`` that gate message passing.

The model factorises over super-communities, so nodes can be split into
shards (graph partitions).  Local updates are independent per shard; only the
global sufficient statistics (stick counts and Normal-Gamma moments, a few
kilobytes) are summed across shards and, in distributed runs, across ranks
through the ``reduce`` callable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.special import digamma, gammaln

_TINY = 1e-300


@dataclass
class NHDPConfig:
    K: int = 64            # global truncation level
    T1: int = 16           # super-community truncation (ignored when levels=2)
    T2: int = 8            # community truncation (levels=2 uses T1 for the community level)
    gamma: float = 1.0
    alpha1: float = 1.0
    alpha0: float = 1.0
    m0: float = 0.0
    kappa0: float = 0.1
    a0: float = 1.0
    b0: float = 1.0
    levels: int = 3
    max_iter: int = 100
    tol: float = 1e-6      # relative ELBO change for convergence
    init_sample: int = 20000
    seed: int = 0


@dataclass
class NHDPResult:
    R: np.ndarray                      # (N, K) responsibilities, rows sum to 1
    elbo_trace: list = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    Nk: np.ndarray | None = None       # expected node count per global atom

    @property
    def effective_k(self) -> int:
        """Atoms carrying at least one expected node."""
        return int((self.Nk >= 1.0).sum()) if self.Nk is not None else 0


# ----------------------------------------------------------------------------- helpers
def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=axis, keepdims=True)


def _xlogx_free(p: np.ndarray, elog: np.ndarray) -> float:
    """sum p * (elog - log p), the categorical prior + entropy terms."""
    return float(np.sum(p * (elog - np.log(np.maximum(p, _TINY)))))


def elog_sticks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """E[log w_t] under truncated stick-breaking, q(v_t)=Beta(a_t, b_t) for t<T, v_T = 1.

    a, b: (..., T-1) -> (..., T)
    """
    dab = digamma(a + b)
    elog_v = digamma(a) - dab
    elog_1mv = digamma(b) - dab
    out = np.zeros(a.shape[:-1] + (a.shape[-1] + 1,))
    out[..., :-1] = elog_v
    out[..., 1:] += np.cumsum(elog_1mv, axis=-1)
    return out


def stick_posterior(counts: np.ndarray, conc: float) -> tuple[np.ndarray, np.ndarray]:
    """Beta parameters of q(v_t), t<T, given expected counts (..., T)."""
    tail = np.cumsum(counts[..., ::-1], axis=-1)[..., ::-1]
    return 1.0 + counts[..., :-1], conc + tail[..., 1:]


def kl_beta(a, b, a0, b0) -> np.ndarray:
    """KL(Beta(a, b) || Beta(a0, b0)), elementwise."""
    return (gammaln(a + b) - gammaln(a) - gammaln(b)
            - gammaln(a0 + b0) + gammaln(a0) + gammaln(b0)
            + (a - a0) * digamma(a) + (b - b0) * digamma(b)
            + (a0 - a + b0 - b) * digamma(a + b))


def kl_normal_gamma(m, kappa, a, b, m0, kappa0, a0, b0) -> float:
    """sum_kd KL(NG(m, kappa, a, b) || NG(m0, kappa0, a0, b0)); m, b: (K, d); kappa, a: (K,)."""
    kappa, a = kappa[:, None], a[:, None]
    kl_norm = 0.5 * (np.log(kappa / kappa0) + kappa0 / kappa - 1.0 + kappa0 * (a / b) * (m - m0) ** 2)
    kl_gam = ((a - a0) * digamma(a) - gammaln(a) + gammaln(a0)
              + a0 * (np.log(b) - np.log(b0)) + a * (b0 - b) / b)
    return float(np.sum(kl_norm + kl_gam))


def expected_loglik(X, X2, m, kappa, a, b) -> np.ndarray:
    """l_ik = E_q[log N(x_i | mu_k, diag(1/lambda_k))], shape (n, K)."""
    d = X.shape[1]
    e_lam = a[:, None] / b
    elog_lam = digamma(a)[:, None] - np.log(b)
    quad = X2 @ e_lam.T - 2.0 * (X @ (e_lam * m).T) + (e_lam * m ** 2).sum(1)[None, :]
    return (0.5 * elog_lam.sum(1)[None, :] - 0.5 * d * np.log(2.0 * np.pi)
            - 0.5 * quad - 0.5 * d / kappa[None, :])


def reduce_features(X, dim: int = 32, seed: int = 0) -> np.ndarray:
    """Project node features to ``dim`` components (TruncatedSVD) and standardise."""
    from sklearn.decomposition import TruncatedSVD

    if X.shape[1] > dim:
        Z = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    else:
        Z = X.toarray() if sparse.issparse(X) else X
    Z = np.asarray(Z, dtype=np.float64)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)


def topk_responsibilities(R: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest responsibilities per node and renormalise (paper: top-k sparse pi)."""
    if k >= R.shape[1]:
        return R / R.sum(1, keepdims=True)
    idx = np.argpartition(-R, k - 1, axis=1)[:, :k]
    rows = np.arange(R.shape[0])[:, None]
    out = np.zeros_like(R)
    out[rows, idx] = R[rows, idx]
    s = out.sum(1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


# ----------------------------------------------------------------------------- shards
class Shard:
    """Nodes of one graph partition, ordered by (super-community, community)."""

    def __init__(self, nodes: np.ndarray, X: np.ndarray, community: np.ndarray,
                 super_of: np.ndarray, cfg: NHDPConfig):
        nodes, community = np.asarray(nodes), np.asarray(community)
        uniq_c, comm0 = np.unique(community, return_inverse=True)
        if cfg.levels == 2:
            sup0 = np.arange(uniq_c.size)               # every community is its own group
        else:
            sup0 = np.unique(np.asarray(super_of)[uniq_c], return_inverse=True)[1]
        order = np.lexsort((comm0, sup0[comm0]))
        comm_sorted = comm0[order]
        # communities relabelled 0..J-1 in sorted order, supers 0..S-1 likewise
        new_c = np.full(uniq_c.size, -1)
        first = np.unique(comm_sorted, return_index=True)[1]
        appear = comm_sorted[np.sort(first)]
        new_c[appear] = np.arange(appear.size)
        self.comm = new_c[comm_sorted]
        sup_of_c = np.empty(appear.size, dtype=np.int64)
        sup_of_c[new_c[appear]] = sup0[appear]
        _, self.sup_of = np.unique(sup_of_c, return_inverse=True)   # supers sorted -> contiguous
        self.levels = cfg.levels
        self.nodes = nodes[order]
        self.X = np.ascontiguousarray(np.asarray(X, dtype=np.float64)[order])
        self.X2 = self.X ** 2
        self.n, self.J = self.X.shape[0], appear.size
        self.S = int(self.sup_of.max()) + 1
        self.s_node = self.sup_of[self.comm]
        self.T1 = cfg.T1
        self.T2 = cfg.T1 if cfg.levels == 2 else cfg.T2
        self.C = sparse.csr_matrix((np.ones(self.n), (self.comm, np.arange(self.n))), shape=(self.J, self.n))
        self.Csup = sparse.csr_matrix((np.ones(self.J), (self.sup_of, np.arange(self.J))), shape=(self.S, self.J))

    # -- initialisation from a hard assignment h (n,) of nodes to global atoms
    def init_structure(self, h: np.ndarray, cfg: NHDPConfig) -> None:
        K, T1, T2, eps = cfg.K, self.T1, self.T2, 0.05
        counts_s = np.zeros((self.S, K))
        np.add.at(counts_s, (self.s_node, h), 1.0)
        top = np.argsort(-counts_s, axis=1, kind="stable")[:, :T1]           # (S, T1) atom per slot
        self.phi = np.full((self.S, T1, K), eps / K)
        self.phi[np.arange(self.S)[:, None], np.arange(T1)[None, :], top] += 1.0 - eps
        if self.levels == 2:
            self.psi = np.broadcast_to(np.eye(T1), (self.J, T1, T1)).copy()
        else:
            slot_of_atom = -np.ones((self.S, K), dtype=np.int64)
            slot_of_atom[np.arange(self.S)[:, None], top] = np.arange(T1)[None, :]
            node_slot = slot_of_atom[self.s_node, h]
            usage = np.zeros((self.J, T1))
            ok = node_slot >= 0
            np.add.at(usage, (self.comm[ok], node_slot[ok]), 1.0)
            top_j = np.argsort(-usage, axis=1, kind="stable")[:, :T2]
            self.psi = np.full((self.J, T2, T1), eps / T1)
            self.psi[np.arange(self.J)[:, None], np.arange(T2)[None, :], top_j] += 1.0 - eps
        self.pj_a, self.pj_b = np.ones((self.J, T2 - 1)), np.full((self.J, T2 - 1), cfg.alpha0)
        self.ps_a, self.ps_b = np.ones((self.S, T1 - 1)), np.full((self.S, T1 - 1), cfg.alpha1)
        self.r = np.zeros((self.n, K))
        self.r[np.arange(self.n), h] = 1.0
        self.zeta = None

    # -- one sweep of local coordinate-ascent updates
    def update(self, ell: np.ndarray, elog_beta: np.ndarray, cfg: NHDPConfig) -> None:
        n, T1, T2, K = self.n, self.T1, self.T2, ell.shape[1]
        M = np.empty((n, T1))                                  # M_it = sum_k phi_{s(i) t k} l_ik
        for t in range(T1):
            M[:, t] = np.einsum("nk,nk->n", ell, self.phi[self.s_node, t])
        A = np.empty((n, T2))                                  # A_iu = sum_t psi_{j(i) u t} M_it
        for u in range(T2):
            A[:, u] = np.einsum("nt,nt->n", M, self.psi[self.comm, u])
        self.zeta = _softmax(A + elog_sticks(self.pj_a, self.pj_b)[self.comm], axis=1)
        self.pj_a, self.pj_b = stick_posterior(self.C @ self.zeta, cfg.alpha0)
        if self.levels == 3:
            Z = np.stack([self.C @ (M * self.zeta[:, [u]]) for u in range(T2)], axis=1)   # (J, T2, T1)
            elog_ps = elog_sticks(self.ps_a, self.ps_b)
            self.psi = _softmax(Z + elog_ps[self.sup_of][:, None, :], axis=2)
            self.ps_a, self.ps_b = stick_posterior(self.Csup @ self.psi.sum(1), cfg.alpha1)
        W = np.stack([self.C @ (ell * self.zeta[:, [u]]) for u in range(T2)], axis=1)      # (J, T2, K)
        Gj = np.einsum("jut,juk->jtk", self.psi, W)
        G = (self.Csup @ Gj.reshape(self.J, -1)).reshape(self.S, T1, K)
        self.phi = _softmax(G + elog_beta[None, None, :], axis=2)
        Q = np.einsum("jut,jtk->juk", self.psi, self.phi[self.sup_of])                    # (J, T2, K)
        r = np.zeros((n, K))
        for u in range(T2):
            r += self.zeta[:, [u]] * Q[self.comm, u]
        self.r = r

    def elbo_local(self, ell: np.ndarray, elog_beta: np.ndarray, cfg: NHDPConfig) -> float:
        """Local ELBO terms at the current state (requires one completed update)."""
        out = _xlogx_free(self.zeta, elog_sticks(self.pj_a, self.pj_b)[self.comm])
        out -= float(kl_beta(self.pj_a, self.pj_b, 1.0, cfg.alpha0).sum())
        if self.levels == 3:
            out += _xlogx_free(self.psi, elog_sticks(self.ps_a, self.ps_b)[self.sup_of][:, None, :])
            out -= float(kl_beta(self.ps_a, self.ps_b, 1.0, cfg.alpha1).sum())
        out += _xlogx_free(self.phi, np.broadcast_to(elog_beta, self.phi.shape))
        out += float(np.sum(self.r * ell))
        return out

    def stats(self) -> dict:
        return {"Nk": self.r.sum(0), "Sx": self.r.T @ self.X, "Sxx": self.r.T @ self.X2,
                "beta_counts": self.phi.sum((0, 1))}


def make_shards(X: np.ndarray, community: np.ndarray, super_of: np.ndarray,
                cfg: NHDPConfig, parts: np.ndarray | None = None) -> list[Shard]:
    """One shard per partition (or one shard for the whole graph)."""
    N = X.shape[0]
    if parts is None:
        return [Shard(np.arange(N), X, community, super_of, cfg)]
    shards = []
    for p in np.unique(parts):
        nodes = np.flatnonzero(parts == p)
        shards.append(Shard(nodes, X[nodes], community[nodes], super_of, cfg))
    return shards


# ----------------------------------------------------------------------------- inference
class _Global:
    def __init__(self, cfg: NHDPConfig, d: int):
        self.cfg = cfg
        self.u = np.ones(cfg.K - 1)
        self.v = np.full(cfg.K - 1, cfg.gamma)
        self.m = np.zeros((cfg.K, d))
        self.kappa = np.full(cfg.K, cfg.kappa0)
        self.a = np.full(cfg.K, cfg.a0)
        self.b = np.full((cfg.K, d), cfg.b0)

    def update(self, st: dict) -> None:
        c = self.cfg
        self.u, self.v = stick_posterior(st["beta_counts"], c.gamma)
        Nk, Sx, Sxx = st["Nk"], st["Sx"], st["Sxx"]
        self.kappa = c.kappa0 + Nk
        self.m = (c.kappa0 * c.m0 + Sx) / self.kappa[:, None]
        self.a = c.a0 + 0.5 * Nk
        bracket = Sxx + c.kappa0 * c.m0 ** 2 - (Sx + c.kappa0 * c.m0) ** 2 / self.kappa[:, None]
        self.b = c.b0 + 0.5 * np.maximum(bracket, 0.0)

    def elbo(self) -> float:
        c = self.cfg
        return (-float(kl_beta(self.u, self.v, 1.0, c.gamma).sum())
                - kl_normal_gamma(self.m, self.kappa, self.a, self.b, c.m0, c.kappa0, c.a0, c.b0))


def _sum_stats(shards: list[Shard]) -> dict:
    keys = ("Nk", "Sx", "Sxx", "beta_counts")
    out = {k: 0.0 for k in keys}
    for sh in shards:
        st = sh.stats()
        for k in keys:
            out[k] = out[k] + st[k]
    return out


def init_atoms(X_init: np.ndarray, K: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    if X_init.shape[0] < K:
        raise ValueError(f"need at least K={K} nodes to initialise the atoms, got {X_init.shape[0]}")
    return KMeans(n_clusters=K, n_init=3, random_state=seed).fit(X_init).cluster_centers_


def fit_nhdp(shards: list[Shard], cfg: NHDPConfig, reduce=None, X_init: np.ndarray | None = None,
             num_nodes: int | None = None, verbose: bool = False) -> NHDPResult:
    """Mean-field coordinate ascent for the nested HDP.

    reduce: callable(dict[str, ndarray]) -> dict summing over ranks (None for one process).
    X_init: rows used for the k-means initialisation of the global atoms.  Must be
            identical on every rank in distributed runs (default: a seeded sample
            of this process's shards).
    """
    if cfg.levels not in (2, 3):
        raise ValueError("levels must be 2 (HDP) or 3 (nested HDP)")
    if not (2 <= cfg.T1 <= cfg.K) or (cfg.levels == 3 and not 2 <= cfg.T2 <= cfg.T1):
        raise ValueError("truncations must satisfy 2 <= T2 <= T1 <= K")
    reduce = reduce or (lambda d: d)
    rng = np.random.default_rng(cfg.seed)
    if X_init is None:
        allX = np.concatenate([sh.X for sh in shards])
        idx = rng.choice(allX.shape[0], size=min(cfg.init_sample, allX.shape[0]), replace=False)
        X_init = allX[np.sort(idx)]
    centers = init_atoms(X_init, cfg.K, cfg.seed)
    glob = _Global(cfg, shards[0].X.shape[1])
    for sh in shards:
        d2 = (sh.X2.sum(1)[:, None] - 2.0 * sh.X @ centers.T + (centers ** 2).sum(1)[None, :])
        sh.init_structure(np.argmin(d2, axis=1), cfg)
    glob.update(reduce(_sum_stats(shards)))

    trace, converged, it = [], False, 0
    for it in range(1, cfg.max_iter + 1):
        elog_beta = elog_sticks(glob.u, glob.v)
        local = 0.0
        for sh in shards:
            ell = expected_loglik(sh.X, sh.X2, glob.m, glob.kappa, glob.a, glob.b)
            if sh.zeta is not None:
                local += sh.elbo_local(ell, elog_beta, cfg)
            sh.update(ell, elog_beta, cfg)
        if it > 1:
            trace.append(float(reduce({"elbo": np.array([local])})["elbo"][0]) + glob.elbo())
            if verbose:
                print(f"[n-HDP] iter {it - 1:3d}  ELBO {trace[-1]:.6e}")
            if len(trace) >= 2 and abs(trace[-1] - trace[-2]) <= cfg.tol * abs(trace[-2]):
                converged = True
                glob.update(reduce(_sum_stats(shards)))
                break
        glob.update(reduce(_sum_stats(shards)))

    N = num_nodes if num_nodes is not None else int(max(sh.nodes.max() for sh in shards)) + 1
    R = np.zeros((N, cfg.K))
    for sh in shards:
        R[sh.nodes] = sh.r
    Nk = reduce({"Nk": sum(sh.r.sum(0) for sh in shards)})["Nk"]
    return NHDPResult(R=R, elbo_trace=trace, n_iter=it, converged=converged, Nk=np.asarray(Nk))
