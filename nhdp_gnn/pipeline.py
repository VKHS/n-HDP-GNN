"""Stages A and B, and the per-partition batches used for training (paper Sect. 10).

Stage A  one-time edge cut into ``num_parts`` parts, then nested Louvain inside each part.
Stage B  n-HDP inference per part (shards); global statistics summed by ``reduce``.
Batches  one ``PartBatch`` per part: its induced subgraph, gate values, GCN
         normalisation, soft assignment S = pi (top-k) and the local super-node graph.
         Cut edges are not used for propagation; with ``num_parts=1`` the batch is
         the whole graph.

Stage outputs depend only on the data, the pipeline configuration and the seed,
so they are cached as .npz files and shared across depths and models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .communities import NestedCommunities, nested_communities
from .gating import (cross_partition_ratio, edge_reduction, gate_weights, normalized_adjacency,
                     num_directed_edges, super_graph)
from .metrics import partition_purity
from .nhdp import NHDPConfig, fit_nhdp, make_shards, reduce_features, topk_responsibilities
from .partition import partition_nodes
from .utils import config_hash


@dataclass
class PipelineConfig:
    num_parts: int = 1
    partition: str = "auto"            # auto | metis | community | bfs | random
    louvain_resolution: float = 1.0
    louvain_backend: str = "auto"      # auto | networkx | igraph
    assign: str = "nhdp"               # nhdp | louvain | random | none
    gate: bool = True                  # responsibility gate on node-level propagation
    topk: int = 4
    feature_dim: int = 32              # SVD dimension of the n-HDP likelihood
    feature_hops: int = 0              # graph smoothing before the n-HDP (0 = node features as in the paper)
    nhdp: NHDPConfig = field(default_factory=NHDPConfig)
    communities: bool = True           # False skips Louvain (baselines that use neither S nor the aux task)
    seed: int = 0


@dataclass
class StageOutput:
    parts: np.ndarray                  # (N,) part id
    community: np.ndarray              # (N,) fine Louvain community
    super_of: np.ndarray               # (J,) super-community of each community
    R: np.ndarray | None               # (N, K) assignment (top-k responsibilities), None if assign='none'
    diagnostics: dict

    @property
    def louvain(self) -> np.ndarray:
        """Coarsest-level Louvain community of each node (auxiliary target)."""
        return self.super_of[self.community]


def _cache_key(data, cfg: PipelineConfig) -> dict:
    d = asdict(cfg)
    return {"dataset": data.name, "N": data.num_nodes, "E": int(data.edge_index.shape[1]),
            "x_sum": float(np.asarray(data.x, dtype=np.float64).sum()), "cfg": d}


def _smooth(data, hops: int) -> np.ndarray:
    x = np.asarray(data.x, dtype=np.float64)
    if hops <= 0:
        return x
    A = normalized_adjacency(data.edge_index, data.num_nodes)
    for _ in range(hops):
        x = A @ x
    return x


def run_stages(data, cfg: PipelineConfig, cache_dir: str | None = "cache", reduce=None,
               only_part: int | None = None, parts: np.ndarray | None = None,
               verbose: bool = False) -> StageOutput:
    """Compute stages A and B.  Distributed workers pass ``only_part`` (their rank),
    ``parts`` (the partition broadcast from rank 0) and ``reduce``: each worker
    computes communities and responsibilities for its own part only, and the
    n-HDP global statistics are summed across ranks."""
    key = config_hash(_cache_key(data, cfg))
    path = Path(cache_dir) / f"stages_{data.name}_{key}.npz" if cache_dir and only_part is None else None
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=False)
        diag = {k[5:]: z[k].item() if z[k].ndim == 0 else z[k].tolist() for k in z.files if k.startswith("diag_")}
        R = z["R"] if z["R"].size else None
        return StageOutput(z["parts"], z["community"], z["super_of"], R, diag)

    N, ei = data.num_nodes, data.edge_index
    rng_seed = cfg.seed
    if parts is not None:
        parts = np.asarray(parts, dtype=np.int64)
    elif cfg.num_parts > 1 and cfg.partition == "community":
        glob = nested_communities(ei, N, seed=rng_seed, resolution=cfg.louvain_resolution, backend=cfg.louvain_backend)
        parts = partition_nodes(ei, N, cfg.num_parts, "community", communities=glob.super_community, seed=rng_seed)
    else:
        parts = partition_nodes(ei, N, cfg.num_parts, cfg.partition, seed=rng_seed)

    sel = np.ones(N, dtype=bool) if only_part is None else parts == only_part
    sub_parts = np.where(sel, parts, -1)
    if not cfg.communities and cfg.assign == "none":
        # one community per part keeps the invariant "communities never cross parts"
        nc = NestedCommunities(parts.copy(), np.arange(int(parts.max()) + 1), 0)
    else:
        nc = nested_communities(ei, N, seed=rng_seed, resolution=cfg.louvain_resolution,
                                backend=cfg.louvain_backend, parts=sub_parts if cfg.num_parts > 1 else None)
    diag = {"num_parts": cfg.num_parts, "louvain_levels": nc.levels,
            "num_communities": nc.num_communities, "num_super_communities": nc.num_super}

    R = None
    if cfg.assign == "nhdp":
        Z = reduce_features(_smooth(data, cfg.feature_hops), cfg.feature_dim, seed=rng_seed)
        ncfg = cfg.nhdp
        shard_parts = sub_parts if cfg.num_parts > 1 else None
        shards = make_shards(Z[sel], nc.community[sel], nc.super_of, ncfg,
                             parts=None if shard_parts is None else shard_parts[sel])
        for sh in shards:                               # make_shards indexed the selected rows
            sh.nodes = np.flatnonzero(sel)[sh.nodes]
        rng = np.random.default_rng(ncfg.seed)
        init_idx = np.sort(rng.choice(N, size=min(ncfg.init_sample, N), replace=False))
        res = fit_nhdp(shards, ncfg, reduce=reduce, X_init=Z[init_idx], num_nodes=N, verbose=verbose)
        R = topk_responsibilities(res.R[sel], cfg.topk) if only_part is not None else topk_responsibilities(res.R, cfg.topk)
        if only_part is not None:
            full = np.zeros((N, res.R.shape[1]))
            full[sel] = R
            R = full
        diag.update({"nhdp_iters": res.n_iter, "nhdp_converged": res.converged, "nhdp_effective_k": res.effective_k,
                     "nhdp_elbo_final": res.elbo_trace[-1] if res.elbo_trace else float("nan")})
    elif cfg.assign in ("louvain", "random"):
        idx = np.flatnonzero(sel)
        lab = np.zeros(N, dtype=np.int64)
        lab[idx] = nc.super_of[nc.community[idx]]
        if cfg.assign == "random":                      # same community sizes, random membership
            lab[idx] = lab[idx][np.random.default_rng(rng_seed).permutation(idx.size)]
        _, lab = np.unique(lab, return_inverse=True)
        R = np.zeros((N, lab.max() + 1))
        R[np.arange(N), lab] = 1.0
        R[~sel] = 0.0
    elif cfg.assign != "none":
        raise ValueError(f"unknown assign mode {cfg.assign!r}")

    if R is not None:
        # The model has no parameter that depends on the number of columns, so each
        # rank may keep its own set of used atoms.
        R = R[:, R[sel].sum(0) > 0].astype(np.float32)
        e_part = ei[:, sel[ei[0]] & sel[ei[1]]]
        _, es = super_graph(e_part, N, R)
        diag.update({"assign_k": int(R.shape[1]), "rho": cross_partition_ratio(ei, parts),
                     "edge_reduction": edge_reduction(num_directed_edges(e_part), es)})
        if only_part is None:                           # needs responsibilities of both endpoints
            g = gate_weights(ei, R)
            diag.update({"rho_gated": cross_partition_ratio(ei, parts, g),
                         "gate_mean": float(g.mean()) if g.size else float("nan")})
    else:
        diag.update({"rho": cross_partition_ratio(ei, parts)})

    if not data.multilabel and nc.levels > 0:
        y = np.asarray(data.y)
        diag["louvain_purity"] = partition_purity(nc.super_of[nc.community][sel], y[sel])
        if R is not None:
            diag["assign_purity"] = partition_purity(R[sel].argmax(1), y[sel])

    out = StageOutput(parts, nc.community, nc.super_of, R, diag)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, parts=parts, community=nc.community, super_of=nc.super_of,
                            R=R if R is not None else np.zeros(0, np.float32),
                            **{f"diag_{k}": np.asarray(v) for k, v in diag.items()})
    return out


# ----------------------------------------------------------------------------- torch batches
def build_batches(data, stages: StageOutput, split, gate: bool = True, parts_to_build=None, device="cpu"):
    """One PartBatch per part (or only ``parts_to_build``)."""
    import torch

    N, ei = data.num_nodes, data.edge_index
    louvain = stages.louvain
    _, louvain = np.unique(louvain, return_inverse=True)
    batches = []
    for p in (parts_to_build if parts_to_build is not None else np.unique(stages.parts)):
        nodes = np.flatnonzero(stages.parts == p)
        local = -np.ones(N, dtype=np.int64)
        local[nodes] = np.arange(nodes.size)
        m = (local[ei[0]] >= 0) & (local[ei[1]] >= 0)
        e = np.stack([local[ei[0][m]], local[ei[1][m]]])
        n = nodes.size
        loops = np.arange(n)
        e_sl = np.concatenate([e, np.stack([loops, loops])], axis=1)
        deg = np.bincount(e_sl[1], minlength=n).astype(np.float64)
        gcn_w = 1.0 / np.sqrt(deg[e_sl[0]] * deg[e_sl[1]])
        S = None if stages.R is None else stages.R[nodes]
        if gate and S is not None:
            g = gate_weights(e_sl, S)
        else:
            g = np.ones(e_sl.shape[1])
        sup = None
        if S is not None:
            _, es = super_graph(e, n, S)
            sup = torch.as_tensor(es, dtype=torch.long)
        y = data.y[nodes]
        batches.append(PartBatch(
            x=torch.as_tensor(np.asarray(data.x[nodes]), dtype=torch.float32),
            y=torch.as_tensor(y, dtype=torch.float32 if data.multilabel else torch.long),
            train_mask=torch.as_tensor(split[0][nodes]), val_mask=torch.as_tensor(split[1][nodes]),
            test_mask=torch.as_tensor(split[2][nodes]),
            edge_index=torch.as_tensor(e, dtype=torch.long),
            edge_index_sl=torch.as_tensor(e_sl, dtype=torch.long),
            gate_sl=torch.as_tensor(g, dtype=torch.float32),
            gcn_weight_sl=torch.as_tensor(gcn_w, dtype=torch.float32),
            S=None if S is None else torch.as_tensor(S, dtype=torch.float32),
            super_edge_index=sup,
            aux_target=torch.as_tensor(louvain[nodes], dtype=torch.long),
            node_ids=torch.as_tensor(nodes, dtype=torch.long)).to(device))
    return batches


class PartBatch:
    """Tensors of one partition's induced subgraph (all indices local)."""

    _fields = ("x", "y", "train_mask", "val_mask", "test_mask", "edge_index", "edge_index_sl", "gate_sl",
               "gcn_weight_sl", "S", "super_edge_index", "aux_target", "node_ids")

    def __init__(self, **kw):
        for f in self._fields:
            setattr(self, f, kw.get(f))

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device):
        for f in self._fields:
            v = getattr(self, f)
            if v is not None:
                setattr(self, f, v.to(device))
        return self
