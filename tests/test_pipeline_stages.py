import threading

import numpy as np
import pytest

from nhdp_gnn.data import load_dataset, undirected
from nhdp_gnn.nhdp import NHDPConfig
from nhdp_gnn.pipeline import PipelineConfig, run_stages

SYN = {"num_super": 3, "comm_per_super": 3, "nodes_per_comm": 30, "seed": 0}


def cfg(**kw):
    base = dict(nhdp=NHDPConfig(K=12, T1=4, T2=3, max_iter=30, seed=0))
    base.update(kw)
    return PipelineConfig(**base)


def test_undirected_dedups_and_drops_self_loops():
    ei = np.array([[0, 1, 1, 2, 2], [1, 0, 1, 0, 0]])
    out = undirected(ei, 3)
    pairs = set(zip(*out.tolist()))
    assert pairs == {(0, 1), (1, 0), (0, 2), (2, 0)}


@pytest.mark.parametrize("assign", ["nhdp", "louvain", "random", "none"])
@pytest.mark.parametrize("num_parts,partition", [(1, "auto"), (3, "community"), (3, "bfs")])
def test_stages_shapes_and_diagnostics(assign, num_parts, partition, tmp_path):
    data = load_dataset("synthetic", synthetic=SYN)
    out = run_stages(data, cfg(assign=assign, num_parts=num_parts, partition=partition), cache_dir=str(tmp_path))
    N = data.num_nodes
    assert out.parts.shape == (N,) and np.unique(out.parts).size == num_parts
    assert out.community.min() >= 0 and out.super_of.shape[0] == out.community.max() + 1
    if assign == "none":
        assert out.R is None
    else:
        assert out.R.shape[0] == N and np.allclose(out.R.sum(1), 1.0, atol=1e-5)
        assert 0.0 <= out.diagnostics["rho_gated"] <= 1.0
    for c in np.unique(out.community):                       # communities never cross parts
        assert np.unique(out.parts[out.community == c]).size == 1
    again = run_stages(data, cfg(assign=assign, num_parts=num_parts, partition=partition), cache_dir=str(tmp_path))
    assert np.array_equal(again.parts, out.parts)            # cache round trip
    if out.R is not None:
        assert np.allclose(again.R, out.R)


def test_per_rank_stages_match_single_process():
    """Two simulated ranks (threads) with an all-reduce reproduce the single-process responsibilities."""
    data = load_dataset("synthetic", synthetic=SYN)
    pc = cfg(assign="nhdp", num_parts=2, partition="bfs")
    ref = run_stages(data, pc, cache_dir=None)
    barrier, box, out = threading.Barrier(2), [None, None], [None, None]

    def make_reduce(rank):
        def reduce(d):
            box[rank] = d
            barrier.wait()
            total = {k: box[0][k] + box[1][k] for k in d}
            barrier.wait()
            return total
        return reduce

    def rank_fn(rank):
        out[rank] = run_stages(data, pc, cache_dir=None, reduce=make_reduce(rank), only_part=rank, parts=ref.parts)

    threads = [threading.Thread(target=rank_fn, args=(r,)) for r in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    for rank in range(2):
        sel = ref.parts == rank
        # columns may be pruned differently per rank; compare per-node assignments in atom space
        a = out[rank].R[sel]
        b = ref.R[sel][:, ref.R[sel].sum(0) > 0]
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-5)
