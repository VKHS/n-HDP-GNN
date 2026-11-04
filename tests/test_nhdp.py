import numpy as np
import pytest
from sklearn.metrics import normalized_mutual_info_score

from nhdp_gnn.nhdp import (NHDPConfig, elog_sticks, fit_nhdp, kl_beta, kl_normal_gamma, make_shards,
                           stick_posterior, topk_responsibilities)


def planted(seed=0, n_super=3, comm_per_super=4, n_per=40, d=8, sep=6.0):
    """Nodes in nested groups; each super-community uses two of six well-separated atoms."""
    rng = np.random.default_rng(seed)
    J = n_super * comm_per_super
    community = np.repeat(np.arange(J), n_per)
    super_of = np.repeat(np.arange(n_super), comm_per_super)
    means = rng.normal(size=(2 * n_super, d)) * sep
    atom = np.empty(community.size, dtype=int)
    for j in range(J):
        s = super_of[j]
        pref = 0.85 if j % 2 == 0 else 0.15
        idx = np.flatnonzero(community == j)
        atom[idx] = np.where(rng.random(idx.size) < pref, 2 * s, 2 * s + 1)
    X = means[atom] + rng.normal(size=(community.size, d))
    return X, community, super_of, atom


def test_elog_sticks_two_atoms_matches_beta_moments():
    from scipy.special import digamma
    a, b = np.array([2.5]), np.array([4.0])
    out = elog_sticks(a, b)
    assert np.allclose(out, [digamma(2.5) - digamma(6.5), digamma(4.0) - digamma(6.5)])


def test_stick_posterior_tail_counts():
    a, b = stick_posterior(np.array([3.0, 2.0, 1.0]), 0.5)
    assert np.allclose(a, [4.0, 3.0]) and np.allclose(b, [0.5 + 3.0, 0.5 + 1.0])


def test_kl_terms_zero_at_prior_and_nonnegative():
    rng = np.random.default_rng(1)
    assert np.allclose(kl_beta(np.array([1.0]), np.array([2.0]), 1.0, 2.0), 0.0)
    a, b = rng.uniform(0.2, 5, 50), rng.uniform(0.2, 5, 50)
    assert np.all(kl_beta(a, b, 1.0, 1.5) >= -1e-10)
    m, kap, A, B = np.zeros((2, 3)), np.full(2, 0.1), np.ones(2), np.ones((2, 3))
    assert abs(kl_normal_gamma(m, kap, A, B, 0.0, 0.1, 1.0, 1.0)) < 1e-12
    m2 = rng.normal(size=(2, 3))
    assert kl_normal_gamma(m2, kap + 3, A + 2, B * 1.7, 0.0, 0.1, 1.0, 1.0) > 0


@pytest.mark.parametrize("levels", [3, 2])
def test_elbo_is_monotone(levels):
    X, comm, sup, _ = planted()
    cfg = NHDPConfig(K=12, T1=4, T2=3, levels=levels, max_iter=60, tol=0.0, seed=0)
    res = fit_nhdp(make_shards(X, comm, sup, cfg), cfg, X_init=X)
    tr = np.asarray(res.elbo_trace)
    assert tr.size > 10
    assert np.all(np.diff(tr) >= -1e-8 * np.abs(tr[:-1])), np.diff(tr).min()


def test_recovers_planted_atoms_and_rows_are_distributions():
    X, comm, sup, atom = planted()
    cfg = NHDPConfig(K=12, T1=4, T2=3, max_iter=100, seed=0)
    res = fit_nhdp(make_shards(X, comm, sup, cfg), cfg, X_init=X)
    assert np.allclose(res.R.sum(1), 1.0)
    assert normalized_mutual_info_score(atom, res.R.argmax(1)) > 0.9
    assert 4 <= res.effective_k <= 12


def test_sharded_inference_equals_single_shard():
    X, comm, sup, _ = planted(seed=3)
    cfg = NHDPConfig(K=10, T1=4, T2=3, max_iter=25, tol=0.0, seed=0)
    single = fit_nhdp(make_shards(X, comm, sup, cfg), cfg, X_init=X)
    parts = sup[comm]                                    # one shard per super-community
    sharded = fit_nhdp(make_shards(X, comm, sup, cfg, parts=parts), cfg, X_init=X)
    assert np.allclose(single.R, sharded.R, atol=1e-6)
    assert np.allclose(single.elbo_trace, sharded.elbo_trace, rtol=1e-9)


def test_reduce_hook_sums_global_statistics_across_ranks():
    """Two 'ranks' simulated in one process: each runs on its own shard, reduce sums both."""
    X, comm, sup, _ = planted(seed=4)
    cfg = NHDPConfig(K=10, T1=4, T2=3, max_iter=15, tol=0.0, seed=0)
    parts = (sup[comm] > 0).astype(int)
    shards = make_shards(X, comm, sup, cfg, parts=parts)
    ref = fit_nhdp(make_shards(X, comm, sup, cfg, parts=parts), cfg, X_init=X)

    import threading
    barrier, box, out = threading.Barrier(2), [None, None], [None, None]

    def make_reduce(rank):
        def reduce(d):
            box[rank] = d
            barrier.wait()
            total = {k: box[0][k] + box[1][k] for k in d}
            barrier.wait()
            return total
        return reduce

    def worker(rank):
        out[rank] = fit_nhdp([shards[rank]], cfg, reduce=make_reduce(rank), X_init=X, num_nodes=X.shape[0])

    th = [threading.Thread(target=worker, args=(r,)) for r in range(2)]
    [t.start() for t in th]
    [t.join() for t in th]
    R = out[0].R + out[1].R
    assert np.allclose(R, ref.R, atol=1e-6)
    assert np.allclose(out[0].elbo_trace, ref.elbo_trace, rtol=1e-9)


def test_topk_rows_sparse_and_normalised():
    rng = np.random.default_rng(0)
    R = rng.dirichlet(np.ones(9), size=30)
    T = topk_responsibilities(R, 3)
    assert np.all((T > 0).sum(1) <= 3)
    assert np.allclose(T.sum(1), 1.0)
    assert np.all(T.argmax(1) == R.argmax(1))
