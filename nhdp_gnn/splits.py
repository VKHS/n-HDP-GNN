"""Train/validation/test splits for the three supervision regimes (paper Sect. 8).

full    stratified random 60/20/20 per seed (OGB: the official split).
semi    Planetoid datasets: the public Planetoid split (20 labels per class,
        500 validation, 1000 test).  Other single-label datasets: 20 labels per
        class drawn from the full-regime training pool.  Multi-label datasets:
        a fraction of the training pool (``semi_ratio``).
scarce  ``scarce_ratio`` (default 5%) of all nodes, stratified, drawn from the
        full-regime training pool.

In every regime the validation and test nodes are those of the full regime
for the same seed (except the public Planetoid split), so regimes differ only
in the labelled training set.
"""
from __future__ import annotations

import numpy as np


def stratified_split(y: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    N = y.shape[0]
    tr, va, te = (np.zeros(N, dtype=bool) for _ in range(3))
    if y.ndim > 1:                                       # multi-label: plain random split
        perm = rng.permutation(N)
        n_tr, n_va = int(round(train_frac * N)), int(round(val_frac * N))
        tr[perm[:n_tr]], va[perm[n_tr:n_tr + n_va]], te[perm[n_tr + n_va:]] = True, True, True
        return tr, va, te
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr = max(1, int(round(train_frac * idx.size)))
        n_va = int(round(val_frac * idx.size))
        tr[idx[:n_tr]] = True
        va[idx[n_tr:n_tr + n_va]] = True
        te[idx[n_tr + n_va:]] = True
    return tr, va, te


def subsample_per_class(y: np.ndarray, pool: np.ndarray, per_class: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(y.shape[0], dtype=bool)
    for c in np.unique(y[pool]):
        idx = np.flatnonzero(pool & (y == c))
        out[rng.choice(idx, size=min(per_class, idx.size), replace=False)] = True
    return out


def subsample_ratio(y: np.ndarray, pool: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """round(ratio * N) training nodes from ``pool``, stratified by class (>= 1 per class)."""
    rng = np.random.default_rng(seed)
    N = y.shape[0]
    target = max(1, int(round(ratio * N)))
    out = np.zeros(N, dtype=bool)
    if y.ndim > 1:
        idx = np.flatnonzero(pool)
        out[rng.choice(idx, size=min(target, idx.size), replace=False)] = True
        return out
    classes, counts = np.unique(y[pool], return_counts=True)
    alloc = np.maximum(1, np.round(target * counts / counts.sum()).astype(int))
    for c, k in zip(classes, alloc):
        idx = np.flatnonzero(pool & (y == c))
        out[rng.choice(idx, size=min(int(k), idx.size), replace=False)] = True
    return out


def regime_split(y: np.ndarray, regime: str, seed: int, full_split=None, public_split=None,
                 semi_per_class: int = 20, semi_ratio: float = 0.2, scarce_ratio: float = 0.05):
    """Return (train, val, test) boolean masks for ``regime`` in {full, semi, scarce}."""
    y = np.asarray(y)
    if full_split is None:
        full_split = stratified_split(y, 0.6, 0.2, seed)
    tr_full, va, te = (np.asarray(m, dtype=bool) for m in full_split)
    if regime == "full":
        return tr_full, va, te
    if regime == "semi":
        if public_split is not None:
            return tuple(np.asarray(m, dtype=bool) for m in public_split)
        if y.ndim > 1:
            tr = np.zeros_like(tr_full)
            idx = np.flatnonzero(tr_full)
            tr[np.random.default_rng(seed).choice(idx, size=max(1, int(semi_ratio * idx.size)), replace=False)] = True
            return tr, va, te
        return subsample_per_class(y, tr_full, semi_per_class, seed), va, te
    if regime == "scarce":
        return subsample_ratio(y, tr_full, scarce_ratio, seed), va, te
    raise ValueError(f"unknown regime {regime!r}")
