"""Single-process training and evaluation.

Protocol: the loss uses training nodes only; model selection uses the
validation metric with early stopping (the best state is deep-copied); the
reported test metric is measured once, on test nodes, with the selected state.

Embedding diagnostics (centroid gap, MADGap, k-means NMI/ARI) are computed on the
logits, the one representation every model has; ``centroid_gap_per_layer``
profiles the hidden layers of models that have them.  With several parts,
BatchNorm normalises every part with its own statistics in training and in
evaluation (see ``part_statistics``).

    python -m nhdp_gnn.train --dataset cora --regime semi --depth 8 --out_dir runs/demo
    python -m nhdp_gnn.train --config configs/example.toml --seed 3
"""
from __future__ import annotations

import copy
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import RunConfig, config_from_dict, parse_args  # noqa: F401  (re-exported)
from .data import load_dataset
from .losses import supervised_loss
from .metrics import centroid_gap, kmeans_nmi_ari, madgap
from .pipeline import build_batches, run_stages
from .splits import regime_split
from .utils import code_version, save_json, set_seed


def build_model(cfg: RunConfig, in_dim: int, out_dim: int, aux_classes: int = 0):
    if cfg.model == "nhdp":
        from .model import NHDPGNN

        use_s = cfg.assign != "none"
        return NHDPGNN(in_dim, cfg.hidden, out_dim, cfg.depth, cfg.heads, cfg.dropout, cfg.attn_dropout,
                       cfg.node_op, cfg.community and use_s, cfg.global_attn and use_s, cfg.residual, cfg.alpha,
                       cfg.norm, aux_classes if cfg.aux_weight > 0 else 0)
    from .baselines import build_baseline

    return build_baseline(cfg.model, in_dim, cfg.hidden, out_dim, cfg.depth, heads=cfg.heads, dropout=cfg.dropout,
                          alpha=cfg.alpha, theta=cfg.theta)


def class_weights(y: np.ndarray, train_mask: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from the training labels (mean weight 1 over classes)."""
    counts = np.bincount(np.asarray(y)[train_mask], minlength=num_classes).astype(np.float64)
    return torch.as_tensor(counts.sum() / np.maximum(counts, 1.0) / num_classes, dtype=torch.float32)


def has_batchnorm(model) -> bool:
    from torch.nn.modules.batchnorm import _BatchNorm

    return any(isinstance(m, _BatchNorm) for m in model.modules())


@contextmanager
def part_statistics(model, enabled: bool = True):
    """Evaluate with the BatchNorm statistics of the part being forwarded.

    Multi-part training normalises every part with that part's batch statistics,
    whereas the running averages mix all parts; evaluating with them would
    normalise a part differently from training.  Inside this context BatchNorm
    layers use the statistics of the current part, and momentum 0 leaves the
    running buffers unchanged.  Dropout follows the model's mode (eval in ``predict``).
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)] if enabled else []
    saved = [(m.training, m.momentum) for m in bns]
    for m in bns:
        m.train()
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, (training, momentum) in zip(bns, saved):
            m.train(training)
            m.momentum = momentum


def score(logits: np.ndarray, y: np.ndarray, mask: np.ndarray, multilabel: bool) -> float:
    if mask.sum() == 0:
        return float("nan")
    if multilabel:
        from sklearn.metrics import roc_auc_score

        yt, yp = y[mask], logits[mask]
        cols = [c for c in range(yt.shape[1]) if 0 < yt[:, c].sum() < yt.shape[0]]
        return float(np.mean([roc_auc_score(yt[:, c], yp[:, c]) for c in cols])) if cols else float("nan")
    return float((logits[mask].argmax(1) == y[mask]).mean())


@torch.no_grad()
def predict(model, batches, num_nodes: int, all_layers: bool = False):
    """Logits for all nodes and, with ``all_layers``, every hidden representation.

    Each part is forwarded on its own subgraph.  With several parts, BatchNorm uses
    each part's own statistics, as in training (single-node parts use the running
    averages, which is all BatchNorm allows for them).
    """
    model.eval()
    multi = len(batches) > 1
    logits, layers = None, []
    for b in batches:
        with part_statistics(model, multi and b.num_nodes > 1):
            lo, hidden, _ = model(b)
        if logits is None:
            logits = np.zeros((num_nodes, lo.shape[1]), dtype=np.float32)
            if all_layers:
                layers = [np.zeros((num_nodes, h.shape[1]), dtype=np.float32) for h in hidden]
        ids = b.node_ids.cpu().numpy()
        logits[ids] = lo.cpu().numpy()
        for store, h in zip(layers, hidden):
            store[ids] = h.cpu().numpy()
    return logits, layers


def run(cfg: RunConfig, verbose: bool = True) -> dict:
    set_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    data = load_dataset(cfg.dataset, cfg.root, cfg.synthetic)
    split = regime_split(data.y, cfg.regime, cfg.seed, full_split=data.official_split,
                         public_split=data.public_split if cfg.regime == "semi" else None,
                         semi_per_class=cfg.semi_per_class, semi_ratio=cfg.semi_ratio, scarce_ratio=cfg.scarce_ratio)
    timings = {}
    t0 = time.perf_counter()
    stages = run_stages(data, cfg.pipeline(), cache_dir=cfg.cache_dir, verbose=False)
    timings["stages_s"] = time.perf_counter() - t0
    batches = build_batches(data, stages, split, gate=cfg.gate, device=cfg.device)

    out_dim = data.num_classes
    aux_classes = int(np.unique(stages.louvain).size)
    model = build_model(cfg, data.x.shape[1], out_dim, aux_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weight = None
    if cfg.class_weights and not data.multilabel:
        weight = class_weights(data.y, split[0], out_dim).to(cfg.device)
    skip_tiny = has_batchnorm(model)                     # BatchNorm cannot train on a one-node part

    y_np, val_mask, test_mask = np.asarray(data.y), split[1], split[2]
    best_val, best_epoch, best_state, curves = -np.inf, 0, None, []
    t_train = time.perf_counter()
    epoch_times = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_ep, total = time.perf_counter(), 0.0
        for b in batches:
            if not bool(b.train_mask.any()) or (skip_tiny and b.num_nodes < 2):
                continue
            opt.zero_grad()
            logits, _, aux = model(b)
            loss = supervised_loss(logits[b.train_mask], b.y[b.train_mask], data.multilabel, cfg.loss,
                                   cfg.focal_gamma, weight)
            if aux is not None:
                loss = loss + cfg.aux_weight * torch.nn.functional.cross_entropy(aux, b.aux_target)
            loss.backward()
            opt.step()
            total += float(loss.item())
        epoch_times.append(time.perf_counter() - t_ep)
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            lo, _ = predict(model, batches, data.num_nodes)
            val = score(lo, y_np, val_mask, data.multilabel)
            curves.append({"epoch": epoch, "loss": total, "val": val})
            if val > best_val:
                best_val, best_epoch = val, epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= cfg.patience:
                break
    timings["train_s"] = time.perf_counter() - t_train
    timings["epoch_s_mean"] = float(np.mean(epoch_times))

    if best_state is not None:
        model.load_state_dict(best_state)
    want_layers = cfg.layer_profile and data.num_nodes <= 200_000 or cfg.save_embeddings
    logits, layers = predict(model, batches, data.num_nodes, all_layers=want_layers)
    result = {
        "key": cfg.key(), "config": asdict(cfg), "metric": data.metric, **code_version(),
        "val": score(logits, y_np, val_mask, data.multilabel),
        "test": score(logits, y_np, test_mask, data.multilabel),
        "best_epoch": best_epoch, "epochs_run": len(epoch_times),
        "propagation_depth": int(getattr(model, "propagation_depth", cfg.depth)),
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_train": int(split[0].sum()), "n_val": int(val_mask.sum()), "n_test": int(test_mask.sum()),
        "timings": timings, "stages": stages.diagnostics,
    }
    if cfg.diagnostics:
        result["embedding"] = {
            "representation": "logits",
            "centroid_gap": centroid_gap(logits, y_np),
            "madgap": madgap(logits, data.edge_index, data.num_nodes, max_sources=cfg.madgap_sources,
                             seed=cfg.seed),
            **kmeans_nmi_ari(logits, y_np, seed=cfg.seed)}
        if want_layers and layers:
            result["embedding"]["centroid_gap_per_layer"] = [centroid_gap(h, y_np) for h in layers]
    if cfg.save_curves:
        result["curves"] = curves

    out = Path(cfg.out_dir)
    if cfg.save_embeddings:
        out.mkdir(parents=True, exist_ok=True)
        files = [f"emb_{cfg.key()}_layer{l:02d}.npy" for l in range(len(layers))] + [f"emb_{cfg.key()}_logits.npy"]
        for name, h in zip(files, layers + [logits]):
            np.save(out / name, h)
        result["embedding_files"] = files
    save_json(out / f"{cfg.key()}.json", result)
    if verbose:
        print(f"[{cfg.dataset} {cfg.regime} {cfg.model} depth={cfg.depth} seed={cfg.seed}] "
              f"val={result['val']:.4f} test={result['test']:.4f} best_epoch={best_epoch}")
    return result


def main(argv=None) -> None:
    run(parse_args(argv, __doc__))


if __name__ == "__main__":
    main(sys.argv[1:])
