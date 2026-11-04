"""Single-process training and evaluation.

Protocol: the loss uses training nodes only; model selection uses the
validation metric with early stopping (the best state is deep-copied); the
reported test metric is measured once, on test nodes, with the selected state.

    python -m nhdp_gnn.train --dataset cora --regime semi --depth 8 --out_dir runs/demo
    python -m nhdp_gnn.train --config configs/example.toml --seed 3
"""
from __future__ import annotations

import copy
import sys
import time
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
from .utils import save_json, set_seed


def build_model(cfg: RunConfig, in_dim: int, out_dim: int, aux_classes: int = 0):
    if cfg.model == "nhdp":
        from .model import NHDPGNN

        use_s = cfg.assign != "none"
        return NHDPGNN(in_dim, cfg.hidden, out_dim, cfg.depth, cfg.heads, cfg.dropout, cfg.attn_dropout,
                       cfg.node_op, cfg.community and use_s, cfg.global_attn and use_s, cfg.residual, cfg.alpha,
                       cfg.norm, aux_classes if cfg.aux_weight > 0 else 0)
    from .baselines import build_baseline

    return build_baseline(cfg.model, in_dim, cfg.hidden, out_dim, cfg.depth, cfg.heads, cfg.dropout)


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
    """Logits and node embeddings for all nodes (each part forwarded on its own subgraph)."""
    model.eval()
    logits, layers = None, None
    for b in batches:
        lo, hidden, _ = model(b)
        if logits is None:
            logits = np.zeros((num_nodes, lo.shape[1]), dtype=np.float32)
            keep = hidden if all_layers else hidden[-1:]
            layers = [np.zeros((num_nodes, h.shape[1]), dtype=np.float32) for h in keep]
        ids = b.node_ids.cpu().numpy()
        logits[ids] = lo.cpu().numpy()
        for store, h in zip(layers, hidden if all_layers else hidden[-1:]):
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
        counts = np.bincount(data.y[split[0]], minlength=out_dim).astype(np.float64)
        weight = torch.as_tensor(counts.sum() / np.maximum(counts, 1.0) / out_dim, dtype=torch.float32,
                                 device=cfg.device)

    y_np, val_mask, test_mask = np.asarray(data.y), split[1], split[2]
    best_val, best_epoch, best_state, curves = -np.inf, 0, None, []
    t_train = time.perf_counter()
    epoch_times = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_ep, total = time.perf_counter(), 0.0
        for b in batches:
            if not bool(b.train_mask.any()):
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
        "key": cfg.key(), "config": asdict(cfg), "metric": data.metric,
        "val": score(logits, y_np, val_mask, data.multilabel),
        "test": score(logits, y_np, test_mask, data.multilabel),
        "best_epoch": best_epoch, "epochs_run": len(epoch_times),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_train": int(split[0].sum()), "n_val": int(val_mask.sum()), "n_test": int(test_mask.sum()),
        "timings": timings, "stages": stages.diagnostics,
    }
    if cfg.diagnostics:
        H = layers[-1]
        result["embedding"] = {
            "centroid_gap": centroid_gap(H, y_np),
            "madgap": madgap(H, data.edge_index, data.num_nodes, max_sources=cfg.madgap_sources, seed=cfg.seed),
            **kmeans_nmi_ari(H, y_np, seed=cfg.seed)}
        if want_layers and len(layers) > 1:
            result["embedding"]["centroid_gap_per_layer"] = [centroid_gap(h, y_np) for h in layers]
    if cfg.save_curves:
        result["curves"] = curves

    out = Path(cfg.out_dir)
    if cfg.save_embeddings:
        out.mkdir(parents=True, exist_ok=True)
        for l, h in enumerate(layers):
            np.save(out / f"emb_{cfg.key()}_layer{l:02d}.npy", h)
        result["embedding_files"] = [f"emb_{cfg.key()}_layer{l:02d}.npy" for l in range(len(layers))]
    save_json(out / f"{cfg.key()}.json", result)
    if verbose:
        print(f"[{cfg.dataset} {cfg.regime} {cfg.model} depth={cfg.depth} seed={cfg.seed}] "
              f"val={result['val']:.4f} test={result['test']:.4f} best_epoch={best_epoch}")
    return result


def main(argv=None) -> None:
    run(parse_args(argv, __doc__))


if __name__ == "__main__":
    main(sys.argv[1:])
