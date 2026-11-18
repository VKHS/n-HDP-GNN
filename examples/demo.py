#!/usr/bin/env python3
"""Demo: what n-HDP-GNN computes, on one planted hierarchical graph (no download).

    python examples/demo.py                  # report, figures and JSON in runs/demo/output/
    python examples/demo.py --train no       # stages A and B and the training-free depth study (numpy only)
    python examples/demo.py --quick          # a smaller graph, fewer depths, seeds and epochs
    python examples/demo.py --jobs 4         # train four runs at a time
    python examples/demo.py --out DIR --work DIR

The graph is a hierarchical stochastic block model whose planted communities and super-communities are
known, so every stage can be compared with the truth:

  1. the graph             sizes, homophily, diameter
  2. stage A               nested Louvain communities against the planted ones
  3. stage B               the n-HDP objective (monotone), the responsibilities and the gate on three kinds of edge
  4. depth, no training    linear probe and centroid gap (Eq. 28) of P^L X for the normalised adjacency and the gated
                           operators, L up to 64
  5. depth, trained        n-HDP-GNN (gate on and off), GCN, GAT, APPNP and GCNII at several depths, mean and 95%
                           interval over seeds (needs PyTorch and PyTorch Geometric)

Everything printed in the report is computed by this script; nothing is typed in.  The training uses the same
generic hyper-parameters for every model and none is tuned.  Runs that fail are listed with their reason and are
not replaced by a value.  The report is a demonstration of the mechanics on one synthetic graph; it is not
evidence about accuracy on real graphs (see results/README.md and artifacts/README.md for those).

The committed copy in examples/output/ was produced by the ``demo`` GitHub Actions workflow (see
examples/README.md); a local run writes to runs/demo/output/ by default, so it never overwrites it.

Writes to --out: demo.md (the report), demo.json (every number, the configuration, code commit and environment),
fig_graph.png, fig_stage_ab.png, fig_depth_free.png and, with training, fig_depth_trained.png, and runs.jsonl.gz (the
result record of every training run, without training curves).  Run records and stage caches go to --work.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if importlib.util.find_spec("nhdp_gnn") is None:            # a checkout that was not pip-installed
    sys.path.insert(0, str(ROOT))

GRAPH = {"num_super": 4, "comm_per_super": 4, "nodes_per_comm": 60, "p_in": 0.2, "p_super": 0.03, "p_out": 0.002,
         "num_classes": 4, "class_alignment": 0.9, "feature_dim": 16, "feature_snr": 1.0, "seed": 0}
QUICK_GRAPH = {"num_super": 3, "comm_per_super": 3, "nodes_per_comm": 30, "class_alignment": 0.9, "seed": 0}
STUDY_DEPTHS = (0, 1, 2, 4, 8, 16, 32, 64)
QUICK_STUDY_DEPTHS = (0, 2, 8, 32)
TRAIN_DEPTHS = (2, 8, 32)
QUICK_TRAIN_DEPTHS = (2, 8)
TRAIN_BASE = {"regime": "semi", "hidden": 64, "heads": 4, "lr": 0.005, "weight_decay": 5e-4, "epochs": 200,
              "patience": 50, "threads": 1}
QUICK_TRAIN_BASE = {"epochs": 60, "patience": 30}
# (id, label, RunConfig overrides): the order is the order of the legend and of the colours
VARIANTS = [("nhdp", "n-HDP-GNN", {"model": "nhdp"}),
            ("nhdp_gate_off", "n-HDP-GNN, gate off", {"model": "nhdp", "gate": False}),
            ("gcn", "GCN", {"model": "gcn"}),
            ("gat", "GAT", {"model": "gat"}),
            ("appnp", "APPNP", {"model": "appnp"}),
            ("gcnii", "GCNII", {"model": "gcnii"})]

# Figure style: the categorical slots of the data-visualisation palette in their fixed order (adjacent pairs pass the
# colour-vision and normal-vision checks); the four-colour node plot uses the subset that also passes all pairs.
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c3"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES = {vid: SLOTS[i] for i, (vid, _, _) in enumerate(VARIANTS)}
MARKERS = {"nhdp": "o", "nhdp_gate_off": "s", "gcn": "^", "gat": "D", "appnp": "v", "gcnii": "P"}
OPERATORS = {"A_hat": ("normalised adjacency (GCN)", SLOTS[0], "o", "-"),
             "gated_nhdp_topk": ("n-HDP top-k gate", SLOTS[1], "s", "-"),
             "gated_nhdp_hard": ("n-HDP hard gate", SLOTS[2], "^", "-"),
             "gated_louvain": ("Louvain gate", SLOTS[3], "D", "-")}
CLUSTER_COLORS = ["#2a78d6", "#eda100", "#e87ba4", "#008300"]
HEAT = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ----------------------------------------------------------------------------- numbers
def mean_ci(values) -> tuple[float, float, int]:
    """Mean, half-width of the 95% t-interval (NaN below two values) and the number of values."""
    from nhdp_gnn.metrics import mean_ci as _mean_ci

    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return float("nan"), float("nan"), 0
    m, h = _mean_ci(v)
    return m, h, len(v)


def graph_facts(data, syn: dict) -> dict:
    from scipy import sparse
    from scipy.sparse import csgraph

    from nhdp_gnn.metrics import edge_homophily

    n, ei = data.num_nodes, data.edge_index
    once = ei[:, ei[0] < ei[1]]
    planted_super = np.repeat(np.arange(syn["num_super"]), syn["comm_per_super"])[data.community]
    A = sparse.csr_matrix((np.ones(once.shape[1]), (once[0], once[1])), shape=(n, n))
    hops = csgraph.shortest_path(A, directed=False, unweighted=True)
    finite = hops[np.isfinite(hops)]
    return {"nodes": n, "edges": int(once.shape[1]), "classes": int(data.num_classes),
            "planted_communities": syn["num_super"] * syn["comm_per_super"], "planted_super_communities": syn["num_super"],
            "nodes_per_community": syn["nodes_per_comm"], "class_alignment": syn["class_alignment"],
            "feature_dim": int(data.x.shape[1]), "feature_snr": float(syn.get("feature_snr", 1.0)),
            "edge_homophily": float(edge_homophily(ei, data.y)), "average_degree": float(ei.shape[1] / n),
            "share_edges_inside_community": float((data.community[once[0]] == data.community[once[1]]).mean()),
            "share_edges_inside_super": float((planted_super[once[0]] == planted_super[once[1]]).mean()),
            "diameter": int(finite.max()), "components": int(csgraph.connected_components(A, directed=False)[0]),
            "share_pairs_8_hops_or_more": float((hops >= 8).mean()), "synthetic": syn}


def structure(data, syn: dict, seed: int = 0):
    """Stages A and B with the defaults of every training configuration, against the planted truth."""
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.metrics import normalized_mutual_info_score as nmi

    from nhdp_gnn.artifacts import stage_ab_report
    from nhdp_gnn.config import RunConfig
    from nhdp_gnn.gating import gate_weights

    report, st = stage_ab_report(data, RunConfig(seed=seed).pipeline())
    planted_super = np.repeat(np.arange(syn["num_super"]), syn["comm_per_super"])[data.community]
    hard = st.R.argmax(1)
    truth = {"nmi_communities": nmi(data.community, st.community), "ari_communities": ari(data.community, st.community),
             "nmi_super_communities": nmi(planted_super, st.louvain), "ari_super_communities": ari(planted_super, st.louvain),
             "nmi_nhdp_hard_vs_super": nmi(planted_super, hard), "ari_nhdp_hard_vs_super": ari(planted_super, hard),
             "nmi_nhdp_hard_vs_class": nmi(data.y, hard), "found_communities": int(st.community.max() + 1),
             "found_super_communities": int(st.louvain.max() + 1)}
    contingency = np.zeros((int(data.community.max() + 1), int(st.community.max() + 1)), dtype=np.int64)
    np.add.at(contingency, (data.community, st.community), 1)
    once = data.edge_index[:, data.edge_index[0] < data.edge_index[1]]
    g = gate_weights(once, st.R)
    same_comm = data.community[once[0]] == data.community[once[1]]
    same_super = planted_super[once[0]] == planted_super[once[1]]
    kinds = {"same community": same_comm, "same super-community, other community": same_super & ~same_comm,
             "other super-community": ~same_super}
    gate = {k: {"edges": int(m.sum()), "mean_gate": float(g[m].mean()) if m.any() else float("nan")}
            for k, m in kinds.items()}
    return report, st, {"truth": truth, "contingency": contingency, "gate_by_edge_kind": gate, "gate": g, "once": once,
                        "planted_super": planted_super}


def depth_study(data, st, depths) -> dict:
    from nhdp_gnn.artifacts import operator_depth_study

    out = operator_depth_study(data, st, depths=depths, probe_seeds=(0, 1, 2, 3, 4), madgap_sources=200)
    out["features"] = "the synthetic Gaussian features as generated (not row-normalised)"
    out["resolution_check"] = probe_resolution(data, depths[-1])
    return out


def probe_resolution(data, depth: int, shrink: float = 1000.0, seeds=(0, 1, 2, 3, 4)) -> dict:
    """How small a class difference the probe can still read: the normalised adjacency's PᴸX at ``depth``, as it is
    and with the deviations of its rows from their mean shrunk by ``shrink`` (the probe's own protocol, same label draws)."""
    from nhdp_gnn.artifacts import linear_probe
    from nhdp_gnn.gating import normalized_adjacency
    from nhdp_gnn.splits import regime_split

    y = np.asarray(data.y)
    P = normalized_adjacency(data.edge_index, data.num_nodes)
    H = np.asarray(data.x, dtype=np.float64)
    for _ in range(depth):
        H = P @ H
    unit = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    centroids = np.stack([unit[y == c].mean(0) for c in np.unique(y)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    iu = np.triu_indices(len(centroids), 1)
    cosine_distance = float(np.mean(1.0 - (centroids @ centroids.T)[iu]))
    shrunk = H.mean(0) + (H - H.mean(0)) / shrink
    classes = np.unique(y)
    plain, small, shares = [], [], []
    for seed in seeds:
        split = regime_split(y, "semi", seed, semi_per_class=20)
        plain.append(linear_probe(H, y, split)["test"])
        small.append(linear_probe(shrunk, y, split)["test"])
        shares.append([float((y[split[2]] == c).mean()) for c in classes])
    per_class = np.mean(shares, axis=0)                      # accuracy of predicting class c for every test node
    return {"depth": int(depth), "shrink": float(shrink), "centroid_cosine_distance": cosine_distance,
            "mean_row_norm": float(np.linalg.norm(H, axis=1).mean()), "probe_test": float(np.mean(plain)),
            "probe_test_shrunk": float(np.mean(small)), "single_class_accuracy": [float(per_class.min()),
                                                                                  float(per_class.max())],
            "label_draws": len(list(seeds))}


# ----------------------------------------------------------------------------- training
def variant_configs(syn: dict, depths, seeds: int, base: dict, work: Path) -> list:
    from nhdp_gnn.config import RunConfig

    cfgs = []
    for vid, _, over in VARIANTS:
        for depth in depths:
            for seed in range(seeds):
                cfgs.append(RunConfig(dataset="synthetic", synthetic=syn, seed=seed, depth=depth, tag=vid,
                                      out_dir=str(work / "runs"), cache_dir=str(work / "cache"), **base, **over))
    return cfgs


def run_training(cfgs: list, jobs: int, runner: str | None = None) -> dict:
    """Run the configurations that have no record yet, as nhdp_gnn.sweep does (one process per run if jobs > 1)."""
    from nhdp_gnn import sweep

    runner = runner or sweep.DEFAULT_RUNNER
    todo = sweep.pending(cfgs)
    print(f"training: {len(cfgs)} runs, {len(cfgs) - len(todo)} already recorded, {len(todo)} to run", flush=True)
    if jobs > 1:
        return sweep.run_pool(todo, jobs, 0.85 * sweep.total_memory_gb(), 0.0, 0.0, runner, 0.25)
    fn = sweep._resolve(runner)
    counts = {"ok": 0, "memory": 0, "time": 0, "diverged": 0, "error": 0}
    for k, cfg in enumerate(todo, 1):
        counts[sweep.attempt(cfg, fn)] += 1
        print(f"  [{k}/{len(todo)}] {cfg.tag} depth {cfg.depth} seed {cfg.seed}", flush=True)
    return counts


def collect(cfgs: list) -> tuple[list, list, list]:
    """Result records, failure records and configurations with neither."""
    from nhdp_gnn import sweep

    results, failures, missing = [], [], []
    for c in cfgs:
        rp, fp = sweep.result_path(c), sweep.failure_path(c)
        if rp.exists():
            results.append(json.loads(rp.read_text()))
        elif fp.exists():
            failures.append(json.loads(fp.read_text()))
        else:
            missing.append(c)
    return results, failures, missing


def aggregate(results: list, depths, value) -> dict:
    """{(variant id, depth): (mean, half-width, n)} of ``value(record)`` over the seeds."""
    cells: dict = {}
    for r in results:
        cells.setdefault((r["config"]["tag"], r["config"]["depth"]), []).append(value(r))
    return {(vid, d): mean_ci(cells.get((vid, d), [])) for vid, _, _ in VARIANTS for d in depths}


def per_seed(results: list, depths, value) -> dict:
    """{(variant id, depth): [value(record) for each seed, in order of seed]}; empty for a cell without records."""
    cells: dict = {}
    for r in sorted(results, key=lambda r: r["config"]["seed"]):
        v = value(r)
        cells.setdefault((r["config"]["tag"], r["config"]["depth"]), []).append(float("nan") if v is None else float(v))
    return {(vid, d): cells.get((vid, d), []) for vid, _, _ in VARIANTS for d in depths}


def median_best_epoch(results: list, depths) -> dict:
    cells: dict = {}
    for r in results:
        cells.setdefault((r["config"]["tag"], r["config"]["depth"]), []).append((r["best_epoch"], r["epochs_run"]))
    return {(vid, d): (float(np.median([b for b, _ in cells[(vid, d)]])), float(np.median([e for _, e in cells[(vid, d)]])))
            for vid, _, _ in VARIANTS for d in depths if (vid, d) in cells}


def trained_summary(cfgs, depths, seeds) -> dict:
    results, failures, missing = collect(cfgs)
    values = {"test": lambda r: 100.0 * r["test"], "centroid_gap": lambda r: r["embedding"]["centroid_gap"],
              "nmi": lambda r: r["embedding"]["nmi"]}
    return {"results": results, "failures": failures, "missing": missing, "depths": list(depths), "seeds": seeds,
            **{k: aggregate(results, depths, f) for k, f in values.items()},
            **{k + "_seeds": per_seed(results, depths, f) for k, f in values.items()},
            "epochs": median_best_epoch(results, depths)}


# ----------------------------------------------------------------------------- figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8.5,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "grid.linestyle": "-", "axes.axisbelow": True, "savefig.dpi": 200, "figure.dpi": 100})
    return plt


def _line(ax, x, y, color, marker, label, ci=None, dashed=False, dodge=0.0):
    """A 2 px line with 8 px markers ringed in the surface colour; ``ci`` adds thin error bars at the points."""
    style = dict(color=color, lw=1.6, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label, zorder=3,
                 ls=(0, (4, 2)) if dashed else "-", clip_on=False)
    x = np.asarray(x, dtype=float) + dodge
    if ci is None:
        ax.plot(x, y, **style)
    else:
        ax.errorbar(x, y, yerr=np.nan_to_num(np.asarray(ci, dtype=float)), elinewidth=1.0, capsize=2.5, capthick=1.0,
                    **style)


def _end_labels(ax, x_last, series, min_gap):
    """Name a line at its right end when no other labelled line ends within ``min_gap``; the legend names the rest."""
    last = None
    for y, text in sorted(series, key=lambda t: t[0]):
        if last is None or y - last >= min_gap:
            ax.annotate(text, (x_last, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=INK2)
            last = y


def figure_graph(path: Path, data, st, extra: dict, syn: dict) -> None:
    import networkx as nx
    from matplotlib.collections import LineCollection

    plt = _style()
    once, g, sup = extra["once"], extra["gate"], extra["planted_super"]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(once.T.tolist())
    xy = np.array([p for _, p in sorted(nx.spring_layout(G, seed=0, iterations=80).items())])
    segs = xy[once.T]                                          # (E, 2, 2)
    colors = [CLUSTER_COLORS[s % len(CLUSTER_COLORS)] for s in sup]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].add_collection(LineCollection(segs, colors=(0.35, 0.35, 0.33, 0.10), linewidths=0.5))
    axes[0].scatter(xy[:, 0], xy[:, 1], s=9, c=colors, linewidths=0, zorder=3)
    shade = np.clip(g, 0.0, 1.0)
    order = np.argsort(shade)
    rgba = np.zeros((order.size, 4))
    rgba[:, :3] = (0.16, 0.16, 0.15)
    rgba[:, 3] = 0.5 * shade[order]
    axes[1].add_collection(LineCollection(segs[order], colors=rgba, linewidths=0.6))
    axes[1].scatter(xy[:, 0], xy[:, 1], s=9, c="#8a8985", linewidths=0, zorder=3)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], ls="none", marker="o", ms=8, mfc=CLUSTER_COLORS[k % len(CLUSTER_COLORS)], mec=SURFACE,
                      mew=1.5, label=f"super-community {k + 1}") for k in range(syn["num_super"])]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=syn["num_super"], frameon=False,
                   fontsize=8.5, handletextpad=0.2, columnspacing=1.2)
    pad = 0.12 * (xy.max(0) - xy.min(0))
    for ax in axes:
        ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
        ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
    axes[0].set_title(f"(a) The graph: {data.num_nodes} nodes, {once.shape[1]} edges\n"
                      "colour = planted super-community", loc="left", fontsize=9.5)
    axes[1].set_title("(b) The same layout with the n-HDP gate\n"
                      "edge opacity = g_ij = ⟨π_i, π_j⟩", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_stage_ab(path: Path, report: dict, extra: dict) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), gridspec_kw={"width_ratios": [1.1, 1.0, 1.25]})
    # (a) planted communities (rows) against the communities found by stage A (columns)
    M = extra["contingency"]
    cols = np.argsort(M.argmax(0) * 10_000 - M.max(0), kind="stable")          # order the columns by dominant planted row
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(M[:, cols], aspect="auto", cmap=LinearSegmentedColormap.from_list("seq", HEAT), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{M.shape[1]} communities found")
    ax.set_ylabel(f"{M.shape[0]} planted communities")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("nodes", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    ax.set_title("(a) Stage A: found vs planted", loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    # (b) the n-HDP objective
    ax = axes[1]
    trace = np.asarray(report["stage_b"]["elbo_trace"], dtype=float)
    it = np.arange(1, trace.size + 1)
    ax.plot(it, trace, color=SLOTS[0], lw=1.6)
    ax.plot(it[[0, -1]], trace[[0, -1]], ls="none", marker="o", ms=7, mfc=SLOTS[0], mec=SURFACE, mew=1.5, zorder=3)
    ax.set_xlabel("update passes completed")
    ax.set_ylabel("objective (ELBO)")
    ax.set_title("(b) Stage B: n-HDP objective", loc="left")
    ax.annotate(f"{trace[-1]:.0f}".replace("-", "\u2212"), (it[-1], trace[-1]), xytext=(-6, -14),
                textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
    ax.annotate(f"{trace[0]:.0f}".replace("-", "\u2212"), (it[0], trace[0]), xytext=(8, -4),
                textcoords="offset points", color=INK2, fontsize=8.5)
    # (c) mean gate by kind of edge (one hue: the kinds are nominal, the bar length carries the value)
    ax = axes[2]
    kinds = list(extra["gate_by_edge_kind"])
    vals = [extra["gate_by_edge_kind"][k]["mean_gate"] for k in kinds]
    ypos = np.arange(len(kinds))[::-1]
    ax.barh(ypos, np.nan_to_num(vals), height=0.38, color=SLOTS[0], zorder=3)
    for p, v in zip(ypos, vals):
        ax.annotate("n/a" if not np.isfinite(v) else f"{v:.3f}", (0 if not np.isfinite(v) else v, p), xytext=(5, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([k.replace(", other", ",\nother") for k in kinds], fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("mean gate ⟨π_i, π_j⟩")
    ax.set_title("(c) The gate by kind of edge", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_depth_free(path: Path, study: dict) -> None:
    plt = _style()
    depths = study["depths"]
    x = np.arange(len(depths))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [1.0, 1.25]})
    for k, (ax, key, ylabel, title) in enumerate(zip(
            axes, ("probe_test_mean", "centroid_gap"), ("linear-probe test accuracy", "centroid gap (Eq. 28)"),
            ("(a) Accuracy of a linear probe on PᴸX", "(b) Class separation of PᴸX (centroid gap)"))):
        ends = []
        for name, (label, color, marker, _) in OPERATORS.items():
            y = np.asarray(study["operators"][name][key], dtype=float)
            _line(ax, x, y, color, marker, label)
            ends.append((y[-1], label))
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlabel("depth L (propagation steps; no training)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if k == 0:                                             # accuracies are close: zoom, name the lines in the legend
            low = min(min(study["operators"][n][key]) for n in OPERATORS)
            ax.set_ylim(max(0.0, low - 0.06), 1.0)
            ax.set_xlim(-0.3, len(depths) - 1 + 0.3)
            ax.legend(loc="lower left", frameon=False)
        else:                                                  # the gaps separate: name each line at its end
            top = max(max(study["operators"][n][key]) for n in OPERATORS)
            ax.set_ylim(0.0, 1.1 * top)
            ax.set_xlim(-0.3, len(depths) - 1 + 2.4)
            _end_labels(ax, x[-1], ends, 0.07 * 1.1 * top)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_depth_trained(path: Path, summary: dict, classes: int) -> None:
    """Mean of the seeds as a marker on a line, every single seed as a small dot behind it.

    The 95% intervals are in the report's tables; they are not drawn, because a t-interval over five seeds reaches
    past 100% accuracy when some seeds trained and others did not, and the dots show that directly."""
    plt = _style()
    depths = summary["depths"]
    x = np.arange(len(depths))
    n = len(VARIANTS)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    panels = [("test", "test accuracy (%)", "(a) Test accuracy by depth"),
              ("centroid_gap", "centroid gap of the logits", "(b) Class separation of the logits by depth")]
    for ax, (key, ylabel, title) in zip(axes, panels):
        every = []
        for i, (vid, label, _) in enumerate(VARIANTS):
            dodge = (i - (n - 1) / 2) * 0.12
            for j, d in enumerate(depths):
                vals = [v for v in summary[key + "_seeds"][(vid, d)] if np.isfinite(v)]
                every += vals
                ax.plot(np.full(len(vals), x[j] + dodge), vals, ls="none", marker="o", ms=3.4, mew=0, color=SERIES[vid],
                        alpha=0.5, zorder=2, clip_on=False)
            means = np.array([summary[key][(vid, d)][0] for d in depths])
            if np.isfinite(means).any():
                _line(ax, x, means, SERIES[vid], MARKERS[vid], label, dashed=vid == "nhdp_gate_off", dodge=dodge)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlim(-0.45, len(depths) - 1 + 0.45)
        ax.set_xlabel("depth (propagation steps); dots: single seeds")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if key == "test":                                      # what predicting at random would give
            chance = 100.0 / classes
            ax.axhline(chance, color=AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
            ax.annotate(f"{chance:.0f}%: one class in {classes}", (-0.45, chance), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color=INK2)
            ax.set_ylim(max(0.0, min(every + [chance]) - 6.0), 101.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def md_table(header: list, rows: list) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def cell(c, scale_digits: int = 1) -> str:
    m, h, n = c
    if n == 0:
        return "no result"
    ci = "" if not np.isfinite(h) else f" ± {h:.{scale_digits}f}"
    return f"{m:.{scale_digits}f}{ci} (n={n})"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def repo_link(text: str, target: str, out) -> str:
    """A Markdown link from the report to a file of the repository, relative to the report's directory. A report written
    outside the repository has nothing to link to, so it names the file."""
    here = Path(out).resolve()
    if here.is_relative_to(ROOT):
        return f"[{text}]({os.path.relpath(ROOT / target, here).replace(os.sep, '/')})"
    return f"`{target}`"


def build_report(ctx: dict) -> str:
    facts, rep, extra, study = ctx["facts"], ctx["stage"], ctx["extra"], ctx["study"]
    env, code, syn, out = ctx["environment"], ctx["code"], ctx["synthetic"], ctx["out"]
    truth, sb, sa = extra["truth"], rep["stage_b"], rep["stage_a"]
    commit = (code.get("code_commit") or "unknown")[:12] + (" (modified)" if code.get("code_dirty") else "")
    torch_v = env["packages"].get("torch") or "not installed"
    ci = ctx.get("ci")
    where = ((f" It ran in the GitHub Actions workflow `{ci['workflow']}`"
              + (f" ([run {ci['run_id']}]({ci['run_url']}))" if ci.get("run_url") else "") + ".") if ci else "")
    L = ["# n-HDP-GNN demo", "",
         f"Generated by `{ctx['command']}` on {ctx['date']} at code commit {commit}, CPU {env.get('cpu_model')}, "
         f"Python {env.get('python')}, numpy {env['packages'].get('numpy')}, torch {torch_v}; run time "
         f"{ctx['total_s'] / 60:.1f} min.{where}", "",
         "One planted synthetic graph, built without a download, so that every stage can be compared with the truth. "
         "It shows what each stage computes and how the pieces behave. It is a demonstration of the mechanics, not "
         f"evidence about accuracy on real graphs: {repo_link('results/README.md', 'results/README.md', out)} and "
         f"{repo_link('artifacts/README.md', 'artifacts/README.md', out)} have those. Every number below was computed by "
         "the script that generated this file. Sections 1 to 4 are computed without PyTorch. Training (section 5) can give "
         "different values on another CPU model, in unstable cells (deep models) by many points "
         f"({repo_link('examples/README.md', 'examples/README.md', out)}, how to read it).", "",
         "## 1. The graph", ""]
    L += md_table(["", ""], [
        ["nodes / undirected edges", f"{facts['nodes']} / {facts['edges']} (average degree {facts['average_degree']:.1f})"],
        ["planted hierarchy", f"{facts['planted_super_communities']} super-communities of "
                              f"{facts['planted_communities'] // facts['planted_super_communities']} communities of "
                              f"{facts['nodes_per_community']} nodes"],
        ["edges inside a community / inside a super-community",
         f"{pct(facts['share_edges_inside_community'])} / {pct(facts['share_edges_inside_super'])}"],
        ["classes", f"{facts['classes']}; a node keeps its super-community's class with probability "
                    f"{facts['class_alignment']}, otherwise it takes another class"],
        ["edge homophily (same class at both ends)", f"{facts['edge_homophily']:.3f}"],
        ["features", f"{facts['feature_dim']}-dimensional Gaussian around a class mean (signal-to-noise {facts['feature_snr']})"],
        ["diameter / connected components", f"{facts['diameter']} / {facts['components']}"],
        ["share of node pairs at least 8 hops apart", pct(facts["share_pairs_8_hops_or_more"])]])
    L += ["", "![The graph and the gate](fig_graph.png)", ""]
    if facts["share_pairs_8_hops_or_more"] == 0:
        L += [f"MADGap (Chen et al.) compares node pairs at least 8 hops apart with neighbours up to 3 hops away. The "
              f"diameter here is {facts['diameter']}, so no pair is 8 hops apart, the remote set is empty and MADGap is "
              "not defined as intended; it is not reported in this demo (the centroid gap of Eq. 28 is). The "
              f"{repo_link('artifacts', 'artifacts/README.md', out)} report MADGap on the real graphs.", ""]
    L += ["## 2. Stage A: nested Louvain communities", "",
          f"Louvain found {sa['louvain_levels']} levels: {truth['found_communities']} communities "
          f"(planted: {facts['planted_communities']}) in {truth['found_super_communities']} super-communities "
          f"(planted: {facts['planted_super_communities']}), in {sa['time_s']:.2f} s.", ""]
    L += md_table(["agreement with the planted structure", "NMI", "ARI"], [
        ["communities", f"{truth['nmi_communities']:.3f}", f"{truth['ari_communities']:.3f}"],
        ["super-communities", f"{truth['nmi_super_communities']:.3f}", f"{truth['ari_super_communities']:.3f}"]])
    L += ["", f"Purity of the super-communities against the class labels: {rep['purity']['louvain_super_communities']:.3f}. "
          "The heat map in panel (a) of the next figure shows which found community holds which planted one.", "",
          "## 3. Stage B: nested HDP", "",
          f"The n-HDP coordinate ascent ran {sb['iterations']} passes ({'converged' if sb['converged'] else 'stopped at the iteration limit'}); "
          f"the objective is evaluated after every pass but the last, {len(sb['elbo_trace'])} values, from "
          f"{sb['elbo_trace'][0]:.1f} to {sb['elbo_trace'][-1]:.1f}; its smallest step was "
          f"{sb['elbo_smallest_step']:.3g} ({'monotone' if sb['elbo_monotone'] else 'NOT monotone'}). "
          f"{sb['effective_k']} atoms carry mass and {sb['atoms_used_after_topk']} are used after the top-k "
          f"truncation; the largest responsibility averages {sb['mean_max_responsibility']:.3f}. Stage B took "
          f"{sb['time_s']:.2f} s.", ""]
    L += md_table(["hard n-HDP assignment against the planted structure", "NMI", "ARI"], [
        ["super-communities", f"{truth['nmi_nhdp_hard_vs_super']:.3f}", f"{truth['ari_nhdp_hard_vs_super']:.3f}"],
        ["class labels", f"{truth['nmi_nhdp_hard_vs_class']:.3f}", ""]])
    L += ["", f"Purity of the hard assignment against the class labels: {rep['purity']['nhdp_hard_clusters']:.3f}.", "",
          "The gate of an edge is g_ij = ⟨π_i, π_j⟩, the overlap of the two nodes' responsibilities:", ""]
    L += md_table(["kind of edge", "edges", "mean gate"],
                  [[k, v["edges"], f"{v['mean_gate']:.3f}"] for k, v in extra["gate_by_edge_kind"].items()])
    kinds = extra["gate_by_edge_kind"]
    inside = [v["mean_gate"] for k, v in kinds.items() if k != "other super-community" and v["edges"]]
    between = kinds["other super-community"]
    L += ["", f"Averaged over all edges the gate is {rep['gates']['nhdp_topk']['kept_mass']:.3f}. The super-node graph "
          f"built from the responsibilities has {100 * (1 - rep['edge_reduction_r_E']):.1f}% of the graph's directed "
          f"edges (edge reduction r_E = {rep['edge_reduction_r_E']:.3f})."]
    if between["edges"] and inside and between["mean_gate"] < 0.1 * min(inside):
        L += ["The gate is close to zero on the edges between super-communities and large on the edges inside them, "
              "whichever community of the super-community they join."]
    L += ["", "![Stages A and B](fig_stage_ab.png)", "",
          "## 4. Depth without training", "",
          "The depth study applies each operator P to the input features L times and measures PᴸX: a logistic-regression "
          "probe on 20 labels per class (test accuracy, mean over 5 label draws, C chosen on validation) and the centroid "
          "gap of Eq. 28 (class separation; 0 means the class centroids coincide). No GNN is trained. The normalised "
          "adjacency is GCN's operator; the gated operators are P_ij = Â_ij g_ij with the n-HDP top-k "
          "responsibilities, their hard assignment, and the coarsest Louvain community.", ""]
    depths = study["depths"]
    L += ["**Linear-probe test accuracy of PᴸX (%), mean ± 95% interval over 5 label draws**", ""]
    L += md_table(["operator \\ depth L"] + [str(d) for d in depths],
                  [[OPERATORS[n][0]] + [cell((100 * m, 100 * h, 5), 1) .replace(" (n=5)", "") for m, h in
                                        zip(study["operators"][n]["probe_test_mean"], study["operators"][n]["probe_test_ci95"])]
                   for n in OPERATORS])
    L += ["", "**Centroid gap of PᴸX**", ""]
    L += md_table(["operator \\ depth L"] + [str(d) for d in depths],
                  [[OPERATORS[n][0]] + [f"{v:.3f}" for v in study["operators"][n]["centroid_gap"]] for n in OPERATORS])
    L.append("")
    rc = study.get("resolution_check")
    gcn = study["operators"]["A_hat"]
    if rc and gcn["centroid_gap"][-1] < 0.1 and rc["probe_test_shrunk"] < rc["probe_test"] - 0.1:   # only if it shows
        L += [f"At depth {rc['depth']} the centroid gap of the normalised adjacency is {gcn['centroid_gap'][-1]:.3f}: the class "
              f"centroids of the unit-length rows of PᴸX differ by a cosine distance of {rc['centroid_cosine_distance']:.1e}. "
              f"The probe reaches {100 * rc['probe_test']:.1f}% there. A probe can read a difference only down to some size: "
              f"with the deviations of the rows from their mean shrunk by a factor of {rc['shrink']:g} it reaches "
              f"{100 * rc['probe_test_shrunk']:.1f}%; predicting one class for every node gives "
              f"{100 * rc['single_class_accuracy'][0]:.1f}% to {100 * rc['single_class_accuracy'][1]:.1f}%, depending on the "
              "class. A probe accuracy that stays flat with depth therefore does not by itself show "
              "that propagation has kept the classes apart; the centroid gap measures how far apart they are.", ""]
    same = [d for d, a, b in zip(depths, gcn["probe_test_mean"], study["operators"]["gated_louvain"]["probe_test_mean"])
            if d >= 1 and abs(a - b) < 1e-9]
    if same and same[-1] == depths[-1] and len(same) >= 2:
        first = next(d for d in same if all(abs(a - b) < 1e-9 for dd, a, b in zip(depths, gcn["probe_test_mean"],
                     study["operators"]["gated_louvain"]["probe_test_mean"]) if dd >= d))
        L += [f"In panel (a) of the next figure the normalised adjacency and the Louvain gate give the same accuracy from "
              f"depth {first} on, so one line hides the other.", ""]
    dl = rep["depth_limit"]
    L += ["In the limit of infinite depth, propagation stays inside the connected components of the gate's support, so "
          "the purity of those components bounds a classifier's accuracy (docs/METHOD.md, Proposition):", ""]
    L += md_table(["gate", "connected components", "purity bound"],
                  [[name, v["components"], f"{v['purity_bound']:.3f}"] for name, v in dl.items()])
    if dl["nhdp_topk"]["components"] == dl["no_gate"]["components"] < dl["nhdp_hard"]["components"]:
        L += ["", "The soft top-k gate is positive, if tiny, on some edges between clusters, so its support is as connected "
              "as the ungated graph and gives the same bound; the hard n-HDP assignment and the Louvain gate cut."]
    L += ["", "![Depth without training](fig_depth_free.png)", ""]
    tr = ctx.get("trained")
    L += ["## 5. Depth with training", ""]
    if tr is None:
        L += [f"Not generated: {ctx['train_note']}", ""]
    else:
        base = ctx["train_base"]
        n_seeds, depths_t = tr["seeds"], tr["depths"]
        L += [f"Six models trained on this graph with the same settings, at depths {', '.join(map(str, depths_t))}, "
              f"{n_seeds} seeds each (the seed changes the split, the initialisation and the stage-B initialisation). "
              f"Semi-supervised: {facts['classes']} classes x 20 labelled nodes; validation and test nodes are those of "
              f"the 60/20/20 split of the same seed. Hidden size {base['hidden']}, {base['heads']} heads, learning rate "
              f"{base['lr']}, weight decay {base['weight_decay']}, at most {base['epochs']} epochs, early stopping on "
              f"validation accuracy with patience {base['patience']}, the selected state evaluated once on the test "
              "nodes. These are generic settings, the same for every model; none was tuned, and none is the article's. "
              "\"Gate off\" is the same n-HDP-GNN with every edge gate set to 1 (the community and global attention "
              "still use the responsibilities).", ""]
        for key, title, digits in (("test", "Test accuracy (%), mean ± 95% interval", 1),
                                   ("centroid_gap", "Centroid gap of the logits (Eq. 28), mean ± 95% interval", 2),
                                   ("nmi", "k-means NMI of the logits, mean ± 95% interval", 2)):
            L += [f"**{title}**", ""]
            L += md_table(["model \\ depth"] + [str(d) for d in depths_t],
                          [[label] + [cell(tr[key][(vid, d)], digits) for d in depths_t] for vid, label, _ in VARIANTS])
            L.append("")
        L += ["**Median best epoch / median epochs run** (a best epoch near 1 means the validation metric never improved "
              "on the first epochs: the model did not train under these settings)", ""]
        L += md_table(["model \\ depth"] + [str(d) for d in depths_t],
                      [[label] + [("%g / %g" % tr["epochs"][(vid, d)]) if (vid, d) in tr["epochs"] else "no result"
                                  for d in depths_t] for vid, label, _ in VARIANTS])
        L += ["", f"With {n_seeds} seeds the intervals are wide and no significance test is reported; the smallest "
              "possible two-sided Wilcoxon p-value would be "
              f"{2 / 2 ** n_seeds:.3f}.", ""]
        total = len(tr["results"]) + len(tr["failures"]) + len(tr["missing"])
        L += [f"{len(tr['results'])} of {total} runs completed, {len(tr['failures'])} failed, {len(tr['missing'])} have no record."]
        if tr["failures"]:
            L += ["", "Failed runs (not replaced by a value):", ""]
            L += md_table(["variant", "depth", "seed", "reason", "detail"],
                          [[f["config"]["tag"], f["config"]["depth"], f["config"]["seed"], f["failure"]["reason"],
                            f["failure"].get("detail", "")[:120].replace("|", "/")] for f in tr["failures"]])
        if tr["missing"]:
            L += ["", "Runs without a record: " + ", ".join(f"{c.tag} depth {c.depth} seed {c.seed}" for c in tr["missing"][:20])]
        L += ["", "![Depth with training](fig_depth_trained.png)", "",
              "In the figure a marker is the mean over the seeds and each small dot is one seed; the 95% intervals are in "
              "the tables above. The dashed line in panel (a) is the accuracy of choosing one class at random.", ""]
        cpus = sorted({r["environment"].get("cpu_model") for r in tr["results"] if r.get("environment")} - {None})
        if len(cpus) > 1:
            L += [f"The runs were made on more than one CPU model ({'; '.join(cpus)}); the same run can give different "
                  "values on different CPU models, so the cells mix them.", ""]
    L += ["## Provenance", "",
          f"`{ctx['command']}`; code commit {commit}; {env.get('platform')}; {env.get('cpu_count')} CPUs; "
          f"scipy {env['packages'].get('scipy')}, scikit-learn {env['packages'].get('scikit-learn')}, "
          f"networkx {env['packages'].get('networkx')}, torch_geometric {env['packages'].get('torch_geometric') or 'not installed'}. "
          "`demo.json` holds every number above, the graph's parameters and the training configuration; "
          "`runs.jsonl.gz` the result record of every training run.", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- driver
def write_jsonl_gz(path: Path, records) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")


def ci_run() -> dict | None:
    """The GitHub Actions run this process is part of, from the variables the runner sets; None anywhere else."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None
    server, repo, run = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return {"workflow": os.environ.get("GITHUB_WORKFLOW"), "event": os.environ.get("GITHUB_EVENT_NAME"),
            "ref": os.environ.get("GITHUB_REF"), "sha": os.environ.get("GITHUB_SHA"), "run_id": run,
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "run_url": f"{server}/{repo}/actions/runs/{run}" if server and repo and run else None}


def have_torch() -> str | None:
    """None if PyTorch and PyTorch Geometric are installed, otherwise what is missing."""
    absent = [m for m in ("torch", "torch_geometric") if importlib.util.find_spec(m) is None]
    return " and ".join(absent) + " not installed" if absent else None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / "demo" / "output"))
    ap.add_argument("--work", default=str(ROOT / "runs" / "demo" / "work"),
                    help="run records and stage caches of the training")
    ap.add_argument("--quick", action="store_true", help="smaller graph, fewer depths, seeds and epochs")
    ap.add_argument("--train", choices=("auto", "yes", "no"), default="auto",
                    help="auto: train if PyTorch is installed; yes: a missing PyTorch is an error; no: skip section 5")
    ap.add_argument("--jobs", type=int, default=1, help="training runs at the same time (one process each if > 1)")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per training cell (default 5, quick 3)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    syn = dict(QUICK_GRAPH if args.quick else GRAPH)
    study_depths = QUICK_STUDY_DEPTHS if args.quick else STUDY_DEPTHS
    train_depths = QUICK_TRAIN_DEPTHS if args.quick else TRAIN_DEPTHS
    seeds = args.seeds or (3 if args.quick else 5)
    base = {**TRAIN_BASE, **(QUICK_TRAIN_BASE if args.quick else {})}

    missing_torch = have_torch()
    train_note = ""
    if args.train == "no":
        train_note = "training was switched off (`--train no`); run the script without it, with PyTorch and PyTorch Geometric installed."
    elif missing_torch:
        if args.train == "yes":
            print(f"error: --train yes needs PyTorch and PyTorch Geometric ({missing_torch})", file=sys.stderr)
            return 2
        train_note = f"{missing_torch}; install PyTorch and PyTorch Geometric and run the script again."
    do_train = not train_note

    from nhdp_gnn.data import load_dataset
    from nhdp_gnn.utils import code_version, environment, save_json

    data = load_dataset("synthetic", synthetic=syn)
    t = time.perf_counter()
    facts = graph_facts(data, syn)
    print(f"graph: {facts['nodes']} nodes, {facts['edges']} edges, diameter {facts['diameter']}", flush=True)
    report, st, extra = structure(data, syn)
    print(f"stages A and B: {time.perf_counter() - t:.1f} s; n-HDP purity {report['purity']['nhdp_hard_clusters']:.3f}", flush=True)
    t = time.perf_counter()
    study = depth_study(data, st, study_depths)
    print(f"training-free depth study: {time.perf_counter() - t:.1f} s", flush=True)

    trained = None
    if do_train:
        cfgs = variant_configs(syn, train_depths, seeds, base, work)
        t = time.perf_counter()
        counts = run_training(cfgs, args.jobs)
        print(f"training finished in {time.perf_counter() - t:.0f} s: "
              + ", ".join(f"{v} {k}" for k, v in counts.items() if v), flush=True)
        trained = trained_summary(cfgs, train_depths, seeds)

    figures = []
    if not args.no_figures:
        if importlib.util.find_spec("matplotlib") is None:
            print("matplotlib is not installed: no figures are drawn", flush=True)
        else:
            figure_graph(out / "fig_graph.png", data, st, extra, syn)
            figure_stage_ab(out / "fig_stage_ab.png", report, extra)
            figure_depth_free(out / "fig_depth_free.png", study)
            figures = ["fig_graph.png", "fig_stage_ab.png", "fig_depth_free.png"]
            if trained is not None:
                figure_depth_trained(out / "fig_depth_trained.png", trained, facts["classes"])
                figures.append("fig_depth_trained.png")

    command = "python examples/demo.py" + (" " + " ".join(argv) if argv else "")
    ctx = {"facts": facts, "stage": report, "extra": extra, "study": study, "trained": trained,
           "train_note": train_note, "train_base": base, "environment": environment(), "code": code_version(),
           "synthetic": syn, "command": command, "date": time.strftime("%Y-%m-%d", time.gmtime()), "out": out,
           "ci": ci_run(),
           "total_s": time.perf_counter() - started}
    (out / "demo.md").write_text(build_report(ctx))
    doc = {"command": command, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "total_s": ctx["total_s"], "code": ctx["code"], "ci": ctx["ci"], "environment": ctx["environment"], "graph": facts,
           "stage_a_b": report, "truth": extra["truth"], "gate_by_edge_kind": extra["gate_by_edge_kind"],
           "contingency": extra["contingency"], "depth_study": study, "figures": figures,
           "training": None if trained is None else {
               "base": base, "depths": list(train_depths), "seeds": seeds, "variants": [v[0] for v in VARIANTS],
               "cells": {k: {f"{vid}|{d}": list(v) for (vid, d), v in trained[k].items()} for k in ("test", "centroid_gap", "nmi")},
               "seed_values": {k: {f"{vid}|{d}": v for (vid, d), v in trained[k + "_seeds"].items()}
                               for k in ("test", "centroid_gap", "nmi")},
               "completed": len(trained["results"]), "failed": len(trained["failures"]), "missing": len(trained["missing"])}}
    save_json(out / "demo.json", doc)
    if trained is not None:
        write_jsonl_gz(out / "runs.jsonl.gz", ({k: v for k, v in r.items() if k != "curves"} for r in trained["results"]))
        if trained["failures"]:
            write_jsonl_gz(out / "failures.jsonl.gz", trained["failures"])
    print(f"report: {out / 'demo.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
