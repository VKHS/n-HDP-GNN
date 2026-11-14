"""Measured artifacts of stages A and B and of the gated propagation operator (numpy only).

    python -m nhdp_gnn.artifacts --out artifacts                       # Cora, CiteSeer, PubMed, Photo, Computers
    python -m nhdp_gnn.artifacts --datasets cora --skip-partition --out /tmp/a

Outputs (every value is measured on the machine that runs the command):

  <out>/stage_ab/<dataset>.json         stages A and B with one part: communities, n-HDP
                                        convergence and ELBO trace, cluster purity, gate
                                        selectivity on same- vs cross-class edges, and the
                                        purity bound of the infinite-depth limit for each gate
                                        (the Proposition in docs/METHOD.md)
  <out>/stage_ab/partition_study.csv    rho, rho_g, r_E, purity and wall time for P parts and
                                        each partitioning method (single process)
  <out>/operator_depth/<dataset>.json   training-free depth study of P^L X for the normalised
                                        adjacency and three gated operators: linear-probe accuracy
                                        (C chosen on validation), centroid gap (Eq. 28), MADGap
  <out>/manifest.json                   code commit, command, platform, library versions, run time

Stage settings are the defaults every training configuration uses (``RunConfig().pipeline()``).
The depth study uses the semi-supervised split (Planetoid: the public split; other
graphs: 20 labels per class drawn per seed), as in configs/table3_semi.toml.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .config import RunConfig
from .data import load_dataset
from .gating import gate_weights, gated_operator, normalized_adjacency
from .metrics import centroid_gap, madgap_from_pairs, madgap_pairs, mean_ci, partition_purity
from .pipeline import PipelineConfig, run_stages
from .splits import regime_split
from .utils import save_json

DATASETS = ("cora", "citeseer", "pubmed", "photo", "computers")
DEPTHS = (0, 1, 2, 4, 8, 16, 32, 64)
PARTS = (2, 4, 8, 14)
METHODS = ("community", "bfs", "random")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


# ----------------------------------------------------------------------------- stage A/B
def _size_stats(labels: np.ndarray) -> dict:
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
    return {"count": int(sizes.size), "median_size": float(np.median(sizes)), "mean_size": float(sizes.mean()),
            "max_size": int(sizes.max())}


def gate_summary(g: np.ndarray, same: np.ndarray) -> dict:
    """Gate values on same-class vs cross-class edges, and the class homophily of the gated mass."""
    g = np.asarray(g, dtype=np.float64)
    total = g.sum()
    cross = ~same
    mean_same = float(g[same].mean()) if same.any() else float("nan")
    mean_cross = float(g[cross].mean()) if cross.any() else float("nan")
    return {"mean_same": mean_same, "mean_cross": mean_cross,
            "cross_to_same": mean_cross / mean_same if mean_same > 0 else float("nan"),
            "zero_share_same": float((g[same] == 0).mean()) if same.any() else float("nan"),
            "zero_share_cross": float((g[cross] == 0).mean()) if cross.any() else float("nan"),
            "kept_mass": float(total / max(g.size, 1)),
            "gated_homophily": float(g[same].sum() / total) if total > 0 else float("nan")}


def component_ceiling(edges: np.ndarray, num_nodes: int, keep: np.ndarray, y: np.ndarray) -> dict:
    """Connected components of the gate support and their purity: the accuracy bound of a
    classifier on the infinite-depth limit of the gated operator (docs/METHOD.md)."""
    src, dst = edges[:, keep]
    A = sparse.csr_matrix((np.ones(src.size), (src, dst)), shape=(num_nodes, num_nodes))
    n, comp = csgraph.connected_components(A, directed=False)
    return {"components": int(n), "purity_bound": partition_purity(comp, y)}


def stage_ab_report(data, cfg: PipelineConfig):
    """Stage A/B diagnostics with ``cfg`` (no cache); returns (report, StageOutput)."""
    t0 = time.perf_counter()
    st = run_stages(data, cfg, cache_dir=None)
    wall = time.perf_counter() - t0
    d = st.diagnostics
    y = np.asarray(data.y)
    N = data.num_nodes
    once = data.edge_index[0] < data.edge_index[1]
    e1 = data.edge_index[:, once]                       # every undirected edge once
    same = y[e1[0]] == y[e1[1]]
    hard = st.R.argmax(1)
    louvain = st.louvain
    gates = {"nhdp_topk": gate_weights(e1, st.R),
             "nhdp_hard": (hard[e1[0]] == hard[e1[1]]).astype(np.float64),
             "louvain": (louvain[e1[0]] == louvain[e1[1]]).astype(np.float64)}
    ceilings = {"no_gate": component_ceiling(e1, N, np.ones(e1.shape[1], dtype=bool), y)}
    ceilings.update({k: component_ceiling(e1, N, g > 0, y) for k, g in gates.items()})
    trace = np.asarray(d["nhdp_elbo_trace"], dtype=np.float64)
    steps = np.diff(trace)
    report = {
        "dataset": data.name, "nodes": N, "edges": int(once.sum()), "features": int(data.x.shape[1]),
        "classes": int(data.num_classes), "edge_homophily": float(same.mean()),
        "stage_a": {"louvain_levels": int(d["louvain_levels"]), "communities": _size_stats(st.community),
                    "super_communities": _size_stats(louvain), "time_s": float(d["time_stage_a_s"])},
        "stage_b": {"iterations": int(d["nhdp_iters"]), "converged": bool(d["nhdp_converged"]),
                    "effective_k": int(d["nhdp_effective_k"]), "atoms_used_after_topk": int(d["assign_k"]),
                    "elbo_trace": trace.tolist(),
                    "elbo_smallest_step": float(steps.min()) if steps.size else float("nan"),
                    "elbo_monotone": bool(np.all(steps >= -1e-8 * np.abs(trace[1:]))) if steps.size else True,
                    "mean_max_responsibility": float(st.R.max(1).mean()), "time_s": float(d["time_stage_b_s"])},
        "purity": {"louvain_super_communities": float(d["louvain_purity"]),
                   "nhdp_hard_clusters": float(d["assign_purity"])},
        "gates": {k: gate_summary(g, same) for k, g in gates.items()},
        "depth_limit": ceilings,
        "edge_reduction_r_E": float(d["edge_reduction"]),
        "wall_s": wall,
        "config": asdict(cfg),
    }
    return report, st


def partition_study(data, base: PipelineConfig, parts=PARTS, methods=METHODS) -> list[dict]:
    rows = []
    for P in parts:
        for method in methods:
            cfg = replace(base, num_parts=P, partition=method)
            t0 = time.perf_counter()
            st = run_stages(data, cfg, cache_dir=None)
            wall = time.perf_counter() - t0
            d = st.diagnostics
            sizes = np.bincount(st.parts, minlength=P)
            rows.append({"dataset": data.name, "parts": P, "method": method, "rho": d["rho"],
                         "rho_gated": d["rho_gated"], "edge_reduction_r_E": d["edge_reduction"],
                         "communities": d["num_communities"], "super_communities": d["num_super_communities"],
                         "louvain_purity": d.get("louvain_purity", float("nan")),
                         "nhdp_purity": d.get("assign_purity", float("nan")),
                         "nhdp_iterations": d["nhdp_iters"], "nhdp_converged": bool(d["nhdp_converged"]),
                         "largest_part_over_mean": float(sizes.max() / sizes.mean()),
                         "time_stage_a_s": d["time_stage_a_s"], "time_stage_b_s": d["time_stage_b_s"],
                         "wall_s": wall})
    return rows


# ----------------------------------------------------------------------------- operator depth study
def _onehot(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    R = np.zeros((inv.size, inv.max() + 1))
    R[np.arange(inv.size), inv] = 1.0
    return R


def linear_probe(H: np.ndarray, y: np.ndarray, split, Cs=C_GRID) -> dict:
    """Logistic regression on L2-normalised rows; C chosen on validation accuracy."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    tr, va, te = split
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    best = {"val": -1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=3000).fit(Hn[tr], y[tr])
            acc_val = float((clf.predict(Hn[va]) == y[va]).mean())
            if acc_val > best["val"]:
                best = {"val": acc_val, "test": float((clf.predict(Hn[te]) == y[te]).mean()), "C": C}
    return best


def operator_depth_study(data, stages, depths=DEPTHS, probe_seeds=(0, 1, 2, 3, 4), Cs=C_GRID,
                         madgap_sources: int = 1000) -> dict:
    """P^L X for the normalised adjacency and three gated operators (no training of the GNN)."""
    N, ei, y = data.num_nodes, data.edge_index, np.asarray(data.y)
    operators = {
        "A_hat": normalized_adjacency(ei, N),
        "gated_nhdp_topk": gated_operator(ei, N, stages.R),
        "gated_nhdp_hard": gated_operator(ei, N, _onehot(stages.R.argmax(1))),
        "gated_louvain": gated_operator(ei, N, _onehot(stages.louvain)),
    }
    if data.public_split is not None:
        splits = [("public", data.public_split)]
    else:
        splits = [(f"semi-seed{s}", regime_split(y, "semi", s, semi_per_class=20)) for s in probe_seeds]
    pairs = madgap_pairs(ei, N, max_sources=madgap_sources, seed=0)
    X = np.asarray(data.x, dtype=np.float64)
    out = {"dataset": data.name, "depths": list(depths), "splits": [name for name, _ in splits],
           "C_grid": list(Cs), "features": "row-normalised input features", "operators": {}}
    for name, P in operators.items():
        rec = {"probe_test_mean": [], "probe_test_ci95": [], "probe_val_mean": [], "probe_per_split": [],
               "centroid_gap": [], "madgap": [], "mean_row_norm": []}
        H, done = X.copy(), 0
        t0 = time.perf_counter()
        for L in depths:
            for _ in range(L - done):
                H = P @ H
            done = L
            res = [linear_probe(H, y, sp, Cs) for _, sp in splits]
            tests = [r["test"] for r in res]
            mean, half = mean_ci(tests) if len(tests) > 1 else (tests[0], float("nan"))
            rec["probe_test_mean"].append(mean)
            rec["probe_test_ci95"].append(half)
            rec["probe_val_mean"].append(float(np.mean([r["val"] for r in res])))
            rec["probe_per_split"].append(res)
            rec["centroid_gap"].append(centroid_gap(H, y))
            rec["madgap"].append(madgap_from_pairs(H, pairs))
            rec["mean_row_norm"].append(float(np.linalg.norm(H, axis=1).mean()))
        rec["time_s"] = time.perf_counter() - t0
        out["operators"][name] = rec
    return out


# ----------------------------------------------------------------------------- provenance
def manifest(argv, started: float) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                                  cwd=Path(__file__).resolve().parent).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no", "--", ".")     # the package code only
    versions = {}
    for mod in ("numpy", "scipy", "sklearn", "networkx"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    return {"command": "python -m nhdp_gnn.artifacts " + " ".join(argv),
            "code_commit": git("rev-parse", "HEAD"), "uncommitted_changes_in_nhdp_gnn": bool(status),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "libraries": versions,
            "louvain_backend": "igraph" if importlib.util.find_spec("igraph") else "networkx",
            "metis_available": importlib.util.find_spec("pymetis") is not None,
            "total_time_s": time.perf_counter() - started,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", nargs="+", type=int, default=list(PARTS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    ap.add_argument("--probe-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--skip-partition", action="store_true")
    ap.add_argument("--skip-operator", action="store_true")
    args = ap.parse_args(argv)
    started = time.perf_counter()
    out = Path(args.out)
    base = RunConfig(seed=args.seed).pipeline()
    methods = list(args.methods) + (["metis"] if importlib.util.find_spec("pymetis") and "metis" not in args.methods
                                    else [])
    partition_rows = []
    for name in args.datasets:
        data = load_dataset(name, args.root)
        report, st = stage_ab_report(data, base)
        save_json(out / "stage_ab" / f"{name}.json", report)
        g = report["gates"]["nhdp_topk"]
        print(f"[{name}] stage A/B {report['wall_s']:.1f}s  J/S={report['stage_a']['communities']['count']}/"
              f"{report['stage_a']['super_communities']['count']}  n-HDP iters={report['stage_b']['iterations']}  "
              f"purity louvain/nhdp={report['purity']['louvain_super_communities']:.3f}/"
              f"{report['purity']['nhdp_hard_clusters']:.3f}  gate same/cross={g['mean_same']:.3f}/"
              f"{g['mean_cross']:.3f}", flush=True)
        if not args.skip_operator:
            study = operator_depth_study(data, st, depths=args.depths, probe_seeds=args.probe_seeds)
            save_json(out / "operator_depth" / f"{name}.json", study)
            for op, rec in study["operators"].items():
                acc = " ".join(f"{a:.3f}" for a in rec["probe_test_mean"])
                print(f"[{name}] {op:16s} probe acc by depth {study['depths']}: {acc}", flush=True)
        if not args.skip_partition:
            rows = partition_study(data, base, parts=args.parts, methods=methods)
            partition_rows += rows
            for r in rows:
                print(f"[{name}] P={r['parts']:2d} {r['method']:9s} rho={r['rho']:.3f} rho_g={r['rho_gated']:.3f} "
                      f"r_E={r['edge_reduction_r_E']:.3f} {r['wall_s']:.1f}s", flush=True)
    if partition_rows:
        path = out / "stage_ab" / "partition_study.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(partition_rows[0]))
            w.writeheader()
            w.writerows(partition_rows)
    save_json(out / "manifest.json", manifest(argv, started))
    print(json.dumps({"written": str(out)}))


if __name__ == "__main__":
    main()
