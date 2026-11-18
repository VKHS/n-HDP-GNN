#!/usr/bin/env python3
"""Smoke check: a fast end-to-end run of the whole pipeline on a small planted hierarchical graph.

    python scripts/smoke.py                  # every check that can run here; the others are listed as skipped
    python scripts/smoke.py --level numpy    # only the checks that need no PyTorch (seconds)
    python scripts/smoke.py --level full     # every check; a missing PyTorch or PyG is a failure (used in CI)
    python scripts/smoke.py --real           # also Cora (downloaded on first use)
    python scripts/smoke.py --only graph,stage_b
    python scripts/smoke.py --list           # the checks and what each needs

The graph is the planted hierarchy of ``configs/smoke.toml`` (270 nodes, no download), so the checks
know what stages A and B should find.  What each check establishes:

  numpy only    the generator, stage A (nested Louvain), stage B (n-HDP: monotone objective, normalised
                responsibilities, cache), the gate (an exact cut for one-hot responsibilities), the
                partitioner, the training-free depth study, and the ``nhdp_gnn.artifacts`` command
  with PyTorch  the ``nhdp_gnn.train`` command, that models learn, every model through ``nhdp_gnn.sweep``
                (``configs/smoke.toml``), the worker pool, resuming, the table builder, a two-part run,
                two-rank DistributedDataParallel training and the time-limit failure record

Each check ends as ``pass``, ``fail`` or ``skip`` (it cannot run here, with the reason).  A skipped check
is not a pass: the summary counts them apart, and at ``--level full`` they fail.  The numeric floors
in the checks are sanity bounds well below what the planted graph gives (the measured values are in the
report); they catch a broken pipeline, not a change of accuracy.  Nothing here measures the method's
accuracy on real data.

Writes ``<out>/report.json`` and ``<out>/report.md`` (the measured values of every check, the code
commit and the environment) and keeps the files of the checks in ``<out>/work``.  Exit status 0 if no
check failed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if importlib.util.find_spec("nhdp_gnn") is None:            # a checkout that was not pip-installed
    sys.path.insert(0, str(ROOT))

# Sanity floors (see the module docstring); the planted graph gives NMI 0.85-0.98 and purities 0.86-0.96.
FLOOR_NMI_COMMUNITIES = 0.6
FLOOR_PURITY_LOUVAIN = 0.75
FLOOR_PURITY_NHDP = 0.80
LEARNS_MARGIN = 0.10                  # accuracy above the share of the largest class


class CheckFailed(Exception):
    """An expectation of a check did not hold; the message is reported."""


class Skipped(Exception):
    """The check cannot run in this environment; the message is the reason."""


def expect(condition, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


class Spec(NamedTuple):
    name: str
    title: str
    needs: tuple
    real: bool
    fn: object


SPECS: list[Spec] = []


def check(name: str, title: str, needs: tuple = (), real: bool = False):
    def register(fn):
        SPECS.append(Spec(name, title, tuple(needs), real, fn))
        return fn
    return register


CAPABILITIES = {"torch": ("torch", "torch_geometric")}


def missing(need: str) -> str | None:
    """None if the capability is installed, otherwise what is missing."""
    absent = [m for m in CAPABILITIES[need] if importlib.util.find_spec(m) is None]
    return " and ".join(absent) + " not installed" if absent else None


class Ctx:
    """Shared state of the checks: the planted graph, its stage outputs and where files go."""

    def __init__(self, out: Path):
        self.out = out
        self.work = out / "work"
        self.spec = tomllib.loads((ROOT / "configs" / "smoke.toml").read_text())
        self.syn = dict(self.spec["base"]["synthetic"])
        self.runner = None              # "module:function" replacing nhdp_gnn.train:run in the sweep checks (tests only)
        self._data = None
        self._stages = None

    @property
    def data(self):
        if self._data is None:
            from nhdp_gnn.data import load_dataset

            self._data = load_dataset("synthetic", synthetic=self.syn)
        return self._data

    def config(self, **overrides):
        """A RunConfig with the base of configs/smoke.toml, writing under the work directory."""
        from nhdp_gnn.config import config_from_dict

        d = dict(self.spec["base"])
        d.update(out_dir=str(self.work / "runs"), cache_dir=str(self.work / "cache"))
        d.update(overrides)
        return config_from_dict(d)

    @property
    def stages(self):
        if self._stages is None:
            from nhdp_gnn.pipeline import run_stages

            self._stages = run_stages(self.data, self.config().pipeline(), cache_dir=str(self.work / "cache"))
        return self._stages


def sh(args, timeout: float = 1200.0) -> subprocess.CompletedProcess:
    """Run a command in the repository root; a non-zero exit is a failure that shows the end of its output."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH")) if p)
    cmd = [str(a) for a in args]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-30:])
        raise CheckFailed(f"`{' '.join(cmd[1:])}` exited with status {proc.returncode}\n{tail}")
    return proc


def record_problems(rec: dict) -> list[str]:
    """Why ``rec`` is not a valid result of ``nhdp_gnn.train.run`` (an empty list if it is)."""
    from nhdp_gnn.config import config_from_dict

    required = ("key", "config", "metric", "val", "test", "best_epoch", "epochs_run", "propagation_depth",
                "n_params", "n_train", "n_val", "n_test", "timings", "stages", "environment", "code_commit")
    absent = [k for k in required if k not in rec]
    if absent:
        return [f"missing {', '.join(absent)}"]

    def finite(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))

    problems = []
    cfg = rec["config"]
    try:
        if config_from_dict(cfg).key() != rec["key"]:
            problems.append("the key does not match the recorded configuration")
    except (TypeError, ValueError) as exc:
        problems.append(f"the recorded configuration is not a RunConfig ({exc})")
    for name in ("val", "test"):
        if not (finite(rec[name]) and 0.0 <= rec[name] <= 1.0):
            problems.append(f"{name} = {rec[name]!r} is not a metric in [0, 1]")
    if not (finite(rec["best_epoch"]) and 1 <= rec["best_epoch"] <= rec["epochs_run"] <= cfg.get("epochs", 10 ** 9)):
        problems.append(f"best_epoch {rec['best_epoch']!r}, epochs_run {rec['epochs_run']!r} are inconsistent")
    if not (finite(rec["n_params"]) and rec["n_params"] > 0):
        problems.append("n_params is not positive")
    if not all(finite(rec[k]) and rec[k] > 0 for k in ("n_train", "n_val", "n_test")):
        problems.append("a split is empty")
    if not (finite(rec["propagation_depth"]) and rec["propagation_depth"] >= 1):
        problems.append("propagation_depth is not positive")
    timings = rec["timings"]
    if not all(finite(timings.get(k)) and timings[k] > 0 for k in ("train_s", "epoch_s_mean")):
        problems.append(f"timings {timings!r} lack positive train_s and epoch_s_mean")
    if not rec["environment"].get("python"):
        problems.append("the environment is not recorded")
    if cfg.get("diagnostics", True):
        emb = rec.get("embedding")
        if not isinstance(emb, dict):
            problems.append("embedding diagnostics are missing")
        else:
            bad = [k for k in ("centroid_gap", "madgap", "nmi", "ari") if not finite(emb.get(k))]
            if bad:
                problems.append(f"embedding {', '.join(bad)} missing or not finite")
            elif not 0.0 <= emb["nmi"] <= 1.0:
                problems.append(f"embedding nmi = {emb['nmi']} is outside [0, 1]")
            if emb.get("representation") != "logits":
                problems.append("embedding diagnostics are not on the logits")
    return problems


def read_results(directory: Path) -> tuple[list[dict], list[Path]]:
    """Result records and failure-record paths in a run directory."""
    names = sorted(p for p in directory.glob("*.json") if p.is_file())
    failures = [p for p in names if p.name.endswith(".failed.json")]
    return [json.loads(p.read_text()) for p in names if p not in failures], failures


# ----------------------------------------------------------------------------- checks without PyTorch
@check("graph", "the planted hierarchical graph is well formed and reproducible")
def check_graph(ctx: Ctx):
    from nhdp_gnn.metrics import edge_homophily
    from nhdp_gnn.synthetic import hierarchical_sbm

    g1, g2 = hierarchical_sbm(**ctx.syn), hierarchical_sbm(**ctx.syn)
    expect(all(np.array_equal(g1[k], g2[k]) for k in ("x", "y", "edge_index", "community", "super_of")),
           "two calls with the same seed gave different graphs")
    s, c, m = ctx.syn["num_super"], ctx.syn["comm_per_super"], ctx.syn["nodes_per_comm"]
    data = ctx.data
    n = data.num_nodes
    expect(n == s * c * m, f"{n} nodes instead of {s * c * m}")
    expect(data.x.dtype == np.float32 and data.x.shape[0] == n, "features are not float32 (N, F)")
    ei = data.edge_index
    expect(not (ei[0] == ei[1]).any(), "self-loops remain")
    fwd, bwd = ei[0] * n + ei[1], ei[1] * n + ei[0]
    expect(np.unique(fwd).size == fwd.size, "duplicate edges")
    expect(np.array_equal(np.sort(fwd), np.sort(bwd)), "edges are not stored in both directions")
    expect(np.array_equal(np.unique(data.y), np.arange(data.num_classes)), "a class has no node")
    h = edge_homophily(ei, data.y)
    return (f"{n} nodes, {ei.shape[1] // 2} undirected edges, {s * c} planted communities in {s} "
            f"super-communities, {data.num_classes} classes, edge homophily {h:.3f}",
            {"nodes": n, "edges": ei.shape[1] // 2, "edge_homophily": h, "classes": data.num_classes})


@check("stage_a", "stage A: nested Louvain finds the planted communities")
def check_stage_a(ctx: Ctx):
    from sklearn.metrics import normalized_mutual_info_score as nmi

    from nhdp_gnn.communities import nested_communities

    d = ctx.data
    nc = nested_communities(d.edge_index, d.num_nodes, seed=0)
    expect(nc.community.shape == (d.num_nodes,) and nc.super_of.shape[0] == nc.community.max() + 1,
           "community and super-community arrays do not fit together")
    expect(nc.num_communities >= 2 and nc.num_super >= 2, "fewer than two communities or super-communities")
    s_nodes = np.repeat(np.arange(ctx.syn["num_super"]), ctx.syn["comm_per_super"])[d.community]
    nmi_c = float(nmi(d.community, nc.community))
    nmi_s = float(nmi(s_nodes, nc.super_community))
    expect(nmi_c >= FLOOR_NMI_COMMUNITIES,
           f"NMI of the Louvain communities with the planted ones is {nmi_c:.3f} (floor {FLOOR_NMI_COMMUNITIES})")
    return (f"{nc.levels} levels, {nc.num_communities} communities in {nc.num_super} super-communities; NMI with the "
            f"planted communities {nmi_c:.3f}, with the planted super-communities {nmi_s:.3f}",
            {"levels": nc.levels, "communities": nc.num_communities, "super_communities": nc.num_super,
             "nmi_communities": nmi_c, "nmi_super": nmi_s})


@check("stage_b", "stage B: n-HDP objective is monotone, responsibilities are normalised, the cache round-trips")
def check_stage_b(ctx: Ctx):
    from nhdp_gnn.pipeline import run_stages

    cache = ctx.work / "cache-stage-b"
    pc = ctx.config().pipeline()
    first = run_stages(ctx.data, pc, cache_dir=str(cache))
    R, diag = first.R, first.diagnostics
    expect(R is not None and np.isfinite(R).all() and np.allclose(R.sum(1), 1.0, atol=1e-5),
           "responsibilities are not finite rows that sum to one")
    expect(R.shape[1] <= pc.nhdp.K, f"{R.shape[1]} atoms in use, more than K = {pc.nhdp.K}")
    trace = np.asarray(diag["nhdp_elbo_trace"], dtype=np.float64)
    steps = np.diff(trace)
    expect(trace.size >= 2 and bool(np.all(steps >= -1e-8 * np.abs(trace[1:]))),
           f"the n-HDP objective decreased (smallest step {steps.min():.3g})")
    expect(diag["louvain_purity"] >= FLOOR_PURITY_LOUVAIN and diag["assign_purity"] >= FLOOR_PURITY_NHDP,
           f"cluster purity {diag['louvain_purity']:.3f} / {diag['assign_purity']:.3f} below the floors "
           f"{FLOOR_PURITY_LOUVAIN} / {FLOOR_PURITY_NHDP}")
    files = sorted(cache.glob("stages_*.npz"))
    expect(len(files) == 1 and not list(cache.glob(".*.tmp")), "the stage cache was not written as one file")
    again = run_stages(ctx.data, pc, cache_dir=str(cache))
    expect(np.array_equal(again.R, R) and np.array_equal(again.community, first.community),
           "the cached stage outputs differ from the computed ones")
    return (f"{int(diag['nhdp_iters'])} iterations, objective {trace[0]:.1f} -> {trace[-1]:.1f} (smallest step "
            f"{steps.min():.3g}), {R.shape[1]} atoms, purity vs labels: Louvain {diag['louvain_purity']:.3f}, "
            f"n-HDP {diag['assign_purity']:.3f}",
            {"iterations": int(diag["nhdp_iters"]), "elbo_first": trace[0], "elbo_last": trace[-1],
             "elbo_smallest_step": float(steps.min()), "atoms": int(R.shape[1]),
             "purity_louvain": diag["louvain_purity"], "purity_nhdp": diag["assign_purity"]})


@check("gate", "the gate keeps edges inside planted super-communities, and one-hot responsibilities cut exactly")
def check_gate(ctx: Ctx):
    from nhdp_gnn.gating import gate_weights, gated_operator

    d, st = ctx.data, ctx.stages
    once = d.edge_index[:, d.edge_index[0] < d.edge_index[1]]
    g = gate_weights(once, st.R)
    expect(bool(((g >= 0) & (g <= 1)).all()), "gate values outside [0, 1]")
    # n-HDP clusters correspond to the planted super-communities, so the gate separates those; it does not
    # separate the finer communities inside one super-community (they share their atoms)
    super_of = np.repeat(np.arange(ctx.syn["num_super"]), ctx.syn["comm_per_super"])[d.community]
    same_comm = d.community[once[0]] == d.community[once[1]]
    same_super = super_of[once[0]] == super_of[once[1]]
    levels = {"same community": g[same_comm], "same super-community, other community": g[same_super & ~same_comm],
              "other super-community": g[~same_super]}
    mean = {k: float(v.mean()) if v.size else float("nan") for k, v in levels.items()}
    mean_inside, mean_between = float(g[same_super].mean()), mean["other super-community"]
    expect(mean_between < 0.25 * mean_inside,
           f"mean gate {mean_inside:.3f} inside planted super-communities, {mean_between:.3f} between them")
    n = d.num_nodes
    onehot = np.zeros((n, int(d.community.max()) + 1))
    onehot[np.arange(n), d.community] = 1.0
    P = gated_operator(d.edge_index, n, onehot)
    x = np.asarray(d.x, dtype=np.float64)
    moved = x.copy()
    moved[d.community == 0] += 5.0                          # change one community only
    h0, h1 = x.copy(), moved.copy()
    for _ in range(8):
        h0, h1 = P @ h0, P @ h1
    outside = d.community != 0
    expect(np.array_equal(h0[outside], h1[outside]), "a change in one community reached another through a zero gate")
    expect(not np.allclose(h0[~outside], h1[~outside]), "the change did not propagate inside its community")
    return ("mean gate: " + "; ".join(f"{k} {v:.3f}" for k, v in mean.items())
            + f" ({100 * float(same_super.mean()):.1f}% of the edges lie inside a super-community); a one-hot gate "
            "leaves the other communities bit-identical after 8 propagation steps",
            {"mean_gate": mean, "share_inside_super": float(same_super.mean())})


@check("partition", "stage A edge cut: community packing cuts fewer edges than BFS and random parts")
def check_partition(ctx: Ctx):
    from dataclasses import replace

    from nhdp_gnn.gating import cross_partition_ratio
    from nhdp_gnn.partition import partition_nodes
    from nhdp_gnn.pipeline import run_stages

    d, st = ctx.data, ctx.stages
    rho = {}
    for method in ("community", "bfs", "random"):
        parts = partition_nodes(d.edge_index, d.num_nodes, 4, method,
                                communities=st.louvain if method == "community" else None, seed=0)
        expect(np.unique(parts).size == 4, f"{method}: not four non-empty parts")
        rho[method] = float(cross_partition_ratio(d.edge_index, parts))
    expect(rho["community"] < min(rho["bfs"], rho["random"]), f"cut ratios {rho}")
    pc = replace(ctx.config().pipeline(), num_parts=3, partition="community")
    out = run_stages(d, pc, cache_dir=None)
    expect(np.unique(out.parts).size == 3, "three parts requested, not three produced")
    for c in np.unique(out.community):
        expect(np.unique(out.parts[out.community == c]).size == 1, f"community {c} spans several parts")
    return ("cut ratio rho with 4 parts: " + ", ".join(f"{k} {v:.3f}" for k, v in rho.items())
            + "; with 3 parts no community crosses a part", {"rho": rho})


@check("depth_study", "training-free depth study: over-smoothing of the normalised adjacency, finite diagnostics")
def check_depth_study(ctx: Ctx):
    from nhdp_gnn.artifacts import operator_depth_study

    depths = (0, 2, 8, 32)
    study = operator_depth_study(ctx.data, ctx.stages, depths=depths, probe_seeds=(0, 1), madgap_sources=100)
    ops = study["operators"]
    expect(set(ops) == {"A_hat", "gated_nhdp_topk", "gated_nhdp_hard", "gated_louvain"}, f"operators {sorted(ops)}")
    for name, rec in ops.items():
        for key in ("probe_test_mean", "centroid_gap", "madgap", "mean_row_norm"):
            expect(bool(np.isfinite(rec[key]).all()), f"{name}: {key} is not finite")
        expect(all(0.0 <= a <= 1.0 for a in rec["probe_test_mean"]), f"{name}: probe accuracy outside [0, 1]")
    gap0 = [rec["centroid_gap"][0] for rec in ops.values()]
    expect(np.allclose(gap0, gap0[0], atol=1e-12), "depth 0 differs between operators")
    a_hat, hard = ops["A_hat"]["centroid_gap"], ops["gated_nhdp_hard"]["centroid_gap"]
    expect(a_hat[-1] < a_hat[0], f"the centroid gap of the normalised adjacency did not fall ({a_hat})")
    expect(hard[-1] > a_hat[-1], f"hard-gated centroid gap {hard[-1]:.3f} not above the adjacency's {a_hat[-1]:.3f}")
    return (f"centroid gap at depth {depths[0]} / {depths[-1]}: normalised adjacency {a_hat[0]:.3f} / {a_hat[-1]:.3f}, "
            f"n-HDP hard gate {hard[0]:.3f} / {hard[-1]:.3f}; probe accuracy at depth {depths[-1]}: "
            f"{ops['A_hat']['probe_test_mean'][-1]:.3f} / {ops['gated_nhdp_hard']['probe_test_mean'][-1]:.3f}",
            {"depths": list(depths), **{name: {"centroid_gap": r["centroid_gap"], "probe": r["probe_test_mean"]}
                                        for name, r in ops.items()}})


@check("artifacts_cli", "python -m nhdp_gnn.artifacts writes its files (generator's default graph)")
def check_artifacts_cli(ctx: Ctx):
    out = ctx.work / "artifacts"
    sh([sys.executable, "-m", "nhdp_gnn.artifacts", "--datasets", "synthetic", "--parts", "2", "--methods",
        "community", "bfs", "--depths", "0", "2", "--probe-seeds", "0", "1", "--out", out])
    stage = json.loads((out / "stage_ab" / "synthetic.json").read_text())
    depth = json.loads((out / "operator_depth" / "synthetic.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    rows = (out / "stage_ab" / "partition_study.csv").read_text().strip().splitlines()
    expect(stage["stage_b"]["elbo_monotone"] is True, "the n-HDP objective in the artifact is not monotone")
    expect({"gates", "depth_limit", "purity"} <= set(stage), "stage_ab report lacks gates, depth_limit or purity")
    expect(depth["depths"] == [0, 2] and "A_hat" in depth["operators"], "operator_depth report is incomplete")
    expect(len(rows) >= 3, f"partition_study.csv has {len(rows) - 1} rows, expected at least 2")
    expect(manifest.get("command", "").startswith("python -m nhdp_gnn.artifacts"), "manifest has no command")
    return (f"{stage['nodes']} nodes: stage report, depth study, {len(rows) - 1} partition rows and a manifest written",
            {"nodes": stage["nodes"], "partition_rows": len(rows) - 1})


# ----------------------------------------------------------------------------- checks with PyTorch
def _grid(ctx: Ctx) -> list[tuple]:
    g = ctx.spec["grid"]
    return [(m, d, s) for m in g["model"] for d in g["depth"] for s in g["seed"]]


def _sweep_args(ctx: Ctx, directory: str, *extra) -> list:
    args = [sys.executable, "-m", "nhdp_gnn.sweep", "configs/smoke.toml", "--set", f"out_dir={ctx.work / directory}",
            "--set", f"cache_dir={ctx.work / 'cache'}", *extra]
    return args + ["--runner", ctx.runner] if ctx.runner else args


@check("train_cli", "python -m nhdp_gnn.train trains n-HDP-GNN and writes a valid record", needs=("torch",))
def check_train_cli(ctx: Ctx):
    out = ctx.work / "train_cli"
    cfg = ctx.config()
    sh([sys.executable, "-m", "nhdp_gnn.train", "--dataset", "synthetic", "--synthetic", json.dumps(ctx.syn),
        "--regime", "full", "--model", "nhdp", "--depth", "3", "--hidden", "16", "--heads", "2", "--epochs", "10",
        "--patience", "50", "--nhdp_K", cfg.nhdp_K, "--nhdp_T1", cfg.nhdp_T1, "--nhdp_T2", cfg.nhdp_T2,
        "--nhdp_max_iter", cfg.nhdp_max_iter, "--madgap_sources", "64", "--threads", "1",
        "--out_dir", out, "--cache_dir", ctx.work / "cache"])
    results, failures = read_results(out)
    expect(len(results) == 1 and not failures, f"{len(results)} results and {len(failures)} failure records")
    problems = record_problems(results[0])
    expect(not problems, "; ".join(problems))
    r = results[0]
    return (f"n-HDP-GNN, depth {r['propagation_depth']}, {r['n_params']} parameters, best epoch {r['best_epoch']} "
            f"of {r['epochs_run']}, validation {r['val']:.3f}, test {r['test']:.3f}",
            {"val": r["val"], "test": r["test"], "best_epoch": r["best_epoch"], "n_params": r["n_params"]})


@check("learns", "models learn: n-HDP-GNN and GCN beat the largest-class rate by a margin", needs=("torch",))
def check_learns(ctx: Ctx):
    from nhdp_gnn import train

    y = ctx.data.y
    majority = float(np.bincount(y).max() / y.size)
    found = {}
    for model in ("nhdp", "gcn"):
        res = train.run(ctx.config(model=model, depth=2, epochs=60, patience=60, lr=0.02,
                                   out_dir=str(ctx.work / "learns")), verbose=False)
        problems = record_problems(res)
        expect(not problems, f"{model}: " + "; ".join(problems))
        expect(res["test"] > majority + LEARNS_MARGIN,
               f"{model}: test accuracy {res['test']:.3f} is not above {majority + LEARNS_MARGIN:.3f} "
               f"(largest class {majority:.3f})")
        found[model] = res["test"]
    return (f"largest class {majority:.3f}; test accuracy after at most 60 epochs: "
            + ", ".join(f"{k} {v:.3f}" for k, v in found.items()), {"majority": majority, **found})


@check("sweep_all_models", "nhdp_gnn.sweep runs configs/smoke.toml: all models, all records valid", needs=("torch",))
def check_sweep_all_models(ctx: Ctx):
    t0 = time.perf_counter()
    proc = sh(_sweep_args(ctx, "sweep"))
    wall = time.perf_counter() - t0
    results, failures = read_results(ctx.work / "sweep")
    expect(not failures, f"{len(failures)} failure records: " + ", ".join(p.name for p in failures[:5]))
    expected = set(_grid(ctx))
    got = {(r["config"]["model"], r["config"]["depth"], r["config"]["seed"]) for r in results}
    expect(got == expected and len(results) == len(expected),
           f"missing runs {sorted(expected - got)[:5]}, unexpected {sorted(got - expected)[:5]}")
    bad = {r["key"]: p for r in results if (p := record_problems(r))}
    expect(not bad, "; ".join(f"{k}: {'; '.join(p)}" for k, p in list(bad.items())[:5]))
    expect("finished: " in proc.stdout and f"{len(expected)} ok" in proc.stdout, "the sweep did not report all runs ok")
    by_model: dict = {}
    for r in results:
        by_model.setdefault(r["config"]["model"], []).append(r["test"])
    return (f"{len(results)} runs (10 models x depths {ctx.spec['grid']['depth']} x seeds {ctx.spec['grid']['seed']}) "
            f"in {wall:.0f} s, no failure record; test accuracy over the 4 runs of each model: "
            + ", ".join(f"{m} {np.mean(v):.2f}" for m, v in by_model.items()),
            {"runs": len(results), "wall_s": wall, "mean_test_by_model": {m: float(np.mean(v)) for m, v in by_model.items()}})


@check("sweep_workers", "nhdp_gnn.sweep --workers: one process per run, peak memory and logs recorded", needs=("torch",))
def check_sweep_workers(ctx: Ctx):
    if not sys.platform.startswith("linux"):
        raise Skipped("the worker pool reads /proc (Linux only)")
    t0 = time.perf_counter()
    sh(_sweep_args(ctx, "workers", "--only", "model=gcn,depth=2", "--workers", "2"))
    wall = time.perf_counter() - t0
    out = ctx.work / "workers"
    results, failures = read_results(out)
    expect(len(results) == 2 and not failures, f"{len(results)} results and {len(failures)} failure records")
    for r in results:
        expect(not record_problems(r), f"{r['key']}: {record_problems(r)}")
        expect(isinstance(r.get("peak_rss_mb"), (int, float)) and r["peak_rss_mb"] > 0, "no peak_rss_mb recorded")
        expect((out / "logs" / f"{r['key']}.log").exists(), "no log file for a run")
    peaks = [round(r["peak_rss_mb"]) for r in results]
    return f"2 runs in separate processes ({wall:.0f} s), peak resident memory {peaks} MB, one log each", {"peak_rss_mb": peaks}


@check("resume", "a second sweep invocation finds every run done and runs none", needs=("torch",))
def check_resume(ctx: Ctx):
    n = len(_grid(ctx))
    if len(list((ctx.work / "sweep").glob("*.json"))) != n:
        raise Skipped("needs the records of sweep_all_models")
    proc = sh(_sweep_args(ctx, "sweep"))
    expect(f", {n} done," in proc.stdout and ", 0 to run" in proc.stdout, "unexpected summary: " + proc.stdout.splitlines()[0])
    expect("finished: 0 ok" in proc.stdout, "the second invocation ran something")
    return f"all {n} runs found done, none repeated", {"done": n}


@check("tables", "scripts/make_tables.py builds the tables from the sweep's records", needs=("torch",))
def check_tables(ctx: Ctx):
    n = len(_grid(ctx))
    if len(list((ctx.work / "sweep").glob("*.json"))) != n:
        raise Skipped("needs the records of sweep_all_models")
    out = ctx.work / "tables"
    sh([sys.executable, "scripts/make_tables.py", "configs/smoke.toml", "--runs", ctx.work / "sweep", "--out", out])
    summary = json.loads((out / "summary.json").read_text())
    expect((summary["expected"], summary["completed"], summary["failed"], summary["missing"]) == (n, n, 0, 0), str(summary))
    readme = (out / "README.md").read_text()
    absent = [m for m in ctx.spec["grid"]["model"] if m not in readme]
    expect(not absent, f"models missing from the tables: {absent}")
    expect((out / "runs.jsonl.gz").exists() and (out / "test.csv").exists(), "tables lack runs.jsonl.gz or test.csv")
    return f"{summary['completed']} of {summary['expected']} runs in the tables, 0 failed, 0 missing", summary


@check("multi_part", "training on two parts (community partition)", needs=("torch",))
def check_multi_part(ctx: Ctx):
    from nhdp_gnn import train

    res = train.run(ctx.config(model="nhdp", depth=2, num_parts=2, partition="community", epochs=8,
                               out_dir=str(ctx.work / "multipart")), verbose=False)
    problems = record_problems(res)
    expect(not problems, "; ".join(problems))
    st = res["stages"]
    expect(st["num_parts"] == 2 and 0.0 < st["rho"] < 1.0, f"num_parts {st['num_parts']}, rho {st['rho']}")
    return (f"2 parts, cut ratio rho {st['rho']:.3f}, gated cut ratio {st['rho_gated']:.3f}, test {res['test']:.3f}",
            {"rho": st["rho"], "rho_gated": st["rho_gated"], "test": res["test"]})


@check("ddp", "two-rank DistributedDataParallel training on this machine (gloo)", needs=("torch",))
def check_ddp(ctx: Ctx):
    from nhdp_gnn.distributed import spawn_local

    out = ctx.work / "ddp"
    spawn_local(ctx.config(model="nhdp", depth=2, epochs=3, partition="community", out_dir=str(out)), world=2)
    files = sorted(out.glob("*-ddp2.json"))
    expect(len(files) == 1, f"{len(files)} DDP result files")
    res = json.loads(files[0].read_text())
    sysm = res["systems"]
    expect(res["world_size"] == 2 and 0.0 <= res["test"] <= 1.0, f"world_size {res['world_size']}, test {res['test']}")
    expect(sysm["bytes_per_step_payload"] > 0 and sysm["time_per_epoch_s"] > 0, f"systems record {sysm}")
    return (f"2 ranks, test {res['test']:.3f}, {sysm['bytes_per_step_payload']:.0f} bytes of gradients per step, "
            f"{sysm['time_per_epoch_s']:.2f} s per epoch", {"test": res["test"], **{k: sysm[k] for k in (
                "bytes_per_step_payload", "time_per_epoch_s")}})


@check("time_limit", "a run that provably cannot finish in its time limit leaves a failure record, not a result",
       needs=("torch",))
def check_time_limit(ctx: Ctx):
    from nhdp_gnn import sweep, train

    cfg = ctx.config(epochs=400, patience=300, max_train_minutes=1e-5, out_dir=str(ctx.work / "limits"))
    outcome = sweep.attempt(cfg, train.run)
    expect(outcome == "time", f"outcome {outcome!r}, expected 'time'")
    info = json.loads(sweep.failure_path(cfg).read_text())["failure"]
    expect(info["reason"] == "time" and info["epochs_run"] >= 1 and info["min_train_s"] > info["limit_s"], str(info))
    expect(not sweep.result_path(cfg).exists(), "a result was written for the stopped run")
    expect(sweep.state(cfg) == "time", f"state {sweep.state(cfg)!r}")
    return (f"stopped after {info['epochs_run']} epoch(s): needs at least {info['min_train_s']:.2f} s, limit "
            f"{info['limit_s']:.4f} s; failure recorded, no result", {k: info[k] for k in ("epochs_run", "min_train_s", "limit_s")})


@check("cora", "Cora (public split): n-HDP-GNN and GCN train a few epochs", needs=("torch",), real=True)
def check_cora(ctx: Ctx):
    from nhdp_gnn import train
    from nhdp_gnn.data import load_dataset

    try:
        data = load_dataset("cora", root=str(ROOT / "data"))
    except Exception as exc:                                # no network, or a changed download location
        raise CheckFailed(f"Cora could not be loaded: {type(exc).__name__}: {exc}") from exc
    expect(data.num_nodes == 2708 and int(data.public_split[0].sum()) == 140, "Cora is not the Planetoid graph")
    found = {}
    for model in ("nhdp", "gcn"):
        res = train.run(ctx.config(dataset="cora", synthetic={}, root=str(ROOT / "data"), regime="semi", model=model,
                                   depth=2, epochs=30, patience=30, out_dir=str(ctx.work / "cora")), verbose=False)
        problems = record_problems(res)
        expect(not problems, f"{model}: " + "; ".join(problems))
        expect((res["n_train"], res["n_val"], res["n_test"]) == (140, 500, 1000), f"{model}: split sizes")
        found[model] = res["test"]
    return ("public split 140/500/1000, test accuracy after at most 30 epochs: "
            + ", ".join(f"{k} {v:.3f}" for k, v in found.items()), found)


# ----------------------------------------------------------------------------- driver
def run_spec(spec: Spec, ctx: Ctx, level: str) -> dict:
    t0 = time.perf_counter()
    status, detail, data = "pass", "", {}
    reasons = [r for need in spec.needs if (r := missing(need))]
    try:
        if reasons:
            if level == "full":
                raise CheckFailed("required at --level full, but " + "; ".join(reasons))
            raise Skipped("; ".join(reasons))
        detail, data = spec.fn(ctx)
    except Skipped as exc:
        status, detail = "skip", str(exc)
    except CheckFailed as exc:
        status, detail = "fail", str(exc)
    except Exception:                                        # an unexpected error is a failure with its traceback
        status, detail = "fail", "unexpected error\n" + "".join(traceback.format_exc().splitlines(True)[-12:])
    return {"name": spec.name, "title": spec.title, "status": status, "seconds": time.perf_counter() - t0,
            "detail": detail, "data": data, "needs": list(spec.needs)}


def summarise(rows: list[dict], not_selected: int, level: str) -> tuple[str, bool]:
    count = {s: sum(r["status"] == s for r in rows) for s in ("pass", "fail", "skip")}
    line = f"{count['pass']} passed, {count['fail']} failed, {count['skip']} skipped"
    if not_selected:
        line += f"; {not_selected} not selected"
    if count["fail"]:
        return f"SMOKE FAILED: {line}", False
    if count["skip"]:
        return (f"SMOKE INCOMPLETE: {line}; skipped checks are neither passed nor failed "
                "(install PyTorch and PyTorch Geometric, or use --level numpy to select only the checks that need neither)"), True
    if not_selected:
        return f"SMOKE OK for the selected checks: {line}", True
    return f"SMOKE OK: {line}", True


def markdown(report: dict) -> str:
    code, env = report["code"], report["environment"]
    commit = (code.get("code_commit") or "unknown")[:12] + (" (modified)" if code.get("code_dirty") else "")
    lines = ["# Smoke check", "",
             f"`{report['command']}` at {report['generated_utc']}, code commit {commit}, "
             f"CPU {env.get('cpu_model')}, Python {env.get('python')}, torch {env['packages'].get('torch')}.", "",
             f"**{report['summary']}**", "", "| check | status | seconds | what was measured |", "|---|---|---|---|"]
    for r in report["checks"]:
        first = (r["detail"].splitlines() or [""])[0].replace("|", "/")
        lines.append(f"| `{r['name']}` | {r['status']} | {r['seconds']:.1f} | {first} |")
    for r in report["checks"]:
        rest = r["detail"].splitlines()[1:]
        if r["status"] == "fail" and rest:
            lines += ["", f"### {r['name']}", "", "```", *rest, "```"]
    lines += ["", "Floors in the checks are sanity bounds well below the values the planted graph gives; "
              "this report says nothing about accuracy on real data."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=("auto", "numpy", "full"), default="auto",
                    help="auto: run what can run and list the rest as skipped; numpy: only the checks that need no "
                         "PyTorch; full: everything, and a missing PyTorch is a failure")
    ap.add_argument("--real", action="store_true", help="also run the checks on real data (Cora; needs the network)")
    ap.add_argument("--only", default="", help="comma-separated check names")
    ap.add_argument("--out", default="runs/smoke-check")
    ap.add_argument("--list", action="store_true", help="list the checks and exit")
    args = ap.parse_args(argv)

    if args.list:
        for s in SPECS:
            tags = ", ".join([*s.needs, *(["--real"] if s.real else [])]) or "numpy only"
            print(f"{s.name:18s} [{tags}]  {s.title}")
        return 0
    names = [n for n in args.only.split(",") if n]
    unknown = sorted(set(names) - {s.name for s in SPECS})
    if unknown:
        ap.error(f"unknown checks {unknown}; --list shows them")
    pool = [s for s in SPECS if args.level != "numpy" or not s.needs]
    pool = [s for s in pool if s.name in names] if names else [s for s in pool if args.real or not s.real]
    left_out = sorted(set(names) - {s.name for s in pool})
    if left_out:
        ap.error(f"{left_out} need PyTorch and cannot be selected with --level numpy")

    out = Path(args.out)
    work = out / "work"
    if work.exists():
        shutil.rmtree(work)                                  # earlier records would make the sweep checks resume
    work.mkdir(parents=True)
    ctx = Ctx(out)
    from nhdp_gnn.utils import code_version, environment, save_json

    started = time.perf_counter()
    rows = []
    for spec in pool:
        row = run_spec(spec, ctx, args.level)
        rows.append(row)
        head, *rest = (row["detail"].splitlines() or [""])
        print(f"[{row['status']:^4}] {row['name']:18s} {row['seconds']:7.1f} s  {head}", flush=True)
        if row["status"] == "fail":
            for line in rest:
                print("           " + line, flush=True)
    summary, ok = summarise(rows, len(SPECS) - len(pool), args.level)
    report = {"command": "python scripts/smoke.py " + " ".join(argv), "level": args.level, "real": args.real,
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": summary,
              "total_s": time.perf_counter() - started, "checks": rows, "code": code_version(),
              "environment": environment(), "synthetic_graph": ctx.syn}
    save_json(out / "report.json", report)
    (out / "report.md").write_text(markdown(report))
    print(summary, flush=True)
    print(f"report: {out / 'report.md'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
