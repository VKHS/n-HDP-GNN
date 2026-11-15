import csv
import glob
import json
import tomllib

import numpy as np

from nhdp_gnn.report import aggregate, load_results, markdown_tables, scaling_table
from nhdp_gnn.sweep import expand, select


def test_every_sweep_file_expands_to_valid_unique_runs():
    for path in glob.glob("configs/*.toml"):
        spec = tomllib.load(open(path, "rb"))
        if "grid" not in spec and "base" not in spec:      # flat single-run config (HPC)
            select([spec])
            continue
        runs = expand(spec)
        cfgs = select(runs)                                 # validates every key
        assert len({c.key() for c in cfgs}) == len(cfgs), path


def test_sweep_sizes_and_sharding():
    spec = tomllib.load(open("configs/table2_full.toml", "rb"))
    runs = expand(spec)
    assert len(runs) == 4 * 10 * 6 * 10
    parts = [select(runs, shard=(i, 7)) for i in range(7)]
    assert sum(len(p) for p in parts) == len(runs)
    only = select(runs, only={"model": "nhdp", "dataset": "cora"})
    assert len(only) == 6 * 10 and all(c.model == "nhdp" for c in only)
    three = select(runs, only={"model": "nhdp", "dataset": "cora", "seed": "0|1|2"})   # alternatives
    assert len(three) == 6 * 3 and {c.seed for c in three} == {0, 1, 2}
    abl = expand(tomllib.load(open("configs/table1_ablation.toml", "rb")))
    assert len(abl) == 4 * 10 * 10 and {r["tag"] for r in abl} >= {"hdp-gnn", "full", "full-gate-off"}


def test_report_aggregates_mean_ci_and_paired_test(tmp_path):
    rng = np.random.default_rng(0)
    for model, mu in (("nhdp", 0.8), ("gcn", 0.7)):
        for depth in (2, 8):
            for seed in range(5):
                res = {"config": {"dataset": "cora", "model": model, "depth": depth, "seed": seed},
                       "test": mu + 0.01 * rng.normal(), "embedding": {"centroid_gap": 0.3}}
                (tmp_path / f"{model}{depth}{seed}.json").write_text(json.dumps(res))
    results = load_results(tmp_path)
    assert len(results) == 20
    table = aggregate(results, ["model"], ["depth"], ["dataset"], "test")
    cells = table[("cora",)]
    assert len(cells[(("nhdp",), (2,))]) == 5
    md = markdown_tables(table, ["model"], ["depth"], ["dataset"], "test", reference="nhdp")
    assert "± " in md and "Wilcoxon" in md and "(n=5)" in md


def test_scaling_table_speedup(tmp_path):
    for p, t in ((1, 8.0), (2, 4.4), (4, 2.5)):
        res = {"config": {"dataset": "ogbn-products", "model": "nhdp", "depth": 10, "seed": 0, "tag": ""},
               "world_size": p, "test": 0.8,
               "systems": {"time_per_epoch_s": t, "edges_per_s": 1e6 * p, "comms_share": 0.1,
                           "allreduce_s_per_step": 0.01, "bytes_per_step_payload": 1e6, "rho": 0.1,
                           "edge_reduction": 0.9}}
        (tmp_path / f"p{p}.json").write_text(json.dumps(res))
    md = scaling_table(load_results(tmp_path))
    assert "| 2 | 4.400 | 1.82 | 0.91 |" in md


def test_code_version_names_the_checked_out_commit():
    import subprocess
    from pathlib import Path

    import nhdp_gnn
    from nhdp_gnn.utils import code_version

    root = Path(nhdp_gnn.__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    v = code_version()
    if head is not None:
        assert v["code_commit"] == head and isinstance(v["code_dirty"], bool)


def _make_tables():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_tables.py"
    spec = importlib.util.spec_from_file_location("make_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tables_report_failed_and_missing_runs(tmp_path):
    from nhdp_gnn.report import load_failures

    runs = tmp_path / "runs"
    (tmp_path / "t.toml").write_text(f'name = "t"\nout_dir = "{runs}"\n[base]\ndataset = "cora"\nregime = "full"\n'
                                     '[grid]\nmodel = ["nhdp", "gcn"]\ndepth = [2, 4]\nseed = [0, 1, 2]\n')
    spec = tomllib.load(open(tmp_path / "t.toml", "rb"))
    runs.mkdir()
    for c in select(expand(spec)):
        rec = {"key": c.key(), "config": {"dataset": c.dataset, "model": c.model, "depth": c.depth, "seed": c.seed,
                                          "tag": ""}}
        if c.model == "gcn" and c.depth == 4:
            rec["failure"] = {"reason": "memory", "detail": "exceeded the 13 GB memory budget"}
            (runs / f"{c.key()}.failed.json").write_text(json.dumps(rec))
        elif c.model == "nhdp" and c.depth == 4 and c.seed == 2:
            rec["failure"] = {"reason": "time", "detail": "stopped after 3 epochs"}
            (runs / f"{c.key()}.failed.json").write_text(json.dumps(rec))
        elif not (c.model == "nhdp" and c.depth == 2 and c.seed == 1):        # one run is missing
            rec.update(test=0.8 + 0.01 * c.seed, embedding={"centroid_gap": 0.1})
            (runs / f"{c.key()}.json").write_text(json.dumps(rec))
    assert len(load_results(runs)) == 7 and len(load_failures(runs)) == 4

    out = tmp_path / "out"
    _make_tables().main([str(tmp_path / "t.toml"), "--runs", str(runs), "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert (summary["expected"], summary["completed"], summary["failed"], summary["missing"]) == (12, 7, 4, 1)
    assert summary["failed_by_reason"] == {"memory": 3, "time": 1}
    md = (out / "README.md").read_text()
    assert "7 completed, 4 failed (3 memory, 1 time), 1 missing" in md
    assert "| gcn | " in md and "failed: 3 memory" in md and "(n=2; failed: 1 time)" in md
    assert "## Failed runs" in md and "## Missing runs" in md and "exceeded the 13 GB memory budget" in md
    rows = list(csv.DictReader(open(out / "test.csv")))
    gcn4 = [r for r in rows if r["model"] == "gcn" and r["depth"] == "4"][0]
    assert gcn4["n"] == "0" and json.loads(gcn4["failed"]) == {"memory": 3}


def test_unpacked_results_continue_a_sweep(tmp_path):
    """A results directory unpacks into records that the sweep and make_tables treat as earlier runs."""
    import importlib.util
    from pathlib import Path

    from nhdp_gnn import sweep

    root = Path(__file__).resolve().parents[1]
    spec_ = importlib.util.spec_from_file_location("unpack_results", root / "scripts" / "unpack_results.py")
    unpack = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(unpack)

    runs = tmp_path / "runs"
    (tmp_path / "t.toml").write_text(f'name = "t"\nout_dir = "{runs}"\n[base]\ndataset = "cora"\n'
                                     '[grid]\nseed = [0, 1, 2, 3]\n')
    spec = tomllib.load(open(tmp_path / "t.toml", "rb"))
    cfgs = select(expand(spec))
    old = tmp_path / "old_runs"
    old.mkdir()
    for c in cfgs[:2]:                                    # seeds 0 and 1 done, seed 2 failed for time
        (old / f"{c.key()}.json").write_text(json.dumps({"key": c.key(), "config": {"seed": c.seed, "tag": ""},
                                                         "test": 0.5}))
    (old / f"{cfgs[2].key()}.failed.json").write_text(json.dumps(
        {"key": cfgs[2].key(), "config": {"seed": 2, "tag": ""}, "failure": {"reason": "time", "detail": "slow"}}))
    mt = _make_tables()
    mt.main([str(tmp_path / "t.toml"), "--runs", str(old), "--out", str(tmp_path / "res")])
    assert unpack.unpack(tmp_path / "res", tmp_path / "prev") == (2, 1)
    assert [c.seed for c in sweep.pending(cfgs, previous=[tmp_path / "prev"])] == [3]
    assert [c.seed for c in sweep.pending(cfgs, retry_failed=True, previous=[tmp_path / "prev"])] == [2, 3]
    # the new campaign completes seeds 2 and 3; its records take precedence over the earlier failure
    runs.mkdir()
    for c in cfgs[2:]:
        (runs / f"{c.key()}.json").write_text(json.dumps({"key": c.key(), "config": {"seed": c.seed, "tag": ""},
                                                          "test": 0.7}))
    mt.main([str(tmp_path / "t.toml"), "--runs", str(runs), "--runs", str(tmp_path / "prev"),
             "--out", str(tmp_path / "res2")])
    summary = json.loads((tmp_path / "res2" / "summary.json").read_text())
    assert (summary["completed"], summary["failed"], summary["missing"]) == (4, 0, 0)


def test_publish_merges_concurrent_pushes_into_the_results_branch(tmp_path):
    """Two jobs publish their own records at the same time; the branch ends up with both."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = Path(__file__).resolve().parents[1]

    def git(*a, cwd):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    origin = tmp_path / "origin.git"
    git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    seed_repo = tmp_path / "seed"
    git("init", "-q", str(seed_repo), cwd=tmp_path)
    for k, v in (("user.name", "t"), ("user.email", "t@example.org")):
        git("config", k, v, cwd=seed_repo)
    (seed_repo / "README").write_text("x\n")
    git("add", "README", cwd=seed_repo)
    git("commit", "-qm", "init", cwd=seed_repo)
    git("push", "-q", str(origin), "HEAD:refs/heads/main", cwd=seed_repo)

    cfg = tmp_path / "t.toml"
    cfg.write_text('name = "t"\n[base]\ndataset = "cora"\n[grid]\nseed = [0, 1, 2, 3]\n')
    cfgs = select(expand(tomllib.load(open(cfg, "rb"))))
    procs = []
    for job, seeds in (("a", (0, 1)), ("b", (2, 3))):
        clone = tmp_path / job
        git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
        for k, v in (("user.name", "bot"), ("user.email", "bot@example.org")):
            git("config", k, v, cwd=clone)
        rec = tmp_path / f"rec_{job}"
        rec.mkdir()
        for c in cfgs:
            if c.seed in seeds:
                (rec / f"{c.key()}.json").write_text(json.dumps(
                    {"key": c.key(), "config": {"dataset": "cora", "model": "nhdp", "depth": 2, "seed": c.seed,
                                                "tag": ""}, "test": 0.5 + 0.1 * c.seed}))
        procs.append(subprocess.Popen([sys.executable, str(root / "scripts" / "publish_results.py"), str(cfg),
                                       "--records", str(rec), "--backoff", "0.5"], cwd=clone,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs

    check = tmp_path / "check"
    git("clone", "-q", "-b", "results/t", str(origin), str(check), cwd=tmp_path)
    summary = json.loads((check / "results" / "t" / "summary.json").read_text())
    assert (summary["completed"], summary["missing"]) == (4, 0)
    assert int(git("rev-list", "--count", "HEAD", cwd=check)) == 2          # one commit per job, no force-push
    # publishing nothing new leaves the branch as it is
    again = subprocess.run([sys.executable, str(root / "scripts" / "publish_results.py"), str(cfg)],
                           cwd=tmp_path / "a", capture_output=True, text=True, check=True)
    assert "already up to date" in again.stdout
