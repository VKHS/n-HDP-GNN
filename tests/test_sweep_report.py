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
