"""Artifact generators on a planted hierarchical graph (numpy only)."""
import json

import numpy as np

from nhdp_gnn.artifacts import gate_summary, main, operator_depth_study, partition_study, stage_ab_report
from nhdp_gnn.config import RunConfig
from nhdp_gnn.data import load_dataset

SYN = dict(num_super=3, comm_per_super=3, nodes_per_comm=30, seed=0)


def test_stage_ab_report_gate_supports_and_depth_limits():
    data = load_dataset("synthetic", synthetic=SYN)
    report, st = stage_ab_report(data, RunConfig(seed=0).pipeline())
    sb = report["stage_b"]                                    # the ELBO is evaluated from the 2nd iteration on
    assert sb["elbo_monotone"] and len(sb["elbo_trace"]) == sb["iterations"] - 1
    hard = report["gates"]["louvain"]                         # 0/1 gate: zero shares are exact complements
    assert 0.0 <= hard["zero_share_same"] <= 1.0 and 0.0 <= hard["gated_homophily"] <= 1.0
    base = report["depth_limit"]["no_gate"]
    for name in ("nhdp_topk", "nhdp_hard", "louvain"):          # gate supports are subsets of the edges
        lim = report["depth_limit"][name]
        assert lim["components"] >= base["components"]
        assert lim["purity_bound"] >= base["purity_bound"] - 1e-12   # finer partition, purity cannot drop
    assert report["edges"] * 2 == data.edge_index.shape[1]


def test_gate_summary_on_known_values():
    s = gate_summary(np.array([1.0, 0.5, 0.0, 0.0]), np.array([True, True, False, False]))
    assert s["mean_same"] == 0.75 and s["mean_cross"] == 0.0 and s["gated_homophily"] == 1.0
    assert s["zero_share_cross"] == 1.0 and s["kept_mass"] == 0.375


def test_operator_depth_study_and_partition_study():
    data = load_dataset("synthetic", synthetic=SYN)
    cfg = RunConfig(seed=0).pipeline()
    _, st = stage_ab_report(data, cfg)
    study = operator_depth_study(data, st, depths=(0, 2), probe_seeds=(0, 1), madgap_sources=64)
    assert set(study["operators"]) == {"A_hat", "gated_nhdp_topk", "gated_nhdp_hard", "gated_louvain"}
    for rec in study["operators"].values():
        assert len(rec["probe_test_mean"]) == 2 and all(0.0 <= a <= 1.0 for a in rec["probe_test_mean"])
        assert rec["mean_row_norm"][1] <= rec["mean_row_norm"][0] + 1e-9    # non-expansive operators
    a0 = study["operators"]["A_hat"]["probe_test_mean"][0]
    assert all(rec["probe_test_mean"][0] == a0 for rec in study["operators"].values())   # depth 0: same features
    rows = partition_study(data, cfg, parts=(3,), methods=("community", "random"))
    assert rows[0]["rho"] < rows[1]["rho"]                   # planted communities: community cut beats random


def test_cli_writes_reports_and_manifest(tmp_path):
    main(["--datasets", "synthetic", "--out", str(tmp_path), "--depths", "0", "1", "--probe-seeds", "0",
          "--parts", "2", "--methods", "community"])
    for rel in ("stage_ab/synthetic.json", "operator_depth/synthetic.json", "stage_ab/partition_study.csv"):
        assert (tmp_path / rel).exists(), rel
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["command"].startswith("python -m nhdp_gnn.artifacts") and manifest["cpu_count"] >= 1
