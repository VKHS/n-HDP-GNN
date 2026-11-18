"""examples/output/ is the output of one run of the demo workflow: complete, from a clean checkout, and consistent.

The report, demo.json and runs.jsonl.gz are written by one run, so every table of the report must follow from the JSON
and from the records.  These tests read the committed files only; they do not run the demo.
"""
import gzip
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "output"
FIGURES = ["fig_graph.png", "fig_stage_ab.png", "fig_depth_free.png", "fig_depth_trained.png"]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_script", ROOT / "examples" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_script"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("demo_script", None)


@pytest.fixture(scope="module")
def doc():
    return json.loads((OUT / "demo.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report():
    return (OUT / "demo.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def runs():
    with gzip.open(OUT / "runs.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_the_folder_holds_what_the_demo_writes_and_nothing_else(doc):
    expected = {"demo.md", "demo.json", "runs.jsonl.gz", *FIGURES}
    if doc["training"]["failed"]:
        expected.add("failures.jsonl.gz")
    assert {p.name for p in OUT.iterdir()} == expected
    assert doc["figures"] == FIGURES
    for name in FIGURES:
        data = (OUT / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 20_000, name


def test_it_was_produced_by_the_demo_workflow_from_a_clean_checkout(doc, report):
    ci, code = doc["ci"], doc["code"]
    assert ci is not None and ci["workflow"] == "demo"
    assert re.fullmatch(r"https://github\.com/VKHS/n-HDP-GNN/actions/runs/\d+", ci["run_url"])
    assert re.fullmatch(r"[0-9a-f]{40}", code["code_commit"]) and ci["sha"] == code["code_commit"]
    assert code["code_dirty"] is False
    assert f"([run {ci['run_id']}]({ci['run_url']}))" in report.splitlines()[2]
    assert doc["command"].startswith("python examples/demo.py --train yes") and "--quick" not in doc["command"]
    assert "--out examples/output" in doc["command"]
    packages = doc["environment"]["packages"]
    assert packages["torch"] and packages["torch_geometric"]


def test_the_commit_it_names_is_in_this_repositorys_history(doc):
    sha = doc["code"]["code_commit"]
    try:
        subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("this checkout has no git history that contains the commit (a shallow clone, or no git)")


def test_the_graph_stage_a_stage_b_and_gate_tables_follow_from_the_json(doc, report):
    g, truth, trace = doc["graph"], doc["truth"], doc["stage_a_b"]["stage_b"]["elbo_trace"]
    assert f"| nodes / undirected edges | {g['nodes']} / {g['edges']} (average degree {g['average_degree']:.1f}) |" in report
    assert f"| communities | {truth['nmi_communities']:.3f} | {truth['ari_communities']:.3f} |" in report
    assert f"| super-communities | {truth['nmi_super_communities']:.3f} | {truth['ari_super_communities']:.3f} |" in report
    assert f"{len(trace)} values, from {trace[0]:.1f} to {trace[-1]:.1f}" in report
    assert doc["stage_a_b"]["stage_b"]["elbo_monotone"] is True and "(monotone)" in report
    for kind, row in doc["gate_by_edge_kind"].items():
        assert f"| {kind} | {row['edges']} | {row['mean_gate']:.3f} |" in report


def test_the_depth_study_tables_follow_from_the_json(doc, report, demo):
    study = doc["depth_study"]
    header = next(line for line in report.splitlines() if line.startswith("| operator \\ depth L |"))
    assert [int(cell) for cell in header.strip("|").split("|")[1:]] == study["depths"]
    for name, (label, *_rest) in demo.OPERATORS.items():
        op = study["operators"][name]
        accuracy = " | ".join(f"{100 * m:.1f} ± {100 * h:.1f}" for m, h in zip(op["probe_test_mean"], op["probe_test_ci95"]))
        assert f"| {label} | {accuracy} |" in report, name
        assert f"| {label} | " + " | ".join(f"{v:.3f}" for v in op["centroid_gap"]) + " |" in report, name


def test_the_training_tables_follow_from_the_json_and_the_records(doc, report, runs, demo):
    t = doc["training"]
    variants, depths, seeds = t["variants"], t["depths"], t["seeds"]
    section = report.split("## 5. Depth with training")[1]
    total = len(variants) * len(depths) * seeds
    assert f"{t['completed']} of {total} runs completed, {t['failed']} failed, {t['missing']} have no record." in section
    assert t["completed"] == len(runs) and t["completed"] + t["failed"] + t["missing"] == total
    labels = {vid: label for vid, label, _ in demo.VARIANTS}
    for key, digits in (("test", 1), ("centroid_gap", 2), ("nmi", 2)):
        for vid in variants:
            row = " | ".join(demo.cell(tuple(t["cells"][key][f"{vid}|{d}"]), digits) for d in depths)
            assert f"| {labels[vid]} | {row} |" in section, (key, vid)
    by_cell = defaultdict(list)
    for r in runs:
        assert "curves" not in r
        by_cell[(r["config"]["tag"], r["config"]["depth"])].append(r)
    for vid in variants:
        for d in depths:
            recs = sorted(by_cell[(vid, d)], key=lambda r: r["config"]["seed"])
            assert [r["config"]["seed"] for r in recs] == sorted({r["config"]["seed"] for r in recs})     # one run per seed
            cell = f"{vid}|{d}"
            assert t["seed_values"]["test"][cell] == pytest.approx([100.0 * r["test"] for r in recs])
            assert t["seed_values"]["centroid_gap"][cell] == pytest.approx([r["embedding"]["centroid_gap"] for r in recs])
            mean, _half_width, n = t["cells"]["test"][cell]
            assert n == len(recs) and mean == pytest.approx(float(np.mean(t["seed_values"]["test"][cell])))


def test_every_failure_is_listed_not_replaced(doc, report):
    t = doc["training"]
    if t["failed"] == 0:
        assert "Failed runs (not replaced by a value):" not in report
        return
    with gzip.open(OUT / "failures.jsonl.gz", "rt", encoding="utf-8") as fh:
        failures = [json.loads(line) for line in fh]
    assert len(failures) == t["failed"] and "Failed runs (not replaced by a value):" in report
