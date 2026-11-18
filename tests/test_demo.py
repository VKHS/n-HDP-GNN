"""examples/demo.py without PyTorch.

The numpy part (graph, stages A and B, the training-free depth study, the report and the figures) is run for real on
the small graph of ``--quick``.  The training section is run with a stand-in for ``nhdp_gnn.train.run`` that writes
records with the schema of the real one, so the aggregation, the tables, the listing of failed runs and the output
files are checked here; what the real training produces is checked by the ``demo`` workflow.
"""
import gzip
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
FIGURES = ["fig_graph.png", "fig_stage_ab.png", "fig_depth_free.png"]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_example", ROOT / "examples" / "demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_example"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("demo_example", None)


@pytest.fixture(scope="module")
def quick(demo, tmp_path_factory):
    root = tmp_path_factory.mktemp("quick")
    assert demo.main(["--quick", "--train", "no", "--out", str(root / "out"), "--work", str(root / "work")]) == 0
    return root / "out"


def load(out):
    return (out / "demo.md").read_text(), json.loads((out / "demo.json").read_text())


# ----------------------------------------------------------------------------- the numpy part, for real
def test_quick_run_writes_the_report_the_json_and_the_figures(quick):
    report, doc = load(quick)
    assert doc["figures"] == (FIGURES if HAS_MPL else [])
    for name in doc["figures"]:
        assert (quick / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert f"]({name})" in report                                    # the report embeds the figures that exist
    assert not list(quick.glob("*.png")) or HAS_MPL
    assert not re.search(r"\bnan\b", report, re.I) and "None" not in report
    assert report.startswith("# n-HDP-GNN demo") and "## Provenance" in report
    assert "Not generated" in report.split("## 5. Depth with training")[1].split("## Provenance")[0]


def test_every_number_in_the_report_is_in_the_json(quick):
    report, doc = load(quick)
    facts, truth = doc["graph"], doc["truth"]
    assert f"{facts['nodes']} / {facts['edges']} (average degree {facts['average_degree']:.1f})" in report
    assert f"| communities | {truth['nmi_communities']:.3f} | {truth['ari_communities']:.3f} |" in report
    for kind, v in doc["gate_by_edge_kind"].items():
        assert f"| {kind} | {v['edges']} | {v['mean_gate']:.3f} |" in report
    study = doc["depth_study"]
    assert study["depths"] == [0, 2, 8, 32]
    for row in study["operators"].values():
        assert " | ".join(f"{v:.3f}" for v in row["centroid_gap"]) in report
        assert len(row["probe_test_mean"]) == len(row["centroid_gap"]) == len(study["depths"])
    assert "not row-normalised" in study["features"]                      # the synthetic features are not the real ones'


def test_stage_b_text_matches_the_recorded_trace(quick):
    report, doc = load(quick)
    sb = doc["stage_a_b"]["stage_b"]
    assert sb["elbo_monotone"] is True and np.all(np.diff(sb["elbo_trace"]) >= 0)
    assert len(sb["elbo_trace"]) == sb["iterations"] - 1                  # evaluated after every pass but the last
    assert f"ran {sb['iterations']} passes" in report and f"{len(sb['elbo_trace'])} values" in report
    assert "(monotone)" in report and "NOT monotone" not in report


def test_the_same_inputs_give_the_same_numbers(demo, quick, tmp_path):
    assert demo.main(["--quick", "--train", "no", "--no-figures", "--out", str(tmp_path / "again"),
                      "--work", str(tmp_path / "work")]) == 0
    _, first = load(quick)
    _, second = load(tmp_path / "again")
    assert first["truth"] == second["truth"] and first["gate_by_edge_kind"] == second["gate_by_edge_kind"]
    for name in first["depth_study"]["operators"]:
        for key in ("probe_test_mean", "centroid_gap"):
            assert first["depth_study"]["operators"][name][key] == second["depth_study"]["operators"][name][key]
    assert second["figures"] == []


def test_the_resolution_check_describes_the_probe_it_ran(demo, tmp_path):
    from nhdp_gnn.data import load_dataset

    data = load_dataset("synthetic", synthetic=dict(demo.QUICK_GRAPH))
    rc = demo.probe_resolution(data, 8, shrink=1e3, seeds=(0, 1))
    assert rc["depth"] == 8 and rc["label_draws"] == 2 and rc["shrink"] == 1000.0
    assert 0.0 <= rc["probe_test_shrunk"] <= rc["probe_test"] + 1e-9 <= 1.0 + 1e-9 or rc["probe_test_shrunk"] >= 0.0
    lo, hi = rc["single_class_accuracy"]
    assert 0.0 < lo <= hi < 1.0 and rc["centroid_cosine_distance"] > 0.0


def test_links_to_the_repository_are_relative_to_the_report(demo, quick):
    assert demo.repo_link("results", "results/README.md", ROOT / "examples" / "output") == "[results](../../results/README.md)"
    assert demo.repo_link("results", "results/README.md", ROOT / "runs" / "demo" / "output") == "[results](../../../results/README.md)"
    assert demo.repo_link("results", "results/README.md", quick) == "`results/README.md`"      # outside the repository
    assert "](../" not in load(quick)[0] and "`results/README.md`" in load(quick)[0]


def test_a_local_run_does_not_write_where_the_committed_output_lives():
    source = (ROOT / "examples" / "demo.py").read_text()
    assert 'default=str(ROOT / "runs" / "demo" / "output")' in source and 'default=str(ROOT / "examples"' not in source


def test_no_figures_and_the_help_text(demo, tmp_path):
    assert demo.main(["--quick", "--train", "no", "--no-figures", "--out", str(tmp_path / "o"),
                      "--work", str(tmp_path / "w")]) == 0
    assert not list((tmp_path / "o").glob("*.png"))
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "demo.py"), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "--quick" in proc.stdout and "--train" in proc.stdout


@pytest.mark.skipif(HAS_TORCH, reason="checks what happens without PyTorch")
def test_without_torch_auto_says_what_is_missing_and_yes_is_an_error(demo, tmp_path, capsys):
    assert demo.main(["--quick", "--no-figures", "--out", str(tmp_path / "auto"), "--work", str(tmp_path / "w")]) == 0
    section = (tmp_path / "auto" / "demo.md").read_text().split("## 5. Depth with training")[1]
    assert "not installed" in section and "Not generated" in section
    assert demo.main(["--quick", "--train", "yes", "--out", str(tmp_path / "yes"), "--work", str(tmp_path / "w")]) == 2
    assert "needs PyTorch" in capsys.readouterr().err
    assert not (tmp_path / "yes" / "demo.md").exists()


# ----------------------------------------------------------------------------- the training part, with a stand-in
RUNNER = textwrap.dedent('''
    import json, os
    from dataclasses import asdict
    from pathlib import Path

    BASE = {"nhdp": 0.70, "nhdp_gate_off": 0.66, "gcn": 0.62, "gat": 0.60, "appnp": 0.64, "gcnii": 0.65}

    def run(cfg, verbose=True):
        if cfg.tag == os.environ.get("FAKE_FAIL_TAG"):
            raise MemoryError("stand-in allocation failure")
        if (cfg.tag, cfg.depth, cfg.seed) == ("gat", 8, 1):
            raise MemoryError("stand-in allocation failure")
        test = BASE[cfg.tag] - 0.01 * cfg.depth + 0.01 * cfg.seed
        rec = {"key": cfg.key(), "config": asdict(cfg), "metric": "acc", "val": test, "test": test,
               "best_epoch": 5 + cfg.seed, "epochs_run": 30, "propagation_depth": cfg.depth, "n_params": 1,
               "n_train": 80, "n_val": 50, "n_test": 50,
               "timings": {"stages_s": 0.1, "train_s": 0.5, "epoch_s_mean": 0.02}, "stages": {},
               "environment": {"python": "3.12", "cpu_model": "Stand-in CPU"}, "code_commit": "abc",
               "curves": {"loss": [1.0, 0.5]},
               "embedding": {"representation": "logits", "centroid_gap": BASE[cfg.tag] - 0.5, "madgap": 0.1,
                             "nmi": BASE[cfg.tag] / 2, "ari": 0.1}}
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{cfg.key()}.json").write_text(json.dumps(rec))
        return rec
''')


@pytest.fixture
def trained(demo, tmp_path, monkeypatch):
    """The whole demo with the training replaced by the stand-in; returns (out dir, the report builder's input)."""
    runner_dir = tmp_path / "runner"
    runner_dir.mkdir()
    (runner_dir / "demo_fake_runner.py").write_text(RUNNER)
    monkeypatch.syspath_prepend(str(runner_dir))
    monkeypatch.setattr(demo, "have_torch", lambda: None)
    real_training, real_report, seen = demo.run_training, demo.build_report, {}
    monkeypatch.setattr(demo, "run_training", lambda cfgs, jobs, runner=None: real_training(cfgs, jobs, "demo_fake_runner:run"))
    monkeypatch.setattr(demo, "build_report", lambda ctx: (seen.update(ctx), real_report(ctx))[1])

    def run(*extra):
        out = tmp_path / "out"
        assert demo.main(["--quick", "--train", "yes", "--out", str(out), "--work", str(tmp_path / "work"), *extra]) == 0
        return out, seen, real_report

    yield run
    sys.modules.pop("demo_fake_runner", None)


def interval(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def test_training_section_aggregates_the_records_and_lists_the_failure(trained):
    out, ctx, _ = trained()
    report, doc = load(out)
    section = report.split("## 5. Depth with training")[1]
    # 6 variants x 2 depths x 3 seeds; the stand-in fails one run with an allocation error
    assert "35 of 36 runs completed, 1 failed, 0 have no record." in section
    assert "Failed runs (not replaced by a value):" in section and "| gat | 8 | 1 | memory |" in section
    m, h = interval([100 * (0.70 - 0.02 + 0.01 * s) for s in range(3)])
    assert f"| n-HDP-GNN | {m:.1f} ± {h:.1f} (n=3) |" in section                # accuracy of nhdp at depth 2
    gat_depth_8 = interval([100 * (0.60 - 0.08 + 0.01 * s) for s in (0, 2)])    # the failed seed is not imputed
    assert f"{gat_depth_8[0]:.1f} ± {gat_depth_8[1]:.1f} (n=2)" in section
    assert "Stand-in CPU" not in section                                          # one CPU model: no warning
    assert (out / "demo.json").exists() and doc["training"]["completed"] == 35 and doc["training"]["failed"] == 1
    runs = [json.loads(l) for l in gzip.open(out / "runs.jsonl.gz", "rt")]
    assert len(runs) == 35 and all("curves" not in r for r in runs)             # the record of every run, without curves
    failures = [json.loads(l) for l in gzip.open(out / "failures.jsonl.gz", "rt")]
    assert len(failures) == 1 and failures[0]["failure"]["reason"] == "memory"
    assert doc["figures"] == (FIGURES + ["fig_depth_trained.png"] if HAS_MPL else [])


def test_a_variant_with_no_result_is_reported_as_such(trained, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_TAG", "appnp")
    out, _, _ = trained()
    section = load(out)[0].split("## 5. Depth with training")[1]
    assert "| APPNP | no result | no result |" in section and "| APPNP | no result | no result |" in section.split("Median best epoch")[1]
    assert "29 of 36 runs completed, 7 failed, 0 have no record." in section       # 6 APPNP runs + the gat run


def test_runs_without_a_record_are_listed_not_filled_in(trained, demo):
    out, ctx, real_report = trained()
    summary = ctx["trained"]
    gone = ctx["trained"]["results"][-1]["config"]
    kept = [r for r in summary["results"] if r["config"] != gone]
    from nhdp_gnn.config import config_from_dict

    missing_cfg = config_from_dict(gone)
    partial = dict(summary, results=kept, missing=[missing_cfg],
                   test=demo.aggregate(kept, summary["depths"], lambda r: 100.0 * r["test"]),
                   centroid_gap=demo.aggregate(kept, summary["depths"], lambda r: r["embedding"]["centroid_gap"]),
                   nmi=demo.aggregate(kept, summary["depths"], lambda r: r["embedding"]["nmi"]),
                   epochs=demo.median_best_epoch(kept, summary["depths"]))
    text = real_report(dict(ctx, trained=partial))
    assert f"34 of 36 runs completed, 1 failed, 1 have no record." in text
    assert f"Runs without a record: {missing_cfg.tag} depth {missing_cfg.depth} seed {missing_cfg.seed}" in text
