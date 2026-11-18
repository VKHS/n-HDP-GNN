"""scripts/smoke.py and configs/smoke.toml (no torch needed).

The checks that train are covered by running the script itself (the ``smoke`` workflow); here a stand-in
runner, written to a temporary module, produces records with the schema of ``train.run`` so that the
logic of the sweep-level checks -- what they accept and what they reject -- is tested without PyTorch.
"""
import importlib.util
import json
import os
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from nhdp_gnn.sweep import expand, select

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


@pytest.fixture(scope="module")
def smoke():
    path = ROOT / "scripts" / "smoke.py"
    spec = importlib.util.spec_from_file_location("smoke", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("smoke", None)


RUNNER = textwrap.dedent('''
    import json, os
    from dataclasses import asdict
    from pathlib import Path

    def run(cfg, verbose=True):
        if os.environ.get("FAKE_FAIL_MODEL") == cfg.model:
            raise ValueError("boom")
        rec = {"key": cfg.key(), "config": asdict(cfg), "metric": "acc", "val": 0.9,
               "test": 1.7 if os.environ.get("FAKE_BAD_MODEL") == cfg.model else 0.8,
               "best_epoch": 3, "epochs_run": 8, "propagation_depth": cfg.depth, "n_params": 123,
               "n_train": 10, "n_val": 5, "n_test": 5,
               "timings": {"stages_s": 0.1, "train_s": 0.5, "epoch_s_mean": 0.05}, "stages": {},
               "environment": {"python": "3.12"}, "code_commit": "abc123",
               "embedding": {"representation": "logits", "centroid_gap": 0.5, "madgap": 0.1, "nmi": 0.4, "ari": 0.3}}
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{cfg.key()}.json").write_text(json.dumps(rec))
        return rec
''')


@pytest.fixture
def ctx(smoke, tmp_path, monkeypatch):
    (tmp_path / "fake_runner.py").write_text(RUNNER)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    c = smoke.Ctx(tmp_path / "out")
    c.work.mkdir(parents=True)
    c.runner = "fake_runner:run"
    return c


def valid_record(smoke, ctx):
    """A record as ``train.run`` writes it, made by the stand-in runner."""
    from nhdp_gnn.sweep import attempt

    cfg = ctx.config(out_dir=str(ctx.work / "one"))
    sys.path.insert(0, str(Path(ctx.work).parent.parent))
    try:
        import fake_runner

        attempt(cfg, fake_runner.run)
    finally:
        sys.path.pop(0)
        sys.modules.pop("fake_runner", None)
    return json.loads((ctx.work / "one" / f"{cfg.key()}.json").read_text())


# ----------------------------------------------------------------------------- the sweep file
def test_smoke_sweep_covers_every_model_and_has_distinct_runs():
    spec = tomllib.loads((ROOT / "configs" / "smoke.toml").read_text())
    cfgs = select(expand(spec), None, (0, 1), None)
    assert len(cfgs) == len({c.key() for c in cfgs}) == 40
    assert {c.model for c in cfgs} == {"nhdp", "gcn", "gat", "sgc", "appnp", "arma", "gcnii", "pairnorm",
                                       "dropedge", "graphunet"}
    assert all(c.dataset == "synthetic" and c.synthetic for c in cfgs)         # no download
    assert all(c.epochs <= 20 and c.hidden % c.heads == 0 for c in cfgs)      # small and valid


# ----------------------------------------------------------------------------- record validation
def test_record_problems_accepts_a_record_and_names_what_is_wrong(smoke, ctx):
    rec = valid_record(smoke, ctx)
    assert smoke.record_problems(rec) == []

    def broken(**changes):
        out = json.loads(json.dumps(rec))
        for k, v in changes.items():
            out[k] = v
        return smoke.record_problems(out)

    assert "missing test" in smoke.record_problems({k: v for k, v in rec.items() if k != "test"})[0]
    assert any("not a metric" in p for p in broken(test=1.2))
    assert any("not a metric" in p for p in broken(val=float("nan")))
    assert any("inconsistent" in p for p in broken(best_epoch=0))
    assert any("inconsistent" in p for p in broken(epochs_run=rec["config"]["epochs"] + 1))
    assert any("key does not match" in p for p in broken(key="0123456789ab"))
    assert any("split is empty" in p for p in broken(n_test=0))
    assert any("not finite" in p for p in broken(embedding={**rec["embedding"], "nmi": float("inf")}))
    assert any("not on the logits" in p for p in broken(embedding={**rec["embedding"], "representation": "h"}))
    assert any("embedding diagnostics are missing" in p for p in broken(embedding=None))
    assert any("timings" in p for p in broken(timings={"train_s": 0.0, "epoch_s_mean": 0.1}))


# ----------------------------------------------------------------------------- sweep-level checks
def test_sweep_checks_pass_on_valid_records(smoke, ctx):
    detail, data = smoke.check_sweep_all_models(ctx)
    assert data["runs"] == 40 and "40 runs" in detail
    assert smoke.check_resume(ctx)[1] == {"done": 40}
    _, summary = smoke.check_tables(ctx)
    assert (summary["expected"], summary["completed"], summary["failed"], summary["missing"]) == (40, 40, 0, 0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the worker pool reads /proc")
def test_worker_check_requires_peak_memory_and_logs(smoke, ctx):
    detail, data = smoke.check_sweep_workers(ctx)
    assert len(data["peak_rss_mb"]) == 2 and all(p > 0 for p in data["peak_rss_mb"])


def test_sweep_check_fails_when_a_model_raises(smoke, ctx, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_MODEL", "arma")
    with pytest.raises(smoke.CheckFailed):
        smoke.check_sweep_all_models(ctx)


def test_sweep_check_fails_on_an_invalid_record(smoke, ctx, monkeypatch):
    monkeypatch.setenv("FAKE_BAD_MODEL", "gat")
    with pytest.raises(smoke.CheckFailed, match="not a metric"):
        smoke.check_sweep_all_models(ctx)


def test_sweep_check_fails_when_runs_are_missing(smoke, ctx):
    smoke.check_sweep_all_models(ctx)
    victim = sorted((ctx.work / "sweep").glob("*.json"))[0]
    victim.unlink()
    assert len(list((ctx.work / "sweep").glob("*.json"))) == 39
    # a record that disappeared is not a pass for the checks that rely on the sweep's records
    with pytest.raises(smoke.Skipped):
        smoke.check_resume(ctx)
    with pytest.raises(smoke.Skipped):
        smoke.check_tables(ctx)


# ----------------------------------------------------------------------------- the command
def test_numpy_level_passes_and_writes_the_report(smoke, tmp_path, capsys):
    assert smoke.main(["--level", "numpy", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    names = [c["name"] for c in report["checks"]]
    assert names == ["graph", "stage_a", "stage_b", "gate", "partition", "depth_study", "artifacts_cli"]
    assert [c["status"] for c in report["checks"]] == ["pass"] * 7
    assert report["summary"].startswith("SMOKE OK for the selected checks: 7 passed, 0 failed, 0 skipped; 10 not")
    assert report["environment"]["python"] and "code_commit" in report["code"]
    text = (tmp_path / "report.md").read_text()
    assert "| `gate` | pass |" in text and "sanity bounds" in text
    assert "SMOKE OK" in capsys.readouterr().out


@pytest.mark.skipif(HAS_TORCH, reason="checks what happens without PyTorch")
def test_without_torch_skips_are_reported_and_full_level_fails(smoke, tmp_path):
    assert smoke.main(["--out", str(tmp_path / "auto"), "--only", "graph,train_cli,ddp"]) == 0
    report = json.loads((tmp_path / "auto" / "report.json").read_text())
    assert [c["status"] for c in report["checks"]] == ["pass", "skip", "skip"]
    assert report["summary"].startswith("SMOKE INCOMPLETE: 1 passed, 0 failed, 2 skipped")
    assert "torch" in report["checks"][1]["detail"]
    # a skipped check is not a pass: at --level full a missing PyTorch fails
    assert smoke.main(["--level", "full", "--out", str(tmp_path / "full"), "--only", "graph,train_cli"]) == 1
    report = json.loads((tmp_path / "full" / "report.json").read_text())
    assert [c["status"] for c in report["checks"]] == ["pass", "fail"]
    assert report["summary"].startswith("SMOKE FAILED")


def test_a_failing_check_gives_exit_status_one_with_its_message(smoke, tmp_path, monkeypatch, capsys):
    def broken(ctx):
        smoke.expect(False, "expectation not met\nsecond line")

    specs = list(smoke.SPECS) + [smoke.Spec("broken", "always fails", (), False, broken)]
    monkeypatch.setattr(smoke, "SPECS", specs)
    assert smoke.main(["--only", "graph,broken", "--out", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "[fail] broken" in out and "second line" in out and "SMOKE FAILED: 1 passed, 1 failed" in out
    assert "### broken" in (tmp_path / "report.md").read_text()


def test_unexpected_errors_are_failures_not_crashes(smoke, tmp_path, monkeypatch):
    def crashes(ctx):
        raise KeyError("nope")

    monkeypatch.setattr(smoke, "SPECS", [smoke.Spec("crash", "raises", (), False, crashes)])
    assert smoke.main(["--out", str(tmp_path)]) == 1
    row = json.loads((tmp_path / "report.json").read_text())["checks"][0]
    assert row["status"] == "fail" and "KeyError" in row["detail"]


def test_list_and_argument_errors(smoke, capsys):
    assert smoke.main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert all(s.name in listing for s in smoke.SPECS) and "[torch, --real]" in listing
    with pytest.raises(SystemExit):
        smoke.main(["--only", "nonexistent"])
    with pytest.raises(SystemExit):
        smoke.main(["--level", "numpy", "--only", "ddp"])


def test_an_earlier_work_directory_is_cleared(smoke, tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "stale.json").write_text("{}")
    assert smoke.main(["--level", "numpy", "--only", "graph", "--out", str(tmp_path)]) == 0
    assert not (tmp_path / "work" / "stale.json").exists()
