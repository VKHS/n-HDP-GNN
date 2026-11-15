"""Worker pool of nhdp_gnn.sweep: one process per run, memory budget, failure records (no torch needed).

A stand-in runner (written to a temporary module) allocates memory, sleeps, raises
or kills itself as its configuration's ``synthetic`` dict asks.
"""
import json
import os
import sys
import textwrap

import pytest

from nhdp_gnn import sweep
from nhdp_gnn.config import RunConfig

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the memory watch reads /proc")

RUNNER = textwrap.dedent('''
    import json, os, signal, time
    from pathlib import Path
    import numpy as np

    def run(cfg):
        p = cfg.synthetic
        hold = np.ones(int(p.get("alloc_mb", 0) * 2**20) // 8)          # touched pages count as resident
        time.sleep(p.get("sleep", 0.0))
        mode = p.get("mode", "ok")
        if mode == "raise":
            raise ValueError("boom")
        if mode == "nomem":
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 99 bytes")
        if mode == "sigkill":
            os.kill(os.getpid(), signal.SIGKILL)
        if mode == "slow":
            from nhdp_gnn.utils import TimeLimitExceeded
            raise TimeLimitExceeded("needs 900 min", epochs_run=3, min_train_s=54000.0)
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{cfg.key()}.json").write_text(json.dumps({"key": cfg.key(), "config": {"seed": cfg.seed},
                                                          "test": 1.0, "held": int(hold.size)}))
''')


@pytest.fixture
def runner(tmp_path, monkeypatch):
    (tmp_path / "fake_runner.py").write_text(RUNNER)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    return "fake_runner:run"


def cfgs(tmp_path, *params):
    return [RunConfig(dataset="synthetic", seed=i, out_dir=str(tmp_path / "runs"), synthetic=p)
            for i, p in enumerate(params)]


def record(cfg):
    return json.loads(sweep.failure_path(cfg).read_text())


def test_runs_in_workers_and_records_their_peak_memory(tmp_path, runner):
    todo = cfgs(tmp_path, {"alloc_mb": 50}, {}, {}, {})
    counts = sweep.run_pool(todo, workers=2, runner=runner, poll=0.1)
    assert counts["ok"] == 4
    for c in todo:
        res = json.loads(sweep.result_path(c).read_text())
        assert res["peak_rss_mb"] > 0
        assert (tmp_path / "runs" / "logs" / f"{c.key()}.log").exists()
    assert json.loads(sweep.result_path(todo[0]).read_text())["peak_rss_mb"] > 50


def test_run_over_the_shared_budget_is_repeated_alone(tmp_path, runner):
    # two runs of ~0.45 GB each exceed 0.8 GB together; alone, each fits
    todo = cfgs(tmp_path, {"alloc_mb": 300, "sleep": 3}, {"alloc_mb": 300, "sleep": 3})
    counts = sweep.run_pool(todo, workers=2, memory_gb=0.8, runner=runner, poll=0.1)
    assert counts["ok"] == 2 and counts["repeated alone"] == 1 and counts["memory"] == 0
    assert all(sweep.result_path(c).exists() and not sweep.failure_path(c).exists() for c in todo)


def test_run_over_the_budget_alone_fails_with_reason_memory(tmp_path, runner):
    todo = cfgs(tmp_path, {"alloc_mb": 900, "sleep": 3}, {})
    counts = sweep.run_pool(todo, workers=2, memory_gb=0.8, runner=runner, poll=0.1)
    assert counts["memory"] == 1 and counts["ok"] == 1
    rec = record(todo[0])
    assert rec["failure"]["reason"] == "memory" and rec["failure"]["peak_mb"] > 800
    assert rec["key"] == todo[0].key() and rec["config"]["seed"] == 0


@pytest.mark.parametrize("mode,reason,needle", [
    ("raise", "error", "ValueError: boom"),
    ("nomem", "memory", "can't allocate memory"),
    ("sigkill", "memory", "SIGKILL"),
    ("slow", "time", "needs 900 min"),
])
def test_failures_are_recorded_with_their_reason(tmp_path, runner, mode, reason, needle):
    todo = cfgs(tmp_path, {"mode": mode}, {})
    counts = sweep.run_pool(todo, workers=2, runner=runner, poll=0.1)
    rec = record(todo[0])
    assert rec["failure"]["reason"] == reason and needle in rec["failure"]["detail"]
    assert counts[reason] == 1 and counts["ok"] == 1
    # memory failures are confirmed by a second attempt on its own
    assert counts["repeated alone"] == (1 if reason == "memory" else 0)
    if mode == "slow":
        assert rec["failure"]["epochs_run"] == 3


def test_wall_clock_limit_and_deadline(tmp_path, runner):
    todo = cfgs(tmp_path, {"sleep": 60})
    counts = sweep.run_pool(todo, workers=1, kill_minutes=0.03, runner=runner, poll=0.1)
    assert counts["time"] == 1 and record(todo[0])["failure"]["reason"] == "time"

    todo = cfgs(tmp_path / "b", {"sleep": 60}, {"sleep": 60}, {"sleep": 60})
    counts = sweep.run_pool(todo, workers=2, deadline_minutes=0.03, runner=runner, poll=0.1)
    assert counts["stopped"] == 2 and counts["ok"] == 0
    assert not any(sweep.failure_path(c).exists() or sweep.result_path(c).exists() for c in todo)


def test_in_process_mode_records_failures_too(tmp_path, runner, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    fn = sweep._resolve(runner)
    ok, bad, slow = cfgs(tmp_path, {}, {"mode": "raise"}, {"mode": "slow"})
    assert sweep.attempt(ok, fn) == "ok" and sweep.attempt(bad, fn) == "error" and sweep.attempt(slow, fn) == "time"
    assert record(bad)["failure"]["reason"] == "error" and record(slow)["failure"]["min_train_s"] == 54000.0


def test_command_line_skips_resource_failures_unless_asked(tmp_path, runner, capsys):
    out = tmp_path / "runs"
    (tmp_path / "s.toml").write_text(textwrap.dedent(f'''
        name = "s"
        out_dir = "{out}"
        [base]
        dataset = "synthetic"
        [grid]
        seed = [0, 1]
        [[variant]]
        name = "ok"
        synthetic = {{mode = "ok"}}
        [[variant]]
        name = "slow"
        synthetic = {{mode = "slow"}}
        [[variant]]
        name = "raise"
        synthetic = {{mode = "raise"}}
    '''))
    argv = [str(tmp_path / "s.toml"), "--workers", "2", "--runner", runner, "--poll", "0.1"]
    with pytest.raises(SystemExit) as e:                # an error failure makes the exit status 1
        sweep.main(argv)
    assert e.value.code == 1
    spec = __import__("tomllib").loads((tmp_path / "s.toml").read_text())
    all_cfgs = sweep.select(sweep.expand(spec))
    by_tag = {t: [c for c in all_cfgs if c.tag == t] for t in ("ok", "slow", "raise")}
    assert all(sweep.result_path(c).exists() for c in by_tag["ok"])
    assert all(sweep.failure_reason(c) == "time" for c in by_tag["slow"])
    assert all(sweep.failure_reason(c) == "error" for c in by_tag["raise"])
    # a second invocation repeats the error failures only
    assert {c.tag for c in sweep.pending(all_cfgs)} == {"raise"}
    assert {c.tag for c in sweep.pending(all_cfgs, retry_failed=True)} == {"slow", "raise"}
    capsys.readouterr()
    sweep.main([str(tmp_path / "s.toml"), "--dry-run"])
    listing = capsys.readouterr().out
    assert "2 done, 4 failed before, 2 to run" in listing and listing.count(" raise") == 2
    sweep.main([str(tmp_path / "s.toml"), "--count", "--only", "seed=1"])
    assert capsys.readouterr().out.strip() == "3"


def test_run_minutes_sets_the_training_limit_outside_the_key(tmp_path):
    base = RunConfig(out_dir=str(tmp_path))
    limited = RunConfig(out_dir=str(tmp_path), max_train_minutes=150)
    assert base.key() == limited.key()


def test_cannot_finish_is_a_lower_bound():
    from nhdp_gnn.utils import cannot_finish

    # 10 epochs in 100 s, fastest 9 s; best at epoch 8, patience 100: at least 98 more epochs, >= 982 s
    assert not cannot_finish(100, 9, 10, 8, 100, 500, limit_s=1000)
    assert cannot_finish(100, 9, 10, 8, 100, 500, limit_s=900)
    assert not cannot_finish(100, 9, 10, 8, 100, 12, limit_s=130)     # only 2 epochs left before max_epochs
    assert not cannot_finish(10 ** 6, 9, 10, 8, 100, 500, limit_s=0)  # no limit
