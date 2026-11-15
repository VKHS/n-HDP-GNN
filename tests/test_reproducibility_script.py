"""scripts/check_reproducibility.py: command line, environments, fingerprints and the summary (no torch needed)."""
import argparse
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import pytest

from nhdp_gnn.config import parse_args


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_reproducibility.py"
    spec = importlib.util.spec_from_file_location("check_reproducibility", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(variant, seed, repeat, test, trace="a", best=10):
    return {"variant": variant, "seed": seed, "repeat": repeat, "val": 0.8, "test": test, "best_epoch": best,
            "epochs_run": best + 100, "trace_sha1": trace}


def test_training_command_is_a_valid_run_configuration(tmp_path):
    mod = _script()
    args = argparse.Namespace(dataset="cora", model="nhdp", depth=2, regime="full", root="data",
                              set=["epochs=7", "hidden=32"])
    cmd = mod.train_command(args, 3, tmp_path / "out", tmp_path / "cache")
    cfg = parse_args(cmd[3:])                                   # after "python -m nhdp_gnn.train"
    assert (cfg.dataset, cfg.model, cfg.depth, cfg.regime, cfg.seed, cfg.threads) == ("cora", "nhdp", 2, "full", 3, 1)
    assert (cfg.epochs, cfg.hidden, cfg.out_dir, cfg.cache_dir) == (7, 32, str(tmp_path / "out"),
                                                                    str(tmp_path / "cache"))


def test_variant_environments(monkeypatch):
    mod = _script()
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx512")
    monkeypatch.setenv("KEEP_ME", "1")
    default, pinned = mod.variant_env("default"), mod.variant_env("pinned")
    assert "ATEN_CPU_CAPABILITY" not in default and default["KEEP_ME"] == "1"
    assert pinned["ATEN_CPU_CAPABILITY"] == "avx2" and pinned["OPENBLAS_CORETYPE"] == "HASWELL"
    assert pinned["MKL_CBWR"] == "AVX2" and pinned["KEEP_ME"] == "1"


def test_fingerprint_sees_the_last_bit():
    mod = _script()
    x = 100711.8233
    nxt = float.fromhex(float(x).hex()) + 2.0 ** -36                    # one ulp above x
    assert nxt != x
    assert mod.fingerprint([x, 2.5]) == mod.fingerprint([x, 2.5])
    assert mod.fingerprint([x, 2.5]) != mod.fingerprint([nxt, 2.5])
    assert mod.fingerprint([]) is None and mod.fingerprint(None) is None


def test_summary_reports_which_repeats_and_variants_agree():
    mod = _script()
    rows = [row("default", 0, 0, .81), row("default", 0, 1, .81), row("pinned", 0, 0, .80), row("pinned", 0, 1, .80),
            row("default", 1, 0, .82), row("default", 1, 1, .83, trace="b"), row("pinned", 1, 0, .82),
            row("pinned", 1, 1, .82)]
    lines = mod.summarize(rows)
    assert "default seed 0: 2 runs identical (test 0.8100, 0.8100)" in lines
    assert "seed 0: variants DIFFER (default 0.8100, pinned 0.8000)" in lines
    assert "default seed 1: 2 runs DIFFER (test 0.8200, 0.8300)" in lines
    assert "pinned seed 1: 2 runs identical (test 0.8200, 0.8200)" in lines
    assert "seed 1: variants agree (default 0.8200, pinned 0.8200)" in lines


def test_single_setting_variants(monkeypatch):
    mod = _script()
    monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
    aten = mod.variant_env("aten")
    assert aten["ATEN_CPU_CAPABILITY"] == "avx2" and "MKL_CBWR" not in aten and "OPENBLAS_CORETYPE" not in aten
    full = mod.variant_env("all")
    assert full["OPENBLAS_CORETYPE"] == "HASWELL" and "X86_V4" in full["NPY_DISABLE_CPU_FEATURES"].split()
    assert full["ONEDNN_MAX_CPU_ISA"] == "AVX2" and "NPY_DISABLE_CPU_FEATURES" not in mod.variant_env("pinned")
    assert mod.variant_env("default")["PYTHONFAULTHANDLER"] == "1"          # a crash prints the Python stack


def test_output_fingerprint_sees_the_last_bit_of_the_diagnostics():
    mod = _script()
    base = {"val": .8, "test": .79, "embedding": {"centroid_gap": 0.5, "madgap": 0.3, "nmi": .2, "ari": .1,
                                                  "centroid_gap_per_layer": [.1, .2]}}
    same, nxt = json.loads(json.dumps(base)), json.loads(json.dumps(base))
    nxt["embedding"]["madgap"] = math.nextafter(0.3, 1.0)
    assert mod.output_fingerprint(base) == mod.output_fingerprint(same) != mod.output_fingerprint(nxt)
    assert mod.output_fingerprint({}) is None


def _child(returncode_of):
    """A stand-in for the training process: a result file, or an exit status chosen from its environment."""
    def run(cmd, env=None, stdout=None, **kwargs):
        code = returncode_of(env)
        if code == 0:
            out = Path(cmd[cmd.index("--out_dir") + 1])
            out.mkdir(parents=True)
            (out / "result.json").write_text(json.dumps({
                "val": .8, "test": .79, "best_epoch": 5, "epochs_run": 105,
                "stages": {"nhdp_elbo_trace": [1.0, 2.0]},
                "embedding": {"centroid_gap": .5, "madgap": .3, "nmi": .2, "ari": .1},
                "environment": {"cpu_model": "Test CPU", "numpy_simd": ["X86_V3"],
                                "blas": [{"internal_api": "openblas", "architecture": "Haswell"}]}}))
        return subprocess.CompletedProcess(cmd, code)
    return run


def test_a_crashing_run_is_recorded_and_the_script_goes_on(monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod.subprocess, "run", _child(lambda env: -11 if "OPENBLAS_CORETYPE" in env else 0))
    mod.main(["--seeds", "0", "1", "--repeats", "2", "--variants", "default", "pinned"])
    out = capsys.readouterr().out
    results = [json.loads(line[len("RESULT "):]) for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 8 and {r["crashed"] for r in results if "crashed" in r} == {"SIGSEGV"}
    assert sum("crashed" in r for r in results) == 4
    finished = [r for r in results if "crashed" not in r]
    assert finished[0]["cpu_model"] == "Test CPU" and finished[0]["blas"][0]["architecture"] == "Haswell"
    assert "pinned seed 0: 2 runs CRASHED (SIGSEGV)" in out
    assert "default seed 1: 2 runs identical (test 0.7900, 0.7900)" in out
    assert "seed 0: crashed: pinned (SIGSEGV)" in out


def test_the_script_fails_if_no_run_finishes(monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod.subprocess, "run", _child(lambda env: 1))
    with pytest.raises(SystemExit) as stopped:
        mod.main(["--seeds", "0", "--repeats", "1", "--variants", "default"])
    assert stopped.value.code == 1


def test_summary_of_crashes():
    mod = _script()
    crashed = {"variant": "pinned", "seed": 0, "repeat": 0, "crashed": "SIGSEGV"}
    lines = mod.summarize([row("default", 0, 0, .81), crashed, row("openblas", 0, 0, .81)])
    assert "pinned seed 0: 1 runs CRASHED (SIGSEGV)" in lines
    assert "seed 0: variants agree (default 0.8100, openblas 0.8100)" in lines
    assert "seed 0: crashed: pinned (SIGSEGV)" in lines
    assert [mod.describe_exit(code) for code in (-11, -4, 1)] == ["SIGSEGV", "SIGILL", "exit 1"]
