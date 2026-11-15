"""scripts/check_reproducibility.py: command line, environments, fingerprints and the summary (no torch needed)."""
import argparse
import importlib.util
from pathlib import Path

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
