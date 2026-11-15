#!/usr/bin/env python3
"""Does a run repeat bit for bit on this machine, and with the kernel selection pinned?

Trains one configuration several times, each in a process of its own, under different kernel
selections ("variants"), and prints one ``RESULT {json}`` line per run: validation and test
metric, best epoch, a fingerprint of the n-HDP objective trace (numpy, double precision), a
fingerprint of the trained model's outputs (torch, single precision), the CPU model and the
kernels that were in use. Every run gets a cache directory of its own, so stages A and B are
recomputed each time.

    python scripts/check_reproducibility.py --dataset cora --seeds 0 1 --repeats 2

Variants (`--variants`; default "default pinned"):

    default   kernels chosen by the CPU
    pinned    ATEN_CPU_CAPABILITY=avx2, MKL_CBWR=AVX2, OPENBLAS_CORETYPE=HASWELL, ONEDNN_MAX_CPU_ISA=AVX2
    aten, mkl, onednn, openblas
              one of the four settings of "pinned" on its own
    numpy     numpy's SIMD dispatch limited to AVX2 (NPY_DISABLE_CPU_FEATURES)
    all       "pinned" and "numpy" together

A run that dies (a segmentation fault, say) is reported as crashed and the script goes on; it
exits with status 1 only if no run finished. Run it on several machines and compare the RESULT
lines: the `determinism` workflow does this on GitHub's runners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

OPENBLAS = {"OPENBLAS_CORETYPE": "HASWELL"}
TORCH = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "AVX2", "ONEDNN_MAX_CPU_ISA": "AVX2"}
PINNED = {**TORCH, **OPENBLAS}
# numpy 2.x dispatches to X86_V3 (AVX2), X86_V4 and AVX512_*; names that a numpy version does not
# know are ignored with a warning, the others are disabled
NUMPY = {"NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR AVX512F AVX512CD AVX512_KNL AVX512_KNM "
                                     "AVX512_SKX AVX512_CLX AVX512_CNL"}
VARIANTS = {"default": {}, "pinned": PINNED, "aten": {"ATEN_CPU_CAPABILITY": "avx2"}, "mkl": {"MKL_CBWR": "AVX2"},
            "onednn": {"ONEDNN_MAX_CPU_ISA": "AVX2"}, "openblas": OPENBLAS, "numpy": NUMPY,
            "all": {**PINNED, **NUMPY}}
MANAGED = sorted({name for settings in VARIANTS.values() for name in settings})


def variant_env(variant: str, base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    for name in MANAGED:                                    # a variable set by the caller must not leak in
        env.pop(name, None)
    env.update(VARIANTS[variant])
    env["PYTHONFAULTHANDLER"] = "1"                         # a crash prints the Python stack
    return env


def fingerprint(trace) -> str | None:
    """Short hash of a list of floats, exact to the last bit."""
    if not trace:
        return None
    return hashlib.sha1(json.dumps([float(v).hex() for v in trace]).encode()).hexdigest()[:12]


def output_fingerprint(result: dict) -> str | None:
    """Hash of the trained model's measured outputs (metrics, centroid gap, MADGap, NMI, ARI)."""
    emb = result.get("embedding") or {}
    values = [result.get("val"), result.get("test"), emb.get("centroid_gap"), emb.get("madgap"), emb.get("nmi"),
              emb.get("ari")] + list(emb.get("centroid_gap_per_layer") or [])
    return fingerprint([v for v in values if v is not None])


def row_of(result: dict, variant: str, seed: int, repeat: int) -> dict:
    env = result.get("environment") or {}
    return {"variant": variant, "seed": seed, "repeat": repeat, "val": result["val"], "test": result["test"],
            "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "trace_sha1": fingerprint(result.get("stages", {}).get("nhdp_elbo_trace")),
            "output_sha1": output_fingerprint(result),
            "cpu_model": env.get("cpu_model"), "cpu_simd": env.get("cpu_simd"),
            "numpy_simd": env.get("numpy_simd"), "blas": env.get("blas"), "packages": env.get("packages")}


def describe_exit(returncode: int) -> str:
    if returncode < 0:
        try:
            return signal.Signals(-returncode).name
        except ValueError:
            return "signal %d" % -returncode
    return "exit %d" % returncode


def crash_row(variant: str, seed: int, repeat: int, returncode: int) -> dict:
    return {"variant": variant, "seed": seed, "repeat": repeat, "crashed": describe_exit(returncode)}


def _key(r: dict):
    if "crashed" in r:
        return ("crashed", r["crashed"])
    return (r["test"], r["val"], r["best_epoch"], r["epochs_run"], r["trace_sha1"], r.get("output_sha1"))


def _same(rows) -> bool:
    return len({_key(r) for r in rows}) == 1


def summarize(rows: list[dict]) -> list[str]:
    """One line per variant and seed (do the repeats agree?) and one per seed (do the variants agree?)."""
    lines = []
    variants = sorted({r["variant"] for r in rows})
    for seed in sorted({r["seed"] for r in rows}):
        firsts = []
        for variant in variants:
            rs = sorted((r for r in rows if r["variant"] == variant and r["seed"] == seed),
                        key=lambda r: r["repeat"])
            if not rs:
                continue
            firsts.append(rs[0])
            if all("crashed" in r for r in rs):
                lines.append("%s seed %d: %d runs CRASHED (%s)"
                             % (variant, seed, len(rs), ", ".join(sorted({r["crashed"] for r in rs}))))
                continue
            tests = ", ".join("crashed (%s)" % r["crashed"] if "crashed" in r else "%.4f" % r["test"] for r in rs)
            lines.append("%s seed %d: %d runs %s (test %s)"
                         % (variant, seed, len(rs), "identical" if _same(rs) else "DIFFER", tests))
        finished = [r for r in firsts if "crashed" not in r]
        if len(finished) > 1:
            tests = ", ".join("%s %.4f" % (r["variant"], r["test"]) for r in finished)
            lines.append("seed %d: variants %s (%s)" % (seed, "agree" if _same(finished) else "DIFFER", tests))
        crashed = [r for r in firsts if "crashed" in r]
        if crashed:
            lines.append("seed %d: crashed: %s" % (seed, ", ".join("%s (%s)" % (r["variant"], r["crashed"])
                                                                   for r in crashed)))
    return lines


def train_command(args, seed: int, out: Path, cache: Path) -> list[str]:
    cmd = [sys.executable, "-m", "nhdp_gnn.train", "--dataset", args.dataset, "--model", args.model,
           "--depth", str(args.depth), "--regime", args.regime, "--seed", str(seed), "--threads", "1",
           "--root", args.root, "--out_dir", str(out), "--cache_dir", str(cache)]
    for kv in args.set:
        key, _, value = kv.partition("=")
        cmd += [f"--{key}", value]
    return cmd


def train_once(args, variant: str, seed: int, repeat: int, work: Path) -> dict:
    out, cache = work / f"{variant}_{seed}_{repeat}", work / f"cache_{variant}_{seed}_{repeat}"
    done = subprocess.run(train_command(args, seed, out, cache), env=variant_env(variant), stdout=subprocess.DEVNULL)
    if done.returncode != 0:
        return crash_row(variant, seed, repeat, done.returncode)
    (path,) = out.glob("*.json")
    return row_of(json.loads(path.read_text()), variant, seed, repeat)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--model", default="nhdp")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--regime", default="full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["default", "pinned"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--set", action="append", default=[], help="extra configuration, key=value")
    ap.add_argument("--work", default=None, help="directory for the runs (default: a temporary one)")
    args = ap.parse_args(argv)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.work or tmp)
        for seed in args.seeds:
            for variant in args.variants:
                for repeat in range(args.repeats):
                    row = train_once(args, variant, seed, repeat, work)
                    rows.append(row)
                    print("RESULT", json.dumps(row), flush=True)
    print("\n".join(summarize(rows)), flush=True)
    if rows and all("crashed" in r for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
