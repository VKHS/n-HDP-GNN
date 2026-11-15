#!/usr/bin/env python3
"""Does a run repeat bit for bit on this machine, and with the instruction set pinned?

Trains one configuration several times, each in a process of its own, with the default
kernel selection ("default") and with the instruction set pinned ("pinned":
ATEN_CPU_CAPABILITY=avx2, MKL_CBWR=AVX2, OPENBLAS_CORETYPE=HASWELL, ONEDNN_MAX_CPU_ISA=AVX2).
Every run gets a cache directory of its own, so stages A and B are recomputed each time.

    python scripts/check_reproducibility.py --dataset cora --seeds 0 1 --repeats 2

Prints one ``RESULT {json}`` line per run (validation and test metric, best epoch, a fingerprint
of the n-HDP objective trace, the CPU model) and a summary of which repeats agree. Run it on
several machines and compare the RESULT lines: the `determinism` workflow does this on GitHub's
runners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PINNED = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "AVX2", "OPENBLAS_CORETYPE": "HASWELL",
          "ONEDNN_MAX_CPU_ISA": "AVX2"}
VARIANTS = {"default": {}, "pinned": PINNED}


def variant_env(variant: str, base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    for name in PINNED:                                     # a variable set by the caller must not leak in
        env.pop(name, None)
    env.update(VARIANTS[variant])
    return env


def fingerprint(trace) -> str | None:
    """Short hash of a list of floats, exact to the last bit."""
    if not trace:
        return None
    return hashlib.sha1(json.dumps([float(v).hex() for v in trace]).encode()).hexdigest()[:12]


def row_of(result: dict, variant: str, seed: int, repeat: int) -> dict:
    env = result.get("environment") or {}
    return {"variant": variant, "seed": seed, "repeat": repeat, "val": result["val"], "test": result["test"],
            "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "trace_sha1": fingerprint(result.get("stages", {}).get("nhdp_elbo_trace")),
            "cpu_model": env.get("cpu_model"), "cpu_simd": env.get("cpu_simd"),
            "packages": env.get("packages")}


def _same(rows) -> bool:
    return len({(r["test"], r["val"], r["best_epoch"], r["epochs_run"], r["trace_sha1"]) for r in rows}) == 1


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
            tests = ", ".join("%.4f" % r["test"] for r in rs)
            lines.append("%s seed %d: %d runs %s (test %s)"
                         % (variant, seed, len(rs), "identical" if _same(rs) else "DIFFER", tests))
        if len(firsts) > 1:
            tests = ", ".join("%s %.4f" % (r["variant"], r["test"]) for r in firsts)
            lines.append("seed %d: variants %s (%s)" % (seed, "agree" if _same(firsts) else "DIFFER", tests))
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
    subprocess.run(train_command(args, seed, out, cache), check=True, env=variant_env(variant),
                   stdout=subprocess.DEVNULL)
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


if __name__ == "__main__":
    main()
