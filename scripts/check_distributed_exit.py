#!/usr/bin/env python
"""How often does a multi-rank run on one machine abort while its processes exit?  (needs PyTorch and PyG)

A run of ``nhdp_gnn.distributed`` ended, now and then, with one process killed by SIGABRT ("terminate called without
an active exception") after the result was written. This script counts such runs for three ways of leaving:

  old      run the worker, destroy the process group, return           (the code before the barrier was added)
  barrier  run the worker, wait at a barrier, destroy the group, return
  exit     what ``nhdp_gnn.distributed`` does now: as barrier, then end the process with os._exit

Each round runs every variant once, ``--world`` processes on this machine, on configs/cluster_check.toml (a small
planted graph that every rank generates itself). The variants rotate their order from round to round. The script
prints, per variant, how many runs ended without a failure and what the others died of. Nothing here says anything
about accuracy; the runs are 10 epochs of a tiny graph, and a run takes a few seconds.

    OMP_NUM_THREADS=1 python scripts/check_distributed_exit.py --world 4 --rounds 25

The abort depends on timing, so the counts depend on the machine, the PyTorch and the Python version; run the same
command again to see how much they move. With 400 runs of a variant and no abort, the 95% upper bound of its rate is
0.9%. The exit status is 0 whatever the counts are.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if importlib.util.find_spec("nhdp_gnn") is None:            # a checkout that was not pip-installed
    sys.path.insert(0, str(ROOT))

VARIANTS = ("old", "barrier", "exit")


def _start(rank, world, port, cfg_dict):
    import torch.distributed as dist

    from nhdp_gnn.config import config_from_dict

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    return config_from_dict(cfg_dict)


def entry_old(rank, world, port, cfg_dict):
    import torch.distributed as dist

    from nhdp_gnn import distributed

    cfg = _start(rank, world, port, cfg_dict)
    try:
        distributed.worker(cfg)
    finally:
        dist.destroy_process_group()


def entry_barrier(rank, world, port, cfg_dict):
    from nhdp_gnn import distributed

    distributed.run_and_leave(_start(rank, world, port, cfg_dict))


def entry_exit(rank, world, port, cfg_dict):
    from nhdp_gnn import distributed

    distributed._local_entry(rank, world, port, cfg_dict)


ENTRIES = {"old": entry_old, "barrier": entry_barrier, "exit": entry_exit}


def one_run(variant: str, world: int, config: Path) -> str:
    """"ok", or what the run died of (the first line of the error that ``mp.spawn`` raised)."""
    import tomllib
    from dataclasses import asdict, replace

    import torch.multiprocessing as mp

    from nhdp_gnn import distributed
    from nhdp_gnn.config import config_from_dict

    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(config_from_dict(tomllib.loads(config.read_text())), threads=1,
                      out_dir=str(Path(tmp) / "runs"), cache_dir=str(Path(tmp) / "cache"))
        try:
            mp.spawn(ENTRIES[variant], args=(world, distributed._free_port(), asdict(cfg)), nprocs=world, join=True)
        except Exception as exc:
            return f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"
    return "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=4, help="processes per run (default 4)")
    ap.add_argument("--rounds", type=int, default=25, help="rounds; every round runs each variant once (default 25)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated, from {', '.join(VARIANTS)}")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "cluster_check.toml")
    args = ap.parse_args(argv)
    variants = [v for v in args.variants.split(",") if v]
    unknown = [v for v in variants if v not in ENTRIES]
    if unknown or not variants:
        ap.error(f"unknown variants {unknown}; choose from {', '.join(VARIANTS)}")
    if args.world < 2 or args.rounds < 1:
        ap.error("--world must be at least 2 and --rounds at least 1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import torch

    tally = {v: collections.Counter() for v in variants}
    started = time.time()
    print(f"torch {torch.__version__}, python {sys.version.split()[0]}, {args.world} processes per run, "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}", flush=True)
    for i in range(args.rounds):
        shift = i % len(variants)
        for v in variants[shift:] + variants[:shift]:
            result = one_run(v, args.world, args.config)
            tally[v][result] += 1
            if result != "ok":
                print(f"round {i} {v}: {result}", flush=True)
        print(f"after {i + 1} rounds ({time.time() - started:.0f} s): " + "; ".join(
            f"{v} {tally[v]['ok']}/{sum(tally[v].values())} ok" for v in variants), flush=True)
    print(f"\nResult, {args.world} processes per run:")
    for v in variants:
        total = sum(tally[v].values())
        died = ", ".join(f"{n} x {what}" for what, n in tally[v].items() if what != "ok")
        print(f"  {v:8s} {tally[v]['ok']:4d} of {total:4d} ended without a failure" + (f"; {died}" if died else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
