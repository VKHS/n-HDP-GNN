"""Temporary diagnostic (branch diag/ddp-teardown only): how often does a 4-rank spawn_local run abort at exit,
and does a final barrier or skipping destroy_process_group change that?  Variants are interleaved trial by trial."""
import collections
import os
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from nhdp_gnn import distributed as D
from nhdp_gnn.config import config_from_dict

CONFIG = Path("configs/cluster_check.toml")


def entry_current(rank, world, port, cfg_dict):
    D._local_entry(rank, world, port, cfg_dict)


def entry_barrier(rank, world, port, cfg_dict):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        D.worker(config_from_dict(cfg_dict))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def entry_nodestroy(rank, world, port, cfg_dict):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    D.worker(config_from_dict(cfg_dict))


VARIANTS = {"current": entry_current, "barrier": entry_barrier, "nodestroy": entry_nodestroy}


def trial(entry, world):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(config_from_dict(tomllib.loads(CONFIG.read_text())), threads=1, out_dir=str(Path(tmp) / "runs"),
                      cache_dir=str(Path(tmp) / "cache"))
        try:
            mp.spawn(entry, args=(world, D._free_port(), asdict(cfg)), nprocs=world, join=True)
            return "ok"
        except Exception as exc:
            return f"{type(exc).__name__}: {str(exc)[:80]}"


if __name__ == "__main__":
    world, n = int(sys.argv[1]), int(sys.argv[2])
    tally = {name: collections.Counter() for name in VARIANTS}
    t0 = time.time()
    for i in range(n):
        for name, entry in VARIANTS.items():
            result = trial(entry, world)
            tally[name][result] += 1
            if result != "ok":
                print(f"trial {i} {name}: {result}", flush=True)
        print(f"after {i + 1} rounds ({time.time() - t0:.0f} s): " + "; ".join(
            f"{name} {c['ok']}/{sum(c.values())} ok" for name, c in tally.items()), flush=True)
    print("SUMMARY world", world)
    for name, c in tally.items():
        print(f"  {name:10s} ok {c['ok']:3d} of {sum(c.values())}  " + "  ".join(f"[{k}: {v}]" for k, v in c.items() if k != "ok"))
