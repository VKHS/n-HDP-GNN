"""Temporary diagnostic (branch diag/ddp-barrier-confirm only): how often does a 4-rank spawn_local run abort at exit
with the previous exit sequence ("old": run the worker, destroy the group) and with the shipped one ("new":
nhdp_gnn.distributed._local_entry, which waits at a barrier before destroying the group)?  The two variants are
interleaved trial by trial and the order alternates from round to round."""
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


def entry_old(rank, world, port, cfg_dict):
    """The exit sequence before the barrier was added: the same code, written out."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        D.worker(config_from_dict(cfg_dict))
    finally:
        dist.destroy_process_group()


def entry_new(rank, world, port, cfg_dict):
    D._local_entry(rank, world, port, cfg_dict)


VARIANTS = {"old": entry_old, "new": entry_new}


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
    world, n, shard = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "-"
    tally = {name: collections.Counter() for name in VARIANTS}
    t0 = time.time()
    for i in range(n):
        names = list(VARIANTS) if i % 2 == 0 else list(VARIANTS)[::-1]
        for name in names:
            result = trial(VARIANTS[name], world)
            tally[name][result] += 1
            if result != "ok":
                print(f"shard {shard} round {i} {name}: {result}", flush=True)
        print(f"shard {shard} after {i + 1} rounds ({time.time() - t0:.0f} s): " + "; ".join(
            f"{name} {c['ok']}/{sum(c.values())} ok" for name, c in tally.items()), flush=True)
    for name, c in tally.items():
        print(f"RESULT shard {shard} world {world} {name} ok {c['ok']} of {sum(c.values())} "
              + " ".join(f"[{k}: {v}]" for k, v in c.items() if k != "ok"), flush=True)
