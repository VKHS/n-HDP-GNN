"""Temporary diagnostic (branch diag/ddp-barrier-confirm only): how often does a 4-rank spawn_local run abort at exit
(SIGABRT while the interpreter shuts down), for four ways of leaving the run?

  old      run the worker, destroy the process group                        (the code before the barrier)
  barrier  run the worker, wait at a barrier, destroy the process group     (nhdp_gnn.distributed._local_entry)
  sleep    as barrier, then sleep half a second before the process ends
  exit     as barrier, then flush and end the process with os._exit(0), without interpreter shutdown

The variants are interleaved trial by trial and the order rotates from round to round.  In round 0 an extra trial
prints how many threads a rank has after each step (DIAG_PROBE)."""
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


def threads() -> int:
    return len(os.listdir("/proc/self/task"))


def probe(rank, label):
    if os.environ.get("DIAG_PROBE") and rank == 1:
        print(f"PROBE rank 1 {label}: {threads()} threads", flush=True)


def _start(rank, world, port, cfg_dict):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    probe(rank, "before init")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    probe(rank, "after init")
    return config_from_dict(cfg_dict)


def entry_old(rank, world, port, cfg_dict):
    cfg = _start(rank, world, port, cfg_dict)
    try:
        D.worker(cfg)
        probe(rank, "after worker")
    finally:
        dist.destroy_process_group()
        probe(rank, "after destroy")


def entry_barrier(rank, world, port, cfg_dict):
    cfg = _start(rank, world, port, cfg_dict)
    try:
        D.worker(cfg)
        probe(rank, "after worker")
        dist.barrier()
        probe(rank, "after barrier")
    finally:
        dist.destroy_process_group()
        probe(rank, "after destroy")


def entry_sleep(rank, world, port, cfg_dict):
    entry_barrier(rank, world, port, cfg_dict)
    time.sleep(0.5)


def entry_exit(rank, world, port, cfg_dict):
    entry_barrier(rank, world, port, cfg_dict)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


VARIANTS = {"old": entry_old, "barrier": entry_barrier, "sleep": entry_sleep, "exit": entry_exit}


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
    os.environ["DIAG_PROBE"] = "1"
    print(f"probe trial: {trial(entry_barrier, world)}", flush=True)
    del os.environ["DIAG_PROBE"]
    tally = {name: collections.Counter() for name in VARIANTS}
    names = list(VARIANTS)
    t0 = time.time()
    for i in range(n):
        order = names[i % len(names):] + names[:i % len(names)]
        for name in order:
            result = trial(VARIANTS[name], world)
            tally[name][result] += 1
            if result != "ok":
                print(f"shard {shard} round {i} {name}: {result}", flush=True)
        print(f"shard {shard} after {i + 1} rounds ({time.time() - t0:.0f} s): " + "; ".join(
            f"{name} {c['ok']}/{sum(c.values())} ok" for name, c in tally.items()), flush=True)
    for name, c in tally.items():
        print(f"RESULT shard {shard} world {world} {name} ok {c['ok']} of {sum(c.values())} "
              + " ".join(f"[{k}: {v}]" for k, v in c.items() if k != "ok"), flush=True)
