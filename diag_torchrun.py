"""Temporary diagnostic (branch diag/ddp-barrier-confirm only): the launch path of the real cluster, torchrun with
``-m nhdp_gnn.distributed``, on one machine: how often does a run end with a failing exit status, with the module as
it was before the barrier ("old") and as it is now ("new": barrier, then os._exit)?  The runs of the two variants are
interleaved and their order alternates.  A run counts as failed when torchrun's exit status is not 0; the number of
result files shows whether the result was written anyway."""
import collections
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODULES = {"old": "nhdp_gnn.distributed_old", "new": "nhdp_gnn.distributed"}


def run(variant, world):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["torchrun", "--standalone", "--nnodes", "1", "--nproc-per-node", str(world), "-m", MODULES[variant],
               "--config", "configs/cluster_check.toml", "--out_dir", f"{tmp}/runs", "--cache_dir", f"{tmp}/cache"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        written = len(list(Path(tmp, "runs").glob("*-ddp*.json")))
        return done.returncode, written, done.stderr[-700:]


if __name__ == "__main__":
    world, n, shard = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    tally = {v: collections.Counter() for v in MODULES}
    t0 = time.time()
    for i in range(n):
        order = list(MODULES) if i % 2 == 0 else list(MODULES)[::-1]
        for v in order:
            code, written, tail = run(v, world)
            tally[v][(code, written)] += 1
            if code != 0 or written != 1:
                print(f"shard {shard} round {i} {v}: exit status {code}, {written} result file(s)\n{tail}", flush=True)
        print(f"shard {shard} after {i + 1} rounds ({time.time() - t0:.0f} s): " + "; ".join(
            f"{v} {sum(c for (code, w), c in t.items() if code == 0 and w == 1)}/{sum(t.values())} clean" for v, t in tally.items()), flush=True)
    for v, t in tally.items():
        print(f"RESULT shard {shard} world {world} {v} clean {sum(c for (code, w), c in t.items() if code == 0 and w == 1)} of {sum(t.values())} "
              + " ".join(f"[status {code}, {w} files: {c}]" for (code, w), c in t.items() if not (code == 0 and w == 1)), flush=True)
