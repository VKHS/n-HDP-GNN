"""Expand a sweep file (TOML) into runs and execute the missing ones.

    python -m nhdp_gnn.sweep configs/table2_full.toml --dry-run
    python -m nhdp_gnn.sweep configs/table2_full.toml --only model=nhdp,dataset=cora --shard 3/16
    python -m nhdp_gnn.sweep configs/table2_full.toml --only "seed=0|1|2"
    python -m nhdp_gnn.sweep configs/table2_full.toml --workers 4 --memory-gb 13 --run-minutes 150

A sweep file has ``name``, a ``[base]`` table of RunConfig values, a ``[grid]``
table of lists (Cartesian product) and optional ``[[variant]]`` tables (named
overrides, stored as the run's ``tag``).  Results are written to
runs/<name>/<key>.json; runs whose result exists are skipped, so a sweep can be
resumed, or split across cluster jobs with --shard i/n.  An ``--only`` value may
list alternatives separated by ``|``.  An optional ``[actions]`` table holds the
settings of the GitHub Actions workflow (.github/workflows/reproduce.yml).

``--workers N`` runs every configuration in a process of its own, N at a time,
and watches their memory (Linux).  When the processes together use more than
``--memory-gb``, the largest one is stopped and its run is repeated later on its
own, with the whole budget.  A run that fails leaves ``<key>.failed.json``
instead of a result, with one of three reasons:

  memory   it needed more than the budget while running alone, or the kernel
           killed it while it ran alone
  time     training provably needs more than ``--run-minutes`` (see train.py),
           or the whole run exceeded ``--kill-minutes``
  diverged the selected model gives non-finite outputs (NaN or infinity)
  error    it raised an exception or exited abnormally

Later invocations skip runs that failed for memory, time or divergence
(``--retry-failed`` repeats them) and repeat runs that failed with an error; ``--previous DIR`` adds
the records of an earlier campaign (unpacked with scripts/unpack_results.py).
Without ``--workers`` the runs execute one after the other in this process, with
the same records.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import RunConfig, _parse_value, config_from_dict

EXIT = {"ok": 0, "error": 1, "time": 3, "memory": 4, "diverged": 5}
DEFAULT_RUNNER = "nhdp_gnn.train:run"


def expand(spec: dict) -> list[dict]:
    base, grid = dict(spec.get("base", {})), dict(spec.get("grid", {}))
    variants = spec.get("variant") or [{}]
    out_dir = spec.get("out_dir", f"runs/{spec.get('name', 'sweep')}")
    keys = list(grid)
    runs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        for var in variants:
            d = dict(base)
            d.update(zip(keys, combo))
            var = dict(var)
            name = var.pop("name", "")
            d.update(var)
            if name:
                d["tag"] = name
            d.setdefault("out_dir", out_dir)
            runs.append(d)
    return runs


def select(runs: list[dict], only: dict | None = None, shard: tuple[int, int] = (0, 1),
           overrides: dict | None = None) -> list[RunConfig]:
    i, n = shard
    allowed = {k: str(v).split("|") for k, v in (only or {}).items()}
    out = []
    for idx, d in enumerate(runs):
        if idx % n != i or any(str(d.get(k)) not in vs for k, vs in allowed.items()):
            continue
        out.append(config_from_dict({**d, **(overrides or {})}))
    return out


# ----------------------------------------------------------------------------- records

def result_path(cfg: RunConfig) -> Path:
    return Path(cfg.out_dir) / f"{cfg.key()}.json"


def failure_path(cfg: RunConfig) -> Path:
    return Path(cfg.out_dir) / f"{cfg.key()}.failed.json"


def _reason(path: Path) -> str | None:
    try:
        return json.loads(path.read_text())["failure"]["reason"]
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return "error"


def failure_reason(cfg: RunConfig) -> str | None:
    return _reason(failure_path(cfg))


def state(cfg: RunConfig, previous=()) -> str | None:
    """"done", the reason of a recorded failure, or None; ``previous`` adds directories of earlier records."""
    dirs = [Path(cfg.out_dir), *map(Path, previous)]
    if any((d / f"{cfg.key()}.json").exists() for d in dirs):
        return "done"
    return next((r for d in dirs if (r := _reason(d / f"{cfg.key()}.failed.json"))), None)


def write_failure(cfg: RunConfig, reason: str, detail: str, log: Path | None = None, **info) -> None:
    from .utils import code_version, save_json

    record = {"key": cfg.key(), "config": asdict(cfg), **code_version(),
              "failure": {"reason": reason, "detail": detail, **info}}
    if log is not None and log.exists():
        record["failure"]["log_tail"] = log.read_text(errors="replace").splitlines()[-40:]
    save_json(failure_path(cfg), record)


def _describe(cfg: RunConfig) -> str:
    text = f"{cfg.dataset} {cfg.regime} {cfg.model} depth={cfg.depth} seed={cfg.seed}"
    return f"{text} {cfg.tag}" if cfg.tag else text


def _is_allocation_error(e: BaseException) -> bool:
    text = str(e).lower()
    return isinstance(e, MemoryError) or any(s in text for s in (
        "can't allocate memory", "cannot allocate memory", "not enough memory", "out of memory", "bad_alloc"))


def _resolve(runner: str):
    module, func = runner.split(":")
    return getattr(importlib.import_module(module), func)


def attempt(cfg: RunConfig, fn) -> str:
    """Run one configuration in this process; returns "ok" or the failure reason (and records it)."""
    from .utils import TimeLimitExceeded, TrainingDiverged

    try:
        fn(cfg)
    except TimeLimitExceeded as e:
        write_failure(cfg, "time", str(e), **e.info)
        return "time"
    except TrainingDiverged as e:
        write_failure(cfg, "diverged", str(e), **e.info)
        return "diverged"
    except Exception as e:                               # noqa: BLE001 -- recorded, the sweep goes on
        if _is_allocation_error(e):
            write_failure(cfg, "memory", f"{type(e).__name__}: {e}")
            return "memory"
        write_failure(cfg, "error", traceback.format_exc(limit=30))
        return "error"
    if not result_path(cfg).exists():
        write_failure(cfg, "error", "the run returned without writing its result")
        return "error"
    failure_path(cfg).unlink(missing_ok=True)
    return "ok"


def child_main(cfg_json: str, runner: str) -> int:
    """Entry point of a worker process: one configuration, exit code from EXIT."""
    try:                                                 # be the kernel's first choice if memory runs out
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("1000")
    except OSError:
        pass
    cfg = config_from_dict(json.loads(cfg_json))
    outcome = attempt(cfg, _resolve(runner))
    if outcome == "ok":
        from .utils import load_json, peak_rss_mb, save_json

        res = load_json(result_path(cfg))
        res["peak_rss_mb"] = peak_rss_mb()               # one run per process: the run's own peak
        save_json(result_path(cfg), res)
    return EXIT[outcome]


# ----------------------------------------------------------------------------- memory

def _memory_mb(pid: int) -> float:
    """Resident plus swapped memory of a process and its descendants (Linux; 0 if unavailable)."""
    total, todo, seen = 0.0, [pid], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(f"/proc/{p}/status") as fh:
                for line in fh:
                    if line.startswith(("VmRSS:", "VmSwap:")):
                        total += int(line.split()[1]) / 1024
            for task in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{task}/children") as fh:
                    todo += [int(c) for c in fh.read().split()]
        except (OSError, ValueError):
            continue
    return total


def total_memory_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


# ----------------------------------------------------------------------------- worker pool

@dataclass
class _Job:
    cfg: RunConfig
    proc: subprocess.Popen
    log: Path
    solo: bool
    start: float = field(default_factory=time.monotonic)
    peak_mb: float = 0.0


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)              # the worker runs in a session of its own
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def run_pool(todo: list[RunConfig], workers: int, memory_gb: float = 0.0, kill_minutes: float = 0.0,
             deadline_minutes: float = 0.0, runner: str = DEFAULT_RUNNER, poll: float = 0.25) -> dict:
    """Run configurations in worker processes; returns counts per outcome."""
    budget_mb = 1024 * memory_gb
    watch = budget_mb > 0 and os.path.exists("/proc/self/status")
    if budget_mb > 0 and not watch:
        print("memory watch needs Linux /proc; running without a memory budget", flush=True)
    queue, solo_queue, running = list(todo), [], {}
    counts = {"ok": 0, "memory": 0, "time": 0, "diverged": 0, "error": 0, "repeated alone": 0, "stopped": 0}
    stop = []
    old_handlers = {s: signal.signal(s, lambda *_: stop.append(1)) for s in (signal.SIGINT, signal.SIGTERM)}
    t0 = time.monotonic()
    n_total, n_started = len(todo), 0

    def launch(cfg, solo):
        nonlocal n_started
        n_started += not solo
        failure_path(cfg).unlink(missing_ok=True)       # a record of an earlier attempt no longer applies
        log = Path(cfg.out_dir) / "logs" / f"{cfg.key()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "nhdp_gnn.sweep", "--child", json.dumps(asdict(cfg)), "--runner", runner]
        with open(log, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        running[proc.pid] = _Job(cfg, proc, log, solo)
        what = "alone" if solo else f"{n_started}/{n_total}"
        print(f"start  [{what}] {cfg.key()}  {_describe(cfg)}", flush=True)

    def finish(job, outcome, note=""):
        minutes = (time.monotonic() - job.start) / 60
        if outcome in counts:
            counts[outcome] += 1
        print(f"{outcome:<6} {job.cfg.key()}  {_describe(job.cfg)}  {minutes:.1f} min, peak {job.peak_mb:.0f} MB"
              + (f"  ({note})" if note else ""), flush=True)

    def again_alone(job, why):
        failure_path(job.cfg).unlink(missing_ok=True)
        solo_queue.append(job.cfg)
        counts["repeated alone"] += 1
        print(f"retry  {job.cfg.key()}  {why}; repeated later on its own", flush=True)

    try:
        while True:
            now = time.monotonic()
            if stop or (deadline_minutes and now - t0 > 60 * deadline_minutes):
                for job in running.values():
                    _kill(job.proc)
                    finish(job, "stopped", "deadline" if not stop else "interrupted")
                running.clear()
                left = len(queue) + len(solo_queue)
                if left:
                    print(f"{left} runs not started", flush=True)
                break

            for pid, job in list(running.items()):           # finished workers
                rc = job.proc.poll()
                if rc is None:
                    continue
                del running[pid]
                if rc == EXIT["ok"] and result_path(job.cfg).exists():
                    finish(job, "ok")
                elif rc == EXIT["memory"] and not job.solo:
                    again_alone(job, "allocation failed while sharing the machine")
                elif rc in (EXIT["error"], EXIT["time"], EXIT["memory"], EXIT["diverged"]) and \
                        failure_path(job.cfg).exists():
                    finish(job, failure_reason(job.cfg) or "error")
                elif rc == -signal.SIGKILL and not job.solo:
                    again_alone(job, "killed by SIGKILL (the kernel's out-of-memory killer)")
                elif rc == -signal.SIGKILL:
                    write_failure(job.cfg, "memory", "killed by SIGKILL while running alone; the kernel's "
                                  "out-of-memory killer is the usual cause", job.log, peak_mb=job.peak_mb)
                    finish(job, "memory")
                else:
                    write_failure(job.cfg, "error", f"worker exited with code {rc}", job.log, peak_mb=job.peak_mb)
                    finish(job, "error")

            if watch and running:                            # memory budget
                use = {pid: _memory_mb(pid) for pid in running}
                for pid, mb in use.items():
                    running[pid].peak_mb = max(running[pid].peak_mb, mb)
                if sum(use.values()) > budget_mb:
                    pid = max(use, key=use.get)
                    job = running.pop(pid)
                    _kill(job.proc)
                    if result_path(job.cfg).exists():         # it had already finished
                        finish(job, "ok")
                    elif len(use) > 1 and not job.solo:
                        again_alone(job, f"{sum(use.values()) / 1024:.1f} GB in use, this run {use[pid]:.0f} MB")
                    else:
                        write_failure(job.cfg, "memory", f"exceeded the {memory_gb:g} GB memory budget while "
                                      f"running alone ({job.peak_mb:.0f} MB)", job.log, peak_mb=job.peak_mb,
                                      budget_gb=memory_gb)
                        finish(job, "memory")

            if kill_minutes:                                  # wall-clock limit per run
                for pid, job in list(running.items()):
                    if now - job.start > 60 * kill_minutes:
                        del running[pid]
                        _kill(job.proc)
                        write_failure(job.cfg, "time", f"stopped after {kill_minutes:g} minutes in total", job.log,
                                      peak_mb=job.peak_mb, kill_minutes=kill_minutes)
                        finish(job, "time")

            if not any(j.solo for j in running.values()):     # start work
                while queue and len(running) < workers:
                    launch(queue.pop(0), solo=False)
                if not queue and solo_queue and not running:
                    launch(solo_queue.pop(0), solo=True)
            if not running and not queue and not solo_queue:
                break
            time.sleep(poll)
    finally:
        for job in running.values():
            _kill(job.proc)
        for s, h in old_handlers.items():
            signal.signal(s, h)
    return counts


# ----------------------------------------------------------------------------- command line

def pending(cfgs: list[RunConfig], retry_failed: bool = False, previous=()) -> list[RunConfig]:
    """Configurations without a result; failed runs only if their failure may be transient."""
    out = []
    for c in cfgs:
        st = state(c, previous)
        if st == "done" or (st in ("memory", "time", "diverged") and not retry_failed):
            continue
        out.append(c)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count", action="store_true", help="print the number of selected runs and exit")
    ap.add_argument("--prefetch", action="store_true", help="load (download) the datasets of the selection and exit")
    ap.add_argument("--only", default="", help="comma-separated key=value filters; a value may be a|b|c")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th configuration starting at i")
    ap.add_argument("--set", action="append", default=[], help="key=value override for every run")
    ap.add_argument("--workers", type=int, default=0, help="run each configuration in its own process, N at a time")
    ap.add_argument("--memory-gb", type=float, default=None,
                    help="with --workers: memory budget of all workers together (default: 85%% of RAM)")
    ap.add_argument("--run-minutes", type=float, default=0.0,
                    help="training time limit per run (sets max_train_minutes; 0: none)")
    ap.add_argument("--kill-minutes", type=float, default=None,
                    help="with --workers: stop a run after this many minutes in total (default: 1.5 x "
                         "--run-minutes + 30, none without --run-minutes)")
    ap.add_argument("--deadline-minutes", type=float, default=0.0,
                    help="with --workers: stop everything after this many minutes (runs in progress are not recorded)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="repeat runs that failed for memory, time or divergence")
    ap.add_argument("--previous", action="append", default=[],
                    help="directory of earlier results and failure records (scripts/unpack_results.py): runs "
                         "recorded there count as done or failed")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--runner", default=DEFAULT_RUNNER, help=argparse.SUPPRESS)
    ap.add_argument("--poll", type=float, default=0.25, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child is not None:
        sys.exit(child_main(args.child, args.runner))
    if not args.config:
        ap.error("the sweep file is required")

    import tomllib

    with open(args.config, "rb") as fh:
        spec = tomllib.load(fh)
    runs = expand(spec)
    only = dict(kv.split("=", 1) for kv in args.only.split(",") if kv)
    defaults = RunConfig()
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = _parse_value(v, getattr(defaults, k))
    if args.run_minutes:
        overrides["max_train_minutes"] = args.run_minutes
    shard = tuple(int(x) for x in args.shard.split("/"))
    cfgs = select(runs, only, shard, overrides)
    if args.count:
        print(len(cfgs))
        return
    if args.prefetch:
        from .data import load_dataset

        for name, root in sorted({(c.dataset, c.root) for c in cfgs if c.dataset != "synthetic"}):
            t = time.perf_counter()
            data = load_dataset(name, root)
            print(f"{name}: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges "
                  f"({time.perf_counter() - t:.0f} s)", flush=True)
        return
    todo = pending(cfgs, args.retry_failed, args.previous)
    states = [state(c, args.previous) for c in cfgs]
    n_done, n_failed = states.count("done"), sum(st not in (None, "done") for st in states)
    print(f"{len(runs)} runs in {args.config}; this selection: {len(cfgs)}, {n_done} done, {n_failed} failed "
          f"before, {len(todo)} to run")
    if args.dry_run:
        for c in todo[:25]:
            print(f"  {c.key()}  {_describe(c)}")
        return

    if args.workers > 0:
        memory_gb = args.memory_gb if args.memory_gb is not None else 0.85 * total_memory_gb()
        kill = args.kill_minutes if args.kill_minutes is not None else (
            1.5 * args.run_minutes + 30 if args.run_minutes else 0.0)
        counts = run_pool(todo, args.workers, memory_gb, kill, args.deadline_minutes, args.runner, args.poll)
    else:
        fn = _resolve(args.runner)
        counts = {"ok": 0, "memory": 0, "time": 0, "diverged": 0, "error": 0}
        for k, cfg in enumerate(todo, 1):
            print(f"--- [{k}/{len(todo)}] {cfg.key()}  {_describe(cfg)}", flush=True)
            outcome = attempt(cfg, fn)
            counts[outcome] += 1
            if outcome != "ok":
                print(f"failed ({outcome}): {json.loads(failure_path(cfg).read_text())['failure']['detail']}",
                      flush=True)
    print("finished: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v or k == "ok"), flush=True)
    sys.exit(1 if counts.get("error") else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
