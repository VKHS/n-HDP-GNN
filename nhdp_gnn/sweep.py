"""Expand a sweep file (TOML) into runs and execute the missing ones.

    python -m nhdp_gnn.sweep configs/table2_full.toml --dry-run
    python -m nhdp_gnn.sweep configs/table2_full.toml --only model=nhdp,dataset=cora --shard 3/16
    python -m nhdp_gnn.sweep configs/table2_full.toml --only "seed=0|1|2"

A sweep file has ``name``, a ``[base]`` table of RunConfig values, a ``[grid]``
table of lists (Cartesian product) and optional ``[[variant]]`` tables (named
overrides, stored as the run's ``tag``).  Results are written to
runs/<name>/<key>.json; runs whose result exists are skipped, so a sweep can be
resumed, or split across cluster jobs with --shard i/n.  An ``--only`` value may
list alternatives separated by ``|``.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path

from .config import RunConfig, _parse_value, config_from_dict


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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated key=value filters; a value may be a|b|c")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th configuration starting at i")
    ap.add_argument("--set", action="append", default=[], help="key=value override for every run")
    args = ap.parse_args(argv)

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
    shard = tuple(int(x) for x in args.shard.split("/"))
    cfgs = select(runs, only, shard, overrides)
    todo = [c for c in cfgs if not (Path(c.out_dir) / f"{c.key()}.json").exists()]
    print(f"{len(runs)} runs in {args.config}; this selection: {len(cfgs)}, {len(cfgs) - len(todo)} done, "
          f"{len(todo)} to run")
    if args.dry_run:
        for c in todo[:25]:
            print(f"  {c.key()}  {c.dataset} {c.regime} {c.model} depth={c.depth} seed={c.seed} {c.tag}")
        return

    from .train import run

    failures = 0
    for k, cfg in enumerate(todo, 1):
        print(f"--- [{k}/{len(todo)}] {cfg.key()}", flush=True)
        try:
            run(cfg)
        except Exception:                                   # keep going; report at the end
            failures += 1
            traceback.print_exc()
    print(f"finished: {len(todo) - failures} ok, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
