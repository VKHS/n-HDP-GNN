#!/usr/bin/env python
"""Compare the training runs of ``examples/demo.py`` made by different runs of the ``demo`` workflow.

GitHub assigns the CPU model of a runner, so two runs of the workflow can train on different CPUs. This script puts
the result records (``runs.jsonl.gz``) of several runs side by side, run by run (same variant, depth and seed), writes
the test accuracy, validation accuracy and best epoch of each into one JSON file and prints how the runs differ.

    python scripts/compare_demo_runs.py --out examples/cpu_comparison.json \\
        44323001112=path/to/runs.jsonl.gz 44325763931=path/to/runs.jsonl.gz 44330815122=examples/output/runs.jsonl.gz

Each argument is ``RUN_ID=PATH``. The CPU model and the code commit are read from the records themselves, and a run
whose records name more than one is refused, as is a set of runs that did not train the same configurations.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

RUN_URL = "https://github.com/VKHS/n-HDP-GNN/actions/runs/{}"
FIELDS = ("test", "val", "best_epoch")


def read_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _one(values: set, what: str, run_id: str):
    if len(values) != 1:
        raise SystemExit(f"run {run_id}: the records name {len(values)} different {what}: {sorted(map(str, values))}")
    return next(iter(values))


def build(inputs: list[tuple[str, Path]]) -> dict:
    """The comparison of the runs in ``inputs`` ([(run id, runs.jsonl.gz)]) as a JSON-ready dictionary."""
    runs, tables, order = [], [], {}
    for run_id, path in inputs:
        records = read_records(path)
        cpu = _one({(r.get("environment") or {}).get("cpu_model") for r in records}, "CPU models", run_id)
        commit = _one({r.get("code_commit") for r in records}, "code commits", run_id)
        if cpu is None:
            raise SystemExit(f"run {run_id}: the records do not name the CPU model")
        table = {}
        for r in records:
            c = r["config"]
            cell = (c["tag"], int(c["depth"]), int(c["seed"]))
            if cell in table:
                raise SystemExit(f"run {run_id}: two records for {cell}")
            table[cell] = {f: r[f] for f in FIELDS}
            order.setdefault(c["tag"], len(order))
        runs.append({"run_id": run_id, "url": RUN_URL.format(run_id), "cpu_model": cpu, "code_commit": commit})
        tables.append(table)
    cells = set(tables[0])
    for (run_id, _), table in zip(inputs, tables):
        if set(table) != cells:
            raise SystemExit(f"run {run_id} trained other configurations than run {inputs[0][0]}")
    rows = [{"variant": v, "depth": d, "seed": s, **{f: [t[(v, d, s)][f] for t in tables] for f in FIELDS}}
            for v, d, s in sorted(cells, key=lambda c: (order[c[0]], c[1], c[2]))]
    return {"description": "Per-run results of examples/demo.py in several runs of the demo workflow. For every field "
                           "the list holds one value per run, in the order of 'runs'. 'test' and 'val' are accuracies "
                           "as fractions. Made by scripts/compare_demo_runs.py from the runs.jsonl.gz of each run.",
            "runs": runs, "rows": rows}


def differing(data: dict, a: int, b: int, field: str = "test") -> list[dict]:
    """The rows whose ``field`` differs between run ``a`` and run ``b`` (indices into ``data['runs']``)."""
    return [r for r in data["rows"] if r[field][a] != r[field][b]]


def cell_means(data: dict, run: int, field: str = "test") -> dict:
    """{(variant, depth): mean of ``field`` over the seeds, in percent} for one run."""
    cells: dict = {}
    for r in data["rows"]:
        cells.setdefault((r["variant"], r["depth"]), []).append(100.0 * r[field][run])
    return {k: sum(v) / len(v) for k, v in cells.items()}


def summary(data: dict) -> str:
    runs, n = data["runs"], len(data["rows"])
    lines = [f"{n} training runs in each of {len(runs)} runs of the workflow:"]
    for i, r in enumerate(runs):
        lines.append(f"  [{i}] {r['run_id']}  {r['cpu_model']}  code commit {r['code_commit'][:7]}")
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            lines.append(f"[{a}] against [{b}]:")
            for field in FIELDS:
                lines.append(f"  {field:10s} differs in {len(differing(data, a, b, field)):3d} of {n} runs")
            diff = differing(data, a, b)
            if diff:
                by_depth = {d: sum(1 for r in diff if r["depth"] == d) for d in sorted({r["depth"] for r in data["rows"]})}
                total = {d: sum(1 for r in data["rows"] if r["depth"] == d) for d in by_depth}
                lines.append("  test differs by depth: " + ", ".join(f"{d}: {by_depth[d]} of {total[d]}" for d in by_depth))
                gaps = sorted(((abs(r["test"][a] - r["test"][b]) * 100, r) for r in diff), key=lambda t: -t[0])
                lines.append("  largest differences in test accuracy (points): " + "; ".join(
                    f"{g:.1f} ({r['variant']}, depth {r['depth']}, seed {r['seed']}: "
                    f"{100 * r['test'][a]:.1f} against {100 * r['test'][b]:.1f})" for g, r in gaps[:4]))
                means_a, means_b = cell_means(data, a), cell_means(data, b)
                moved = [(k, means_a[k], means_b[k]) for k in means_a if means_a[k] != means_b[k]]
                lines.append(f"  mean test accuracy over the seeds differs in {len(moved)} of {len(means_a)} cells: " + "; ".join(
                    f"{k[0]} depth {k[1]}: {x:.1f} against {y:.1f}" for k, x, y in sorted(moved, key=lambda t: -abs(t[1] - t[2]))))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="RUN_ID=PATH", help="a workflow run and its runs.jsonl.gz")
    ap.add_argument("--out", type=Path, help="write the comparison to this JSON file")
    args = ap.parse_args(argv)
    inputs = []
    for item in args.runs:
        run_id, sep, path = item.partition("=")
        if not sep or not run_id.isdigit():
            ap.error(f"{item!r}: expected RUN_ID=PATH with a numeric run id")
        inputs.append((run_id, Path(path)))
    if len(inputs) < 2:
        ap.error("give at least two runs")
    data = build(inputs)
    if args.out:
        args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(summary(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
