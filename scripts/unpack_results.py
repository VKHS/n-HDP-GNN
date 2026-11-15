"""Unpack a results directory (written by scripts/make_tables.py) into per-run records.

    python scripts/unpack_results.py results/table2_full previous/table2_full

Writes ``<key>.json`` for every line of runs.jsonl.gz and ``<key>.failed.json`` for
every line of failures.jsonl.gz, the layout ``nhdp_gnn.sweep --previous`` and
``make_tables.py --runs`` read.  The reproduce workflow uses it to continue a
sweep from its results branch: runs recorded there are not repeated.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def unpack(results_dir, out_dir) -> tuple[int, int]:
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for name, suffix in (("runs.jsonl.gz", ".json"), ("failures.jsonl.gz", ".failed.json")):
        n = 0
        if (results_dir / name).exists():
            with gzip.open(results_dir / name, "rt") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        (out_dir / f"{rec['key']}{suffix}").write_text(json.dumps(rec))
                        n += 1
        counts.append(n)
    return counts[0], counts[1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)
    runs, failures = unpack(args.results_dir, args.out_dir)
    print(f"{args.results_dir}: {runs} results, {failures} failure records -> {args.out_dir}")


if __name__ == "__main__":
    main()
