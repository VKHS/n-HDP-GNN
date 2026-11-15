"""Aggregate run results into tables: mean +- 95% CI over seeds, paired tests, scaling.

    python -m nhdp_gnn.report runs/table2_full --by dataset --rows model --cols depth --value test
    python -m nhdp_gnn.report runs/table2_full --value embedding.centroid_gap --reference nhdp
    python -m nhdp_gnn.report runs/hpc --scaling

``--rows/--cols/--by`` name RunConfig fields (or ``tag``); ``--value`` is a dotted
path into the result JSON.  ``--reference ROW`` adds two-sided Wilcoxon p-values of
every row against ROW, pairing runs by seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import mean_ci, paired_wilcoxon


def _records(path):
    for f in sorted(Path(path).rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and "config" in d:
            yield d


def load_results(path) -> list[dict]:
    """Run results under ``path`` (failure records are left out)."""
    return [d for d in _records(path) if "failure" not in d]


def load_failures(path) -> list[dict]:
    """Failure records (``<key>.failed.json``, written by ``nhdp_gnn.sweep``) under ``path``."""
    return [d for d in _records(path) if "failure" in d]


def field_of(result: dict, name: str):
    if name in result.get("config", {}):
        return result["config"][name]
    return result.get(name)


def value_of(result: dict, dotted: str):
    cur = result
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _key(result, names):
    return tuple(field_of(result, n) for n in names) if names else ()


def aggregate(results, rows, cols, by, value):
    """{by_key: {(row_key, col_key): {seed: value}}}"""
    table = defaultdict(lambda: defaultdict(dict))
    for r in results:
        v = value_of(r, value)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        table[_key(r, by)][(_key(r, rows), _key(r, cols))][field_of(r, "seed")] = float(v)
    return table


def aggregate_failures(failures, rows, cols, by):
    """{by_key: {(row_key, col_key): {reason: count}}}"""
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in failures:
        table[_key(r, by)][(_key(r, rows), _key(r, cols))][r["failure"]["reason"]] += 1
    return table


def _failed_text(counts) -> str:
    return ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items()))


def _fmt_key(k):
    return " / ".join(str(x) for x in k) if k else "-"


def _order(k):
    return tuple((0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in k)


def _sort(keys):
    return sorted(keys, key=_order)


def markdown_tables(table, rows, cols, by, value, reference=None, scale=1.0, failures=None) -> str:
    """Mean ± 95% CI (n) per cell; ``failures`` (from ``aggregate_failures``) adds failed runs by reason."""
    failures = failures or {}
    lines = []
    for bk in _sort(set(table) | set(failures)):
        cells, failed = table.get(bk, {}), failures.get(bk, {})
        row_keys = _sort({rk for rk, _ in cells} | {rk for rk, _ in failed})
        col_keys = _sort({ck for _, ck in cells} | {ck for _, ck in failed})
        title = f"**{_fmt_key(bk)}** -- {value}" if by else f"**{value}**"
        lines += [title, "", "| " + " / ".join(rows) + " | " + " | ".join(_fmt_key(c) for c in col_keys) + " |",
                  "|---" * (len(col_keys) + 1) + "|"]
        for rk in row_keys:
            out = []
            for ck in col_keys:
                seeds, bad = cells.get((rk, ck), {}), failed.get((rk, ck), {})
                if not seeds:
                    out.append(f"failed: {_failed_text(bad)}" if bad else "")
                    continue
                m, h = mean_ci(list(seeds.values()))
                ci = "" if np.isnan(h) else f" ± {h * scale:.2f}"
                extra = f"; failed: {_failed_text(bad)}" if bad else ""
                out.append(f"{m * scale:.2f}{ci} (n={len(seeds)}{extra})")
            lines.append(f"| {_fmt_key(rk)} | " + " | ".join(out) + " |")
        if reference is not None:
            ref = (reference,) if not isinstance(reference, tuple) else reference
            lines += ["", f"Wilcoxon p-value vs {_fmt_key(ref)} (paired by seed)", "",
                      "| " + " / ".join(rows) + " | " + " | ".join(_fmt_key(c) for c in col_keys) + " |",
                      "|---" * (len(col_keys) + 1) + "|"]
            for rk in row_keys:
                if rk == ref:
                    continue
                ps = []
                for ck in col_keys:
                    a, b = cells.get((rk, ck), {}), cells.get((ref, ck), {})
                    common = sorted(set(a) & set(b))
                    p = paired_wilcoxon([a[s] for s in common], [b[s] for s in common]) if common else float("nan")
                    ps.append("" if np.isnan(p) else f"{p:.3g}")
                lines.append(f"| {_fmt_key(rk)} | " + " | ".join(ps) + " |")
        lines.append("")
    return "\n".join(lines)


def scaling_table(results) -> str:
    """Speedup S(p) = T1/Tp and efficiency E(p) = S(p)/p from distributed runs."""
    groups = defaultdict(dict)
    for r in results:
        if "systems" not in r:
            continue
        c = r["config"]
        groups[(c["dataset"], c["model"], c["depth"], c.get("tag", ""))][r["world_size"]] = r
    lines = []
    for g in sorted(groups, key=str):
        runs = groups[g]
        t1 = runs[1]["systems"]["time_per_epoch_s"] if 1 in runs else None
        lines += [f"**{g[0]} -- {g[1]} depth {g[2]} {g[3]}**", "",
                  "| p | time/epoch (s) | S(p) | E(p) | edges/s | Comms% | all-reduce (ms) | bytes/step | rho | r_E | test |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in sorted(runs):
            s, r = runs[p]["systems"], runs[p]
            sp = t1 / s["time_per_epoch_s"] if t1 else float("nan")
            er = s.get("edge_reduction")
            lines.append(
                f"| {p} | {s['time_per_epoch_s']:.3f} | {sp:.2f} | {sp / p:.2f} | {s['edges_per_s']:.3g} | "
                f"{100 * s['comms_share']:.1f} | {1000 * s['allreduce_s_per_step']:.1f} | "
                f"{s['bytes_per_step_payload']:.3g} | {s['rho']:.3f} | "
                f"{'' if er is None else f'{er:.3f}'} | {r['test']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def write_csv(table, rows, cols, by, path, failures=None) -> None:
    """One row per cell: mean, CI half-width, n, per-seed values and failed runs by reason (JSON)."""
    failures = failures or {}
    keys = {(bk, cell) for bk, cells in table.items() for cell in cells}
    keys |= {(bk, cell) for bk, cells in failures.items() for cell in cells}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(by) + list(rows) + list(cols) + ["mean", "ci95", "n", "seeds", "failed"])
        for bk, (rk, ck) in sorted(keys, key=lambda k: _order(k[0] + k[1][0] + k[1][1])):
            seeds = table.get(bk, {}).get((rk, ck), {})
            m, h = mean_ci(list(seeds.values())) if seeds else (float("nan"), float("nan"))
            bad = dict(failures.get(bk, {}).get((rk, ck), {}))
            w.writerow(list(bk) + list(rk) + list(ck) + [m, h, len(seeds), json.dumps(seeds),
                                                         json.dumps(bad) if bad else ""])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--rows", default="model")
    ap.add_argument("--cols", default="depth")
    ap.add_argument("--by", default="dataset")
    ap.add_argument("--value", default="test")
    ap.add_argument("--scale", type=float, default=100.0, help="multiply values (100 for percentages)")
    ap.add_argument("--reference", default=None, help="row value to test the others against")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--scaling", action="store_true")
    args = ap.parse_args(argv)
    results = load_results(args.path)
    if args.scaling:
        print(scaling_table(results))
        return
    rows = [x for x in args.rows.split(",") if x]
    cols = [x for x in args.cols.split(",") if x]
    by = [x for x in args.by.split(",") if x]
    table = aggregate(results, rows, cols, by, args.value)
    print(markdown_tables(table, rows, cols, by, args.value, args.reference, args.scale))
    if args.csv:
        write_csv(table, rows, cols, by, args.csv)


if __name__ == "__main__":
    main(sys.argv[1:])
