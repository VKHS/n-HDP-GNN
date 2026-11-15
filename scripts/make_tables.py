"""Aggregate a sweep's run results into the article's table layouts.

    python scripts/make_tables.py configs/table2_full.toml --runs runs/table2_full --out results/table2_full

Writes to --out:
  README.md          every table of the sweep (mean ± 95% CI over seeds, n per cell, Wilcoxon
                     p-values against the reference row, failed runs per cell), and the runs
                     that failed or are missing
  <value>.csv        one CSV per reported value, with the per-seed values and failed runs
  runs.jsonl.gz      every run's result JSON, one per line (training curves dropped)
  failures.jsonl.gz  every failure record (runs stopped for memory or time, or that raised)
  summary.json       expected / completed / failed / missing run counts and the code commit of the runs

A cell is reported only from completed runs; nothing is imputed.  Failed and missing
runs are listed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import tomllib
from collections import Counter
from pathlib import Path

from nhdp_gnn.config import RunConfig, _parse_value
from nhdp_gnn.report import aggregate, aggregate_failures, load_failures, load_results, markdown_tables, write_csv
from nhdp_gnn.sweep import expand, select

ACCURACY = dict(rows="model", cols="depth", by="dataset", value="test", reference="nhdp", scale=100.0)
DEPTH_EMBEDDING = [dict(rows="model", cols="depth", by="dataset", value="embedding.centroid_gap", scale=1.0),
                   dict(rows="model", cols="depth", by="dataset", value="embedding.madgap", scale=1.0)]
DEPTH_COST = [dict(rows="model", cols="depth", by="dataset", value="timings.epoch_s_mean", scale=1.0),
              dict(rows="model", cols="depth", by="dataset", value="peak_rss_mb", scale=1.0)]
LAYOUTS = {
    "table1_ablation": [dict(rows="tag", cols="dataset", by="", value="test", reference="full", scale=100.0)],
    "table2_full": [ACCURACY, *DEPTH_EMBEDDING, *DEPTH_COST],
    "table3_semi": [ACCURACY, *DEPTH_EMBEDDING, *DEPTH_COST],
    "table4_scarce": [ACCURACY, *DEPTH_EMBEDDING, *DEPTH_COST],
    "table5_clustering": [dict(rows="model", cols="dataset", by="", value="test", scale=100.0),
                          dict(rows="model", cols="dataset", by="", value="embedding.nmi", scale=1.0),
                          dict(rows="model", cols="dataset", by="", value="embedding.ari", scale=1.0)],
    "fig11_heads": [dict(rows="tag", cols="dataset", by="", value="test", scale=100.0)],
    "ogb_depth": [ACCURACY, *DEPTH_EMBEDDING],
}
UNITS = {"test": "accuracy in %", "timings.epoch_s_mean": "seconds per training epoch (one thread)",
         "peak_rss_mb": "peak resident memory of the run's process in MB"}
REASONS = {
    "memory": "the run needed more memory than the budget while running alone",
    "time": "training provably needed more than the time limit (or the run exceeded the wall-clock limit)",
    "diverged": "the selected model gave non-finite outputs (NaN or infinity)",
    "error": "the run raised an exception",
}


def _split(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def _summary_line(text: str, n: int = 160) -> str:
    """One line of a failure detail: for a traceback the exception line, otherwise the last line."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    exc = [ln for ln in lines if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exceeded|Diverged)\b", ln)]
    line = exc[-1] if exc else (lines[-1] if lines else "")
    return (line[:n] + "...") if len(line) > n else line


def _write_jsonl_gz(path: Path, records) -> None:
    """One JSON object per line, sorted by key; the gzip header carries no timestamp, so the file
    changes only when its content does."""
    lines = [json.dumps(r, sort_keys=True) for r in sorted(records, key=lambda r: r.get("key", ""))]
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write("".join(line + "\n" for line in lines).encode())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the sweep file the runs came from")
    ap.add_argument("--runs", action="append", default=None,
                    help="directory of run records (default: the sweep's out_dir); may be repeated, and for a run "
                         "recorded in several directories the first one given wins")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="the same --only filter given to the sweep")
    ap.add_argument("--set", action="append", default=[], help="the same --set overrides given to the sweep")
    args = ap.parse_args(argv)
    spec = tomllib.loads(Path(args.config).read_text())
    name = spec.get("name", Path(args.config).stem)
    run_dirs = [Path(d) for d in (args.runs or [spec.get("out_dir", f"runs/{name}")])]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    only = dict(kv.split("=", 1) for kv in args.only.split(",") if kv)
    defaults = RunConfig()
    overrides = {k: _parse_value(v, getattr(defaults, k)) for k, v in (kv.split("=", 1) for kv in args.set)}
    expected = {c.key(): c for c in select(expand(spec), only, (0, 1), overrides)}
    found, recorded = {}, {}
    for d in run_dirs:
        for r in load_results(d):
            if r.get("key") in expected:
                found.setdefault(r["key"], r)
        for f in load_failures(d):
            if f.get("key") in expected:
                recorded.setdefault(f["key"], f)
    # sorted, so that the same records always give byte-identical tables
    results, done = [found[k] for k in sorted(found)], set(found)
    failed = {k: recorded[k] for k in sorted(recorded) if k not in done}
    failures = list(failed.values())
    missing = [expected[k] for k in expected if k not in done and k not in failed]
    commits = sorted({r.get("code_commit") for r in results + failures if r.get("code_commit")})
    by_reason = Counter(f["failure"]["reason"] for f in failures)

    selection = "".join(f" --only {args.only}" if args.only else "") + "".join(f" --set {s}" for s in args.set)
    failed_text = f"{len(failed)} failed" + (f" ({', '.join(f'{n} {r}' for r, n in sorted(by_reason.items()))})"
                                             if failed else "")
    lines = [f"# {name}", "",
             f"Sweep `{args.config}{selection}`: {len(expected)} runs expected, {len(done)} completed, "
             f"{failed_text}, {len(missing)} missing.", ""]
    note = spec.get("actions", {}).get("note")
    if note and args.only and args.only == spec.get("actions", {}).get("only"):
        lines += [f"Selection: {note}", ""]
    lines += ["Values are mean ± 95% t-interval over seeds; n is the number of completed seeds in the cell.", ""]
    if failed:
        lines += ["Failed runs are counted per cell by reason: "
                  + "; ".join(f"*{r}*: {REASONS.get(r, r)}" for r in sorted(by_reason)) + ".", ""]
    if commits:
        lines += [f"Code commit(s) of the runs: {', '.join(commits)}", ""]
    for lay in LAYOUTS.get(name, [ACCURACY]):
        rows, cols, by = _split(lay["rows"]), _split(lay["cols"]), _split(lay["by"])
        table = aggregate(results, rows, cols, by, lay["value"])
        bad = aggregate_failures(failures, rows, cols, by)
        unit = f" ({UNITS[lay['value']]})" if lay["value"] in UNITS else ""
        lines += [f"## {lay['value']}{unit}", "",
                  markdown_tables(table, rows, cols, by, lay["value"], lay.get("reference"), lay["scale"], bad)]
        write_csv(table, rows, cols, by, out / f"{lay['value'].replace('.', '_')}.csv", bad)
    if failed:
        lines += ["## Failed runs", "", "| key | dataset | model | tag | depth | seed | reason | detail |",
                  "|---|---|---|---|---|---|---|---|"]
        fields = ("dataset", "model", "tag", "depth", "seed")
        order = [tuple((0, v) if isinstance(v, (int, float)) else (1, str(v)) for v in
                       (f["config"].get(k, "") for k in fields)) for f in failures]
        for _, f in sorted(zip(order, failures), key=lambda t: t[0]):
            c, info = f["config"], f["failure"]
            lines.append(f"| {f['key']} | " + " | ".join(str(c.get(k, "")) for k in fields) + " | "
                         f"{info['reason']} | {_summary_line(info.get('detail', '')).replace('|', '/')} |")
        lines.append("")
    if missing:
        lines += ["## Missing runs", "", "Not attempted, or stopped by the job's deadline.", "",
                  "| key | dataset | model | tag | depth | seed |", "|---|---|---|---|---|---|"]
        lines += [f"| {c.key()} | {c.dataset} | {c.model} | {c.tag} | {c.depth} | {c.seed} |" for c in missing]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    _write_jsonl_gz(out / "runs.jsonl.gz", ({k: v for k, v in r.items() if k != "curves"} for r in results))
    _write_jsonl_gz(out / "failures.jsonl.gz", failures)
    (out / "summary.json").write_text(json.dumps(
        {"sweep": args.config, "name": name, "only": only, "overrides": overrides, "expected": len(expected),
         "completed": len(done), "failed": len(failed), "failed_by_reason": dict(sorted(by_reason.items())),
         "missing": len(missing), "failed_keys": sorted(failed), "missing_keys": [c.key() for c in missing],
         "code_commits": commits, "base": spec.get("base", {}), "grid": spec.get("grid", {})},
        indent=2, default=str) + "\n")
    print(f"{name}: {len(done)}/{len(expected)} runs completed, {len(failed)} failed, {len(missing)} missing -> {out}")


if __name__ == "__main__":
    main()
