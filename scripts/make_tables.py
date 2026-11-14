"""Aggregate a sweep's run results into the article's table layouts.

    python scripts/make_tables.py configs/table2_full.toml --runs runs/table2_full --out results/table2_full

Writes to --out:
  README.md       every table of the sweep (mean ± 95% CI over seeds, n per cell, Wilcoxon
                  p-values against the reference row), and how many expected runs are missing
  <value>.csv     one CSV per reported value, with the per-seed values
  runs.jsonl.gz   every run's result JSON, one per line (training curves dropped)
  summary.json    expected / completed / missing run counts and the code commit of the runs

A cell is reported only from completed runs; nothing is imputed.  Missing runs are listed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import tomllib
from pathlib import Path

from nhdp_gnn.config import RunConfig, _parse_value
from nhdp_gnn.report import aggregate, load_results, markdown_tables, write_csv
from nhdp_gnn.sweep import expand, select

ACCURACY = dict(rows="model", cols="depth", by="dataset", value="test", reference="nhdp", scale=100.0)
DEPTH_EMBEDDING = [dict(rows="model", cols="depth", by="dataset", value="embedding.centroid_gap", scale=1.0),
                   dict(rows="model", cols="depth", by="dataset", value="embedding.madgap", scale=1.0)]
LAYOUTS = {
    "table1_ablation": [dict(rows="tag", cols="dataset", by="", value="test", reference="full", scale=100.0)],
    "table2_full": [ACCURACY, *DEPTH_EMBEDDING],
    "table3_semi": [ACCURACY, *DEPTH_EMBEDDING],
    "table4_scarce": [ACCURACY, *DEPTH_EMBEDDING],
    "table5_clustering": [dict(rows="model", cols="dataset", by="", value="test", scale=100.0),
                          dict(rows="model", cols="dataset", by="", value="embedding.nmi", scale=1.0),
                          dict(rows="model", cols="dataset", by="", value="embedding.ari", scale=1.0)],
    "fig11_heads": [dict(rows="tag", cols="dataset", by="", value="test", scale=100.0)],
    "ogb_depth": [ACCURACY, *DEPTH_EMBEDDING],
}


def _split(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the sweep file the runs came from")
    ap.add_argument("--runs", default=None, help="directory of run JSON files (default: the sweep's out_dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="the same --only filter given to the sweep")
    ap.add_argument("--set", action="append", default=[], help="the same --set overrides given to the sweep")
    args = ap.parse_args(argv)
    spec = tomllib.loads(Path(args.config).read_text())
    name = spec.get("name", Path(args.config).stem)
    runs_dir = Path(args.runs or spec.get("out_dir", f"runs/{name}"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    only = dict(kv.split("=", 1) for kv in args.only.split(",") if kv)
    defaults = RunConfig()
    overrides = {k: _parse_value(v, getattr(defaults, k)) for k, v in (kv.split("=", 1) for kv in args.set)}
    expected = {c.key(): c for c in select(expand(spec), only, (0, 1), overrides)}
    results = [r for r in load_results(runs_dir) if r.get("key") in expected]
    done = {r["key"] for r in results}
    missing = [expected[k] for k in expected if k not in done]
    commits = sorted({r.get("code_commit") for r in results if r.get("code_commit")})

    selection = "".join(f" --only {args.only}" if args.only else "") + "".join(f" --set {s}" for s in args.set)
    lines = [f"# {name}", "",
             f"Sweep `{args.config}{selection}`: {len(expected)} runs expected, {len(done)} completed, "
             f"{len(missing)} missing.",
             "Values are mean ± 95% t-interval over seeds (n = completed seeds per cell); accuracy in %.", ""]
    if commits:
        lines += [f"Code commit(s) of the runs: {', '.join(commits)}", ""]
    for lay in LAYOUTS.get(name, [ACCURACY]):
        rows, cols, by = _split(lay["rows"]), _split(lay["cols"]), _split(lay["by"])
        table = aggregate(results, rows, cols, by, lay["value"])
        lines += [f"## {lay['value']}", "",
                  markdown_tables(table, rows, cols, by, lay["value"], lay.get("reference"), lay["scale"])]
        write_csv(table, rows, cols, by, out / f"{lay['value'].replace('.', '_')}.csv")
    if missing:
        lines += ["## Missing runs", "", "| key | dataset | model | tag | depth | seed |", "|---|---|---|---|---|---|"]
        lines += [f"| {c.key()} | {c.dataset} | {c.model} | {c.tag} | {c.depth} | {c.seed} |" for c in missing]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    with gzip.open(out / "runs.jsonl.gz", "wt") as fh:
        for r in sorted(results, key=lambda r: r["key"]):
            fh.write(json.dumps({k: v for k, v in r.items() if k != "curves"}, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"sweep": args.config, "name": name, "only": only, "overrides": overrides, "expected": len(expected),
         "completed": len(done), "missing": len(missing), "missing_keys": [c.key() for c in missing],
         "code_commits": commits, "base": spec.get("base", {}), "grid": spec.get("grid", {})},
        indent=2, default=str) + "\n")
    print(f"{name}: {len(done)}/{len(expected)} runs -> {out}")


if __name__ == "__main__":
    main()
