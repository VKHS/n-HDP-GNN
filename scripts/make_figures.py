"""Figures from the measured artifacts written by ``python -m nhdp_gnn.artifacts``.

    python scripts/make_figures.py --artifacts artifacts --out artifacts/figures

fig_stage_ab.png        (a) n-HDP ELBO per node by iteration, (b) cluster purity,
                        (c) gate on same- vs cross-class edges, (d) purity bound of the
                        infinite-depth limit for each gate
fig_operator_depth.png  linear-probe accuracy, centroid gap (Eq. 28) and MADGap of P^L X
                        against depth L for the normalised adjacency and three gated operators
fig_partition.png       cut ratio rho and gated cut mass rho_g, and stage A+B wall time,
                        against the number of parts for each partitioning method

Every plotted value is read from the artifact files; nothing is interpolated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["cora", "citeseer", "pubmed", "photo", "computers"]
LABEL = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "photo": "Photo", "computers": "Computers"}
# Okabe-Ito palette (colour-blind safe)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#666666"}
DS_COLOR = dict(zip(ORDER, [C["blue"], C["orange"], C["green"], C["red"], C["purple"]]))
OPS = {"A_hat": ("normalised adjacency", C["grey"], "-"),
       "gated_nhdp_topk": ("gated: n-HDP top-k", C["blue"], "-"),
       "gated_nhdp_hard": ("gated: n-HDP hard", C["sky"], "--"),
       "gated_louvain": ("gated: Louvain", C["orange"], "-")}
METHOD_STYLE = {"community": (C["blue"], "o"), "bfs": (C["orange"], "s"), "random": (C["grey"], "^"),
                "metis": (C["green"], "D")}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "savefig.dpi": 200, "figure.dpi": 100})


def _load(art: Path):
    stage = {p.stem: json.loads(p.read_text()) for p in sorted((art / "stage_ab").glob("*.json"))}
    depth = {p.stem: json.loads(p.read_text()) for p in sorted((art / "operator_depth").glob("*.json"))}
    rows = []
    csv_path = art / "stage_ab" / "partition_study.csv"
    if csv_path.exists():
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
    names = [n for n in ORDER if n in stage] + [n for n in stage if n not in ORDER]
    return names, stage, depth, rows


def _panel_label(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


def fig_stage_ab(names, stage, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
    ax = axes[0]
    for n in names:
        s = stage[n]
        tr = np.asarray(s["stage_b"]["elbo_trace"]) / s["nodes"]
        ax.plot(np.arange(2, tr.size + 2), tr, color=DS_COLOR.get(n), label=LABEL.get(n, n), lw=1.4)
    ax.set_xlabel("CAVI iteration")
    ax.set_ylabel("ELBO / node")
    ax.set_title("n-HDP inference (monotone ELBO)")
    ax.legend(frameon=False)
    _panel_label(ax, "a")

    x = np.arange(len(names))
    w = 0.26
    ax = axes[1]
    hom = [stage[n]["edge_homophily"] for n in names]
    lou = [stage[n]["purity"]["louvain_super_communities"] for n in names]
    nh = [stage[n]["purity"]["nhdp_hard_clusters"] for n in names]
    ax.bar(x - w, hom, w, color=C["grey"], label="edge homophily")
    ax.bar(x, lou, w, color=C["orange"], label="Louvain purity")
    ax.bar(x + w, nh, w, color=C["blue"], label="n-HDP purity (hard)")
    ax.set_xticks(x, [LABEL.get(n, n) for n in names])
    ax.set_ylim(0, 1.32)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title("Label agreement of the structure")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    _panel_label(ax, "b")

    ax = axes[2]
    g = [stage[n]["gates"]["nhdp_topk"] for n in names]
    top = max(max(v["mean_same"], v["mean_cross"]) for v in g)
    ax.bar(x - w / 2, [v["mean_same"] for v in g], w, color=C["blue"], label="same-class edges")
    ax.bar(x + w / 2, [v["mean_cross"] for v in g], w, color=C["red"], label="cross-class edges")
    for i, v in enumerate(g):
        ax.text(i, max(v["mean_same"], v["mean_cross"]) + 0.03 * top, f"{v['cross_to_same']:.2f}", ha="center",
                fontsize=7.5, color=C["grey"])
    ax.set_xticks(x, [LABEL.get(n, n) for n in names])
    ax.set_ylim(0, 1.45 * top)
    ax.set_ylabel(r"mean gate $\langle\pi_i,\pi_j\rangle$")
    ax.set_title("Gate selectivity, n-HDP top-k (number: cross/same)")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    _panel_label(ax, "c")

    ax = axes[3]
    keys = [("no_gate", "no gate", C["grey"]), ("louvain", "Louvain gate", C["orange"]),
            ("nhdp_topk", "n-HDP top-k gate", C["blue"]), ("nhdp_hard", "n-HDP hard gate", C["sky"])]
    w4 = 0.2
    for k, (key, lab, col) in enumerate(keys):
        vals = [stage[n]["depth_limit"][key]["purity_bound"] for n in names]
        ax.bar(x + (k - 1.5) * w4, vals, w4, color=col, label=lab)
    ax.set_xticks(x, [LABEL.get(n, n) for n in names])
    ax.set_ylim(0, 1.32)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("purity of gate components")
    ax.set_title("Accuracy bound of the infinite-depth limit")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    _panel_label(ax, "d")
    fig.savefig(out / "fig_stage_ab.png")
    plt.close(fig)


def fig_operator_depth(names, depth, out: Path):
    names = [n for n in names if n in depth]
    if not names:
        return
    metrics = [("probe_test_mean", "linear-probe test accuracy"), ("centroid_gap", "centroid gap (Eq. 28)"),
               ("madgap", "MADGap")]
    fig, axes = plt.subplots(len(metrics), len(names), figsize=(3.1 * len(names), 2.4 * len(metrics)),
                             constrained_layout=True, squeeze=False)
    for j, n in enumerate(names):
        d = depth[n]
        L = np.asarray(d["depths"], dtype=float)
        xs = np.where(L == 0, 0.5, L)                  # depth 0 drawn at x = 0.5 on the log axis
        for i, (key, ylab) in enumerate(metrics):
            ax = axes[i, j]
            for op, (lab, col, ls) in OPS.items():
                if op not in d["operators"]:
                    continue
                rec = d["operators"][op]
                y = np.asarray(rec[key], dtype=float)
                ax.plot(xs, y, ls, color=col, marker="o", ms=3, lw=1.4, label=lab)
                if key == "probe_test_mean":
                    ci = np.asarray(rec["probe_test_ci95"], dtype=float)
                    if np.isfinite(ci).any():
                        ax.fill_between(xs, y - ci, y + ci, color=col, alpha=0.15, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, ["0" if v == 0 else str(int(v)) for v in L])
            ax.minorticks_off()
            if i == 0:
                split = "public split" if d["splits"] == ["public"] else f"{len(d['splits'])} semi splits, 95% CI"
                ax.set_title(f"{LABEL.get(n, n)} ({split})")
            if i == len(metrics) - 1:
                ax.set_xlabel("depth L (propagation steps)")
            if j == 0:
                ax.set_ylabel(ylab)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    fig.savefig(out / "fig_operator_depth.png")
    plt.close(fig)


def fig_partition(names, rows, out: Path):
    if not rows:
        return
    names = [n for n in names if any(r["dataset"] == n for r in rows)]
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 5.0), constrained_layout=True, squeeze=False)
    for j, n in enumerate(names):
        rs = [r for r in rows if r["dataset"] == n]
        methods = [m for m in METHOD_STYLE if any(r["method"] == m for r in rs)]
        for m in methods:
            col, mk = METHOD_STYLE[m]
            sel = sorted((r for r in rs if r["method"] == m), key=lambda r: int(r["parts"]))
            P = [int(r["parts"]) for r in sel]
            axes[0, j].plot(P, [float(r["rho"]) for r in sel], "-", color=col, marker=mk, ms=4,
                            label=rf"{m}: $\rho$")
            axes[0, j].plot(P, [float(r["rho_gated"]) for r in sel], "--", color=col, marker=mk, ms=4,
                            mfc="white", label=rf"{m}: $\rho_g$")
            axes[1, j].plot(P, [float(r["wall_s"]) for r in sel], "-", color=col, marker=mk, ms=4, label=m)
        axes[0, j].set_title(LABEL.get(n, n))
        axes[0, j].set_ylim(0, 1)
        axes[1, j].set_xlabel("parts P")
        for ax in axes[:, j]:
            ax.set_xticks(sorted({int(r["parts"]) for r in rs}))
    axes[0, 0].set_ylabel("share of edges / gated mass cut")
    axes[1, 0].set_ylabel("stage A+B wall time (s, 1 process)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    fig.savefig(out / "fig_partition.png")
    plt.close(fig)


def _f(v, nd=3):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def summary_markdown(names, stage, depth, rows, manifest) -> str:
    """Markdown tables generated from the artifact files (no hand-copied numbers)."""
    L = []
    if manifest:
        L += [f"Generated by `{manifest.get('command')}` at code commit `{manifest.get('code_commit')}` "
              f"({manifest.get('generated_utc')}; {manifest.get('cpu_count')} CPUs, {manifest.get('platform')}; "
              f"Louvain backend: {manifest.get('louvain_backend')}; METIS available: "
              f"{manifest.get('metis_available')}).", ""]
    L += ["## Stages A and B (one part)", "",
          "| graph | nodes | edges | classes | edge homophily | communities (fine / super) | n-HDP iterations "
          "(converged) | atoms used | purity Louvain | purity n-HDP (hard) | r_E | stage A (s) | stage B (s) |",
          "|---" * 13 + "|"]
    for n in names:
        s = stage[n]
        a, b = s["stage_a"], s["stage_b"]
        L.append(f"| {LABEL.get(n, n)} | {s['nodes']} | {s['edges']} | {s['classes']} | {_f(s['edge_homophily'])} | "
                 f"{a['communities']['count']} / {a['super_communities']['count']} | {b['iterations']} "
                 f"({'yes' if b['converged'] else 'no'}) | {b['atoms_used_after_topk']} | "
                 f"{_f(s['purity']['louvain_super_communities'])} | {_f(s['purity']['nhdp_hard_clusters'])} | "
                 f"{_f(s['edge_reduction_r_E'])} | {_f(a['time_s'], 1)} | {_f(b['time_s'], 1)} |")
    L += ["", "## Gates: selectivity and the purity bound of the infinite-depth limit", "",
          "Mean gate on same-class and cross-class edges, the class homophily of the gated mass "
          "(edge homophily for no gate), and the purity of the gate-connected components with their number.", "",
          "| graph | gate | mean gate same-class | mean gate cross-class | cross / same | gated homophily | "
          "components | purity bound |", "|---" * 8 + "|"]
    gate_names = [("no_gate", "none"), ("nhdp_topk", "n-HDP top-k"), ("nhdp_hard", "n-HDP hard"),
                  ("louvain", "Louvain")]
    for n in names:
        s = stage[n]
        for key, lab in gate_names:
            lim = s["depth_limit"][key]
            if key == "no_gate":
                L.append(f"| {LABEL.get(n, n)} | {lab} | 1 | 1 | 1 | {_f(s['edge_homophily'])} | "
                         f"{lim['components']} | {_f(lim['purity_bound'])} |")
                continue
            g = s["gates"][key]
            L.append(f"| {LABEL.get(n, n)} | {lab} | {_f(g['mean_same'])} | {_f(g['mean_cross'])} | "
                     f"{_f(g['cross_to_same'], 2)} | {_f(g['gated_homophily'])} | {lim['components']} | "
                     f"{_f(lim['purity_bound'])} |")
    names_d = [n for n in names if n in depth]
    if names_d:
        ds = depth[names_d[0]]["depths"]
        L += ["", "## Training-free depth study: linear-probe test accuracy of P^L X", "",
              "Planetoid graphs: public split (one value). Amazon graphs: mean ± 95% CI over the semi-supervised "
              "splits listed in the JSON.", "",
              "| graph | operator | " + " | ".join(f"L={d}" for d in ds) + " |", "|---" * (len(ds) + 2) + "|"]
        for n in names_d:
            for op, (lab, _, _) in OPS.items():
                rec = depth[n]["operators"].get(op)
                if rec is None:
                    continue
                cells = []
                for m, h in zip(rec["probe_test_mean"], rec["probe_test_ci95"]):
                    cells.append(f"{100 * m:.1f}" + (f" ± {100 * h:.1f}" if h is not None and np.isfinite(h) else ""))
                L.append(f"| {LABEL.get(n, n)} | {lab} | " + " | ".join(cells) + " |")
    if rows:
        L += ["", "## Partition study (single process)", "",
              "| graph | parts | method | rho | rho_g | r_E | n-HDP purity | largest part / mean | wall (s) |",
              "|---" * 9 + "|"]
        order = {m: i for i, m in enumerate(METHOD_STYLE)}
        for r in sorted(rows, key=lambda r: (ORDER.index(r["dataset"]) if r["dataset"] in ORDER else 99,
                                             int(r["parts"]), order.get(r["method"], 99))):
            L.append(f"| {LABEL.get(r['dataset'], r['dataset'])} | {r['parts']} | {r['method']} | "
                     f"{_f(float(r['rho']))} | {_f(float(r['rho_gated']))} | {_f(float(r['edge_reduction_r_E']))} | "
                     f"{_f(float(r['nhdp_purity']))} | {_f(float(r['largest_part_over_mean']), 2)} | "
                     f"{_f(float(r['wall_s']), 1)} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default=None, help="default: <artifacts>/figures")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)
    out = Path(args.out) if args.out else art / "figures"
    out.mkdir(parents=True, exist_ok=True)
    names, stage, depth, rows = _load(art)
    if not names:
        raise SystemExit(f"no stage_ab/*.json under {art}")
    fig_stage_ab(names, stage, out)
    fig_operator_depth(names, depth, out)
    fig_partition(names, rows, out)
    manifest = json.loads((art / "manifest.json").read_text()) if (art / "manifest.json").exists() else None
    (art / "SUMMARY.md").write_text("# Artifact summary\n\n" + summary_markdown(names, stage, depth, rows, manifest))
    print(f"figures written to {out}; tables to {art / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
