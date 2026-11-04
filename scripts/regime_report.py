"""Structure diagnostics per dataset (stages A and B only, no training).

Per dataset: edge homophily; Louvain levels and community counts; purity of the
Louvain communities and of the n-HDP clusters (the accuracy ceiling of
propagation confined to them, see docs/METHOD.md); gate selectivity (mean gate
on same-class vs cross-class edges); super-graph edge reduction r_E; and the
cross-partition ratio rho and its gated counterpart rho_g for --parts parts.

    python scripts/regime_report.py --datasets cora citeseer pubmed photo --parts 14
    python scripts/regime_report.py --datasets cora --set feature_hops=2 --set nhdp_alpha0=0.1
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from nhdp_gnn.config import RunConfig, _parse_value
from nhdp_gnn.data import load_dataset
from nhdp_gnn.gating import gate_weights
from nhdp_gnn.metrics import edge_homophily
from nhdp_gnn.pipeline import run_stages


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "photo"])
    ap.add_argument("--parts", type=int, default=14)
    ap.add_argument("--partition", default="auto")
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="RunConfig key=value")
    args = ap.parse_args(argv)
    defaults = RunConfig()
    extra = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        extra[k] = _parse_value(v, getattr(defaults, k))

    cols = ["dataset", "N", "homophily", "louvain levels", "communities (fine/coarse)", "Louvain purity",
            "n-HDP K", "n-HDP purity", "gate same/cross", "r_E", f"rho ({args.parts} parts)", "rho_g"]
    print("| " + " | ".join(cols) + " |")
    print("|---" * len(cols) + "|")
    for name in args.datasets:
        data = load_dataset(name, args.root)
        cfg = RunConfig(dataset=name, seed=args.seed, num_parts=args.parts, partition=args.partition, **extra)
        st = run_stages(data, cfg.pipeline(), cache_dir=cfg.cache_dir)
        d = st.diagnostics
        gate = ""
        hom = ""
        if not data.multilabel:
            y = np.asarray(data.y)
            src, dst = data.edge_index
            same = y[src] == y[dst]
            g = gate_weights(data.edge_index, st.R)
            gate = f"{g[same].mean():.3f} / {g[~same].mean():.3f}"
            hom = f"{edge_homophily(data.edge_index, y):.3f}"
        cells = [name, data.num_nodes, hom, d.get("louvain_levels"),
                 f"{d.get('num_communities')} / {d.get('num_super_communities')}",
                 f"{d.get('louvain_purity', float('nan')):.3f}", d.get("assign_k"),
                 f"{d.get('assign_purity', float('nan')):.3f}", gate, f"{d.get('edge_reduction', float('nan')):.3f}",
                 f"{d.get('rho', float('nan')):.3f}", f"{d.get('rho_gated', float('nan')):.3f}"]
        print("| " + " | ".join(str(c) for c in cells) + " |", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
