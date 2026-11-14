# Changelog

## 2.0.0

The repository now implements the pipeline described in the article.

- `nhdp_gnn` package:
  - stage A: one-time edge cut (METIS, community packing, BFS, random) and nested Louvain communities;
  - stage B: nested-HDP inference (coordinate ascent with an exact, monotone ELBO), sharded across parts;
  - stage C: π-gated node attention with community- and global-level attention; residual variants;
  - single-process and DistributedDataParallel training with communication accounting;
  - baselines at any depth (GCN, GAT, SGC, APPNP, ARMA, GCNII, PairNorm, DropEdge, Graph U-Net);
  - metrics (centroid gap of Eq. 28, topological MADGap, k-means NMI/ARI, ρ, ρ_g, r_E, purity).
- Train/validation/test protocol for the three regimes (full, semi, scarce) with early stopping
  that restores the selected state; results written per run as JSON.
- One sweep file per table and figure (`configs/`), resumable and shardable; `nhdp_gnn.report`
  and `scripts/make_tables.py` aggregate seeds into mean ± 95% CI with paired Wilcoxon tests.
- Planetoid and Amazon graphs are read from the original raw files, so stages A and B run without
  PyTorch; equality with PyTorch Geometric's processing is tested.
- Measured artifacts (`artifacts/`): stage A/B diagnostics, partition study and a training-free
  depth study on five graphs, with figures and a provenance manifest.
- GitHub Actions: tests on every push; a manual workflow that runs a sweep on CPU runners and
  publishes its tables.
- Documentation: `docs/METHOD.md` (equations, symbol map, depth-limit proposition, open choices),
  `docs/REPRODUCING.md` (commands per table, protocol, cluster and Actions runs).

## 1.0 (May 2025)

First release: `MAIN-CORA.py`, `MAIN-CITESEER.py`, `MAIN-NETWORKX.py` and `MAIN-PHOTOS.py`.
They are kept in `legacy/` (MAIN-PHOTOS.py in the git history); `legacy/README.md` describes
what they implement.
