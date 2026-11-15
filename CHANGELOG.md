# Changelog

## 2.1.0

- `nhdp_gnn.sweep --workers N`: every run executes in a process of its own, N at a time, within a
  memory budget (`--memory-gb`). A run that does not fit next to the others is repeated on its own;
  a run that needs more than the budget alone, or whose training provably needs more than
  `--run-minutes`, leaves a failure record (`<key>.failed.json`, reason memory, time or error)
  instead of a result. Results of runs in their own process include `peak_rss_mb`.
- A run whose selected model gives non-finite outputs is recorded as failed with reason
  `diverged` instead of failing later in the embedding diagnostics (Graph U-Net with 10 or 15
  pooling levels on Cora and CiteSeer).
- `max_train_minutes` (not part of the run key): training stops as soon as early stopping cannot
  end it within the limit.
- `scripts/make_tables.py` reports failed runs per cell and by reason, and lists them;
  `failures.jsonl.gz` holds the records. Tables 2-4 also report seconds per epoch and peak memory.
- The reproduce workflow uses the worker pool (13 GB of a runner's 16 GB, swap off), reads an
  optional `[actions]` table from the sweep file (selection, runs per job, memory, time limit) and
  continues a sweep from its results branch (`scripts/unpack_results.py`) unless started with
  `fresh`. Every job merges its records into the results branch when it finishes
  (`scripts/publish_results.py`, retried when another job pushed first), so results appear while
  a sweep runs instead of after its last job.
- Every run records its `environment` (CPU model and instruction sets, Python and library versions,
  thread and BLAS variables, numpy's SIMD targets and the kernel family OpenBLAS selected), because
  executions of the same configuration are not always bit-identical (`results/README.md`).
- `scripts/check_reproducibility.py` and the manual `determinism` workflow repeat one configuration
  in separate processes under different kernel selections (default, pinned instruction sets, each
  setting on its own, numpy's SIMD dispatch), fingerprint the n-HDP objective trace and the trained
  model's outputs, report a run that crashes and go on, and compare the results.
- Tables are byte-identical for the same records, whatever order the records are read in, and
  order labels naturally (heads-2 before heads-10).
- Table 5 and Fig. 11 run on GitHub-hosted runners for Cora, CiteSeer, PubMed, Photo and
  ogbn-arxiv; ogbn-products and ogbn-proteins are left to a cluster.
- The first launch of Tables 2-5 and Fig. 11 (commit 4f00db1) was cancelled. The runners of the
  jobs that reached an OGB graph, or ran four Graph U-Nets with four or more pooling levels at once,
  shut down mid-job (most likely out of memory), which discarded the finished runs of those jobs.

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
