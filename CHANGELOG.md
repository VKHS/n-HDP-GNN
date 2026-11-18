# Changelog

## Unreleased

- Smoke check: `scripts/smoke.py` and `configs/smoke.toml` run the whole pipeline on a 270-node planted hierarchical
  graph and report the measured value of every check. `--level numpy` (7 checks, no PyTorch, seconds) and `--level
  full` (16 checks: trainer, every model through the sweep runner, worker pool, resuming, tables, two parts, two-rank
  DDP, time limit); `--real` adds Cora. A check that cannot run is reported as skipped, and a missing PyTorch fails
  the full level. The `smoke` workflow runs the full level on GitHub's runners.
- Demo: `examples/demo.py` runs the stages on one planted graph against its known truth (stage A, stage B and the
  gate, the training-free depth study with a linear probe and the centroid gap, six models trained at several depths)
  and writes a report, figures and JSON. The `demo` workflow runs it with training on a runner and publishes the output
  to the branch `results/demo`. `make smoke`, `make smoke-full`, `make demo`, `make demo-quick`, `make diagrams`.
- Block diagrams in `docs/diagrams/` (Mermaid sources and SVGs rendered by `scripts/make_diagrams.py`) shown in
  `docs/ARCHITECTURE.md`: the pipeline, one layer, the modules and their layers (checked against the imports by a
  test), the experiments, the cluster, a distributed run and the checks.
- `docs/CLUSTER.md`: what the article states about the 14-node platform and what it leaves out, an example addressing
  plan (marked as an example), the preparation and checks of a new cluster, and `scripts/cluster_inventory.py`, which
  records the inventory the machines report about themselves. `configs/cluster_check.toml` and `configs/hosts.example.txt`.
- `scripts/launch_cluster.sh` ignores comments and blank lines in the host file, tolerates CRLF line ends, can pin
  gloo's network interface (`GLOO_SOCKET_IFNAME`), fails when nothing ran, and is tested with a stand-in `ssh`.
- The `tests` workflow installs the `figures` extra, so the tests of the demo's figures run on CI.
- `nhdp_gnn.distributed`: a run in which every process succeeded could still end with one of them killed by SIGABRT
  ("terminate called without an active exception") while the interpreter shut down, after the result file was written;
  it showed up as a failed four-rank test of the `tests` workflow. Core files of such processes give the cause: the
  worker threads of the gloo process group are still alive after `destroy_process_group()` (a probe counted 1 thread
  before `init_process_group` and 4 after `destroy_process_group`), and the one that has just finished a collective
  drops the tensors of that work, which needs the GIL. The main thread is by then shutting the interpreter down, so
  Python ends the worker thread inside a destructor that must not throw, and the process aborts. Now every rank waits
  at a barrier before it leaves the process group (`run_and_leave`) and then ends its process with `os._exit` after
  flushing the standard streams (`leave_process`, used by `spawn_local` and by `python -m nhdp_gnn.distributed`). The
  package registers no `atexit` handler and the result is written before the barrier, so nothing is lost.
  Measured on GitHub runners (torch 2.15.0, Python 3.12.16; throwaway workflow runs
  [44330801025](https://github.com/VKHS/n-HDP-GNN/actions/runs/44330801025),
  [44332940231](https://github.com/VKHS/n-HDP-GNN/actions/runs/44332940231),
  [44335278607](https://github.com/VKHS/n-HDP-GNN/actions/runs/44335278607) and
  [44339044832](https://github.com/VKHS/n-HDP-GNN/actions/runs/44339044832)): with four processes on one machine
  (`spawn_local`), the previous exit aborted in 32 of 800 runs (4.0%), the barrier alone in 9 of 800 (1.1%) and the
  barrier with `os._exit` in 0 of 400 (the 95% upper bound of its rate is 0.9%); a half-second sleep before the
  process ended instead of `os._exit` also gave 0 of 400. With the launch path of the cluster (`torchrun`, four
  processes on one machine) the previous module ended with exit status 1 (SIGABRT, result file written) in 2 of 192
  runs and the new one in 0 of 192, too few to tell them apart by themselves (p = 0.5). Not measured: other PyTorch
  or Python versions, and processes on several machines. `scripts/check_distributed_exit.py` repeats the first
  measurement; the counts depend on the machine and the versions.
- `examples/output/` holds the output of run 44330815122 of the `demo` workflow, copied unchanged (code commit c9540fe,
  AMD EPYC 9V74 runner, 90 of 90 training runs, 9.9 minutes). `tests/test_committed_outputs.py` checks that it records a
  workflow run on a clean checkout of a commit in the history and that its tables follow from its JSON and run records;
  `tests/test_docs_links.py` checks the links of the documents. The `tests` workflow checks out the full history.
- Demo results on three CPU models: the NumPy sections are identical, the training differs between the two AMD EPYC
  runs and the Intel Xeon run in 27 of 90 runs (three by more than 20 points), and the two AMD runs agree in all 90.
  `examples/cpu_comparison.json` holds the per-run values, `scripts/compare_demo_runs.py` makes it, and
  `examples/README.md` says how to read the demo with this in mind.

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
  model's outputs, report a run that crashes and go on, and compare the results. Two executions on
  GitHub's runners (`results/README.md`) found repeats bit-identical on one CPU model, different
  results across CPU models whatever kernel selection was used, and a segmentation fault with
  `OPENBLAS_CORETYPE=HASWELL` on AMD EPYC 9V45 (and, in the first execution, 9V74).
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
