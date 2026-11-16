# Results of trained models

Each directory holds one sweep from `configs/`, run on GitHub-hosted CPU runners by
`.github/workflows/reproduce.yml` (one torch thread per run) and aggregated by
`scripts/make_tables.py`:

| File | Content |
|---|---|
| `README.md` | the article's table layout: mean ± 95% t-interval over seeds with n per cell, paired two-sided Wilcoxon p-values against the reference row, failed runs per cell, and the expected, completed, failed and missing runs |
| `<value>.csv` | one row per cell with the mean, the CI half-width, n, the per-seed values and the failed runs by reason |
| `runs.jsonl.gz` | every run's result JSON: configuration, validation and test metrics, best epoch, timings, peak memory, stage A/B diagnostics, embedding diagnostics, code commit and, in later runs, the environment (CPU model and instruction sets, library versions, thread and BLAS variables) |
| `failures.jsonl.gz` | every failure record: configuration, reason (memory, time, diverged or error) and the measurements behind it |
| `summary.json` | counts, the sweep's `[base]` and `[grid]`, and the code commit(s) of the runs |

**Protocol.** The sweeps use the hyper-parameters in `configs/` (hidden 64, learning
rate 0.005, weight decay 5·10⁻⁴, at most 500 epochs with early stopping after 100
epochs without validation improvement, 10 seeds). These are generic values; where the
article used other values, its numbers can differ from these.

**Execution.** Table 1 ran in four processes per runner, five runs each in sequence. From
Tables 2-5 and Fig. 11 on, every run executes in a process of its own, four at a time per
runner, within a memory budget of 13 GB of the runner's 16 GB (`nhdp_gnn.sweep --workers`).
A run that does not fit next to the others is repeated on its own. A run that needs more than
13 GB on its own, or whose training provably needs more than 150 minutes, is reported as
failed with that reason instead of a value, and so is a run whose selected model gives
non-finite outputs (reason `diverged`); each table lists its failed runs with the
measurements behind the decision. While a sweep runs, every finished job adds its runs to the
branch `results/<sweep>`, whose tables are therefore current but incomplete.

Runs completed by the first launch of Tables 2-5 and Fig. 11 (commit 4f00db1, cancelled) were
kept: the code that trains and evaluates the models is the same at both commits (only the
sweep runner changed), and every run records its commit. Runs of the same configuration are
not always bit-identical between processes, also at the same commit. Of the 155 configurations
that Tables 2 and 5 share, 126 gave the same test accuracy. The 29 that differ are 13 of 20 for
n-HDP-GNN (largest difference 3.47 points), 7 of 20 for GCNII (0.53), 4 of 20 for GAT (0.20),
2 of 15 for Graph U-Net (0.10), 2 of 20 for APPNP (0.08) and 1 of 20 for ARMA (0.03); GCN and
SGC gave the same test accuracy in all 20.

For n-HDP-GNN the trace of the Stage-B objective (double precision, numpy) separates the two cases:
in all 13 pairs whose test accuracy differs, the trace differs too (from iteration 0 or 2 on; the
largest relative difference along a trace is between 6·10⁻¹⁶ and 9·10⁻¹⁴, and both executions need
the same number of iterations), and in all 7 pairs with the same test accuracy the traces are
identical. The two executions of a differing pair
therefore differ in their floating-point arithmetic, which shows in Stage B and, in the same pairs,
in the trained model. These runs did not record their CPU or library versions, so what differed
between their two executions is not known. The manual `determinism` workflow measures what the CPU
model does: it trains n-HDP-GNN (Cora, full supervision, depth 2, one thread, a process per run)
twice per seed and kernel selection on each of eight GitHub runners (two executions:
[43746860276](https://github.com/VKHS/n-HDP-GNN/actions/runs/43746860276), seeds 0-2, default and
pinned kernels, and
[43764052765](https://github.com/VKHS/n-HDP-GNN/actions/runs/43764052765), seeds 0-1, eight kernel
selections).

* On one CPU model, repeats are bit-identical. In the second execution all 110 groups of finished
  repeats agree in test accuracy, best epoch, Stage-B trace and the fingerprint of the trained
  model's outputs, runners of the same CPU model agree with each other, and their test accuracy,
  best epoch and Stage-B trace equal those of the first execution (numpy 2.5.4 there, 2.5.5 in the
  second).
* Different CPU models give different results. With the default kernels seed 0 reached 76.30% on
  AMD EPYC 7763, 79.26% on AMD EPYC 9V45 and 78.15% on Intel Xeon Platinum 8573C and on Xeon
  6975P-C (2.96 points between the highest and the lowest); seed 1 reached 81.48%, 80.19% and
  80.74% (1.30 points). The two Intel models agree in every kernel selection.
* No kernel selection made the four models agree. Limiting ATen, MKL, oneDNN or OpenBLAS to AVX2
  (Haswell kernels in OpenBLAS), and numpy's SIMD dispatch to AVX2, each alone and all together,
  still gave at least two different results per seed. The one agreement beyond the Intel pair is
  EPYC 7763 with EPYC 9V45 under `ATEN_CPU_CAPABILITY=avx2`: the same test accuracy, best epoch and
  output fingerprint, but not the same Stage-B trace.
* `OPENBLAS_CORETYPE=HASWELL` ends the process with a segmentation fault on AMD EPYC 9V45: all
  12 runs of the three kernel selections that contain it (`openblas`, `pinned`, `all`) in each of
  three jobs, and no run of the other five. The first execution's `pinned` selection also died on
  AMD EPYC 9V74. The other CPU models did not crash. The Python stack of the second execution ends
  in scikit-learn's randomized SVD, called from `reduce_features` (`nhdp_gnn/nhdp.py`); why it
  crashes is not known.

The CPU model that GitHub assigns to a runner is therefore a source of variation in every sweep
here. Which CPU produced which of the 29 differences above is not known. Later runs record their
`environment` (CPU model and instruction sets, library versions, thread and BLAS variables).
Single runs should be compared through means over seeds, as the tables do.

| Sweep | Article | Runs | Status | Workflow run |
|---|---|---|---|---|
| [`table1_ablation`](table1_ablation/README.md) | Table 1 | 400 / 400 | complete | [43643974046](https://github.com/VKHS/n-HDP-GNN/actions/runs/43643974046) |
| [`table2_full`](table2_full/README.md) | Tables 2, 6-8 | 2248 / 2400 completed, 152 failed (87 memory, 25 time, 40 diverged) | complete | [43678475781](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678475781), continuing [43658901273](https://github.com/VKHS/n-HDP-GNN/actions/runs/43658901273); the 40 diverged runs re-run in [43731956974](https://github.com/VKHS/n-HDP-GNN/actions/runs/43731956974) |
| [`table3_semi`](table3_semi/README.md) | Tables 3, 6-8 | 2241 / 2400 completed, 159 failed (88 memory, 31 time, 40 diverged) | complete | [43678483352](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678483352), continuing [43645884923](https://github.com/VKHS/n-HDP-GNN/actions/runs/43645884923); the 40 diverged runs re-run in [43763047138](https://github.com/VKHS/n-HDP-GNN/actions/runs/43763047138) |
| [`table4_scarce`](table4_scarce/README.md) | Tables 4, 6-8 | 2248 / 2400 completed, 152 failed (92 memory, 20 time, 40 diverged) | complete | [43678490785](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678490785), continuing [43645890551](https://github.com/VKHS/n-HDP-GNN/actions/runs/43645890551); the 40 diverged runs re-run in [43855293851](https://github.com/VKHS/n-HDP-GNN/actions/runs/43855293851) |
| [`table5_clustering`](table5_clustering/README.md) | Table 5 | 190 / 200 completed, 10 failed (memory); without ogbn-products and ogbn-proteins | complete | [43658886877](https://github.com/VKHS/n-HDP-GNN/actions/runs/43658886877) |
| [`fig11_heads`](fig11_heads/README.md) | Fig. 11 | 210 / 250 completed, 40 failed (30 memory, 10 time); without ogbn-products and ogbn-proteins | complete | [43678422172](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678422172), continuing [43658893455](https://github.com/VKHS/n-HDP-GNN/actions/runs/43658893455) |
| multi-node scaling, OGB depth runs, ogbn-products and ogbn-proteins | Tables 5, 9-12, Fig. 11 | | need a CPU cluster (`scripts/launch_cluster.sh`) | |

## Summary

What the finished sweeps show, with the generic hyper-parameters of `configs/`, which are the same
for every model (the article's own settings were not used, and n-HDP-GNN was not tuned). The
sections below have the numbers and the exceptions.

* **Components (Table 1).** Each component in the article's progression raises the mean accuracy
  up to the one-head gated attention on all four graphs. The full model does not exceed the
  two-head variant, and GAT with residual connections and no hierarchy is above it on Cora,
  CiteSeer and Photo (p = 0.002 each).
* **Shallow depth (Tables 2-4).** At depth 2 n-HDP-GNN is below the highest other model on every
  graph in every label regime: by 8.46, 6.30, 1.35 and 7.59 points on Cora, CiteSeer, PubMed and
  Photo with full supervision, by 19.32, 9.22, 5.85 and 19.85 with the public split and by 16.09,
  7.80, 3.73 and 13.00 with 5% labels.
* **Large depth (Tables 2-4).** On Cora, CiteSeer and PubMed n-HDP-GNN's accuracy changes by at
  most 2.8 points between depth 2 and depth 30 (CiteSeer with the public split: a fall of 6.9),
  and its centroid gap and MADGap stay clearly above zero at depth 30. GAT and ARMA fall below
  50% on all four graphs in all three regimes, and GCN and DropEdge on Cora and CiteSeer; the
  centroid gaps of GAT, ARMA and DropEdge are near zero or negative on Cora, CiteSeer and PubMed.
  This agrees with the article's claim of resilience to over-smoothing, but it is not unique to
  n-HDP-GNN: APPNP and GCNII also hold with depth, and at depth 30 both are above n-HDP-GNN's
  mean on all four graphs in all three regimes (p ≤ 0.014, except APPNP on PubMed with full
  supervision, p = 0.85, and GCNII on Photo with full supervision, p = 0.084). On Photo
  n-HDP-GNN's accuracy falls at depth 20 and 30 in all three regimes, with large spreads between
  seeds.
* **Clustering (Table 5).** The logits of n-HDP-GNN give the highest NMI and ARI on CiteSeer and
  PubMed and the second highest on Photo (five seeds, so these are differences of means).
* **Heads (Fig. 11).** One head is the worst setting on Cora, CiteSeer, PubMed and ogbn-arxiv.
  From two heads on the means are within 3.7 points on Cora, CiteSeer and PubMed; on Photo
  accuracy is highest with two heads and unstable with more.
* **Cells without an accuracy.** Graph U-Net has none where it needed more than 13 GB or 150
  minutes or gave NaN outputs, and SGC's accuracy is low where early stopping ended its training
  early (Tables 2-5); failed runs are listed with their reason, not filled in.
* **Not measured here.** Multi-node scaling and communication (Tables 9-12), the OGB-scale depth
  runs, ogbn-products and ogbn-proteins need a cluster; the article's own hyper-parameters were
  not run.

## Table 1: component ablation (full supervision, depth 2)

Test accuracy (%), mean over 10 seeds; Wilcoxon p-values against `full` are in
[table1_ablation/README.md](table1_ablation/README.md). With 10 seeds the smallest
possible p-value is 0.002 (all ten pairs in the same direction).

| Variant | What changes | Cora | CiteSeer | PubMed | Photo |
|---|---|---|---|---|---|
| hdp-gnn | two-level HDP, GCN layers, no community/global level, no residual | 75.54 | 69.95 | 87.43 | 84.00 |
| nhdp-gnn | nested (three-level) HDP | 77.20 | 70.09 | 87.50 | 85.76 |
| nhdp-gnn+res | + residual connections | 79.04 | 70.72 | 87.60 | 88.53 |
| nhdp-gnn+res+gat1 | + gated attention, 1 head | 82.22 | 71.79 | 87.82 | 92.50 |
| nhdp-gnn+res+gat2 | 2 heads | 80.48 | 71.85 | 88.35 | 92.33 |
| full | 4 heads + community and global attention | 80.44 | 71.33 | 87.66 | 86.80 |
| full-gate-off | full, node attention not gated | 86.37 | 72.00 | 87.77 | 78.48 |
| full-louvain-hard | full, hard Louvain super-communities instead of n-HDP | 85.94 | 71.88 | 87.27 | 88.92 |
| full-random-partition | full, random community membership (Louvain sizes) | 74.67 | 70.35 | 87.56 | 82.60 |
| gat+res-no-hierarchy | no gate, no community or global level (GAT + residual) | 86.80 | 73.51 | 87.86 | 93.68 |

* Following the article's progression, each added component raises the mean
  accuracy up to the one-head gated attention on all four graphs (Cora 75.54 to
  82.22, Photo 84.00 to 92.50; on PubMed by 0.39 points in total).
* The full model is below the two-head variant on Photo (86.80 vs 92.33) and does
  not exceed it on the other graphs.
* Without the hierarchy (GAT with residual connections) accuracy is higher than the
  full model on Cora (+6.35), CiteSeer (+2.18) and Photo (+6.88), each with p = 0.002,
  and not different on PubMed (+0.20, p = 0.625).
* Turning the gate off in the full model raises Cora accuracy from 80.44 to 86.37
  (p = 0.002); on Photo it is lower but highly variable (78.48 ± 7.37, p = 0.11).
  This matches the training-free depth study in `artifacts/`, where the n-HDP gate
  preserves less class information than the ungated operator on Cora.

## Table 2: depth, full supervision (60/20/20 split)

Test accuracy (%), mean over 10 seeds; [table2_full/README.md](table2_full/README.md) has every
depth and model with 95% intervals and Wilcoxon p-values against n-HDP-GNN, and the same tables
for the centroid gap (Eq. 28) and MADGap of the logits (Tables 6-8), seconds per epoch and peak
memory. With 10 seeds the smallest possible p-value is 0.002.

| Dataset | n-HDP-GNN, depth 2 / 10 / 30 | Highest other models at depth 30 | Below 50% at depth 30 |
|---|---|---|---|
| Cora | 80.30 / 82.91 / 82.87 | APPNP 88.35, GCNII 87.67 | GCN 30.19, GAT 30.44, ARMA 30.11, DropEdge 30.19, SGC 34.98 |
| CiteSeer | 71.49 / 70.77 / 71.79 | APPNP 76.67, GCNII 75.84 | GCN 21.79, GAT 21.87, ARMA 20.69, DropEdge 22.58, PairNorm 37.83 |
| PubMed | 87.59 / 87.66 / 87.83 | GCNII 88.35, APPNP 87.87 | GCN 47.55, GAT 41.48, ARMA 40.25, DropEdge 40.48 |
| Photo | 86.94 / 86.30 / 55.28 | APPNP 91.25, PairNorm 83.30 | GAT 25.71, ARMA 26.21, SGC 27.93 |

* n-HDP-GNN's accuracy changes little with depth on Cora, CiteSeer and PubMed (within 2.6, 1.3
  and 0.3 points between depths 2 and 30). On Photo it falls at depths 20 and 30
  (55.78 ± 19.63 and 55.28 ± 21.19) with a large spread between seeds.
* GCN, GAT, ARMA and DropEdge fall to 20-48% by depth 30 on Cora, CiteSeer and PubMed (on Cora
  to the share of the largest class, 30.2%); there n-HDP-GNN is higher than all four at depth 30
  (p = 0.002 each).
* APPNP and GCNII hardly degrade with depth and are above n-HDP-GNN at every depth on Cora and
  CiteSeer (p ≤ 0.004 at depth 30). On PubMed at depth 30 GCNII is higher (p = 0.014) and APPNP
  level (p = 0.85).
* Centroid gap and MADGap of the logits at depth 30: n-HDP-GNN keeps them (centroid gap / MADGap
  1.02 / 0.65 on Cora, 0.94 / 0.72 on CiteSeer, 1.11 / 0.71 on PubMed), as do APPNP and GCNII
  (Cora 0.91 / 0.69 and 0.86 / 0.63), while for GAT, ARMA and DropEdge the centroid gap is within
  0.003 of zero on these three graphs (GCN: 0.22 on PubMed, within 0.001 of zero on Cora and
  CiteSeer). On Photo n-HDP-GNN's centroid gap drops to 0.32 at depth 30.
* SGC reaches 46.41 on Cora and 44.09 on Photo at depth 2 (GCN: 88.28 and 92.30). On Cora nine of
  its ten runs stop 100 epochs after a best validation epoch between 1 and 9 and the tenth trains
  all 500 epochs; on Photo seven stop after a best epoch between 1 and 67 and three train all 500
  (best epoch 489-499). The Table 5 bullet below says what this points to.
* Graph U-Net exceeded 13 GB on Photo at every depth and on PubMed from depth 4 (or needed more
  than 150 minutes); with 10 and 15 pooling levels (depth 20 and 30) it gives NaN outputs on Cora
  and CiteSeer: the model selected on validation accuracy (epoch 1; 101 epochs were run) gives
  non-finite outputs for every node (2708 of 2708 on Cora, 3327 of 3327 on CiteSeer). These 40
  runs have no accuracy and are listed as `diverged` in `failures.jsonl.gz`, not as results. Earlier
  code had recorded them as errors; the current code records them as diverged, and they were re-run
  with it.
* Cost at depth 30 (one thread, four runs per 4-vCPU runner): n-HDP-GNN takes 0.70, 0.93, 4.0 and
  9.2 s per epoch on Cora, CiteSeer, PubMed and Photo, about as much as GAT (0.51, 0.76, 3.6, 9.2)
  and more than GCN (0.32, 0.60, 3.2, 5.4) and APPNP (0.13, 0.35, 0.45, 0.74); SGC recomputes its
  propagation in every epoch (`cached=False`) and is the slowest. Its peak memory at depth 30 is
  3814 MB on PubMed and 4440 MB on Photo. Peak memory is recorded only for runs that had a process
  of their own (from commit 20d9d0f on), so some cells have fewer than 10 values.

## Table 3: depth, semi-supervised (public split)

Test accuracy (%), mean over 10 seeds, with the public Planetoid split (20 labels per class, 500
validation and 1000 test nodes) on Cora, CiteSeer and PubMed and 20 labels per class on Photo;
[table3_semi/README.md](table3_semi/README.md) has every depth and model with 95% intervals and
Wilcoxon p-values against n-HDP-GNN, and the same tables for the centroid gap and MADGap of the
logits (Tables 6-8), seconds per epoch and peak memory. With 10 seeds the smallest possible
p-value is 0.002.

| Dataset | n-HDP-GNN, depth 2 / 10 / 30 | Highest other model at depth 2 | Highest other models at depth 30 | Below 50% at depth 30 |
|---|---|---|---|---|
| Cora | 62.53 / 63.88 / 59.77 | GAT 81.85 | APPNP 82.59, GCNII 79.84 | DropEdge 23.91, ARMA 24.52, GAT 25.34, PairNorm 29.56, GCN 30.91 |
| CiteSeer | 61.49 / 57.33 / 54.60 | GAT 70.71 | APPNP 70.20, GCNII 69.24 | ARMA 18.59, GAT 19.15, DropEdge 19.82, GCN 34.07, PairNorm 36.21 |
| PubMed | 72.10 / 72.10 / 71.69 | ARMA 77.95 | GCNII 78.96, APPNP 78.08 | DropEdge 40.46, ARMA 40.54, GAT 42.67 |
| Photo | 70.29 / 51.37 / 33.62 | ARMA 90.14 | APPNP 88.97, GCNII 68.80 | GAT 19.56, ARMA 21.08, DropEdge 31.62, SGC 34.21 (n=5), GCN 35.81 |

* At depth 2 n-HDP-GNN has the lowest mean on Cora, CiteSeer and PubMed, 19.32, 9.22 and 5.85
  points below the highest model (GAT, GAT, ARMA), with p = 0.002 against every model except
  GCNII and Graph U-Net on CiteSeer (p = 0.029 and 0.049). On Photo only SGC is lower (68.04,
  p = 0.49) and ARMA is 19.85 points above. At depth 4 the gap to the highest model (APPNP) is
  17.51, 9.46, 4.53 and 26.01 points. These are the shared hyper-parameters of
  `configs/table3_semi.toml`; other settings for n-HDP-GNN were not tried here.
* n-HDP-GNN's accuracy changes little with depth on Cora (59.45-64.19 over the six depths) and
  PubMed (71.69-73.31); on CiteSeer it falls by 6.9 points between depths 2 and 30 (61.49 to
  54.60). On Photo it falls from 70.29 to 51.37 at depth 10 and 33.62 at depth 30, with intervals
  from ± 6.5 (depth 2) to ± 14.5 (depth 30): at depth 20 seven of ten seeds end at 22.0-25.4%
  with a best validation epoch between 1 and 19, and at depth 30 six of ten end at 12.0-25.4%
  with a best epoch between 1 and 9.
* GAT, ARMA and DropEdge are below 50% at depth 30 on all four graphs, GCN on Cora, CiteSeer and
  Photo, PairNorm on Cora and CiteSeer, and SGC on Photo. n-HDP-GNN is higher than every model
  below 50% at depth 30 on Cora, CiteSeer and PubMed (p = 0.002 each). On Photo it does not
  differ from GCN, GAT, ARMA, SGC, PairNorm or DropEdge at depth 30 (p between 0.074 and 0.92).
* APPNP is above n-HDP-GNN in all 24 dataset-depth cells and its mean varies by at most 1.2
  points over the six depths on each graph. GCNII also holds on Cora, CiteSeer and PubMed (79.84,
  69.24 and 78.96 at depth 30) and falls on Photo from 88.69 at depth 2 to 68.80 at depth 30. At
  depth 30 both are above n-HDP-GNN on all four graphs (p = 0.002; GCNII on Photo p = 0.006).
* Centroid gap and MADGap of the logits at depth 30: n-HDP-GNN keeps them on Cora (0.56 / 0.37),
  CiteSeer (0.45 / 0.56) and PubMed (0.78 / 0.55), as do APPNP (0.81 / 0.70, 0.75 / 0.93,
  1.07 / 0.92) and GCNII (0.72 / 0.66, 0.46 / 0.70, 1.05 / 0.92), while for GAT and ARMA the
  centroid gap is within 0.02 of zero on these three graphs and for DropEdge it is negative
  (-0.17, -0.09, -0.12). On Photo n-HDP-GNN's centroid gap falls from 0.43 at depth 2 to 0.13 at
  depth 30.
* SGC reaches 77.82, 67.35 and 74.43 at depth 2 on Cora, CiteSeer and PubMed (GCN: 81.70, 69.54
  and 77.04) and 71.16, 68.35 and 68.61 at depth 30. On Photo it reaches 68.04 at depth 2 (GCN:
  89.22): in six of ten seeds its best validation epoch is between 20 and 32, and the other four
  run all 500 epochs (best epoch 490-500). At depth 20 and 30 on Photo 4 and 5 of its 10 runs needed more than 150
  minutes, leaving n = 6 and 5.
* Graph U-Net finished depths 2-10 on Cora and CiteSeer but on PubMed only depth 2. It exceeded
  13 GB on Photo at every depth, and on PubMed from depth 4 it either exceeded 13 GB (28 of 50
  runs) or needed more than 150 minutes (22 of 50). With 10 and 15 pooling levels (depth 20 and
  30) it gives NaN outputs on Cora and CiteSeer: the model selected on validation accuracy
  (epoch 1; 101 epochs were run) gives non-finite outputs for every node (2708 of 2708 on Cora,
  3327 of 3327 on CiteSeer). These 40 runs have no accuracy and are listed as `diverged` in
  `failures.jsonl.gz`, not as results. Earlier code had recorded them as errors (k-means in the
  diagnostics raised `Input X contains NaN`); they were re-run with the current code.
* Cost at depth 30 (one thread, four runs per 4-vCPU runner): n-HDP-GNN takes 0.64, 0.87, 3.95
  and 8.86 s per epoch on Cora, CiteSeer, PubMed and Photo, against 0.43, 0.76, 3.68 and 8.65 for
  GAT, 0.28, 0.53, 2.96 and 5.25 for GCN and 0.13, 0.36, 0.40 and 0.74 for APPNP. SGC recomputes
  its propagation in every epoch (`cached=False`) and is the slowest (3.07, 7.91, 9.51 and 24.66
  s; the Photo value is from the five runs that finished). The peak memory of n-HDP-GNN at depth
  30 is 3847 MB on PubMed and 4431 MB on Photo. Peak memory is recorded only for runs that had a
  process of their own (from commit 20d9d0f on), so some cells have fewer than 10 values.

## Table 4: depth, scarce supervision (5% labels)

Test accuracy (%), mean over 10 seeds, with 5% of the nodes labelled (stratified, drawn from the
training pool of the full regime: 136, 165, 986 and 382 labelled nodes on Cora, CiteSeer, PubMed
and Photo; the validation and test nodes are those of the full regime);
[table4_scarce/README.md](table4_scarce/README.md) has every depth and model with 95% intervals and
Wilcoxon p-values against n-HDP-GNN, and the same tables for the centroid gap and MADGap of the
logits (Tables 6-8), seconds per epoch and peak memory. With 10 seeds the smallest possible
p-value is 0.002.

| Dataset | n-HDP-GNN, depth 2 / 10 / 30 | Highest other model at depth 2 | Highest other models at depth 30 | Below 50% at depth 30 |
|---|---|---|---|---|
| Cora | 64.57 / 64.69 / 62.02 | GAT 80.67 | APPNP 82.37, GCNII 79.30 | ARMA 28.61, DropEdge 30.19, GAT 30.19, GCN 32.76, SGC 36.76, PairNorm 40.96 |
| CiteSeer | 61.99 / 60.74 / 59.73 | DropEdge 69.79 | APPNP 69.73, GCNII 67.92 | ARMA 19.92, GAT 22.94, DropEdge 23.21, PairNorm 26.02, GCN 38.61 |
| PubMed | 81.71 / 82.82 / 83.00 | APPNP 85.44 | APPNP 85.19, GCNII 84.71 | ARMA 40.06, DropEdge 40.84, GCN 42.14, GAT 43.27 |
| Photo | 79.87 / 68.07 / 38.41 | ARMA 92.87 | APPNP 90.55, GCNII 80.49 | GAT 25.38, ARMA 26.31, SGC 28.23, DropEdge 48.79 |

* At depth 2 n-HDP-GNN is below every other model on CiteSeer, and below every other model except
  SGC on Cora, PubMed and Photo (SGC: 40.22, 79.65 and 38.34). The highest model (GAT, DropEdge,
  APPNP and ARMA) is 16.09, 7.80, 3.73 and 13.00 points above it on Cora, CiteSeer, PubMed and
  Photo, with p = 0.002 against every model except SGC on CiteSeer and PubMed (p = 0.020 and
  0.275) and Graph U-Net on CiteSeer (p = 0.004). At depth 4 the gap to the highest model (APPNP
  on Cora, CiteSeer and PubMed, ARMA on Photo) is 16.11, 7.86, 3.21 and 18.95 points. These are
  the shared hyper-parameters of `configs/table4_scarce.toml`; other settings for n-HDP-GNN were
  not tried here.
* n-HDP-GNN's accuracy changes little with depth on Cora (62.02-64.98 over the six depths),
  CiteSeer (59.73-62.08) and PubMed (81.71-83.00). On Photo it falls from 79.87 at depth 2 to
  68.07 at depth 10 and 38.41 at depth 30, with intervals from ± 4.14 (depth 2) to ± 17.90
  (depth 20): at depth 20 six of ten seeds end at 25.4-27.3% with a best validation epoch between
  2 and 119, and at depth 30 six of ten end at 25.4% with a best epoch between 1 and 6.
* GAT, ARMA and DropEdge are below 50% at depth 30 on all four graphs, GCN on Cora, CiteSeer and
  PubMed, PairNorm on Cora and CiteSeer, and SGC on Cora and Photo. n-HDP-GNN is higher than every
  model below 50% at depth 30 on Cora, CiteSeer and PubMed (p = 0.002 each). On Photo it does not
  differ from GAT, ARMA, SGC or DropEdge at depth 30 (p between 0.125 and 0.25), and GCN and
  PairNorm are above it (p = 0.014 and 0.020).
* APPNP is above n-HDP-GNN in all 24 dataset-depth cells (p = 0.002 in each); its mean varies by
  0.35-1.78 points over the six depths on CiteSeer, PubMed and Photo and by 3.00 on Cora (79.37 at
  depth 2, 82.37 at depth 30). GCNII is also above n-HDP-GNN in all 24 cells (p ≤ 0.05 in 23; the
  exception is PubMed at depth 10, p = 0.131), with p = 0.002 at depth 30 on all four graphs; its
  mean falls on Photo from 91.04 at depth 2 to 80.49 at depth 30. At depth 30 n-HDP-GNN is third
  highest of nine models on Cora and PubMed, fourth on CiteSeer and sixth on Photo (Graph U-Net
  has no result at depth 30).
* Centroid gap and MADGap of the logits at depth 30: n-HDP-GNN keeps them on Cora (0.70 / 0.46),
  CiteSeer (0.57 / 0.61) and PubMed (0.97 / 0.64), as do APPNP (0.65 / 0.60, 0.58 / 0.81,
  1.16 / 0.87) and GCNII (0.66 / 0.64, 0.50 / 0.78, 1.19 / 0.88), while for GAT, ARMA and DropEdge
  the centroid gap is within 0.005 of zero on these three graphs (and for GAT and ARMA on Photo).
  On Photo n-HDP-GNN's centroid gap falls from 0.49 at depth 2 to 0.07 at depth 30 (MADGap 0.22
  to 0.05), while APPNP (0.84), GCNII (0.65), PairNorm (0.60) and GCN (0.50) are well above it.
* SGC reaches 40.22 and 38.34 at depth 2 on Cora and Photo, against 64.79 and 79.65 on CiteSeer
  and PubMed (GCN: 79.69, 91.09, 69.31 and 84.81). On Cora its best validation epoch lies between
  2 and 14 in all ten seeds and training stops 100 epochs later (102-114 epochs run); on Photo it
  lies between 1 and 57 in nine of ten seeds (the tenth: 245). The sweep of Table 3 differs from
  this one only in the regime, and there the same SGC reached 77.82 on Cora and 68.04 on Photo at
  depth 2. The cause of the difference is not established. On PubMed at depth 30, 2 of its 10 runs
  needed more than 150 minutes, leaving n = 8.
* Graph U-Net finished depths 2-10 on Cora and CiteSeer, but on PubMed only depth 2 and on Photo no
  depth. It exceeded 13 GB in all 60 Photo runs, and on PubMed from depth 4 it either exceeded
  13 GB (32 of 50 runs) or needed more than 150 minutes (18 of 50). With 10 and 15 pooling levels
  (depth 20 and 30) it gives NaN outputs on Cora and CiteSeer: the model selected on validation
  accuracy (epoch 1; 101 epochs were run) gives non-finite outputs for every node (2708 of 2708 on
  Cora, 3327 of 3327 on CiteSeer). These 40 runs have no accuracy and are listed as `diverged` in
  `failures.jsonl.gz`, not as results. Earlier code had recorded them as errors (k-means in the
  diagnostics raised `Input X contains NaN`); they were re-run with the current code.
* Cost at depth 30 (one thread, four runs per 4-vCPU runner): n-HDP-GNN takes 0.60, 0.91, 3.87
  and 8.86 s per epoch on Cora, CiteSeer, PubMed and Photo, against 0.50, 0.73, 3.54 and 8.37 for
  GAT, 0.30, 0.56, 3.02 and 5.01 for GCN and 0.12, 0.34, 0.42 and 0.73 for APPNP. SGC recomputes
  its propagation in every epoch (`cached=False`) and is the slowest (3.17, 7.75, 9.32 and 28.24
  s; the PubMed value is from the eight runs that finished). The peak memory of n-HDP-GNN at depth
  30 is 3839 MB on PubMed and 4446 MB on Photo; every completed run recorded its peak memory.

## Table 5: classification and clustering (full supervision, depth 2)

Five seeds per cell; NMI and ARI compare k-means clusters of each model's logits with the
classes. The tables with intervals are in [table5_clustering/README.md](table5_clustering/README.md).
Ranks count the models with results (Graph U-Net has none on Photo and ogbn-arxiv).

| Dataset | n-HDP-GNN accuracy % (rank) | Highest accuracy % | n-HDP-GNN NMI / ARI (ranks) | Highest NMI / ARI of the other models |
|---|---|---|---|---|
| Cora | 80.93 (7 of 8) | APPNP 88.59 | 0.78 / 0.82 (4, 3 of 8) | ARMA 0.80 / ARMA 0.83 |
| CiteSeer | 71.42 (8 of 8) | ARMA 78.04 | 0.67 / 0.72 (1, 1 of 8) | ARMA 0.61 / GCNII 0.65 |
| PubMed | 88.06 (4 of 8) | GCNII 89.07 | 0.68 / 0.74 (1, 1 of 8) | GCNII 0.62 / GCNII 0.67 |
| Photo | 85.04 (6 of 7) | ARMA 94.19 | 0.82 / 0.83 (2, 2 of 7) | ARMA 0.85 / ARMA 0.86 |
| ogbn-arxiv | 61.64 (7 of 7) | GCN 67.02 | 0.43 / 0.25 (7, 6 of 7) | GCNII 0.47 / GCNII 0.29 |

* n-HDP-GNN's accuracy is the lowest or second lowest on Cora, CiteSeer, Photo and
  ogbn-arxiv, and 1.0 point below the highest on PubMed.
* Its logits give the highest NMI and ARI on CiteSeer and PubMed and the second highest on
  Photo. With five seeds no paired Wilcoxon test can reach p < 0.05 (the smallest two-sided
  p-value is 0.0625), so these are differences of means: on CiteSeer the 95% intervals do not
  overlap (NMI 0.67 ± 0.02 against ARMA's 0.61 ± 0.02); on PubMed they overlap slightly
  (0.68 ± 0.05 against GCNII's 0.62 ± 0.01).
* SGC does poorly under the shared protocol (Cora 51.04 ± 22.88, Photo 40.89 ± 12.11). In four
  of five seeds on each of these graphs training stops 100 epochs after a best validation epoch
  between 1 and 67 (test accuracy 33.33-55.19 on Cora, 25.38-46.44 on Photo), and the fifth run
  trains all 500 epochs (80.93 on Cora, 49.84 on Photo). Where it runs all 500 epochs (CiteSeer,
  PubMed, ogbn-arxiv) the best epoch lies in the last 100, i.e. validation accuracy was still
  improving when training ended; on PubMed the one early-stopped run reaches 69.08 and the other
  four 80.75-82.20. A logistic regression on the same propagated features (rows L2-normalised,
  regularisation chosen on validation) reaches 80.5% on Cora with the 140 labels of the public
  split (`artifacts/`, L = 2). This points to the shared optimiser settings (learning rate
  0.005, patience 100) rather than to SGC itself; SGC was not re-run with other settings here.
* Graph U-Net exceeded the 13 GB budget on Photo and ogbn-arxiv in all five seeds.

## Fig. 11: number of attention heads (n-HDP-GNN, full supervision, depth 10)

Test accuracy (%), mean ± 95% interval over five seeds, with 8 channels per head; every
head count is in [fig11_heads/README.md](fig11_heads/README.md).

| Heads | Cora | CiteSeer | PubMed | Photo | ogbn-arxiv |
|---|---|---|---|---|---|
| 1 | 67.85 ± 9.42 | 46.93 ± 22.47 | 85.53 ± 2.41 | 63.30 ± 21.21 | 35.68 ± 11.36 |
| 2 | 80.89 ± 2.11 | 69.97 ± 3.94 | 87.26 ± 1.04 | 89.01 ± 2.47 | 52.53 ± 5.09 |
| 4 | 82.04 ± 2.66 | 70.33 ± 2.88 | 87.59 ± 0.89 | 88.08 ± 5.06 | failed: time |
| 8 | 82.37 ± 1.12 | 71.05 ± 2.41 | 87.77 ± 0.67 | 81.87 ± 12.36 | failed: memory |
| 12 | 82.96 ± 0.91 | 71.48 ± 3.41 | 87.55 ± 0.51 | 64.41 ± 30.06 | failed: memory |
| 18 | 82.81 ± 1.53 | 67.74 ± 6.56 | 87.21 ± 0.95 | 62.88 ± 31.03 | failed: memory |

* One head is the worst setting on Cora, CiteSeer and PubMed (CiteSeer seeds range from 21% to
  69%) and on ogbn-arxiv, where only one and two heads finished. On Photo it gives 63.30 ± 21.21,
  above the means of 16 heads (55.34 ± 22.29) and 18 heads (62.88 ± 31.03).
* From two heads on, the means vary within 2.1 points on Cora (80.89-82.96), 3.7 on
  CiteSeer (67.74-71.48) and 0.6 on PubMed (87.21-87.77).
* On Photo, accuracy is highest with two heads and becomes unstable with more: from 8 heads
  on, at least one seed ends below 66%, and from 12 heads on, one seed ends between 25% and
  32%.
* On ogbn-arxiv only one and two heads finished: with 4 and 6 heads an epoch took 16-52 s,
  so training provably needed more than 150 minutes; with 8 or more heads a run needed more
  than 13 GB.
