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
in the trained model. What differs between the executions is not known: these runs do not record
their CPU or library versions. A different CPU model, which selects other BLAS and PyTorch kernels,
is a candidate that has not been tested. Later runs record their `environment` (CPU model and
instruction sets, library versions, thread and BLAS variables). Single runs should be compared
through means over seeds, as the tables do.

| Sweep | Article | Runs | Status | Workflow run |
|---|---|---|---|---|
| [`table1_ablation`](table1_ablation/README.md) | Table 1 | 400 / 400 | complete | [43643974046](https://github.com/VKHS/n-HDP-GNN/actions/runs/43643974046) |
| [`table2_full`](table2_full/README.md) | Tables 2, 6-8 | 2248 / 2400 completed, 152 failed (87 memory, 25 time, 40 diverged) | complete | [43678475781](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678475781), continuing [43658901273](https://github.com/VKHS/n-HDP-GNN/actions/runs/43658901273); the 40 diverged runs re-run in [43731956974](https://github.com/VKHS/n-HDP-GNN/actions/runs/43731956974) |
| [`table3_semi`](table3_semi/README.md) | Tables 3, 6-8 | 2241 / 2400 completed, 159 failed (88 memory, 31 time, 40 diverged) | complete | [43678483352](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678483352), continuing [43645884923](https://github.com/VKHS/n-HDP-GNN/actions/runs/43645884923); the 40 diverged runs re-run in [43763047138](https://github.com/VKHS/n-HDP-GNN/actions/runs/43763047138) |
| `table4_scarce` | Tables 4, 6-8 | 2400 | running ([live tables](https://github.com/VKHS/n-HDP-GNN/tree/results/table4_scarce/results/table4_scarce)) | [43678490785](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678490785) |
| [`table5_clustering`](table5_clustering/README.md) | Table 5 | 190 / 200 completed, 10 failed (memory); without ogbn-products and ogbn-proteins | complete | [43658886877](https://github.com/VKHS/n-HDP-GNN/actions/runs/43658886877) |
| [`fig11_heads`](fig11_heads/README.md) | Fig. 11 | 210 / 250 completed, 40 failed (30 memory, 10 time); without ogbn-products and ogbn-proteins | complete | [43678422172](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678422172), continuing [43658893455](https://github.com/VKHS/n-HDP-GNN/actions/runs/43658893455) |
| multi-node scaling, OGB depth runs, ogbn-products and ogbn-proteins | Tables 5, 9-12, Fig. 11 | | need a CPU cluster (`scripts/launch_cluster.sh`) | |

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
* SGC underfits under the shared protocol here too (Cora 46.41, Photo 44.09 at depth 2).
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
* SGC does not underfit on Cora, CiteSeer and PubMed here (77.82, 67.35 and 74.43 at depth 2;
  71.16, 68.35 and 68.61 at depth 30). On Photo it reaches 68.04 at depth 2: in six of ten seeds
  its best validation epoch is between 20 and 32, and the other four run all 500 epochs (best
  epoch 490-500). At depth 20 and 30 on Photo 4 and 5 of its 10 runs needed more than 150
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
* SGC underfits under the shared protocol (Cora 51.04 ± 22.88, Photo 40.89 ± 12.11). In four
  of five seeds on each of these graphs its best validation epoch is between 1 and 67, so
  early stopping ends training before the learning rate of 0.005 has moved its single linear
  layer far; where it runs all 500 epochs (CiteSeer, PubMed, ogbn-arxiv) the best epoch lies
  in the last 100, i.e. it is still improving. A logistic regression on the same propagated
  features (rows L2-normalised, regularisation chosen on validation) reaches 80.5% on Cora
  with the 140 labels of the public split (`artifacts/`, L = 2). The SGC rows reflect the
  shared optimiser settings rather than SGC.
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
