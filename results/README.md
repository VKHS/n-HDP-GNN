# Results of trained models

Each directory holds one sweep from `configs/`, run on GitHub-hosted CPU runners by
`.github/workflows/reproduce.yml` (one torch thread per run) and aggregated by
`scripts/make_tables.py`:

| File | Content |
|---|---|
| `README.md` | the article's table layout: mean ± 95% t-interval over seeds with n per cell, paired two-sided Wilcoxon p-values against the reference row, failed runs per cell, and the expected, completed, failed and missing runs |
| `<value>.csv` | one row per cell with the mean, the CI half-width, n, the per-seed values and the failed runs by reason |
| `runs.jsonl.gz` | every run's result JSON: configuration, validation and test metrics, best epoch, timings, peak memory, stage A/B diagnostics, embedding diagnostics, code commit |
| `failures.jsonl.gz` | every failure record: configuration, reason (memory, time or error) and the measurements behind it |
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
failed with that reason instead of a value; each table lists its failed runs with the
measurements behind the decision. While a sweep runs, every finished job adds its runs to the
branch `results/<sweep>`, whose tables are therefore current but incomplete.

Runs completed by the first launch of Tables 2-5 and Fig. 11 (commit 4f00db1, cancelled) were
kept: the code that trains and evaluates the models is the same at both commits (only the
sweep runner changed), and every run records its commit. Runs of the same configuration are
not always bit-identical between processes. Of the 98 configurations that Tables 2 and 5
share, 73 gave the same test accuracy; the largest differences were 3.47 points for
n-HDP-GNN, 0.53 for GCNII and 0.20 for GAT, while GCN, SGC and Graph U-Net were identical.

| Sweep | Article | Runs | Status | Workflow run |
|---|---|---|---|---|
| [`table1_ablation`](table1_ablation/README.md) | Table 1 | 400 / 400 | complete | [43643974046](https://github.com/VKHS/n-HDP-GNN/actions/runs/43643974046) |
| `table2_full` | Tables 2, 6-8 | 2400 | running ([live tables](https://github.com/VKHS/n-HDP-GNN/tree/results/table2_full/results/table2_full)) | [43678475781](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678475781) |
| `table3_semi` | Tables 3, 6-8 | 2400 | running ([live tables](https://github.com/VKHS/n-HDP-GNN/tree/results/table3_semi/results/table3_semi)) | [43678483352](https://github.com/VKHS/n-HDP-GNN/actions/runs/43678483352) |
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
  full model on Cora (+6.36), CiteSeer (+2.18) and Photo (+6.88), each with p = 0.002,
  and equal on PubMed (p = 0.63).
* Turning the gate off in the full model raises Cora accuracy from 80.44 to 86.37
  (p = 0.002); on Photo it is lower but highly variable (78.48 ± 7.37, p = 0.11).
  This matches the training-free depth study in `artifacts/`, where the n-HDP gate
  preserves less class information than the ungated operator on Cora.

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

* One head is the worst setting on every graph (CiteSeer seeds range from 21% to 69%).
* From two heads on, the means vary within 2.1 points on Cora (80.89-82.96), 3.7 on
  CiteSeer (67.74-71.48) and 0.6 on PubMed (87.21-87.77).
* On Photo, accuracy is highest with two heads and becomes unstable with more: from 8 heads
  on, at least one seed ends below 66%, and from 12 heads on, one seed ends between 25% and
  32%.
* On ogbn-arxiv only one and two heads finished: with 4 and 6 heads an epoch took 16-52 s,
  so training provably needed more than 150 minutes; with 8 or more heads a run needed more
  than 13 GB.
