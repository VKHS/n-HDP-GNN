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
measurements behind the decision.

| Sweep | Article | Runs | Status | Workflow run |
|---|---|---|---|---|
| [`table1_ablation`](table1_ablation/README.md) | Table 1 | 400 / 400 | complete | [43643974046](https://github.com/VKHS/n-HDP-GNN/actions/runs/43643974046) |
| `table2_full` | Tables 2, 6-8 | 2400 | running | |
| `table3_semi` | Tables 3, 6-8 | 2400 | running | |
| `table4_scarce` | Tables 4, 6-8 | 2400 | running | |
| `table5_clustering` | Table 5 | 200 of 280 (without ogbn-products and ogbn-proteins) | running | |
| `fig11_heads` | Fig. 11 | 250 of 350 (without ogbn-products and ogbn-proteins) | running | |
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
