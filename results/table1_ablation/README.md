# table1_ablation

Sweep `configs/table1_ablation.toml`: 400 runs expected, 400 completed, 0 missing.
Values are mean ± 95% t-interval over seeds (n = completed seeds per cell); accuracy in %.

Code commit(s) of the runs: 4f00db1371643ba63fd5a37ce4d9e24d08bf1f24

## test

**test**

| tag | citeseer | cora | photo | pubmed |
|---|---|---|---|---|
| full | 71.33 ± 1.34 (n=10) | 80.44 ± 0.74 (n=10) | 86.80 ± 3.64 (n=10) | 87.66 ± 0.30 (n=10) |
| full-gate-off | 72.00 ± 1.29 (n=10) | 86.37 ± 0.87 (n=10) | 78.48 ± 7.37 (n=10) | 87.77 ± 0.34 (n=10) |
| full-louvain-hard | 71.88 ± 1.25 (n=10) | 85.94 ± 1.19 (n=10) | 88.92 ± 2.40 (n=10) | 87.27 ± 0.28 (n=10) |
| full-random-partition | 70.35 ± 1.06 (n=10) | 74.67 ± 0.61 (n=10) | 82.60 ± 4.59 (n=10) | 87.56 ± 0.41 (n=10) |
| gat+res-no-hierarchy | 73.51 ± 1.54 (n=10) | 86.80 ± 1.07 (n=10) | 93.68 ± 0.49 (n=10) | 87.86 ± 0.25 (n=10) |
| hdp-gnn | 69.95 ± 1.15 (n=10) | 75.54 ± 0.71 (n=10) | 84.00 ± 2.34 (n=10) | 87.43 ± 0.24 (n=10) |
| nhdp-gnn | 70.09 ± 1.59 (n=10) | 77.20 ± 1.38 (n=10) | 85.76 ± 1.52 (n=10) | 87.50 ± 0.37 (n=10) |
| nhdp-gnn+res | 70.72 ± 1.75 (n=10) | 79.04 ± 0.65 (n=10) | 88.53 ± 1.28 (n=10) | 87.60 ± 0.36 (n=10) |
| nhdp-gnn+res+gat1 | 71.79 ± 1.22 (n=10) | 82.22 ± 0.89 (n=10) | 92.50 ± 0.53 (n=10) | 87.82 ± 0.30 (n=10) |
| nhdp-gnn+res+gat2 | 71.85 ± 1.37 (n=10) | 80.48 ± 0.99 (n=10) | 92.33 ± 0.51 (n=10) | 88.35 ± 0.24 (n=10) |

Wilcoxon p-value vs full (paired by seed)

| tag | citeseer | cora | photo | pubmed |
|---|---|---|---|---|
| full-gate-off | 0.193 | 0.00195 | 0.105 | 0.475 |
| full-louvain-hard | 0.227 | 0.00195 | 0.322 | 0.0195 |
| full-random-partition | 0.0684 | 0.00195 | 0.193 | 0.629 |
| gat+res-no-hierarchy | 0.00195 | 0.00195 | 0.00195 | 0.625 |
| hdp-gnn | 0.0488 | 0.00195 | 0.131 | 0.16 |
| nhdp-gnn | 0.0156 | 0.00195 | 0.416 | 0.186 |
| nhdp-gnn+res | 0.264 | 0.00391 | 0.695 | 0.359 |
| nhdp-gnn+res+gat1 | 0.129 | 0.00195 | 0.00195 | 0.221 |
| nhdp-gnn+res+gat2 | 0.131 | 0.844 | 0.00391 | 0.00195 |

