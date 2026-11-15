# table2_full

Sweep `configs/table2_full.toml`: 2400 runs expected, 2239 completed, 148 failed (38 error, 85 memory, 25 time), 13 missing.

Values are mean ± 95% t-interval over seeds; n is the number of completed seeds in the cell.

Failed runs are counted per cell by reason: *error*: the run raised an exception; *memory*: the run needed more memory than the budget while running alone; *time*: training provably needed more than the time limit (or the run exceeded the wall-clock limit).

Code commit(s) of the runs: 0f7045972e60a86d7966841bc3e5c4bfe0caa016, 20d9d0fac98f9a83bb60e54bf36ddf1c98869b65, 4f00db1371643ba63fd5a37ce4d9e24d08bf1f24

## test (accuracy in %)

**citeseer** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 77.33 ± 1.60 (n=10) | 77.08 ± 1.55 (n=10) | 76.84 ± 1.32 (n=10) | 76.73 ± 1.30 (n=10) | 76.67 ± 1.84 (n=9) | 76.67 ± 1.63 (n=10) |
| arma | 77.79 ± 1.09 (n=10) | 74.76 ± 1.58 (n=10) | 50.38 ± 9.72 (n=10) | 24.97 ± 3.61 (n=10) | 20.72 ± 1.03 (n=10) | 20.69 ± 0.59 (n=10) |
| dropedge | 77.42 ± 1.53 (n=10) | 76.28 ± 1.19 (n=10) | 73.51 ± 1.44 (n=10) | 73.22 ± 1.26 (n=10) | 34.26 ± 12.26 (n=10) | 22.58 ± 1.03 (n=10) |
| gat | 77.18 ± 1.27 (n=10) | 75.44 ± 2.02 (n=9) | 74.10 ± 1.53 (n=10) | 73.89 ± 1.68 (n=10) | 22.06 ± 2.20 (n=10) | 21.87 ± 1.49 (n=10) |
| gcn | 76.96 ± 1.76 (n=10) | 76.04 ± 1.19 (n=10) | 73.13 ± 1.41 (n=10) | 73.04 ± 1.63 (n=10) | 66.07 ± 4.96 (n=10) | 21.79 ± 0.71 (n=10) |
| gcnii | 75.66 ± 1.07 (n=10) | 74.94 ± 0.98 (n=10) | 74.01 ± 1.39 (n=10) | 74.34 ± 1.06 (n=10) | 75.17 ± 1.31 (n=10) | 75.84 ± 1.70 (n=10) |
| graphunet | 76.22 ± 1.45 (n=10) | 75.44 ± 1.66 (n=10) | 75.03 ± 1.80 (n=10) | 75.06 ± 1.45 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 71.49 ± 1.37 (n=10) | 71.46 ± 1.75 (n=10) | 70.83 ± 1.73 (n=10) | 70.77 ± 1.13 (n=10) | 72.03 ± 1.54 (n=10) | 71.79 ± 1.46 (n=10) |
| pairnorm | 76.30 ± 1.19 (n=10) | 73.80 ± 1.46 (n=9) | 73.03 ± 1.43 (n=10) | 71.11 ± 2.56 (n=10) | 64.32 ± 5.60 (n=10) | 37.83 ± 11.19 (n=10) |
| sgc | 75.50 ± 1.11 (n=10) | 74.85 ± 1.13 (n=10) | 73.46 ± 1.16 (n=10) | 73.03 ± 1.43 (n=10) | 71.99 ± 0.97 (n=10) | 71.36 ± 1.24 (n=10) |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00781 | 0.00195 |
| arma | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |
| dropedge | 0.00195 | 0.00195 | 0.0273 | 0.0117 | 0.00195 | 0.00195 |
| gat | 0.00195 | 0.00391 | 0.0117 | 0.00586 | 0.00195 | 0.00195 |
| gcn | 0.00195 | 0.00195 | 0.0488 | 0.0332 | 0.00977 | 0.00195 |
| gcnii | 0.00195 | 0.00195 | 0.0352 | 0.00195 | 0.00195 | 0.00195 |
| graphunet | 0.00195 | 0.00195 | 0.00391 | 0.00195 |  |  |
| pairnorm | 0.00195 | 0.0312 | 0.041 | 0.492 | 0.00586 | 0.00195 |
| sgc | 0.00195 | 0.00195 | 0.00977 | 0.0254 | 0.922 | 0.322 |

**cora** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 88.76 ± 0.72 (n=10) | 88.57 ± 1.07 (n=10) | 88.41 ± 0.90 (n=10) | 88.39 ± 0.69 (n=10) | 88.42 ± 0.91 (n=9) | 88.35 ± 0.83 (n=10) |
| arma | 88.46 ± 0.67 (n=10) | 86.39 ± 0.91 (n=10) | 65.69 ± 11.91 (n=10) | 37.96 ± 2.42 (n=10) | 30.19 ± 0.00 (n=10) | 30.11 ± 0.09 (n=10) |
| dropedge | 88.56 ± 0.71 (n=10) | 87.43 ± 0.64 (n=10) | 86.89 ± 1.02 (n=10) | 82.00 ± 3.27 (n=10) | 30.19 ± 0.00 (n=10) | 30.19 ± 0.00 (n=10) |
| gat | 88.11 ± 0.79 (n=10) | 87.28 ± 0.98 (n=10) | 86.69 ± 1.28 (n=10) | 86.19 ± 0.95 (n=10) | 30.28 ± 0.21 (n=10) | 30.44 ± 0.63 (n=10) |
| gcn | 88.28 ± 0.89 (n=10) | 87.33 ± 0.79 (n=10) | 86.06 ± 1.23 (n=10) | 85.56 ± 1.07 (n=10) | 53.48 ± 15.01 (n=10) | 30.19 ± 0.00 (n=10) |
| gcnii | 87.57 ± 0.49 (n=10) | 86.98 ± 0.96 (n=10) | 86.30 ± 1.68 (n=10) | 85.74 ± 1.89 (n=10) | 87.31 ± 0.96 (n=10) | 87.67 ± 1.29 (n=10) |
| graphunet | 88.33 ± 0.84 (n=10) | 87.63 ± 0.93 (n=10) | 87.67 ± 1.03 (n=10) | 88.04 ± 0.82 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 80.30 ± 1.12 (n=10) | 81.35 ± 1.28 (n=10) | 82.93 ± 0.71 (n=10) | 82.91 ± 1.21 (n=10) | 82.81 ± 0.78 (n=10) | 82.87 ± 0.87 (n=10) |
| pairnorm | 87.33 ± 1.18 (n=10) | 86.79 ± 1.18 (n=9) | 86.07 ± 1.30 (n=10) | 85.26 ± 2.07 (n=10) | 79.17 ± 1.16 (n=10) | 71.70 ± 5.00 (n=10) |
| sgc | 46.41 ± 9.62 (n=10) | 45.59 ± 9.52 (n=10) | 39.78 ± 4.78 (n=10) | 39.33 ± 4.66 (n=10) | 37.44 ± 4.33 (n=10) | 34.98 ± 3.43 (n=10) |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00391 | 0.00195 |
| arma | 0.00195 | 0.00391 | 0.00391 | 0.00195 | 0.00195 | 0.00195 |
| dropedge | 0.00195 | 0.00195 | 0.00195 | 0.51 | 0.00195 | 0.00195 |
| gat | 0.00195 | 0.00195 | 0.00586 | 0.00391 | 0.00195 | 0.00195 |
| gcn | 0.00195 | 0.00195 | 0.00391 | 0.00781 | 0.00195 | 0.00195 |
| gcnii | 0.00195 | 0.00391 | 0.00977 | 0.0488 | 0.00195 | 0.00391 |
| graphunet | 0.00195 | 0.00195 | 0.00195 | 0.00195 |  |  |
| pairnorm | 0.00195 | 0.00391 | 0.00586 | 0.0488 | 0.00195 | 0.00195 |
| sgc | 0.00391 | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |

**photo** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 93.26 ± 0.42 (n=10) | 92.47 ± 0.39 (n=10) | 91.66 ± 0.42 (n=10) | 91.51 ± 0.45 (n=10) | 91.28 ± 0.67 (n=9) | 91.25 ± 0.59 (n=10) |
| arma | 94.53 ± 0.44 (n=10) | 93.85 ± 0.49 (n=10) | 78.14 ± 5.47 (n=10) | 46.78 ± 7.29 (n=10) | 29.94 ± 6.76 (n=10) | 26.21 ± 0.88 (n=10) |
| dropedge | 92.46 ± 0.32 (n=10) | 92.33 ± 0.61 (n=10) | 90.00 ± 0.26 (n=10) | 84.57 ± 3.59 (n=10) | 55.64 ± 5.78 (n=10) | 55.72 ± 7.03 (n=10) |
| gat | 92.91 ± 0.50 (n=10) | 92.24 ± 0.56 (n=10) | 90.63 ± 0.61 (n=10) | 89.31 ± 1.03 (n=10) | 26.66 ± 2.95 (n=10) | 25.71 ± 0.75 (n=10) |
| gcn | 92.30 ± 0.42 (n=10) | 92.41 ± 0.41 (n=10) | 89.08 ± 1.29 (n=10) | 83.39 ± 3.43 (n=10) | 63.17 ± 3.82 (n=10) | 60.65 ± 6.46 (n=10) |
| gcnii | 93.11 ± 0.46 (n=10) | 91.31 ± 0.54 (n=10) | 88.43 ± 1.33 (n=10) | 85.97 ± 1.74 (n=10) | 82.37 ± 1.39 (n=10) | 82.22 ± 1.81 (n=10) |
| graphunet | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 9 memory | failed: 10 memory |
| nhdp | 86.94 ± 3.71 (n=10) | 84.47 ± 7.95 (n=10) | 85.98 ± 3.71 (n=10) | 86.30 ± 4.35 (n=10) | 55.78 ± 19.63 (n=10) | 55.28 ± 21.19 (n=10) |
| pairnorm | 93.53 ± 0.55 (n=10) | 93.29 ± 0.48 (n=9) | 92.69 ± 0.51 (n=10) | 92.73 ± 0.52 (n=10) | 88.32 ± 1.51 (n=10) | 83.30 ± 1.38 (n=10) |
| sgc | 44.09 ± 5.27 (n=10) | 37.70 ± 7.32 (n=10) | 33.30 ± 6.07 (n=10) | 31.92 ± 5.11 (n=10) | 29.19 ± 3.07 (n=10) | 27.93 ± 2.12 (n=10) |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.00195 | 0.00195 | 0.00391 | 0.00391 | 0.00391 | 0.00391 |
| arma | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.0195 | 0.0137 |
| dropedge | 0.00195 | 0.00195 | 0.0645 | 0.322 | 0.922 | 1 |
| gat | 0.00195 | 0.00195 | 0.0371 | 0.389 | 0.00781 | 0.0156 |
| gcn | 0.00195 | 0.00195 | 0.232 | 0.221 | 0.492 | 0.695 |
| gcnii | 0.00195 | 0.0137 | 0.275 | 0.77 | 0.0273 | 0.084 |
| graphunet |  |  |  |  |  |  |
| pairnorm | 0.00195 | 0.00391 | 0.00195 | 0.00195 | 0.0156 | 0.084 |
| sgc | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.0156 | 0.0195 |

**pubmed** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 88.74 ± 0.39 (n=10) | 87.96 ± 0.33 (n=10) | 87.82 ± 0.32 (n=10) | 87.84 ± 0.32 (n=10) | 87.90 ± 0.30 (n=10) | 87.87 ± 0.29 (n=10) |
| arma | 87.86 ± 0.31 (n=10) | 87.02 ± 0.37 (n=10) | 81.20 ± 1.72 (n=10) | 47.01 ± 7.88 (n=10) | 40.35 ± 0.36 (n=10) | 40.25 ± 0.26 (n=10) |
| dropedge | 87.90 ± 0.33 (n=10) | 87.93 ± 0.38 (n=10) | 85.78 ± 0.54 (n=10) | 84.55 ± 0.61 (n=10) | 78.62 ± 5.78 (n=10) | 40.48 ± 0.37 (n=10) |
| gat | 86.77 ± 0.31 (n=10) | 85.12 ± 0.43 (n=9) | 84.03 ± 0.39 (n=10) | 83.91 ± 0.40 (n=10) | 53.52 ± 11.07 (n=10) | 41.48 ± 3.09 (n=10) |
| gcn | 87.86 ± 0.38 (n=10) | 87.25 ± 0.51 (n=10) | 84.73 ± 0.49 (n=10) | 84.27 ± 0.36 (n=10) | 68.26 ± 9.41 (n=10) | 47.55 ± 8.36 (n=10) |
| gcnii | 88.95 ± 0.34 (n=10) | 88.90 ± 0.40 (n=10) | 88.85 ± 0.35 (n=10) | 88.84 ± 0.38 (n=10) | 88.49 ± 0.39 (n=10) | 88.35 ± 0.32 (n=10) |
| graphunet | 88.70 ± 0.36 (n=10) | failed: 10 memory | failed: 5 memory, 5 time | failed: 4 memory, 6 time | failed: 4 memory, 5 time | failed: 3 memory, 7 time |
| nhdp | 87.59 ± 0.29 (n=10) | 87.57 ± 1.12 (n=10) | 87.67 ± 0.50 (n=10) | 87.66 ± 0.37 (n=10) | 87.72 ± 0.39 (n=10) | 87.83 ± 0.42 (n=10) |
| pairnorm | 88.00 ± 0.30 (n=10) | 87.10 ± 0.41 (n=9) | 85.87 ± 0.43 (n=10) | 85.28 ± 0.51 (n=10) | 84.29 ± 0.34 (n=10) | 83.54 ± 0.39 (n=10) |
| sgc | 80.46 ± 2.89 (n=10) | 78.83 ± 2.82 (n=10) | 77.03 ± 3.09 (n=10) | 76.50 ± 3.06 (n=10) | 70.65 ± 4.19 (n=10) | 65.19 ± 4.55 (n=8; failed: 2 time) |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.00195 | 0.375 | 1 | 0.201 | 0.152 | 0.846 |
| arma | 0.0645 | 0.105 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |
| dropedge | 0.0645 | 0.375 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |
| gat | 0.00195 | 0.00781 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |
| gcn | 0.129 | 0.203 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |
| gcnii | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00586 | 0.0137 |
| graphunet | 0.00195 |  |  |  |  |  |
| pairnorm | 0.00781 | 0.164 | 0.00195 | 0.00195 | 0.00195 | 0.00195 |
| sgc | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00195 | 0.00781 |

## embedding.centroid_gap

**citeseer** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.95 ± 0.08 (n=10) | 0.94 ± 0.08 (n=10) | 0.94 ± 0.07 (n=10) | 0.94 ± 0.07 (n=10) | 0.93 ± 0.09 (n=9) | 0.93 ± 0.09 (n=10) |
| arma | 0.99 ± 0.01 (n=10) | 0.93 ± 0.04 (n=10) | 0.46 ± 0.22 (n=10) | 0.01 ± 0.01 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| dropedge | 0.95 ± 0.05 (n=10) | 0.97 ± 0.03 (n=10) | 0.93 ± 0.03 (n=10) | 0.89 ± 0.06 (n=10) | 0.23 ± 0.22 (n=10) | -0.00 ± 0.00 (n=10) |
| gat | 0.95 ± 0.02 (n=10) | 0.94 ± 0.03 (n=9) | 0.94 ± 0.03 (n=10) | 0.93 ± 0.03 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| gcn | 0.95 ± 0.05 (n=10) | 0.95 ± 0.03 (n=10) | 0.93 ± 0.02 (n=10) | 0.93 ± 0.02 (n=10) | 0.79 ± 0.05 (n=10) | -0.00 ± 0.00 (n=10) |
| gcnii | 0.98 ± 0.01 (n=10) | 0.93 ± 0.03 (n=10) | 0.92 ± 0.04 (n=10) | 0.95 ± 0.03 (n=10) | 0.92 ± 0.03 (n=10) | 0.89 ± 0.03 (n=10) |
| graphunet | 0.90 ± 0.04 (n=10) | 0.90 ± 0.06 (n=10) | 0.89 ± 0.08 (n=10) | 0.86 ± 0.07 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 0.96 ± 0.08 (n=10) | 0.99 ± 0.04 (n=10) | 0.94 ± 0.07 (n=10) | 0.95 ± 0.03 (n=10) | 0.99 ± 0.02 (n=10) | 0.94 ± 0.03 (n=10) |
| pairnorm | 0.82 ± 0.05 (n=10) | 0.67 ± 0.04 (n=9) | 0.60 ± 0.05 (n=10) | 0.72 ± 0.06 (n=10) | 0.83 ± 0.10 (n=10) | 0.30 ± 0.23 (n=10) |
| sgc | 0.51 ± 0.07 (n=10) | 0.51 ± 0.06 (n=10) | 0.45 ± 0.07 (n=10) | 0.42 ± 0.08 (n=10) | 0.38 ± 0.07 (n=10) | 0.38 ± 0.07 (n=10) |

**cora** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.93 ± 0.03 (n=10) | 0.91 ± 0.04 (n=10) | 0.91 ± 0.04 (n=10) | 0.91 ± 0.04 (n=10) | 0.91 ± 0.04 (n=9) | 0.91 ± 0.03 (n=10) |
| arma | 0.94 ± 0.02 (n=10) | 0.91 ± 0.04 (n=10) | 0.62 ± 0.21 (n=10) | 0.03 ± 0.02 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| dropedge | 0.96 ± 0.02 (n=10) | 0.92 ± 0.01 (n=10) | 0.83 ± 0.03 (n=10) | 0.81 ± 0.04 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| gat | 0.99 ± 0.01 (n=10) | 0.98 ± 0.03 (n=10) | 0.97 ± 0.02 (n=10) | 0.98 ± 0.02 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| gcn | 0.96 ± 0.02 (n=10) | 0.91 ± 0.03 (n=10) | 0.75 ± 0.04 (n=10) | 0.79 ± 0.04 (n=10) | 0.47 ± 0.29 (n=10) | -0.00 ± 0.00 (n=10) |
| gcnii | 0.94 ± 0.02 (n=10) | 0.86 ± 0.03 (n=10) | 0.81 ± 0.03 (n=10) | 0.82 ± 0.03 (n=10) | 0.84 ± 0.03 (n=10) | 0.86 ± 0.02 (n=10) |
| graphunet | 0.90 ± 0.05 (n=10) | 0.78 ± 0.05 (n=10) | 0.79 ± 0.06 (n=10) | 0.84 ± 0.05 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 1.00 ± 0.02 (n=10) | 1.02 ± 0.02 (n=10) | 1.02 ± 0.02 (n=10) | 1.03 ± 0.01 (n=10) | 1.03 ± 0.02 (n=10) | 1.02 ± 0.02 (n=10) |
| pairnorm | 0.71 ± 0.03 (n=10) | 0.67 ± 0.03 (n=9) | 0.66 ± 0.02 (n=10) | 0.69 ± 0.03 (n=10) | 0.84 ± 0.03 (n=10) | 0.76 ± 0.05 (n=10) |
| sgc | 0.04 ± 0.07 (n=10) | 0.04 ± 0.06 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) |

**photo** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.93 ± 0.01 (n=10) | 0.91 ± 0.01 (n=10) | 0.89 ± 0.01 (n=10) | 0.88 ± 0.01 (n=10) | 0.87 ± 0.02 (n=9) | 0.88 ± 0.01 (n=10) |
| arma | 0.95 ± 0.01 (n=10) | 0.93 ± 0.03 (n=10) | 0.77 ± 0.06 (n=10) | 0.39 ± 0.14 (n=10) | 0.08 ± 0.13 (n=10) | -0.00 ± 0.00 (n=10) |
| dropedge | 0.92 ± 0.01 (n=10) | 0.89 ± 0.02 (n=10) | 0.82 ± 0.03 (n=10) | 0.80 ± 0.03 (n=10) | 0.56 ± 0.15 (n=10) | 0.56 ± 0.15 (n=10) |
| gat | 0.97 ± 0.01 (n=10) | 0.98 ± 0.02 (n=10) | 0.90 ± 0.04 (n=10) | 0.89 ± 0.04 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) |
| gcn | 0.91 ± 0.01 (n=10) | 0.89 ± 0.02 (n=10) | 0.82 ± 0.03 (n=10) | 0.80 ± 0.05 (n=10) | 0.66 ± 0.04 (n=10) | 0.59 ± 0.15 (n=10) |
| gcnii | 0.94 ± 0.02 (n=10) | 0.86 ± 0.04 (n=10) | 0.80 ± 0.04 (n=10) | 0.78 ± 0.05 (n=10) | 0.72 ± 0.05 (n=10) | 0.75 ± 0.03 (n=10) |
| graphunet | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 9 memory | failed: 10 memory |
| nhdp | 0.69 ± 0.18 (n=10) | 0.75 ± 0.16 (n=10) | 0.75 ± 0.15 (n=10) | 0.84 ± 0.09 (n=10) | 0.33 ± 0.32 (n=10) | 0.32 ± 0.30 (n=10) |
| pairnorm | 0.67 ± 0.02 (n=10) | 0.67 ± 0.01 (n=9) | 0.66 ± 0.02 (n=10) | 0.64 ± 0.02 (n=10) | 0.80 ± 0.02 (n=10) | 0.77 ± 0.02 (n=10) |
| sgc | 0.02 ± 0.02 (n=10) | 0.01 ± 0.01 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |

**pubmed** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 1.20 ± 0.03 (n=10) | 1.20 ± 0.04 (n=10) | 1.20 ± 0.05 (n=10) | 1.20 ± 0.05 (n=10) | 1.20 ± 0.04 (n=10) | 1.20 ± 0.05 (n=10) |
| arma | 1.23 ± 0.01 (n=10) | 1.21 ± 0.02 (n=10) | 1.09 ± 0.06 (n=10) | 0.11 ± 0.18 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| dropedge | 1.13 ± 0.02 (n=10) | 1.23 ± 0.03 (n=10) | 1.18 ± 0.05 (n=10) | 1.16 ± 0.04 (n=10) | 0.93 ± 0.24 (n=10) | -0.00 ± 0.00 (n=10) |
| gat | 1.10 ± 0.04 (n=10) | 1.17 ± 0.07 (n=9) | 1.10 ± 0.08 (n=10) | 1.15 ± 0.03 (n=10) | 0.12 ± 0.20 (n=10) | -0.00 ± 0.00 (n=10) |
| gcn | 1.16 ± 0.03 (n=10) | 1.21 ± 0.03 (n=10) | 1.16 ± 0.06 (n=10) | 1.15 ± 0.04 (n=10) | 0.80 ± 0.17 (n=10) | 0.22 ± 0.27 (n=10) |
| gcnii | 1.26 ± 0.01 (n=10) | 1.26 ± 0.01 (n=10) | 1.25 ± 0.01 (n=10) | 1.25 ± 0.01 (n=10) | 1.25 ± 0.02 (n=10) | 1.26 ± 0.01 (n=10) |
| graphunet | 1.16 ± 0.06 (n=10) | failed: 10 memory | failed: 5 memory, 5 time | failed: 4 memory, 6 time | failed: 4 memory, 5 time | failed: 3 memory, 7 time |
| nhdp | 1.27 ± 0.04 (n=10) | 1.17 ± 0.21 (n=10) | 1.23 ± 0.08 (n=10) | 1.23 ± 0.05 (n=10) | 1.21 ± 0.07 (n=10) | 1.11 ± 0.12 (n=10) |
| pairnorm | 1.26 ± 0.01 (n=10) | 1.26 ± 0.02 (n=9) | 1.22 ± 0.01 (n=10) | 1.21 ± 0.02 (n=10) | 1.18 ± 0.02 (n=10) | 1.22 ± 0.01 (n=10) |
| sgc | 1.10 ± 0.12 (n=10) | 1.05 ± 0.15 (n=10) | 0.94 ± 0.24 (n=10) | 0.90 ± 0.23 (n=10) | 0.47 ± 0.24 (n=10) | 0.11 ± 0.10 (n=8; failed: 2 time) |

## embedding.madgap

**citeseer** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.89 ± 0.05 (n=10) | 0.92 ± 0.05 (n=10) | 0.93 ± 0.04 (n=10) | 0.93 ± 0.04 (n=10) | 0.93 ± 0.05 (n=9) | 0.93 ± 0.06 (n=10) |
| arma | 0.91 ± 0.01 (n=10) | 0.92 ± 0.02 (n=10) | 0.57 ± 0.22 (n=10) | 0.04 ± 0.04 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| dropedge | 0.90 ± 0.02 (n=10) | 0.91 ± 0.01 (n=10) | 0.93 ± 0.01 (n=10) | 0.92 ± 0.03 (n=10) | 0.32 ± 0.29 (n=10) | -0.00 ± 0.00 (n=10) |
| gat | 0.91 ± 0.01 (n=10) | 0.94 ± 0.02 (n=9) | 0.94 ± 0.02 (n=10) | 0.94 ± 0.02 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) |
| gcn | 0.90 ± 0.02 (n=10) | 0.91 ± 0.02 (n=10) | 0.91 ± 0.02 (n=10) | 0.92 ± 0.02 (n=10) | 0.91 ± 0.02 (n=10) | -0.00 ± 0.00 (n=10) |
| gcnii | 0.89 ± 0.02 (n=10) | 0.89 ± 0.03 (n=10) | 0.92 ± 0.03 (n=10) | 0.92 ± 0.03 (n=10) | 0.92 ± 0.02 (n=10) | 0.90 ± 0.03 (n=10) |
| graphunet | 0.89 ± 0.02 (n=10) | 0.87 ± 0.04 (n=10) | 0.87 ± 0.04 (n=10) | 0.85 ± 0.04 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 0.66 ± 0.04 (n=10) | 0.69 ± 0.03 (n=10) | 0.68 ± 0.04 (n=10) | 0.68 ± 0.02 (n=10) | 0.74 ± 0.02 (n=10) | 0.72 ± 0.03 (n=10) |
| pairnorm | 0.82 ± 0.04 (n=10) | 0.76 ± 0.02 (n=9) | 0.74 ± 0.02 (n=10) | 0.77 ± 0.03 (n=10) | 0.89 ± 0.05 (n=10) | 0.51 ± 0.28 (n=10) |
| sgc | 0.61 ± 0.05 (n=10) | 0.65 ± 0.04 (n=10) | 0.63 ± 0.06 (n=10) | 0.62 ± 0.07 (n=10) | 0.61 ± 0.07 (n=10) | 0.63 ± 0.06 (n=10) |

**cora** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.63 ± 0.02 (n=10) | 0.66 ± 0.02 (n=10) | 0.68 ± 0.02 (n=10) | 0.68 ± 0.02 (n=10) | 0.68 ± 0.03 (n=9) | 0.69 ± 0.02 (n=10) |
| arma | 0.62 ± 0.01 (n=10) | 0.63 ± 0.03 (n=10) | 0.57 ± 0.17 (n=10) | 0.11 ± 0.06 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| dropedge | 0.65 ± 0.02 (n=10) | 0.63 ± 0.02 (n=10) | 0.62 ± 0.02 (n=10) | 0.63 ± 0.03 (n=10) | -0.00 ± 0.00 (n=10) | -0.00 ± 0.00 (n=10) |
| gat | 0.69 ± 0.01 (n=10) | 0.71 ± 0.03 (n=10) | 0.72 ± 0.03 (n=10) | 0.71 ± 0.04 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) |
| gcn | 0.65 ± 0.02 (n=10) | 0.63 ± 0.02 (n=10) | 0.54 ± 0.03 (n=10) | 0.58 ± 0.03 (n=10) | 0.40 ± 0.25 (n=10) | -0.00 ± 0.00 (n=10) |
| gcnii | 0.63 ± 0.02 (n=10) | 0.59 ± 0.03 (n=10) | 0.59 ± 0.03 (n=10) | 0.59 ± 0.03 (n=10) | 0.60 ± 0.03 (n=10) | 0.63 ± 0.04 (n=10) |
| graphunet | 0.64 ± 0.04 (n=10) | 0.59 ± 0.05 (n=10) | 0.58 ± 0.03 (n=10) | 0.61 ± 0.03 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 0.57 ± 0.03 (n=10) | 0.59 ± 0.02 (n=10) | 0.61 ± 0.02 (n=10) | 0.63 ± 0.02 (n=10) | 0.64 ± 0.01 (n=10) | 0.65 ± 0.03 (n=10) |
| pairnorm | 0.48 ± 0.02 (n=10) | 0.47 ± 0.02 (n=9) | 0.46 ± 0.02 (n=10) | 0.48 ± 0.03 (n=10) | 0.63 ± 0.02 (n=10) | 0.64 ± 0.02 (n=10) |
| sgc | 0.05 ± 0.06 (n=10) | 0.05 ± 0.06 (n=10) | 0.02 ± 0.01 (n=10) | 0.02 ± 0.01 (n=10) | 0.01 ± 0.01 (n=10) | 0.01 ± 0.00 (n=10) |

**photo** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.43 ± 0.01 (n=10) | 0.47 ± 0.01 (n=10) | 0.51 ± 0.01 (n=10) | 0.52 ± 0.01 (n=10) | 0.54 ± 0.01 (n=9) | 0.54 ± 0.01 (n=10) |
| arma | 0.35 ± 0.01 (n=10) | 0.36 ± 0.03 (n=10) | 0.53 ± 0.08 (n=10) | 0.18 ± 0.12 (n=10) | 0.05 ± 0.05 (n=10) | 0.01 ± 0.00 (n=10) |
| dropedge | 0.45 ± 0.01 (n=10) | 0.45 ± 0.02 (n=10) | 0.44 ± 0.02 (n=10) | 0.48 ± 0.04 (n=10) | 0.41 ± 0.12 (n=10) | 0.44 ± 0.12 (n=10) |
| gat | 0.43 ± 0.01 (n=10) | 0.47 ± 0.02 (n=10) | 0.44 ± 0.03 (n=10) | 0.47 ± 0.05 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) |
| gcn | 0.45 ± 0.01 (n=10) | 0.44 ± 0.02 (n=10) | 0.45 ± 0.04 (n=10) | 0.46 ± 0.06 (n=10) | 0.51 ± 0.03 (n=10) | 0.50 ± 0.10 (n=10) |
| gcnii | 0.39 ± 0.03 (n=10) | 0.44 ± 0.05 (n=10) | 0.45 ± 0.07 (n=10) | 0.49 ± 0.07 (n=10) | 0.49 ± 0.04 (n=10) | 0.53 ± 0.04 (n=10) |
| graphunet | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 9 memory | failed: 10 memory |
| nhdp | 0.28 ± 0.06 (n=10) | 0.36 ± 0.07 (n=10) | 0.31 ± 0.06 (n=10) | 0.39 ± 0.05 (n=10) | 0.15 ± 0.15 (n=10) | 0.18 ± 0.17 (n=10) |
| pairnorm | 0.26 ± 0.01 (n=10) | 0.25 ± 0.01 (n=9) | 0.24 ± 0.01 (n=10) | 0.23 ± 0.01 (n=10) | 0.36 ± 0.04 (n=10) | 0.30 ± 0.05 (n=10) |
| sgc | 0.03 ± 0.03 (n=10) | 0.01 ± 0.02 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) | 0.00 ± 0.00 (n=10) |

**pubmed** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.81 ± 0.02 (n=10) | 0.85 ± 0.02 (n=10) | 0.87 ± 0.02 (n=10) | 0.87 ± 0.02 (n=10) | 0.87 ± 0.02 (n=10) | 0.87 ± 0.02 (n=10) |
| arma | 0.82 ± 0.01 (n=10) | 0.81 ± 0.01 (n=10) | 0.85 ± 0.04 (n=10) | 0.13 ± 0.19 (n=10) | -0.01 ± 0.00 (n=10) | -0.01 ± 0.01 (n=10) |
| dropedge | 0.81 ± 0.02 (n=10) | 0.85 ± 0.02 (n=10) | 0.90 ± 0.03 (n=10) | 0.91 ± 0.02 (n=10) | 0.80 ± 0.17 (n=10) | -0.00 ± 0.00 (n=10) |
| gat | 0.84 ± 0.02 (n=10) | 0.90 ± 0.03 (n=9) | 0.88 ± 0.04 (n=10) | 0.92 ± 0.01 (n=10) | 0.15 ± 0.22 (n=10) | 0.00 ± 0.00 (n=10) |
| gcn | 0.82 ± 0.02 (n=10) | 0.82 ± 0.02 (n=10) | 0.89 ± 0.05 (n=10) | 0.91 ± 0.03 (n=10) | 0.77 ± 0.06 (n=10) | 0.23 ± 0.27 (n=10) |
| gcnii | 0.82 ± 0.01 (n=10) | 0.81 ± 0.01 (n=10) | 0.82 ± 0.01 (n=10) | 0.82 ± 0.01 (n=10) | 0.84 ± 0.01 (n=10) | 0.85 ± 0.01 (n=10) |
| graphunet | 0.79 ± 0.02 (n=10) | failed: 10 memory | failed: 5 memory, 5 time | failed: 4 memory, 6 time | failed: 4 memory, 5 time | failed: 3 memory, 7 time |
| nhdp | 0.75 ± 0.02 (n=10) | 0.70 ± 0.11 (n=10) | 0.76 ± 0.04 (n=10) | 0.76 ± 0.03 (n=10) | 0.76 ± 0.04 (n=10) | 0.71 ± 0.06 (n=10) |
| pairnorm | 0.84 ± 0.01 (n=10) | 0.83 ± 0.01 (n=9) | 0.84 ± 0.01 (n=10) | 0.85 ± 0.01 (n=10) | 0.88 ± 0.01 (n=10) | 0.93 ± 0.01 (n=10) |
| sgc | 0.77 ± 0.05 (n=10) | 0.79 ± 0.06 (n=10) | 0.74 ± 0.18 (n=10) | 0.73 ± 0.18 (n=10) | 0.52 ± 0.25 (n=10) | 0.31 ± 0.27 (n=8; failed: 2 time) |

## timings.epoch_s_mean (seconds per training epoch (one thread))

**citeseer** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.33 ± 0.02 (n=10) | 0.30 ± 0.03 (n=10) | 0.35 ± 0.03 (n=10) | 0.32 ± 0.04 (n=10) | 0.31 ± 0.04 (n=9) | 0.35 ± 0.04 (n=10) |
| arma | 0.62 ± 0.07 (n=10) | 0.68 ± 0.08 (n=10) | 0.75 ± 0.08 (n=10) | 0.75 ± 0.07 (n=10) | 0.91 ± 0.13 (n=10) | 1.11 ± 0.12 (n=10) |
| dropedge | 0.32 ± 0.03 (n=10) | 0.32 ± 0.04 (n=10) | 0.37 ± 0.04 (n=10) | 0.32 ± 0.04 (n=10) | 0.43 ± 0.05 (n=10) | 0.54 ± 0.03 (n=10) |
| gat | 0.32 ± 0.04 (n=10) | 0.35 ± 0.04 (n=9) | 0.42 ± 0.05 (n=10) | 0.42 ± 0.05 (n=10) | 0.59 ± 0.07 (n=10) | 0.76 ± 0.08 (n=10) |
| gcn | 0.28 ± 0.04 (n=10) | 0.32 ± 0.03 (n=10) | 0.35 ± 0.04 (n=10) | 0.40 ± 0.03 (n=10) | 0.46 ± 0.04 (n=10) | 0.60 ± 0.04 (n=10) |
| gcnii | 0.36 ± 0.02 (n=10) | 0.32 ± 0.04 (n=10) | 0.38 ± 0.04 (n=10) | 0.35 ± 0.03 (n=10) | 0.44 ± 0.02 (n=10) | 0.62 ± 0.03 (n=10) |
| graphunet | 0.35 ± 0.03 (n=10) | 0.34 ± 0.04 (n=10) | 0.46 ± 0.05 (n=10) | 0.45 ± 0.08 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 0.36 ± 0.04 (n=10) | 0.39 ± 0.05 (n=10) | 0.51 ± 0.06 (n=10) | 0.51 ± 0.06 (n=10) | 0.76 ± 0.05 (n=10) | 0.93 ± 0.06 (n=10) |
| pairnorm | 0.32 ± 0.02 (n=10) | 0.36 ± 0.04 (n=9) | 0.39 ± 0.04 (n=10) | 0.39 ± 0.05 (n=10) | 0.51 ± 0.05 (n=10) | 0.62 ± 0.06 (n=10) |
| sgc | 0.56 ± 0.05 (n=10) | 1.09 ± 0.12 (n=10) | 2.29 ± 0.19 (n=10) | 2.42 ± 0.25 (n=10) | 4.81 ± 0.43 (n=10) | 7.48 ± 0.68 (n=10) |

**cora** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.09 ± 0.01 (n=10) | 0.09 ± 0.01 (n=10) | 0.11 ± 0.01 (n=10) | 0.10 ± 0.01 (n=10) | 0.10 ± 0.02 (n=9) | 0.13 ± 0.02 (n=10) |
| arma | 0.20 ± 0.02 (n=10) | 0.24 ± 0.03 (n=10) | 0.29 ± 0.03 (n=10) | 0.31 ± 0.04 (n=10) | 0.44 ± 0.07 (n=10) | 0.61 ± 0.09 (n=10) |
| dropedge | 0.10 ± 0.01 (n=10) | 0.09 ± 0.01 (n=10) | 0.13 ± 0.01 (n=10) | 0.13 ± 0.01 (n=10) | 0.19 ± 0.01 (n=10) | 0.31 ± 0.02 (n=10) |
| gat | 0.11 ± 0.01 (n=10) | 0.12 ± 0.01 (n=10) | 0.19 ± 0.02 (n=10) | 0.20 ± 0.02 (n=10) | 0.35 ± 0.03 (n=10) | 0.51 ± 0.04 (n=10) |
| gcn | 0.09 ± 0.01 (n=10) | 0.11 ± 0.01 (n=10) | 0.13 ± 0.01 (n=10) | 0.16 ± 0.01 (n=10) | 0.22 ± 0.02 (n=10) | 0.32 ± 0.02 (n=10) |
| gcnii | 0.09 ± 0.01 (n=10) | 0.10 ± 0.01 (n=10) | 0.14 ± 0.01 (n=10) | 0.17 ± 0.01 (n=10) | 0.22 ± 0.02 (n=10) | 0.34 ± 0.02 (n=10) |
| graphunet | 0.11 ± 0.01 (n=10) | 0.18 ± 0.03 (n=10) | 0.40 ± 0.09 (n=10) | 0.36 ± 0.07 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 0.13 ± 0.01 (n=10) | 0.14 ± 0.02 (n=10) | 0.23 ± 0.02 (n=10) | 0.24 ± 0.02 (n=10) | 0.40 ± 0.03 (n=10) | 0.70 ± 0.04 (n=10) |
| pairnorm | 0.09 ± 0.01 (n=10) | 0.10 ± 0.01 (n=9) | 0.15 ± 0.02 (n=10) | 0.15 ± 0.02 (n=10) | 0.25 ± 0.03 (n=10) | 0.33 ± 0.03 (n=10) |
| sgc | 0.22 ± 0.02 (n=10) | 0.43 ± 0.04 (n=10) | 0.84 ± 0.08 (n=10) | 0.92 ± 0.10 (n=10) | 1.96 ± 0.20 (n=10) | 3.15 ± 0.33 (n=10) |

**photo** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.20 ± 0.02 (n=10) | 0.24 ± 0.02 (n=10) | 0.35 ± 0.04 (n=10) | 0.34 ± 0.05 (n=10) | 0.50 ± 0.12 (n=9) | 0.74 ± 0.14 (n=10) |
| arma | 0.59 ± 0.08 (n=10) | 1.18 ± 0.16 (n=10) | 2.18 ± 0.29 (n=10) | 2.23 ± 0.27 (n=10) | 4.83 ± 0.80 (n=10) | 8.50 ± 1.18 (n=10) |
| dropedge | 0.29 ± 0.03 (n=10) | 0.56 ± 0.08 (n=10) | 1.14 ± 0.11 (n=10) | 1.31 ± 0.12 (n=10) | 2.65 ± 0.23 (n=10) | 4.16 ± 0.40 (n=10) |
| gat | 0.50 ± 0.04 (n=10) | 0.97 ± 0.08 (n=10) | 2.28 ± 0.23 (n=10) | 2.50 ± 0.22 (n=10) | 5.17 ± 0.29 (n=10) | 9.16 ± 0.81 (n=10) |
| gcn | 0.33 ± 0.04 (n=10) | 0.71 ± 0.05 (n=10) | 1.55 ± 0.17 (n=10) | 1.91 ± 0.15 (n=10) | 3.22 ± 0.34 (n=10) | 5.38 ± 0.52 (n=10) |
| gcnii | 0.49 ± 0.05 (n=10) | 0.87 ± 0.07 (n=10) | 1.76 ± 0.19 (n=10) | 1.80 ± 0.12 (n=10) | 3.21 ± 0.30 (n=10) | 5.16 ± 0.45 (n=10) |
| graphunet | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 9 memory | failed: 10 memory |
| nhdp | 0.76 ± 0.09 (n=10) | 1.44 ± 0.17 (n=10) | 2.63 ± 0.27 (n=10) | 2.59 ± 0.24 (n=10) | 5.24 ± 0.78 (n=10) | 9.17 ± 0.86 (n=10) |
| pairnorm | 0.33 ± 0.03 (n=10) | 0.64 ± 0.07 (n=9) | 1.43 ± 0.15 (n=10) | 1.53 ± 0.17 (n=10) | 3.11 ± 0.34 (n=10) | 4.51 ± 0.56 (n=10) |
| sgc | 1.92 ± 0.19 (n=10) | 3.93 ± 0.53 (n=10) | 8.25 ± 0.73 (n=10) | 8.55 ± 0.87 (n=10) | 17.00 ± 1.64 (n=10) | 27.17 ± 2.74 (n=10) |

**pubmed** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.31 ± 0.02 (n=10) | 0.27 ± 0.04 (n=10) | 0.31 ± 0.03 (n=10) | 0.28 ± 0.02 (n=10) | 0.33 ± 0.02 (n=10) | 0.45 ± 0.04 (n=10) |
| arma | 0.60 ± 0.03 (n=10) | 1.07 ± 0.07 (n=10) | 1.42 ± 0.17 (n=10) | 1.62 ± 0.17 (n=10) | 2.79 ± 0.45 (n=10) | 4.31 ± 0.33 (n=10) |
| dropedge | 0.32 ± 0.03 (n=10) | 0.42 ± 0.05 (n=10) | 0.68 ± 0.05 (n=10) | 0.65 ± 0.07 (n=10) | 1.18 ± 0.11 (n=10) | 1.81 ± 0.22 (n=10) |
| gat | 0.42 ± 0.05 (n=10) | 0.62 ± 0.08 (n=9) | 1.16 ± 0.15 (n=10) | 1.24 ± 0.15 (n=10) | 2.32 ± 0.29 (n=10) | 3.58 ± 0.42 (n=10) |
| gcn | 0.32 ± 0.04 (n=10) | 0.54 ± 0.05 (n=10) | 0.95 ± 0.11 (n=10) | 1.20 ± 0.08 (n=10) | 1.99 ± 0.17 (n=10) | 3.22 ± 0.24 (n=10) |
| gcnii | 0.45 ± 0.05 (n=10) | 0.59 ± 0.04 (n=10) | 1.01 ± 0.11 (n=10) | 1.22 ± 0.09 (n=10) | 1.75 ± 0.16 (n=10) | 2.60 ± 0.21 (n=10) |
| graphunet | 0.88 ± 0.13 (n=10) | failed: 10 memory | failed: 5 memory, 5 time | failed: 4 memory, 6 time | failed: 4 memory, 5 time | failed: 3 memory, 7 time |
| nhdp | 0.58 ± 0.06 (n=10) | 0.89 ± 0.10 (n=10) | 1.30 ± 0.12 (n=10) | 1.48 ± 0.19 (n=10) | 2.60 ± 0.36 (n=10) | 4.00 ± 0.33 (n=10) |
| pairnorm | 0.37 ± 0.04 (n=10) | 0.62 ± 0.07 (n=9) | 1.11 ± 0.10 (n=10) | 1.18 ± 0.12 (n=10) | 2.11 ± 0.28 (n=10) | 2.84 ± 0.28 (n=10) |
| sgc | 0.66 ± 0.07 (n=10) | 1.27 ± 0.13 (n=10) | 2.72 ± 0.23 (n=10) | 2.73 ± 0.25 (n=10) | 5.63 ± 0.41 (n=10) | 9.07 ± 1.59 (n=8; failed: 2 time) |

## peak_rss_mb (peak resident memory of the run's process in MB)

**citeseer** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 746.92 ± 0.67 (n=10) | 747.08 ± 0.46 (n=10) | 748.63 ± 0.81 (n=10) | 750.01 ± 0.62 (n=10) | 750.93 ± 0.67 (n=9) | 749.79 ± 0.67 (n=10) |
| arma | 804.08 ± 1.04 (n=10) | 814.20 ± 2.18 (n=10) | 850.96 ± 1.81 (n=10) | 865.91 ± 2.16 (n=10) | 943.99 ± 3.05 (n=10) | 1034.49 ± 4.01 (n=10) |
| dropedge | 752.55 ± 1.16 (n=6) | 759.54 ± 2.66 (n=2) | 771.59 ± 1.03 (n=6) | 779.14 ± 0.75 (n=10) | 805.57 ± 14.40 (n=6) | 836.85 ± 25.02 (n=6) |
| gat | 764.77 ± 0.51 (n=9) | 787.03 ± 3.45 (n=7) | 838.35 ± 1.75 (n=10) | 864.31 ± 2.24 (n=10) | 986.95 ± 1.92 (n=10) | 1104.63 ± 3.75 (n=10) |
| gcn | 753.30 ± 0.56 (n=8) | 758.65 ± 0.44 (n=8) | 772.36 ± 0.76 (n=10) | 779.05 ± 0.53 (n=10) | 815.65 ± 8.92 (n=8) | 848.22 ± 14.80 (n=8) |
| graphunet | 941.79 ± 7.54 (n=8) | 1269.67 ± 50.76 (n=8) | 1734.05 ± 171.16 (n=10) | 1900.31 ± 321.34 (n=10) | failed: 9 error | failed: 10 error |
| nhdp | 821.78 ± 4.16 (n=6) | 854.99 ± 18.89 (n=2) | 922.75 ± 3.41 (n=6) | 952.60 ± 2.92 (n=10) | 1101.86 ± 6.14 (n=6) | 1259.50 ± 11.12 (n=6) |
| pairnorm |  | 765.96 ± 3.50 (n=3) | 794.01 ± 1.17 (n=10) | 809.58 ± 2.07 (n=10) | 882.56 ± 3.89 (n=8) | 959.15 ± 11.43 (n=4) |
| sgc | 1087.91 ± 0.51 (n=10) | 1087.62 ± 0.56 (n=10) | 1087.52 ± 0.56 (n=10) | 1087.80 ± 0.50 (n=10) | 1088.24 ± 0.20 (n=10) | 1087.74 ± 0.50 (n=10) |

**cora** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 632.65 ± 0.99 (n=8) | 633.13 ± 0.60 (n=8) | 633.35 ± 1.40 (n=10) | 635.05 ± 1.10 (n=10) | 636.28 ± 1.29 (n=9) | 637.07 ± 0.77 (n=10) |
| arma | 668.85 ± 6.05 (n=10) | 662.93 ± 6.84 (n=10) | 708.84 ± 4.46 (n=10) | 713.03 ± 4.26 (n=10) | 784.26 ± 7.38 (n=10) | 855.87 ± 5.27 (n=10) |
| gat | 637.48 ± 0.96 (n=6) | 645.25 ± 4.59 (n=2) | 701.57 ± 3.39 (n=6) | 723.84 ± 3.34 (n=10) | 846.27 ± 5.90 (n=6) | 965.33 ± 12.66 (n=6) |
| gcn |  | 638.34 ± 2.44 (n=4) | 650.54 ± 2.97 (n=10) | 651.77 ± 3.80 (n=10) | 672.95 ± 3.44 (n=8) | 702.41 ± 11.41 (n=4) |
| gcnii | 637.84 ± 1.02 (n=10) | 642.39 ± 4.18 (n=10) | 657.32 ± 4.26 (n=10) | 663.39 ± 2.37 (n=10) | 696.11 ± 5.85 (n=10) | 734.77 ± 3.02 (n=10) |
| graphunet |  | 2087.17 ± 307.50 (n=4) | 3714.57 ± 376.89 (n=10) | 3924.19 ± 692.08 (n=10) | failed: 9 error | failed: 10 error |
| pairnorm | 634.39 ± 0.86 (n=10) | 638.60 ± 2.73 (n=9) | 661.89 ± 3.03 (n=10) | 672.39 ± 4.82 (n=10) | 728.33 ± 4.51 (n=10) | 788.00 ± 6.32 (n=10) |
| sgc | 751.18 ± 0.81 (n=8) | 779.50 ± 0.58 (n=8) | 765.67 ± 0.97 (n=10) | 776.07 ± 4.21 (n=10) | 766.37 ± 0.78 (n=8) | 766.03 ± 1.00 (n=8) |

**photo** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 755.34 ± 3.64 (n=6) | 756.99 ± 2.51 (n=6) | 757.84 ± 1.21 (n=10) | 760.82 ± 2.22 (n=10) | 797.91 ± 16.78 (n=9) | 835.13 ± 20.96 (n=6) |
| arma | 825.08 ± 3.68 (n=8) | 874.66 ± 7.12 (n=8) | 977.03 ± 6.17 (n=10) | 1025.58 ± 5.00 (n=10) | 1202.92 ± 11.91 (n=10) | 1400.38 ± 17.11 (n=10) |
| dropedge | 784.13 ± 1.51 (n=10) | 804.94 ± 0.94 (n=10) | 844.53 ± 1.88 (n=10) | 865.86 ± 2.28 (n=10) | 984.60 ± 4.16 (n=10) | 1093.54 ± 4.93 (n=10) |
| gat |  |  |  | 1904.83 ± 28.37 (n=3) |  |  |
| gcn | 797.26 ± 2.26 (n=10) | 820.70 ± 1.12 (n=10) | 866.42 ± 1.48 (n=10) | 888.81 ± 1.84 (n=10) | 1013.15 ± 5.94 (n=10) | 1155.03 ± 10.52 (n=10) |
| gcnii | 807.00 ± 3.06 (n=10) | 831.32 ± 3.58 (n=10) | 888.16 ± 1.54 (n=10) | 920.91 ± 2.23 (n=10) | 1075.63 ± 3.57 (n=10) | 1228.57 ± 3.23 (n=10) |
| graphunet | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 10 memory | failed: 9 memory | failed: 10 memory |
| nhdp | 1105.97 ± 7.08 (n=10) | 1324.94 ± 12.77 (n=10) | 1828.97 ± 9.07 (n=10) | 2076.66 ± 17.36 (n=10) | 3256.03 ± 13.20 (n=10) | 4439.86 ± 21.33 (n=10) |
| pairnorm | 799.44 ± 1.64 (n=10) | 833.97 ± 1.73 (n=9) | 913.19 ± 4.86 (n=10) | 963.53 ± 4.36 (n=10) | 1174.95 ± 2.55 (n=10) | 1381.91 ± 3.02 (n=10) |
| sgc | 2056.75 (n=1) | 2075.53 ± 3.59 (n=4) | 2075.08 ± 1.50 (n=10) | 2075.30 ± 1.21 (n=10) | 2075.27 ± 0.91 (n=10) | 2075.00 ± 1.13 (n=10) |

**pubmed** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma |  | 1022.89 ± 22.37 (n=4) | 1254.19 ± 10.43 (n=10) | 1360.07 ± 13.99 (n=10) | 1887.96 ± 25.15 (n=8) | 2368.25 ± 54.43 (n=4) |
| dropedge | 1002.55 ± 1.84 (n=10) | 1025.17 ± 2.25 (n=10) | 1069.71 ± 6.13 (n=10) | 1101.96 ± 18.91 (n=10) | 1239.99 ± 30.99 (n=10) | 1427.00 ± 42.49 (n=10) |
| gat | 1023.00 ± 3.66 (n=10) | 1155.10 ± 35.92 (n=9) | 1478.57 ± 48.66 (n=10) | 1616.41 ± 27.50 (n=10) | 2559.49 ± 44.11 (n=10) | 3390.74 ± 44.50 (n=10) |
| gcn | 1004.03 ± 1.63 (n=10) | 1027.55 ± 2.10 (n=10) | 1081.99 ± 15.21 (n=10) | 1103.65 ± 6.44 (n=10) | 1297.34 ± 16.68 (n=10) | 1485.74 ± 19.00 (n=10) |
| gcnii | 1023.53 ± 2.22 (n=6) | 1056.52 ± 59.01 (n=2) | 1125.39 ± 25.29 (n=10) | 1192.46 ± 20.29 (n=10) | 1479.36 ± 22.11 (n=10) | 1734.58 ± 30.53 (n=6) |
| graphunet | 10021.67 ± 336.44 (n=10) | failed: 10 memory | failed: 5 memory, 5 time | failed: 4 memory, 6 time | failed: 4 memory, 5 time | failed: 3 memory, 7 time |
| nhdp | 1137.86 ± 15.73 (n=10) | 1303.80 ± 16.84 (n=10) | 1811.96 ± 32.92 (n=10) | 2008.08 ± 42.94 (n=10) | 2912.78 ± 40.23 (n=10) | 3814.11 ± 53.54 (n=10) |
| pairnorm | 1002.37 ± 2.25 (n=8) | 1034.39 ± 4.38 (n=7) | 1100.55 ± 24.82 (n=10) | 1215.35 ± 21.13 (n=10) | 1581.94 ± 34.81 (n=10) | 2015.14 ± 44.27 (n=10) |
| sgc | 1122.25 ± 1.73 (n=10) | 1121.76 ± 1.54 (n=10) | 1122.06 ± 1.78 (n=10) | 1121.89 ± 1.66 (n=10) | 1122.06 ± 1.36 (n=10) | 1122.22 ± 2.02 (n=8; failed: 2 time) |

## Failed runs

| key | dataset | model | tag | depth | seed | reason | detail |
|---|---|---|---|---|---|---|---|
| f92dbee9fa48 | citeseer | graphunet |  | 20 | 0 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 688eac4e4501 | citeseer | graphunet |  | 20 | 1 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 420c82084ee0 | citeseer | graphunet |  | 20 | 2 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 9ab9502a8a8e | citeseer | graphunet |  | 20 | 4 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 6ba919485506 | citeseer | graphunet |  | 20 | 5 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| d36859cdfeba | citeseer | graphunet |  | 20 | 6 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| ad44d5caf93a | citeseer | graphunet |  | 20 | 7 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| f8b0241be37f | citeseer | graphunet |  | 20 | 8 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| ef88aea2a737 | citeseer | graphunet |  | 20 | 9 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| a4643ff2f4d1 | citeseer | graphunet |  | 30 | 0 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| d3aca2e1c397 | citeseer | graphunet |  | 30 | 1 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 1468d6466526 | citeseer | graphunet |  | 30 | 2 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 18c1ed9f1c72 | citeseer | graphunet |  | 30 | 3 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| e2e2ca517494 | citeseer | graphunet |  | 30 | 4 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 2fe1d840db00 | citeseer | graphunet |  | 30 | 5 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 1f18c7bbecb6 | citeseer | graphunet |  | 30 | 6 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 47d0296f725d | citeseer | graphunet |  | 30 | 7 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 6b1d407f6146 | citeseer | graphunet |  | 30 | 8 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| b79ed2b611c0 | citeseer | graphunet |  | 30 | 9 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| f6260c918d5d | cora | graphunet |  | 20 | 0 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 48e774657f3e | cora | graphunet |  | 20 | 1 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 8743bdd4648d | cora | graphunet |  | 20 | 2 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 1a7d7f9f3a8a | cora | graphunet |  | 20 | 4 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 40c831f0dd86 | cora | graphunet |  | 20 | 5 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| a9f6bd595418 | cora | graphunet |  | 20 | 6 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 946e4ffa7d6f | cora | graphunet |  | 20 | 7 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 1a6162836290 | cora | graphunet |  | 20 | 8 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 8735029592cc | cora | graphunet |  | 20 | 9 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 10f4e6c8f9b7 | cora | graphunet |  | 30 | 0 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| fab52af26d71 | cora | graphunet |  | 30 | 1 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 5aef4542cdad | cora | graphunet |  | 30 | 2 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 76cbccfc79ea | cora | graphunet |  | 30 | 3 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 18905db7b581 | cora | graphunet |  | 30 | 4 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 98d64bfe8463 | cora | graphunet |  | 30 | 5 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 4a080cfc7a7e | cora | graphunet |  | 30 | 6 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 591ed36173bc | cora | graphunet |  | 30 | 7 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| 2d21673050f0 | cora | graphunet |  | 30 | 8 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| caba6821980e | cora | graphunet |  | 30 | 9 | error | KMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifi... |
| d04acbdcf3ae | photo | graphunet |  | 2 | 0 | memory | exceeded the 13 GB memory budget while running alone (13321 MB) |
| ed3c73db5cc1 | photo | graphunet |  | 2 | 1 | memory | exceeded the 13 GB memory budget while running alone (13322 MB) |
| 7c763923d92a | photo | graphunet |  | 2 | 2 | memory | exceeded the 13 GB memory budget while running alone (13321 MB) |
| 1f7c1ef65992 | photo | graphunet |  | 2 | 3 | memory | exceeded the 13 GB memory budget while running alone (13317 MB) |
| cbe73d7635b4 | photo | graphunet |  | 2 | 4 | memory | exceeded the 13 GB memory budget while running alone (13339 MB) |
| 77f35b2c5574 | photo | graphunet |  | 2 | 5 | memory | exceeded the 13 GB memory budget while running alone (13358 MB) |
| 44ff87b56e95 | photo | graphunet |  | 2 | 6 | memory | exceeded the 13 GB memory budget while running alone (13400 MB) |
| 79e73dd734e6 | photo | graphunet |  | 2 | 7 | memory | exceeded the 13 GB memory budget while running alone (13360 MB) |
| eae862da7157 | photo | graphunet |  | 2 | 8 | memory | exceeded the 13 GB memory budget while running alone (13329 MB) |
| b52ca065390d | photo | graphunet |  | 2 | 9 | memory | exceeded the 13 GB memory budget while running alone (13411 MB) |
| 54ae5ab5b793 | photo | graphunet |  | 4 | 0 | memory | exceeded the 13 GB memory budget while running alone (13434 MB) |
| 7f0aebd411b1 | photo | graphunet |  | 4 | 1 | memory | exceeded the 13 GB memory budget while running alone (13371 MB) |
| 50c744191907 | photo | graphunet |  | 4 | 2 | memory | exceeded the 13 GB memory budget while running alone (13314 MB) |
| c4529e087041 | photo | graphunet |  | 4 | 3 | memory | exceeded the 13 GB memory budget while running alone (13465 MB) |
| 6487a9255a81 | photo | graphunet |  | 4 | 4 | memory | exceeded the 13 GB memory budget while running alone (13316 MB) |
| 58813763e82a | photo | graphunet |  | 4 | 5 | memory | exceeded the 13 GB memory budget while running alone (13343 MB) |
| 2b213b7d8d4e | photo | graphunet |  | 4 | 6 | memory | exceeded the 13 GB memory budget while running alone (13492 MB) |
| 7c8c5dde1830 | photo | graphunet |  | 4 | 7 | memory | exceeded the 13 GB memory budget while running alone (13341 MB) |
| 8e4f664f81a7 | photo | graphunet |  | 4 | 8 | memory | exceeded the 13 GB memory budget while running alone (13322 MB) |
| a9e13b1c20bf | photo | graphunet |  | 4 | 9 | memory | exceeded the 13 GB memory budget while running alone (13353 MB) |
| 9e3001786bf9 | photo | graphunet |  | 8 | 0 | memory | exceeded the 13 GB memory budget while running alone (13325 MB) |
| 0cf01d46b185 | photo | graphunet |  | 8 | 1 | memory | exceeded the 13 GB memory budget while running alone (13368 MB) |
| 86f7db14744a | photo | graphunet |  | 8 | 2 | memory | exceeded the 13 GB memory budget while running alone (13444 MB) |
| 9555a0ce8466 | photo | graphunet |  | 8 | 3 | memory | exceeded the 13 GB memory budget while running alone (13402 MB) |
| 1f139a18bd08 | photo | graphunet |  | 8 | 4 | memory | exceeded the 13 GB memory budget while running alone (13397 MB) |
| 92cdf815c06c | photo | graphunet |  | 8 | 5 | memory | exceeded the 13 GB memory budget while running alone (13425 MB) |
| 0658a2cc8dc6 | photo | graphunet |  | 8 | 6 | memory | exceeded the 13 GB memory budget while running alone (13544 MB) |
| 4288b6391087 | photo | graphunet |  | 8 | 7 | memory | exceeded the 13 GB memory budget while running alone (13371 MB) |
| 521768692b18 | photo | graphunet |  | 8 | 8 | memory | exceeded the 13 GB memory budget while running alone (13323 MB) |
| 994942289d1b | photo | graphunet |  | 8 | 9 | memory | exceeded the 13 GB memory budget while running alone (13327 MB) |
| 5282108069f2 | photo | graphunet |  | 10 | 0 | memory | exceeded the 13 GB memory budget while running alone (13390 MB) |
| 7e730943522c | photo | graphunet |  | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13340 MB) |
| 6499d1869ac8 | photo | graphunet |  | 10 | 2 | memory | exceeded the 13 GB memory budget while running alone (13406 MB) |
| 1ec4fa8c8d37 | photo | graphunet |  | 10 | 3 | memory | exceeded the 13 GB memory budget while running alone (13321 MB) |
| 138eac5c588c | photo | graphunet |  | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13367 MB) |
| 343785e934ff | photo | graphunet |  | 10 | 5 | memory | exceeded the 13 GB memory budget while running alone (13446 MB) |
| d8b2a1bf693c | photo | graphunet |  | 10 | 6 | memory | exceeded the 13 GB memory budget while running alone (13494 MB) |
| d0605e2e95b9 | photo | graphunet |  | 10 | 7 | memory | exceeded the 13 GB memory budget while running alone (13360 MB) |
| 01ad419c2e5f | photo | graphunet |  | 10 | 8 | memory | exceeded the 13 GB memory budget while running alone (13316 MB) |
| 989ec8f8b32b | photo | graphunet |  | 10 | 9 | memory | exceeded the 13 GB memory budget while running alone (13319 MB) |
| 6bac370b127f | photo | graphunet |  | 20 | 0 | memory | exceeded the 13 GB memory budget while running alone (13494 MB) |
| 10a7a558c458 | photo | graphunet |  | 20 | 1 | memory | exceeded the 13 GB memory budget while running alone (13519 MB) |
| c4ddcc9d9b58 | photo | graphunet |  | 20 | 2 | memory | exceeded the 13 GB memory budget while running alone (13314 MB) |
| a12eea0fc072 | photo | graphunet |  | 20 | 4 | memory | exceeded the 13 GB memory budget while running alone (13510 MB) |
| 2162be720c63 | photo | graphunet |  | 20 | 5 | memory | exceeded the 13 GB memory budget while running alone (13343 MB) |
| c633c0100e66 | photo | graphunet |  | 20 | 6 | memory | exceeded the 13 GB memory budget while running alone (13340 MB) |
| 7c5d75af43f2 | photo | graphunet |  | 20 | 7 | memory | exceeded the 13 GB memory budget while running alone (13380 MB) |
| 1d2bebb11ed5 | photo | graphunet |  | 20 | 8 | memory | exceeded the 13 GB memory budget while running alone (13409 MB) |
| 3765686351ec | photo | graphunet |  | 20 | 9 | memory | exceeded the 13 GB memory budget while running alone (13393 MB) |
| f20184a6e020 | photo | graphunet |  | 30 | 0 | memory | exceeded the 13 GB memory budget while running alone (13425 MB) |
| 002f16678a09 | photo | graphunet |  | 30 | 1 | memory | exceeded the 13 GB memory budget while running alone (13422 MB) |
| c73828059805 | photo | graphunet |  | 30 | 2 | memory | exceeded the 13 GB memory budget while running alone (13353 MB) |
| f3c948365437 | photo | graphunet |  | 30 | 3 | memory | exceeded the 13 GB memory budget while running alone (13318 MB) |
| 8b5334c7073f | photo | graphunet |  | 30 | 4 | memory | exceeded the 13 GB memory budget while running alone (13355 MB) |
| 89517b1e815e | photo | graphunet |  | 30 | 5 | memory | exceeded the 13 GB memory budget while running alone (13438 MB) |
| 8c029e0b4ec5 | photo | graphunet |  | 30 | 6 | memory | exceeded the 13 GB memory budget while running alone (13323 MB) |
| cc43bccf1f34 | photo | graphunet |  | 30 | 7 | memory | exceeded the 13 GB memory budget while running alone (13466 MB) |
| 0771651a06d2 | photo | graphunet |  | 30 | 8 | memory | exceeded the 13 GB memory budget while running alone (13474 MB) |
| 2254a4026ea6 | photo | graphunet |  | 30 | 9 | memory | exceeded the 13 GB memory budget while running alone (13323 MB) |
| 83730b9bdc57 | pubmed | graphunet |  | 4 | 0 | memory | exceeded the 13 GB memory budget while running alone (13412 MB) |
| aacf3d819f5b | pubmed | graphunet |  | 4 | 1 | memory | exceeded the 13 GB memory budget while running alone (13357 MB) |
| c3a05e006f7a | pubmed | graphunet |  | 4 | 2 | memory | exceeded the 13 GB memory budget while running alone (13322 MB) |
| 1be0ec268234 | pubmed | graphunet |  | 4 | 3 | memory | exceeded the 13 GB memory budget while running alone (13340 MB) |
| 10423a74067f | pubmed | graphunet |  | 4 | 4 | memory | exceeded the 13 GB memory budget while running alone (13469 MB) |
| 99a1725deded | pubmed | graphunet |  | 4 | 5 | memory | exceeded the 13 GB memory budget while running alone (13313 MB) |
| af7a824b039a | pubmed | graphunet |  | 4 | 6 | memory | exceeded the 13 GB memory budget while running alone (13351 MB) |
| bf7d955edcea | pubmed | graphunet |  | 4 | 7 | memory | exceeded the 13 GB memory budget while running alone (13515 MB) |
| 492bf077cddc | pubmed | graphunet |  | 4 | 8 | memory | exceeded the 13 GB memory budget while running alone (13382 MB) |
| a35fac3e4831 | pubmed | graphunet |  | 4 | 9 | memory | exceeded the 13 GB memory budget while running alone (13321 MB) |
| b7a826d14169 | pubmed | graphunet |  | 8 | 0 | memory | exceeded the 13 GB memory budget while running alone (13464 MB) |
| 6b1deba82ddb | pubmed | graphunet |  | 8 | 1 | memory | exceeded the 13 GB memory budget while running alone (13343 MB) |
| f409e43524c5 | pubmed | graphunet |  | 8 | 2 | time | stopped after 7 epochs (16.5 min): early stopping needs at least 100 more epochs at >= 84.8 s each, i.e. at least 158 min of training, more than the 150-minute ... |
| 3cb7f902d4f5 | pubmed | graphunet |  | 8 | 3 | time | stopped after 1 epochs (1.6 min): early stopping needs at least 100 more epochs at >= 96.7 s each, i.e. at least 163 min of training, more than the 150-minute l... |
| 3a9b4fac3ba1 | pubmed | graphunet |  | 8 | 4 | memory | exceeded the 13 GB memory budget while running alone (13360 MB) |
| bc4ff8f77271 | pubmed | graphunet |  | 8 | 5 | time | stopped after 1 epochs (1.7 min): early stopping needs at least 100 more epochs at >= 104.6 s each, i.e. at least 176 min of training, more than the 150-minute ... |
| 9619d9b2d0e7 | pubmed | graphunet |  | 8 | 6 | memory | exceeded the 13 GB memory budget while running alone (13357 MB) |
| 3e8ccacb6eb0 | pubmed | graphunet |  | 8 | 7 | time | stopped after 1 epochs (1.7 min): early stopping needs at least 100 more epochs at >= 104.9 s each, i.e. at least 177 min of training, more than the 150-minute ... |
| 3d781356e58c | pubmed | graphunet |  | 8 | 8 | time | stopped after 1 epochs (1.5 min): early stopping needs at least 100 more epochs at >= 93.0 s each, i.e. at least 157 min of training, more than the 150-minute l... |
| 09390d1a7f7c | pubmed | graphunet |  | 8 | 9 | memory | exceeded the 13 GB memory budget while running alone (13505 MB) |
| 87cd14e103e7 | pubmed | graphunet |  | 10 | 0 | time | stopped after 1 epochs (1.6 min): early stopping needs at least 100 more epochs at >= 94.1 s each, i.e. at least 158 min of training, more than the 150-minute l... |
| 24ab331ef50c | pubmed | graphunet |  | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13339 MB) |
| a79935624167 | pubmed | graphunet |  | 10 | 2 | time | stopped after 1 epochs (1.9 min): early stopping needs at least 100 more epochs at >= 111.3 s each, i.e. at least 187 min of training, more than the 150-minute ... |
| 4737e983b5e9 | pubmed | graphunet |  | 10 | 3 | time | stopped after 1 epochs (2.0 min): early stopping needs at least 100 more epochs at >= 119.8 s each, i.e. at least 202 min of training, more than the 150-minute ... |
| a6bb8f9fb423 | pubmed | graphunet |  | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13351 MB) |
| cd42dbc7e1eb | pubmed | graphunet |  | 10 | 5 | memory | exceeded the 13 GB memory budget while running alone (13319 MB) |
| 59f51539ad72 | pubmed | graphunet |  | 10 | 6 | time | stopped after 1 epochs (2.3 min): early stopping needs at least 100 more epochs at >= 136.8 s each, i.e. at least 230 min of training, more than the 150-minute ... |
| 1a776f39cbe5 | pubmed | graphunet |  | 10 | 7 | memory | exceeded the 13 GB memory budget while running alone (13336 MB) |
| 18c93a450c5d | pubmed | graphunet |  | 10 | 8 | time | stopped after 1 epochs (1.5 min): early stopping needs at least 100 more epochs at >= 92.0 s each, i.e. at least 155 min of training, more than the 150-minute l... |
| eb88d1231a3d | pubmed | graphunet |  | 10 | 9 | time | stopped after 1 epochs (1.7 min): early stopping needs at least 100 more epochs at >= 104.9 s each, i.e. at least 176 min of training, more than the 150-minute ... |
| f37ea68b418d | pubmed | graphunet |  | 20 | 0 | memory | exceeded the 13 GB memory budget while running alone (13435 MB) |
| fabcfabe18d7 | pubmed | graphunet |  | 20 | 1 | time | stopped after 1 epochs (1.6 min): early stopping needs at least 100 more epochs at >= 96.2 s each, i.e. at least 162 min of training, more than the 150-minute l... |
| a95a0846684c | pubmed | graphunet |  | 20 | 2 | memory | exceeded the 13 GB memory budget while running alone (13348 MB) |
| 4989f078b38e | pubmed | graphunet |  | 20 | 4 | time | stopped after 1 epochs (1.7 min): early stopping needs at least 100 more epochs at >= 104.2 s each, i.e. at least 175 min of training, more than the 150-minute ... |
| 68717453d23c | pubmed | graphunet |  | 20 | 5 | memory | exceeded the 13 GB memory budget while running alone (13372 MB) |
| c01521e6c250 | pubmed | graphunet |  | 20 | 6 | memory | exceeded the 13 GB memory budget while running alone (13336 MB) |
| 061fce8bd3a1 | pubmed | graphunet |  | 20 | 7 | time | stopped after 1 epochs (2.2 min): early stopping needs at least 100 more epochs at >= 133.7 s each, i.e. at least 225 min of training, more than the 150-minute ... |
| 21f74fafadf4 | pubmed | graphunet |  | 20 | 8 | time | stopped after 1 epochs (1.7 min): early stopping needs at least 100 more epochs at >= 103.2 s each, i.e. at least 174 min of training, more than the 150-minute ... |
| 6dad7ce0ba90 | pubmed | graphunet |  | 20 | 9 | time | stopped after 1 epochs (1.5 min): early stopping needs at least 100 more epochs at >= 92.4 s each, i.e. at least 156 min of training, more than the 150-minute l... |
| 434e2414c08f | pubmed | graphunet |  | 30 | 0 | time | stopped after 1 epochs (1.7 min): early stopping needs at least 100 more epochs at >= 104.9 s each, i.e. at least 177 min of training, more than the 150-minute ... |
| 1b3cd3766f0c | pubmed | graphunet |  | 30 | 1 | memory | exceeded the 13 GB memory budget while running alone (13353 MB) |
| b4c0acd7b106 | pubmed | graphunet |  | 30 | 2 | time | stopped after 1 epochs (1.6 min): early stopping needs at least 100 more epochs at >= 98.6 s each, i.e. at least 166 min of training, more than the 150-minute l... |
| 1fbe1850fa99 | pubmed | graphunet |  | 30 | 3 | time | stopped after 1 epochs (1.6 min): early stopping needs at least 100 more epochs at >= 94.2 s each, i.e. at least 159 min of training, more than the 150-minute l... |
| 2d6321bd0df3 | pubmed | graphunet |  | 30 | 4 | memory | exceeded the 13 GB memory budget while running alone (13331 MB) |
| e959d38bcb86 | pubmed | graphunet |  | 30 | 5 | time | stopped after 1 epochs (1.6 min): early stopping needs at least 100 more epochs at >= 96.8 s each, i.e. at least 163 min of training, more than the 150-minute l... |
| 6c657d68314e | pubmed | graphunet |  | 30 | 6 | memory | exceeded the 13 GB memory budget while running alone (13363 MB) |
| 4ec4bb3c7604 | pubmed | graphunet |  | 30 | 7 | time | stopped after 1 epochs (1.8 min): early stopping needs at least 100 more epochs at >= 106.8 s each, i.e. at least 180 min of training, more than the 150-minute ... |
| 91bcb1b87507 | pubmed | graphunet |  | 30 | 8 | time | stopped after 1 epochs (1.7 min): early stopping needs at least 100 more epochs at >= 103.5 s each, i.e. at least 174 min of training, more than the 150-minute ... |
| ca7ebbcf0305 | pubmed | graphunet |  | 30 | 9 | time | stopped after 1 epochs (1.5 min): early stopping needs at least 100 more epochs at >= 91.8 s each, i.e. at least 154 min of training, more than the 150-minute l... |
| 2919d61dfce2 | pubmed | sgc |  | 30 | 1 | time | stopped after 342 epochs (122.4 min): early stopping needs at least 100 more epochs at >= 16.7 s each, i.e. at least 150 min of training, more than the 150-minu... |
| c5d4d93b3b39 | pubmed | sgc |  | 30 | 6 | time | stopped after 401 epochs (126.8 min): early stopping needs at least 97 more epochs at >= 14.4 s each, i.e. at least 150 min of training, more than the 150-minut... |

## Missing runs

Not attempted, or stopped by the job's deadline.

| key | dataset | model | tag | depth | seed |
|---|---|---|---|---|---|
| 0353dfc589be | cora | appnp |  | 20 | 3 |
| 9141e9decd47 | cora | pairnorm |  | 4 | 3 |
| 49928c3f264b | cora | graphunet |  | 20 | 3 |
| e3b0d5eb732a | citeseer | gat |  | 4 | 3 |
| a6ab1de5fce8 | citeseer | appnp |  | 20 | 3 |
| 6a62ce0fc730 | citeseer | pairnorm |  | 4 | 3 |
| 63da490d5c15 | citeseer | graphunet |  | 20 | 3 |
| 16101f9e4a15 | pubmed | gat |  | 4 | 3 |
| 1fcf1c0e2519 | pubmed | pairnorm |  | 4 | 3 |
| a1a5c069ea8e | pubmed | graphunet |  | 20 | 3 |
| 902510711f5b | photo | appnp |  | 20 | 3 |
| 9691174f51f3 | photo | pairnorm |  | 4 | 3 |
| fe66f6a244b7 | photo | graphunet |  | 20 | 3 |
