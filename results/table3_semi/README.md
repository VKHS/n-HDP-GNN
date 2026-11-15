# table3_semi

Sweep `configs/table3_semi.toml`: 2400 runs expected, 553 completed, 0 failed, 1847 missing.

Values are mean ± 95% t-interval over seeds; n is the number of completed seeds in the cell.

Code commit(s) of the runs: 0f7045972e60a86d7966841bc3e5c4bfe0caa016, 4f00db1371643ba63fd5a37ce4d9e24d08bf1f24

## test (accuracy in %)

**citeseer** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 68.37 ± 3.74 (n=3) | 64.84 ± 1.52 (n=5) | 31.35 ± 55.27 (n=2) | 19.68 ± 1.66 (n=5) | 20.10 ± 2.24 (n=3) |  |
| dropedge | 70.20 ± 0.94 (n=5) | 66.76 ± 1.45 (n=7) | 56.08 ± 4.16 (n=4) |  | 25.65 ± 107.37 (n=2) | 18.78 ± 0.30 (n=4) |
| gat |  |  |  | 62.47 ± 11.43 (n=3) | 21.82 ± 6.38 (n=5) | 20.00 ± 39.39 (n=2) |
| gcn |  |  | 57.10 ± 13.98 (n=2) | 51.10 ± 20.33 (n=2) |  |  |
| gcnii | 63.87 ± 1.85 (n=6) | 63.94 ± 1.98 (n=10) | 64.51 ± 1.25 (n=10) | 63.80 ± 1.43 (n=10) |  | 69.27 ± 3.66 (n=4) |
| graphunet |  |  | 61.15 ± 36.21 (n=2) | 60.55 ± 34.94 (n=2) |  |  |
| nhdp | 62.43 ± 5.56 (n=3) | 60.28 ± 4.75 (n=6) | 57.48 ± 3.45 (n=5) | 58.26 ± 3.86 (n=5) | 57.80 ± 2.79 (n=5) | 55.44 ± 2.81 (n=5) |
| pairnorm | 67.30 ± 12.71 (n=2) | 63.45 ± 10.80 (n=2) |  | 44.20 ± 2.90 (n=6) | 35.08 ± 8.46 (n=5) | 35.55 ± 13.34 (n=2) |
| sgc | 67.34 ± 1.13 (n=5) | 68.73 ± 2.02 (n=3) |  |  |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.25 | 0.0625 | 0.5 | 0.0625 | 0.25 |  |
| dropedge |  | 0.0625 | 0.875 |  | 0.5 | 0.125 |
| gat |  |  |  |  | 0.25 | 0.5 |
| gcn |  |  |  |  |  |  |
| gcnii |  | 0.0312 | 0.0625 | 0.0625 |  |  |
| graphunet |  |  |  |  |  |  |
| pairnorm |  |  |  | 0.25 | 0.25 | 0.5 |
| sgc |  |  |  |  |  |  |

**cora** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 80.63 ± 0.76 (n=3) | 78.30 ± 0.69 (n=5) | 37.00 ± 171.53 (n=2) | 23.76 ± 7.37 (n=5) | 26.10 ± 24.09 (n=3) |  |
| dropedge | 81.39 ± 0.72 (n=7) | 79.73 ± 0.95 (n=10) | 74.92 ± 1.23 (n=10) | 73.14 ± 1.32 (n=10) |  | 23.65 ± 6.95 (n=4) |
| gat |  | 78.95 ± 2.24 (n=4) | 75.45 ± 3.56 (n=4) | 77.03 ± 2.68 (n=3) | 28.46 ± 9.48 (n=5) | 23.37 ± 9.81 (n=6) |
| gcn | 82.10 ± 7.62 (n=2) | 79.60 ± 3.81 (n=2) |  | 69.45 ± 3.37 (n=4) |  |  |
| graphunet | 78.90 ± 7.62 (n=2) | 79.65 ± 1.91 (n=2) |  | 78.38 ± 2.03 (n=4) |  |  |
| nhdp | 62.76 ± 2.03 (n=8) | 64.19 ± 1.31 (n=10) | 61.96 ± 2.26 (n=10) | 63.88 ± 1.69 (n=10) | 59.07 ± 8.38 (n=3) | 58.38 ± 5.64 (n=4) |
| pairnorm |  |  |  | 69.23 ± 6.90 (n=3) | 55.70 ± 13.22 (n=5) | 25.90 ± 12.71 (n=2) |
| sgc | 78.00 ± 0.93 (n=5) | 79.50 ± 3.44 (n=3) | 78.60 ± 19.06 (n=2) | 78.30 ± 11.44 (n=2) |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.25 | 0.0625 | 0.5 | 0.0625 | 0.25 |  |
| dropedge | 0.0156 | 0.00195 | 0.00195 | 0.00195 |  | 0.125 |
| gat |  | 0.125 | 0.125 | 0.25 |  |  |
| gcn | 0.5 | 0.5 |  | 0.125 |  |  |
| graphunet | 0.5 | 0.5 |  | 0.125 |  |  |
| pairnorm |  |  |  | 0.25 |  |  |
| sgc | 0.0625 | 0.25 | 0.5 | 0.5 |  |  |

**photo** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp |  | 88.96 ± 3.48 (n=4) |  |  |  |  |
| arma | 90.78 ± 1.46 (n=3) | 89.37 ± 1.11 (n=5) | 46.73 ± 22.03 (n=4) | 23.15 ± 9.69 (n=6) | 23.74 ± 8.50 (n=3) |  |
| dropedge | 88.49 ± 2.63 (n=5) | 87.81 ± 6.80 (n=3) |  |  |  |  |
| gat | 88.21 ± 2.43 (n=6) | 89.01 ± 7.08 (n=3) | 86.67 ± 1.64 (n=8) | 85.40 ± 1.25 (n=9) | 26.33 ± 1.30 (n=5) | 20.11 ± 6.90 (n=6) |
| nhdp | 62.59 ± 7.66 (n=3) | 69.63 ± 22.64 (n=5) | 69.98 ± 46.54 (n=2) | 48.08 ± 29.51 (n=5) | 32.13 ± 29.08 (n=3) |  |
| pairnorm |  |  |  | 86.05 ± 0.83 (n=3) | 55.20 ± 39.03 (n=5) | 47.81 ± 365.65 (n=2) |
| sgc | 69.55 ± 5.17 (n=5) | 55.59 ± 0.71 (n=3) |  |  |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp |  | 0.25 |  |  |  |  |
| arma | 0.25 | 0.0625 | 1 | 0.0625 | 0.25 |  |
| dropedge |  |  |  |  |  |  |
| gat |  |  | 0.5 | 0.125 |  |  |
| pairnorm |  |  |  |  |  |  |
| sgc |  |  |  |  |  |  |

**pubmed** -- test

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 76.78 ± 0.50 (n=6) | 77.14 ± 0.42 (n=10) | 77.69 ± 0.32 (n=10) | 77.71 ± 0.27 (n=10) |  | 78.18 ± 0.56 (n=4) |
| arma | 77.86 ± 0.47 (n=5) | 76.11 ± 0.54 (n=7) | 51.65 ± 15.88 (n=2) | 46.26 ± 1.00 (n=5) | 40.60 ± 1.94 (n=3) |  |
| dropedge | 76.84 ± 0.96 (n=5) | 76.73 ± 1.41 (n=3) |  |  |  |  |
| gat |  |  |  | 77.37 ± 3.01 (n=3) | 41.90 ± 1.83 (n=5) | 40.70 ± 0.00 (n=2) |
| gcnii |  | 76.72 ± 0.68 (n=4) | 76.43 ± 1.33 (n=4) |  |  | 79.20 ± 0.43 (n=4) |
| nhdp | 72.20 ± 1.49 (n=3) | 72.10 ± 0.98 (n=5) | 74.25 ± 20.97 (n=2) | 71.74 ± 1.41 (n=5) | 73.23 ± 2.17 (n=3) |  |
| pairnorm |  |  | 75.25 ± 5.72 (n=2) | 75.85 ± 3.20 (n=4) | 73.24 ± 2.90 (n=5) | 72.35 ± 14.61 (n=2) |
| sgc | 73.86 ± 1.83 (n=5) | 75.23 ± 0.14 (n=3) |  |  |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp |  | 0.0625 | 0.5 | 0.0625 |  |  |
| arma | 0.25 | 0.0625 | 0.5 | 0.0625 | 0.25 |  |
| dropedge |  |  |  |  |  |  |
| gat |  |  |  |  |  |  |
| gcnii |  | 0.25 |  |  |  |  |
| pairnorm |  |  |  | 0.5 |  |  |
| sgc |  |  |  |  |  |  |

## embedding.centroid_gap

**citeseer** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.70 ± 0.18 (n=3) | 0.62 ± 0.08 (n=5) | 0.04 ± 0.77 (n=2) | -0.04 ± 0.07 (n=5) | -0.13 ± 0.45 (n=3) |  |
| dropedge | 0.69 ± 0.18 (n=5) | 0.67 ± 0.10 (n=7) | 0.37 ± 0.10 (n=4) |  | 0.06 ± 1.19 (n=2) | -0.04 ± 0.03 (n=4) |
| gat |  |  |  | 0.71 ± 0.06 (n=3) | 0.00 ± 0.00 (n=5) | -0.00 ± 0.00 (n=2) |
| gcn |  |  | 0.50 ± 1.49 (n=2) | 0.32 ± 0.04 (n=2) |  |  |
| gcnii | 0.62 ± 0.09 (n=6) | 0.52 ± 0.18 (n=10) | 0.58 ± 0.06 (n=10) | 0.53 ± 0.15 (n=10) |  | 0.37 ± 0.45 (n=4) |
| graphunet |  |  | 0.33 ± 0.91 (n=2) | 0.42 ± 0.46 (n=2) |  |  |
| nhdp | 0.60 ± 0.28 (n=3) | 0.51 ± 0.25 (n=6) | 0.49 ± 0.23 (n=5) | 0.47 ± 0.20 (n=5) | 0.56 ± 0.21 (n=5) | 0.49 ± 0.10 (n=5) |
| pairnorm | 0.78 ± 0.19 (n=2) | 0.72 ± 0.26 (n=2) |  | 0.19 ± 0.21 (n=6) | 0.04 ± 0.10 (n=5) | 0.01 ± 0.91 (n=2) |
| sgc | 0.13 ± 0.03 (n=5) | 0.21 ± 0.30 (n=3) |  |  |  |  |

**cora** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.70 ± 0.06 (n=3) | 0.81 ± 0.06 (n=5) | 0.46 ± 1.42 (n=2) | 0.01 ± 0.21 (n=5) | -0.00 ± 0.00 (n=3) |  |
| dropedge | 0.87 ± 0.08 (n=7) | 0.84 ± 0.06 (n=10) | 0.69 ± 0.05 (n=10) | 0.69 ± 0.06 (n=10) |  | -0.14 ± 0.12 (n=4) |
| gat |  | 0.90 ± 0.02 (n=4) | 0.88 ± 0.02 (n=4) | 0.89 ± 0.11 (n=3) | -0.00 ± 0.00 (n=5) | -0.00 ± 0.00 (n=6) |
| gcn | 0.91 ± 0.43 (n=2) | 0.80 ± 0.35 (n=2) |  | 0.62 ± 0.11 (n=4) |  |  |
| graphunet | 0.83 ± 0.66 (n=2) | 0.82 ± 0.05 (n=2) |  | 0.81 ± 0.07 (n=4) |  |  |
| nhdp | 0.49 ± 0.12 (n=8) | 0.62 ± 0.07 (n=10) | 0.67 ± 0.05 (n=10) | 0.73 ± 0.04 (n=10) | 0.71 ± 0.10 (n=3) | 0.60 ± 0.20 (n=4) |
| pairnorm |  |  |  | 0.57 ± 0.29 (n=3) | 0.57 ± 0.12 (n=5) | -0.07 ± 0.23 (n=2) |
| sgc | 0.33 ± 0.11 (n=5) | 0.41 ± 0.75 (n=3) | 0.29 ± 0.41 (n=2) | 0.22 ± 0.40 (n=2) |  |  |

**photo** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp |  | 0.86 ± 0.12 (n=4) |  |  |  |  |
| arma | 0.83 ± 0.08 (n=3) | 0.85 ± 0.05 (n=5) | 0.58 ± 0.25 (n=4) | 0.19 ± 0.42 (n=6) | -0.01 ± 0.01 (n=3) |  |
| dropedge | 0.91 ± 0.08 (n=5) | 0.75 ± 0.14 (n=3) |  |  |  |  |
| gat | 0.92 ± 0.07 (n=6) | 0.91 ± 0.09 (n=3) | 0.84 ± 0.09 (n=8) | 0.85 ± 0.04 (n=9) | 0.01 ± 0.03 (n=5) | 0.00 ± 0.00 (n=6) |
| nhdp | 0.33 ± 0.41 (n=3) | 0.43 ± 0.48 (n=5) | 0.36 ± 3.77 (n=2) | 0.32 ± 0.47 (n=5) | 0.01 ± 0.04 (n=3) |  |
| pairnorm |  |  |  | 0.75 ± 0.24 (n=3) | 0.41 ± 0.55 (n=5) | 0.35 ± 5.08 (n=2) |
| sgc | 0.27 ± 0.39 (n=5) | 0.05 ± 0.07 (n=3) |  |  |  |  |

**pubmed** -- embedding.centroid_gap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 1.09 ± 0.01 (n=6) | 1.09 ± 0.03 (n=10) | 1.07 ± 0.04 (n=10) | 1.07 ± 0.04 (n=10) |  | 1.06 ± 0.10 (n=4) |
| arma | 1.06 ± 0.04 (n=5) | 0.98 ± 0.10 (n=7) | 0.51 ± 0.69 (n=2) | 0.18 ± 0.10 (n=5) | -0.11 ± 0.30 (n=3) |  |
| dropedge | 1.09 ± 0.05 (n=5) | 1.13 ± 0.04 (n=3) |  |  |  |  |
| gat |  |  |  | 1.07 ± 0.13 (n=3) | -0.00 ± 0.00 (n=5) | -0.00 ± 0.00 (n=2) |
| gcnii |  | 0.98 ± 0.14 (n=4) | 0.90 ± 0.19 (n=4) |  |  | 1.10 ± 0.05 (n=4) |
| nhdp | 1.03 ± 0.08 (n=3) | 1.05 ± 0.05 (n=5) | 1.05 ± 0.31 (n=2) | 0.78 ± 0.33 (n=5) | 0.91 ± 0.44 (n=3) |  |
| pairnorm |  |  | 1.13 ± 0.13 (n=2) | 1.11 ± 0.05 (n=4) | 0.93 ± 0.16 (n=5) | 1.00 ± 0.24 (n=2) |
| sgc | 0.69 ± 0.30 (n=5) | 0.98 ± 0.50 (n=3) |  |  |  |  |

## embedding.madgap

**citeseer** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.80 ± 0.09 (n=3) | 0.83 ± 0.04 (n=5) | 0.58 ± 0.93 (n=2) | -0.02 ± 0.03 (n=5) | -0.04 ± 0.15 (n=3) |  |
| dropedge | 0.81 ± 0.09 (n=5) | 0.85 ± 0.06 (n=7) | 0.73 ± 0.04 (n=4) |  | 0.29 ± 4.30 (n=2) | -0.04 ± 0.03 (n=4) |
| gat |  |  |  | 0.95 ± 0.01 (n=3) | 0.01 ± 0.01 (n=5) | 0.00 ± 0.00 (n=2) |
| gcn |  |  | 0.83 ± 0.88 (n=2) | 0.72 ± 0.03 (n=2) |  |  |
| gcnii | 0.79 ± 0.04 (n=6) | 0.71 ± 0.21 (n=10) | 0.83 ± 0.04 (n=10) | 0.76 ± 0.19 (n=10) |  | 0.55 ± 0.60 (n=4) |
| graphunet |  |  | 0.61 ± 0.06 (n=2) | 0.73 ± 0.09 (n=2) |  |  |
| nhdp | 0.62 ± 0.20 (n=3) | 0.56 ± 0.15 (n=6) | 0.55 ± 0.12 (n=5) | 0.56 ± 0.08 (n=5) | 0.59 ± 0.08 (n=5) | 0.58 ± 0.06 (n=5) |
| pairnorm | 0.84 ± 0.01 (n=2) | 0.90 ± 0.21 (n=2) |  | 0.68 ± 0.11 (n=6) | 0.48 ± 0.28 (n=5) | 0.59 ± 0.98 (n=2) |
| sgc | 0.37 ± 0.05 (n=5) | 0.48 ± 0.39 (n=3) |  |  |  |  |

**cora** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.57 ± 0.10 (n=3) | 0.64 ± 0.04 (n=5) | 0.67 ± 0.71 (n=2) | 0.05 ± 0.25 (n=5) | -0.00 ± 0.00 (n=3) |  |
| dropedge | 0.66 ± 0.04 (n=7) | 0.68 ± 0.03 (n=10) | 0.58 ± 0.05 (n=10) | 0.61 ± 0.04 (n=10) |  | -0.06 ± 0.04 (n=4) |
| gat |  | 0.70 ± 0.03 (n=4) | 0.71 ± 0.08 (n=4) | 0.76 ± 0.12 (n=3) | 0.00 ± 0.00 (n=5) | 0.00 ± 0.00 (n=6) |
| gcn | 0.68 ± 0.19 (n=2) | 0.63 ± 0.19 (n=2) |  | 0.57 ± 0.11 (n=4) |  |  |
| graphunet | 0.69 ± 0.62 (n=2) | 0.62 ± 1.03 (n=2) |  | 0.72 ± 0.07 (n=4) |  |  |
| nhdp | 0.34 ± 0.07 (n=8) | 0.37 ± 0.02 (n=10) | 0.39 ± 0.02 (n=10) | 0.44 ± 0.03 (n=10) | 0.43 ± 0.15 (n=3) | 0.37 ± 0.11 (n=4) |
| pairnorm |  |  |  | 0.49 ± 0.07 (n=3) | 0.50 ± 0.18 (n=5) | 0.20 ± 1.53 (n=2) |
| sgc | 0.38 ± 0.09 (n=5) | 0.42 ± 0.46 (n=3) | 0.41 ± 0.50 (n=2) | 0.34 ± 0.39 (n=2) |  |  |

**photo** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp |  | 0.47 ± 0.07 (n=4) |  |  |  |  |
| arma | 0.33 ± 0.05 (n=3) | 0.38 ± 0.08 (n=5) | 0.17 ± 0.27 (n=4) | 0.12 ± 0.13 (n=6) | 0.06 ± 0.10 (n=3) |  |
| dropedge | 0.42 ± 0.02 (n=5) | 0.43 ± 0.02 (n=3) |  |  |  |  |
| gat | 0.44 ± 0.04 (n=6) | 0.45 ± 0.17 (n=3) | 0.58 ± 0.08 (n=8) | 0.61 ± 0.07 (n=9) | 0.03 ± 0.05 (n=5) | 0.00 ± 0.00 (n=6) |
| nhdp | 0.18 ± 0.15 (n=3) | 0.19 ± 0.18 (n=5) | 0.15 ± 1.35 (n=2) | 0.12 ± 0.15 (n=5) | 0.01 ± 0.04 (n=3) |  |
| pairnorm |  |  |  | 0.37 ± 0.16 (n=3) | 0.59 ± 0.47 (n=5) | 0.60 ± 0.82 (n=2) |
| sgc | 0.21 ± 0.12 (n=5) | 0.16 ± 0.06 (n=3) |  |  |  |  |

**pubmed** -- embedding.madgap

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.85 ± 0.02 (n=6) | 0.89 ± 0.01 (n=10) | 0.91 ± 0.01 (n=10) | 0.92 ± 0.01 (n=10) |  | 0.92 ± 0.04 (n=4) |
| arma | 0.85 ± 0.02 (n=5) | 0.82 ± 0.05 (n=7) | 0.66 ± 0.15 (n=2) | 0.54 ± 0.07 (n=5) | -0.26 ± 0.63 (n=3) |  |
| dropedge | 0.88 ± 0.03 (n=5) | 0.93 ± 0.06 (n=3) |  |  |  |  |
| gat |  |  |  | 0.96 ± 0.05 (n=3) | 0.00 ± 0.00 (n=5) | 0.00 ± 0.00 (n=2) |
| gcnii |  | 0.86 ± 0.07 (n=4) | 0.82 ± 0.08 (n=4) |  |  | 0.96 ± 0.05 (n=4) |
| nhdp | 0.67 ± 0.01 (n=3) | 0.68 ± 0.02 (n=5) | 0.67 ± 0.04 (n=2) | 0.58 ± 0.12 (n=5) | 0.62 ± 0.09 (n=3) |  |
| pairnorm |  |  | 0.88 ± 0.25 (n=2) | 0.87 ± 0.11 (n=4) | 0.78 ± 0.12 (n=5) | 0.82 ± 0.38 (n=2) |
| sgc | 0.70 ± 0.17 (n=5) | 0.88 ± 0.22 (n=3) |  |  |  |  |

## timings.epoch_s_mean (seconds per training epoch (one thread))

**citeseer** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.67 ± 0.28 (n=3) | 0.73 ± 0.08 (n=5) | 0.73 ± 0.06 (n=2) | 0.70 ± 0.04 (n=5) | 1.01 ± 0.44 (n=3) |  |
| dropedge | 0.29 ± 0.03 (n=5) | 0.30 ± 0.05 (n=7) | 0.39 ± 0.03 (n=4) |  | 0.45 ± 0.01 (n=2) | 0.44 ± 0.01 (n=4) |
| gat |  |  |  | 0.47 ± 0.21 (n=3) | 0.62 ± 0.09 (n=5) | 0.68 ± 0.02 (n=2) |
| gcn |  |  | 0.41 ± 0.05 (n=2) | 0.43 ± 0.01 (n=2) |  |  |
| gcnii | 0.31 ± 0.06 (n=6) | 0.35 ± 0.02 (n=10) | 0.32 ± 0.03 (n=10) | 0.44 ± 0.02 (n=10) |  | 0.55 ± 0.01 (n=4) |
| graphunet |  |  | 0.44 ± 0.04 (n=2) | 0.47 ± 0.39 (n=2) |  |  |
| nhdp | 0.39 ± 0.18 (n=3) | 0.38 ± 0.05 (n=6) | 0.54 ± 0.05 (n=5) | 0.48 ± 0.02 (n=5) | 0.79 ± 0.17 (n=5) | 0.76 ± 0.10 (n=5) |
| pairnorm | 0.36 ± 0.01 (n=2) | 0.39 ± 0.08 (n=2) |  | 0.41 ± 0.05 (n=6) | 0.55 ± 0.06 (n=5) | 0.60 ± 0.10 (n=2) |
| sgc | 0.51 ± 0.06 (n=5) | 1.15 ± 0.40 (n=3) |  |  |  |  |

**cora** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 0.22 ± 0.10 (n=3) | 0.25 ± 0.04 (n=5) | 0.28 ± 0.11 (n=2) | 0.29 ± 0.01 (n=5) | 0.54 ± 0.28 (n=3) |  |
| dropedge | 0.09 ± 0.01 (n=7) | 0.10 ± 0.01 (n=10) | 0.11 ± 0.01 (n=10) | 0.16 ± 0.01 (n=10) |  | 0.27 ± 0.01 (n=4) |
| gat |  | 0.12 ± 0.00 (n=4) | 0.21 ± 0.00 (n=4) | 0.23 ± 0.10 (n=3) | 0.34 ± 0.04 (n=5) | 0.41 ± 0.04 (n=6) |
| gcn | 0.10 ± 0.00 (n=2) | 0.12 ± 0.02 (n=2) |  | 0.14 ± 0.00 (n=4) |  |  |
| graphunet | 0.12 ± 0.01 (n=2) | 0.18 ± 0.09 (n=2) |  | 0.37 ± 0.05 (n=4) |  |  |
| nhdp | 0.12 ± 0.01 (n=8) | 0.16 ± 0.01 (n=10) | 0.19 ± 0.02 (n=10) | 0.30 ± 0.02 (n=10) | 0.53 ± 0.29 (n=3) | 0.61 ± 0.00 (n=4) |
| pairnorm |  |  |  | 0.18 ± 0.08 (n=3) | 0.25 ± 0.03 (n=5) | 0.30 ± 0.02 (n=2) |
| sgc | 0.20 ± 0.03 (n=5) | 0.45 ± 0.16 (n=3) | 0.94 ± 0.04 (n=2) | 1.18 ± 0.01 (n=2) |  |  |

**photo** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp |  | 0.17 ± 0.05 (n=4) |  |  |  |  |
| arma | 0.53 ± 0.15 (n=3) | 1.23 ± 0.20 (n=5) | 2.46 ± 0.82 (n=4) | 2.43 ± 0.72 (n=6) | 5.54 ± 2.52 (n=3) |  |
| dropedge | 0.27 ± 0.02 (n=5) | 0.61 ± 0.19 (n=3) |  |  |  |  |
| gat | 0.44 ± 0.09 (n=6) | 1.11 ± 0.14 (n=3) | 1.81 ± 0.13 (n=8) | 3.11 ± 0.19 (n=9) | 5.76 ± 0.81 (n=5) | 8.41 ± 0.44 (n=6) |
| nhdp | 0.80 ± 0.47 (n=3) | 1.51 ± 0.18 (n=5) | 2.60 ± 0.44 (n=2) | 2.76 ± 0.35 (n=5) | 7.02 ± 2.53 (n=3) |  |
| pairnorm |  |  |  | 1.64 ± 0.71 (n=3) | 3.44 ± 0.20 (n=5) | 4.54 ± 0.15 (n=2) |
| sgc | 1.76 ± 0.32 (n=5) | 4.47 ± 0.77 (n=3) |  |  |  |  |

**pubmed** -- timings.epoch_s_mean

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| appnp | 0.27 ± 0.05 (n=6) | 0.30 ± 0.02 (n=10) | 0.25 ± 0.02 (n=10) | 0.36 ± 0.02 (n=10) |  | 0.39 ± 0.02 (n=4) |
| arma | 0.68 ± 0.10 (n=5) | 1.08 ± 0.09 (n=7) | 1.42 ± 0.04 (n=2) | 1.50 ± 0.06 (n=5) | 3.19 ± 1.33 (n=3) |  |
| dropedge | 0.30 ± 0.03 (n=5) | 0.48 ± 0.18 (n=3) |  |  |  |  |
| gat |  |  |  | 1.48 ± 0.52 (n=3) | 2.37 ± 0.35 (n=5) | 3.16 ± 0.06 (n=2) |
| gcnii |  | 0.55 ± 0.04 (n=4) | 0.95 ± 0.03 (n=4) |  |  | 1.93 ± 0.03 (n=4) |
| nhdp | 0.64 ± 0.23 (n=3) | 0.94 ± 0.11 (n=5) | 1.21 ± 0.18 (n=2) | 1.39 ± 0.12 (n=5) | 3.23 ± 1.33 (n=3) |  |
| pairnorm |  |  | 1.00 ± 0.79 (n=2) | 1.30 ± 0.39 (n=4) | 2.47 ± 0.32 (n=5) | 3.03 ± 1.19 (n=2) |
| sgc | 0.62 ± 0.09 (n=5) | 1.32 ± 0.47 (n=3) |  |  |  |  |

## peak_rss_mb (peak resident memory of the run's process in MB)

**citeseer** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 804.89 ± 3.06 (n=3) | 813.56 ± 2.34 (n=5) | 848.45 ± 20.92 (n=2) | 865.55 ± 2.92 (n=5) | 951.05 ± 5.41 (n=3) |  |
| dropedge | 753.03 ± 1.11 (n=5) | 758.65 ± 2.04 (n=3) |  |  |  |  |
| gat |  |  |  | 863.23 ± 5.19 (n=3) | 984.30 ± 7.42 (n=5) | 1102.62 ± 12.31 (n=2) |
| nhdp | 824.35 ± 2.33 (n=3) | 851.62 ± 3.15 (n=2) | 924.26 (n=1) | 953.40 ± 4.57 (n=5) | 1108.31 ± 27.19 (n=3) |  |
| pairnorm |  |  |  | 806.85 ± 9.11 (n=2) | 875.48 ± 9.05 (n=5) | 959.37 ± 41.87 (n=2) |
| sgc | 1087.99 ± 0.24 (n=5) | 1087.49 ± 2.28 (n=3) |  |  |  |  |

**cora** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 657.66 ± 35.68 (n=3) | 664.31 ± 10.28 (n=5) | 698.62 ± 38.47 (n=2) | 713.68 ± 16.27 (n=5) | 782.33 ± 14.82 (n=3) |  |
| dropedge | 633.88 (n=1) |  |  |  |  |  |
| gat |  |  |  | 726.39 ± 7.06 (n=3) | 840.72 ± 13.86 (n=3) |  |
| nhdp | 667.80 ± 48.72 (n=2) |  |  |  | 954.86 ± 32.12 (n=3) |  |
| pairnorm |  |  |  | 668.79 ± 18.76 (n=3) | 730.85 ± 15.83 (n=5) | 783.74 ± 121.33 (n=2) |
| sgc | 752.00 ± 0.88 (n=5) | 771.37 ± 16.30 (n=3) |  |  |  |  |

**photo** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 828.01 ± 8.74 (n=3) | 851.59 ± 6.98 (n=5) | 986.97 ± 133.59 (n=2) | 1028.43 ± 13.32 (n=5) | 1193.03 ± 38.08 (n=3) |  |
| dropedge | 785.78 ± 2.16 (n=5) | 806.27 ± 6.29 (n=3) |  |  |  |  |
| gat |  |  |  | 1935.62 (n=1) | 3091.65 ± 27.65 (n=5) | 4308.78 ± 403.27 (n=2) |
| nhdp | 1110.18 ± 19.81 (n=3) | 1334.18 ± 14.63 (n=5) | 1835.25 ± 26.08 (n=2) | 2060.42 ± 25.67 (n=5) | 3241.53 ± 49.57 (n=3) |  |
| pairnorm |  |  |  | 957.90 ± 10.84 (n=3) | 1172.83 ± 4.78 (n=5) | 1379.22 ± 48.62 (n=2) |
| sgc | 2057.56 ± 1.73 (n=5) | 2079.04 (n=1) |  |  |  |  |

**pubmed** -- peak_rss_mb

| model | 2 | 4 | 8 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| arma | 1003.42 ± 9.26 (n=3) | 1026.22 ± 10.22 (n=5) | 1226.97 ± 347.31 (n=2) | 1375.52 ± 25.09 (n=5) | 1881.41 ± 116.02 (n=3) |  |
| dropedge | 1003.72 ± 3.06 (n=5) | 1025.98 ± 8.22 (n=3) |  |  |  |  |
| gat |  |  |  | 1589.45 ± 106.83 (n=3) | 2523.12 ± 105.11 (n=5) | 3353.59 ± 211.94 (n=2) |
| nhdp | 1116.59 ± 82.74 (n=3) | 1280.64 ± 56.01 (n=5) | 1814.12 ± 671.29 (n=2) | 2003.58 ± 45.06 (n=5) | 2974.33 ± 25.81 (n=3) |  |
| pairnorm |  |  |  | 1119.99 ± 6.18 (n=2) | 1553.39 ± 49.63 (n=5) | 2009.93 ± 81.23 (n=2) |
| sgc | 1122.86 ± 2.96 (n=5) | 1123.60 ± 8.26 (n=3) |  |  |  |  |

## Missing runs

Not attempted, or stopped by the job's deadline.

| key | dataset | model | tag | depth | seed |
|---|---|---|---|---|---|
| b92c918cbff7 | cora | nhdp |  | 2 | 5 |
| 43c4265feca7 | cora | nhdp |  | 2 | 7 |
| 28ad5fab40af | cora | nhdp |  | 20 | 3 |
| 4bd70f7c9ad8 | cora | nhdp |  | 20 | 4 |
| 8abb99d58dd9 | cora | nhdp |  | 20 | 5 |
| 4f32d1b37ed2 | cora | nhdp |  | 20 | 6 |
| 08021a9c674c | cora | nhdp |  | 20 | 7 |
| c07937acdbe0 | cora | nhdp |  | 20 | 8 |
| 30a51bf56c14 | cora | nhdp |  | 20 | 9 |
| f62fbc0a15b5 | cora | nhdp |  | 30 | 0 |
| 69e4295a0ffd | cora | nhdp |  | 30 | 1 |
| d75cd01acb13 | cora | nhdp |  | 30 | 6 |
| 7341564ff669 | cora | nhdp |  | 30 | 7 |
| f4327a4e05d8 | cora | nhdp |  | 30 | 8 |
| 73a850385bf7 | cora | nhdp |  | 30 | 9 |
| 3f744fa0a7e5 | cora | gcn |  | 2 | 0 |
| e5edeb9e8581 | cora | gcn |  | 2 | 1 |
| d7d5ddc85a5d | cora | gcn |  | 2 | 2 |
| 3d3305ca2566 | cora | gcn |  | 2 | 3 |
| 14d649a27154 | cora | gcn |  | 2 | 4 |
| fbad0bbe371e | cora | gcn |  | 2 | 5 |
| d2558d6a4cc8 | cora | gcn |  | 2 | 6 |
| 5e8f486392b0 | cora | gcn |  | 2 | 7 |
| d4259e04fec3 | cora | gcn |  | 4 | 2 |
| 73b393293565 | cora | gcn |  | 4 | 3 |
| 6edcf27c909e | cora | gcn |  | 4 | 4 |
| a0682790ca2d | cora | gcn |  | 4 | 5 |
| 4dfa33fdd959 | cora | gcn |  | 4 | 6 |
| dcae57581705 | cora | gcn |  | 4 | 7 |
| bb188cd79bcd | cora | gcn |  | 4 | 8 |
| 838d756803d8 | cora | gcn |  | 4 | 9 |
| ffcc0997576f | cora | gcn |  | 8 | 0 |
| fb6d6fd38748 | cora | gcn |  | 8 | 1 |
| 5f3cf39ce6df | cora | gcn |  | 8 | 2 |
| 17d7c9ab76f2 | cora | gcn |  | 8 | 3 |
| 7dc6b70dd002 | cora | gcn |  | 8 | 4 |
| a4daaf3c1702 | cora | gcn |  | 8 | 5 |
| cfe841a37978 | cora | gcn |  | 8 | 6 |
| 5473ad924ae3 | cora | gcn |  | 8 | 7 |
| f03fd1ba02c2 | cora | gcn |  | 8 | 8 |
| bb10997784e9 | cora | gcn |  | 8 | 9 |
| da4346f7c801 | cora | gcn |  | 10 | 0 |
| d7c3c325c973 | cora | gcn |  | 10 | 1 |
| 2737ba28a40d | cora | gcn |  | 10 | 2 |
| 1c133f98330a | cora | gcn |  | 10 | 3 |
| eb43a1d308b9 | cora | gcn |  | 10 | 4 |
| 9641a47b7c53 | cora | gcn |  | 10 | 5 |
| daf7d5af4630 | cora | gcn |  | 20 | 0 |
| 90a81ac566ab | cora | gcn |  | 20 | 1 |
| b15ece41bbfe | cora | gcn |  | 20 | 2 |
| 9d53481bd83a | cora | gcn |  | 20 | 3 |
| 14e9d2788b10 | cora | gcn |  | 20 | 4 |
| 3863aa98afc7 | cora | gcn |  | 20 | 5 |
| e13dcae5e66a | cora | gcn |  | 20 | 6 |
| 78aea18a8938 | cora | gcn |  | 20 | 7 |
| 10704553cc0c | cora | gcn |  | 20 | 8 |
| 06fceab0fa65 | cora | gcn |  | 20 | 9 |
| 00cc77524901 | cora | gcn |  | 30 | 0 |
| 0ba6effea57b | cora | gcn |  | 30 | 1 |
| 564f27bd23c9 | cora | gcn |  | 30 | 2 |
| a53a4a68f032 | cora | gcn |  | 30 | 3 |
| d2121dbc677e | cora | gcn |  | 30 | 4 |
| b6ce3be16174 | cora | gcn |  | 30 | 5 |
| 1b8b01f471d4 | cora | gcn |  | 30 | 6 |
| 8418488e2b0e | cora | gcn |  | 30 | 7 |
| 63571305e867 | cora | gcn |  | 30 | 8 |
| db5041a42fcf | cora | gcn |  | 30 | 9 |
| 684f61929cc2 | cora | gat |  | 2 | 0 |
| a900b9dfd8cf | cora | gat |  | 2 | 1 |
| 84c0a05fc3d4 | cora | gat |  | 2 | 2 |
| e56c4bcb986f | cora | gat |  | 2 | 3 |
| 58beadd76080 | cora | gat |  | 2 | 4 |
| 76be69c50a79 | cora | gat |  | 2 | 5 |
| a279d18325a7 | cora | gat |  | 2 | 6 |
| 754de18c6562 | cora | gat |  | 2 | 7 |
| b97be140e30e | cora | gat |  | 2 | 8 |
| c5f79e646ef7 | cora | gat |  | 2 | 9 |
| 13a1ba62f6a5 | cora | gat |  | 4 | 0 |
| 5a7c26124f4e | cora | gat |  | 4 | 1 |
| be2337544f5d | cora | gat |  | 4 | 2 |
| eac16f484a26 | cora | gat |  | 4 | 3 |
| d39f854df49f | cora | gat |  | 4 | 4 |
| edd60757be0c | cora | gat |  | 4 | 5 |
| 4e75492a7729 | cora | gat |  | 8 | 4 |
| 86251dc1fe6b | cora | gat |  | 8 | 5 |
| 24bc5c930c93 | cora | gat |  | 8 | 6 |
| 63477250ab4f | cora | gat |  | 8 | 7 |
| 99124e820df7 | cora | gat |  | 8 | 8 |
| 04db741139b9 | cora | gat |  | 8 | 9 |
| f6156fd5cc05 | cora | gat |  | 10 | 0 |
| a211cd974911 | cora | gat |  | 10 | 2 |
| e6d8d5d91804 | cora | gat |  | 10 | 3 |
| d7e20132e4cb | cora | gat |  | 10 | 5 |
| 8c35b1f24483 | cora | gat |  | 10 | 7 |
| f5b5ac2b6ddb | cora | gat |  | 10 | 8 |
| 3b32b1ae06c2 | cora | gat |  | 10 | 9 |
| 56137b9cd35f | cora | gat |  | 20 | 0 |
| 24358914ffbe | cora | gat |  | 20 | 1 |
| d964cf0e9917 | cora | gat |  | 20 | 3 |
| 7f016c43e89c | cora | gat |  | 20 | 4 |
| d02dc805a1c2 | cora | gat |  | 20 | 6 |
| 4fef4f2d84ba | cora | gat |  | 30 | 2 |
| 5a3a50f5efcd | cora | gat |  | 30 | 3 |
| 797f5f4a0469 | cora | gat |  | 30 | 4 |
| 9b798d29c347 | cora | gat |  | 30 | 5 |
| c28bb0180f97 | cora | sgc |  | 2 | 1 |
| f11c004a2d7b | cora | sgc |  | 2 | 2 |
| 6ad175b21fde | cora | sgc |  | 2 | 4 |
| 1599c13e72e1 | cora | sgc |  | 2 | 5 |
| 4169dd6c8db5 | cora | sgc |  | 2 | 7 |
| 6ef5edea5d32 | cora | sgc |  | 4 | 3 |
| 2470934b29b3 | cora | sgc |  | 4 | 4 |
| bacaef34efae | cora | sgc |  | 4 | 5 |
| e3183ddc9721 | cora | sgc |  | 4 | 6 |
| 48c67244c472 | cora | sgc |  | 4 | 7 |
| d5350b3d7225 | cora | sgc |  | 4 | 8 |
| 7855a39ce2d9 | cora | sgc |  | 4 | 9 |
| 3be40c549471 | cora | sgc |  | 8 | 0 |
| f3065de0127a | cora | sgc |  | 8 | 1 |
| 17d325e5a6d1 | cora | sgc |  | 8 | 2 |
| b1b36e0106b7 | cora | sgc |  | 8 | 3 |
| 9cbfa8591fc5 | cora | sgc |  | 8 | 4 |
| c6a249f982ed | cora | sgc |  | 8 | 5 |
| 69d3bc1d437a | cora | sgc |  | 8 | 6 |
| 79f2fd70c6e5 | cora | sgc |  | 8 | 7 |
| 1ad11bdfab1e | cora | sgc |  | 10 | 2 |
| 0a908e10f99a | cora | sgc |  | 10 | 3 |
| 360b370f80fb | cora | sgc |  | 10 | 4 |
| 2a3ca63a2be4 | cora | sgc |  | 10 | 5 |
| 8e7abedf571e | cora | sgc |  | 10 | 6 |
| 48d3c0ecc173 | cora | sgc |  | 10 | 7 |
| b2cd033f09f4 | cora | sgc |  | 10 | 8 |
| 94df9b06fc20 | cora | sgc |  | 10 | 9 |
| e6afe3ff5a25 | cora | sgc |  | 20 | 0 |
| 228263778732 | cora | sgc |  | 20 | 1 |
| b7c40f535b59 | cora | sgc |  | 20 | 2 |
| 4dacafb384ef | cora | sgc |  | 20 | 3 |
| 4f8486b10c04 | cora | sgc |  | 20 | 4 |
| 11c409e19dcc | cora | sgc |  | 20 | 5 |
| 621a83187be7 | cora | sgc |  | 20 | 6 |
| 218580c1fa2b | cora | sgc |  | 20 | 7 |
| ea2f99a507f0 | cora | sgc |  | 20 | 8 |
| f9b895722426 | cora | sgc |  | 20 | 9 |
| 245e94df05ad | cora | sgc |  | 30 | 0 |
| e77ca47c4d31 | cora | sgc |  | 30 | 1 |
| 59639ae9028f | cora | sgc |  | 30 | 2 |
| a634460ab536 | cora | sgc |  | 30 | 3 |
| ff33a5538b8c | cora | sgc |  | 30 | 4 |
| 4de97aac2472 | cora | sgc |  | 30 | 5 |
| df1f524cf738 | cora | sgc |  | 30 | 6 |
| c205eaae62b4 | cora | sgc |  | 30 | 7 |
| 0a6ed22eb2cd | cora | sgc |  | 30 | 8 |
| 3a4e1ab9143a | cora | sgc |  | 30 | 9 |
| 4228c5c431c4 | cora | appnp |  | 2 | 0 |
| 7093f4143658 | cora | appnp |  | 2 | 1 |
| c8d85688eb59 | cora | appnp |  | 2 | 2 |
| 02927f292eaa | cora | appnp |  | 2 | 3 |
| e0ec123cd068 | cora | appnp |  | 2 | 4 |
| 3ffb55e56b77 | cora | appnp |  | 2 | 5 |
| a56df312bd0b | cora | appnp |  | 2 | 6 |
| 4a61c2fe785f | cora | appnp |  | 2 | 7 |
| b5f5b3000f74 | cora | appnp |  | 2 | 8 |
| 8ce5408b2e93 | cora | appnp |  | 2 | 9 |
| 4aac2ed11a7a | cora | appnp |  | 4 | 0 |
| c7713c5657bd | cora | appnp |  | 4 | 1 |
| c546b1fa4821 | cora | appnp |  | 4 | 2 |
| 3cb519cf001e | cora | appnp |  | 4 | 3 |
| bd7c7586baf2 | cora | appnp |  | 4 | 4 |
| 9444d6a0e40f | cora | appnp |  | 4 | 5 |
| 02d5aee0eb48 | cora | appnp |  | 4 | 6 |
| 812ccca569bc | cora | appnp |  | 4 | 7 |
| 3a58019a2152 | cora | appnp |  | 4 | 8 |
| c73af3586933 | cora | appnp |  | 4 | 9 |
| d8f21bdcec6b | cora | appnp |  | 8 | 0 |
| 0d1fb00393ce | cora | appnp |  | 8 | 1 |
| 70e219446346 | cora | appnp |  | 8 | 2 |
| e62bca6f557d | cora | appnp |  | 8 | 3 |
| ce37c4c02457 | cora | appnp |  | 8 | 4 |
| 165617df9bd9 | cora | appnp |  | 8 | 5 |
| e3fbd3cb5cde | cora | appnp |  | 8 | 6 |
| c79a3d0f2702 | cora | appnp |  | 8 | 7 |
| 2f18eecbb16b | cora | appnp |  | 8 | 8 |
| a41b9c9a4ae0 | cora | appnp |  | 8 | 9 |
| 466042c06b84 | cora | appnp |  | 10 | 0 |
| 5f0a0ac4a778 | cora | appnp |  | 10 | 1 |
| 9d54d3016c1d | cora | appnp |  | 10 | 2 |
| 8df16ebfa144 | cora | appnp |  | 10 | 3 |
| 1c3c58200eb6 | cora | appnp |  | 10 | 4 |
| a21679cea2fc | cora | appnp |  | 10 | 5 |
| 2f1b5af8598b | cora | appnp |  | 10 | 6 |
| ca6e6dc954a1 | cora | appnp |  | 10 | 7 |
| 7bddd90fafe8 | cora | appnp |  | 10 | 8 |
| d1b60a90d001 | cora | appnp |  | 10 | 9 |
| 2b98599be74b | cora | appnp |  | 20 | 0 |
| a79b4de6954f | cora | appnp |  | 20 | 1 |
| e43876b25144 | cora | appnp |  | 20 | 2 |
| 12a4de3bc20e | cora | appnp |  | 20 | 3 |
| e9e95eed1c01 | cora | appnp |  | 20 | 4 |
| 5dad7a88397f | cora | appnp |  | 20 | 5 |
| 8d81debd572d | cora | appnp |  | 20 | 6 |
| bc3940c67482 | cora | appnp |  | 20 | 7 |
| 3cd183c6757a | cora | appnp |  | 20 | 8 |
| cbd38f028186 | cora | appnp |  | 20 | 9 |
| 546ae627c25f | cora | appnp |  | 30 | 0 |
| 8e3c5fc2f935 | cora | appnp |  | 30 | 1 |
| 0aca699ba155 | cora | appnp |  | 30 | 2 |
| 22051d6e88cc | cora | appnp |  | 30 | 3 |
| e1e0f6b2c343 | cora | appnp |  | 30 | 4 |
| 5f10340bbb0b | cora | appnp |  | 30 | 5 |
| 7164ab67a163 | cora | appnp |  | 30 | 6 |
| 9df2e6452e08 | cora | appnp |  | 30 | 7 |
| b7f251286749 | cora | appnp |  | 30 | 8 |
| 179cf4c97d80 | cora | appnp |  | 30 | 9 |
| 43a1c9cd3b98 | cora | arma |  | 2 | 0 |
| e57724686b41 | cora | arma |  | 2 | 2 |
| 2d7310e5c6d7 | cora | arma |  | 2 | 3 |
| f81d3e2d37a5 | cora | arma |  | 2 | 5 |
| 3a02b8067188 | cora | arma |  | 2 | 7 |
| 391c06066fff | cora | arma |  | 2 | 8 |
| e4193b5ac5aa | cora | arma |  | 2 | 9 |
| 31d51a7848de | cora | arma |  | 4 | 0 |
| 7518d0501759 | cora | arma |  | 4 | 1 |
| 9843603dbf13 | cora | arma |  | 4 | 3 |
| 67a800e8ccb6 | cora | arma |  | 4 | 4 |
| 5ea2064a3260 | cora | arma |  | 4 | 6 |
| b87fcb1c685e | cora | arma |  | 8 | 0 |
| 17a9ab90a6f2 | cora | arma |  | 8 | 2 |
| e8f38c21e63b | cora | arma |  | 8 | 3 |
| 3abbffba2970 | cora | arma |  | 8 | 4 |
| f262d6e9fbf0 | cora | arma |  | 8 | 5 |
| ac978611c35d | cora | arma |  | 8 | 6 |
| 67a4b4f888b3 | cora | arma |  | 8 | 8 |
| 992674e74b57 | cora | arma |  | 8 | 9 |
| 68b54cec1d21 | cora | arma |  | 10 | 1 |
| 43e3b15a350e | cora | arma |  | 10 | 2 |
| b2172050228d | cora | arma |  | 10 | 4 |
| 1a877f9ab0bc | cora | arma |  | 10 | 5 |
| 6277e63fc659 | cora | arma |  | 10 | 7 |
| 0cfcf14073b7 | cora | arma |  | 20 | 3 |
| 3d51c7a3e4b4 | cora | arma |  | 20 | 4 |
| 2a26eef3b311 | cora | arma |  | 20 | 5 |
| 2505b44543b9 | cora | arma |  | 20 | 6 |
| c126ad8b734a | cora | arma |  | 20 | 7 |
| 29c00415c3a8 | cora | arma |  | 20 | 8 |
| 56d644ecb334 | cora | arma |  | 20 | 9 |
| bef4f6ee6a0a | cora | arma |  | 30 | 0 |
| 61ee6fe537fb | cora | arma |  | 30 | 1 |
| ec9076785c28 | cora | arma |  | 30 | 2 |
| f49f4eca40a6 | cora | arma |  | 30 | 3 |
| 53e3b7ced9e5 | cora | arma |  | 30 | 4 |
| 73e5a4e104d2 | cora | arma |  | 30 | 5 |
| cf2ae99e2d16 | cora | arma |  | 30 | 6 |
| 731243c8fc52 | cora | arma |  | 30 | 7 |
| 4c1905f7e8b3 | cora | arma |  | 30 | 8 |
| d5bc70909050 | cora | arma |  | 30 | 9 |
| 8a8ce776f2fa | cora | gcnii |  | 2 | 0 |
| 7fdc48395708 | cora | gcnii |  | 2 | 1 |
| 8c347fc1cf77 | cora | gcnii |  | 2 | 2 |
| ca0d5207c33a | cora | gcnii |  | 2 | 3 |
| b4eec9193883 | cora | gcnii |  | 2 | 4 |
| 43d5b4ad3ee6 | cora | gcnii |  | 2 | 5 |
| 82910a6c1fe2 | cora | gcnii |  | 2 | 6 |
| e1ff88ce2acf | cora | gcnii |  | 2 | 7 |
| 33984aba3489 | cora | gcnii |  | 2 | 8 |
| 615a736f530b | cora | gcnii |  | 2 | 9 |
| 67b63f2de5c4 | cora | gcnii |  | 4 | 0 |
| 36c314ed606b | cora | gcnii |  | 4 | 1 |
| 741cf30397c9 | cora | gcnii |  | 4 | 2 |
| b50f2aa6778e | cora | gcnii |  | 4 | 3 |
| 2aa90823bf1a | cora | gcnii |  | 4 | 4 |
| 38a097bf5578 | cora | gcnii |  | 4 | 5 |
| 120185f9658f | cora | gcnii |  | 4 | 6 |
| 167329041e4d | cora | gcnii |  | 4 | 7 |
| ed5ef5f8224c | cora | gcnii |  | 4 | 8 |
| b81fb2e8b5f8 | cora | gcnii |  | 4 | 9 |
| 7f6cee710928 | cora | gcnii |  | 8 | 0 |
| 3f2d34496fc6 | cora | gcnii |  | 8 | 1 |
| 586d1e3a5fe8 | cora | gcnii |  | 8 | 2 |
| d0882e929bf3 | cora | gcnii |  | 8 | 3 |
| a52793ff50f7 | cora | gcnii |  | 8 | 4 |
| 04f9cc8f018e | cora | gcnii |  | 8 | 5 |
| f6080fa64c10 | cora | gcnii |  | 8 | 6 |
| 4b00878df0f7 | cora | gcnii |  | 8 | 7 |
| fc154165303f | cora | gcnii |  | 8 | 8 |
| 47057bd15715 | cora | gcnii |  | 8 | 9 |
| ee5445a4c53f | cora | gcnii |  | 10 | 0 |
| ce04850b8522 | cora | gcnii |  | 10 | 1 |
| ef010de059f1 | cora | gcnii |  | 10 | 2 |
| 62354befda3d | cora | gcnii |  | 10 | 3 |
| f6a9bd3eacdb | cora | gcnii |  | 10 | 4 |
| 8917d2e0a951 | cora | gcnii |  | 10 | 5 |
| dd6ff757f4de | cora | gcnii |  | 10 | 6 |
| 77331c3451b6 | cora | gcnii |  | 10 | 7 |
| 4b265f4ca3c6 | cora | gcnii |  | 10 | 8 |
| c4458b9d70a6 | cora | gcnii |  | 10 | 9 |
| af5db8416cd9 | cora | gcnii |  | 20 | 0 |
| 3f16c1cdf1ed | cora | gcnii |  | 20 | 1 |
| ddcbc6a57fcd | cora | gcnii |  | 20 | 2 |
| a9e8e243ab85 | cora | gcnii |  | 20 | 3 |
| b11ed044478a | cora | gcnii |  | 20 | 4 |
| 58691bb02d9d | cora | gcnii |  | 20 | 5 |
| d2f64e6eda55 | cora | gcnii |  | 20 | 6 |
| 48a71328af8c | cora | gcnii |  | 20 | 7 |
| b9d4d43c83a2 | cora | gcnii |  | 20 | 8 |
| 74fb92f13e0f | cora | gcnii |  | 20 | 9 |
| a84ca5f81823 | cora | gcnii |  | 30 | 0 |
| faf83275684d | cora | gcnii |  | 30 | 1 |
| b70d088b83ea | cora | gcnii |  | 30 | 2 |
| 63a69317e37a | cora | gcnii |  | 30 | 3 |
| 3b1b48da644b | cora | gcnii |  | 30 | 4 |
| 561efa5f9f7b | cora | gcnii |  | 30 | 5 |
| 5435676abdec | cora | gcnii |  | 30 | 6 |
| c8d97935b5e9 | cora | gcnii |  | 30 | 7 |
| b814934f6fc4 | cora | gcnii |  | 30 | 8 |
| af65a75e646a | cora | gcnii |  | 30 | 9 |
| 17c8370966d7 | cora | pairnorm |  | 2 | 0 |
| c5cc1cd8728e | cora | pairnorm |  | 2 | 1 |
| 27bc0f3f68e7 | cora | pairnorm |  | 2 | 2 |
| fd969ac02f99 | cora | pairnorm |  | 2 | 3 |
| d6748d98b598 | cora | pairnorm |  | 2 | 4 |
| 330a5ab48361 | cora | pairnorm |  | 2 | 5 |
| ce4fff19ae6f | cora | pairnorm |  | 2 | 6 |
| 64e16f12da39 | cora | pairnorm |  | 2 | 7 |
| e01ff4441d9f | cora | pairnorm |  | 2 | 8 |
| 5b0d34d455be | cora | pairnorm |  | 2 | 9 |
| d56bc37d3da8 | cora | pairnorm |  | 4 | 0 |
| 3dbb3feff65e | cora | pairnorm |  | 4 | 1 |
| fbcf4857dad3 | cora | pairnorm |  | 4 | 2 |
| 37a87c8f7826 | cora | pairnorm |  | 4 | 3 |
| 940c984c0660 | cora | pairnorm |  | 4 | 4 |
| b58ee9eec5a7 | cora | pairnorm |  | 4 | 5 |
| 872ececbff75 | cora | pairnorm |  | 4 | 6 |
| e84f87d48a5d | cora | pairnorm |  | 4 | 7 |
| d2a46d476c56 | cora | pairnorm |  | 4 | 8 |
| 9dc52bb7624b | cora | pairnorm |  | 4 | 9 |
| 2c0d1e7a2888 | cora | pairnorm |  | 8 | 0 |
| 201e7ef3aa0c | cora | pairnorm |  | 8 | 1 |
| 2626f9e8b231 | cora | pairnorm |  | 8 | 2 |
| d0031e43e5ab | cora | pairnorm |  | 8 | 3 |
| 633b4959c405 | cora | pairnorm |  | 8 | 4 |
| f5dcfcf67800 | cora | pairnorm |  | 8 | 5 |
| 7bd361a4bfce | cora | pairnorm |  | 8 | 6 |
| 2cd3da045ec6 | cora | pairnorm |  | 8 | 7 |
| 5a9ed2638a03 | cora | pairnorm |  | 8 | 8 |
| c19aa637594b | cora | pairnorm |  | 8 | 9 |
| 7b280af95429 | cora | pairnorm |  | 10 | 0 |
| e3e1d1a0fc94 | cora | pairnorm |  | 10 | 2 |
| fef6beff62c2 | cora | pairnorm |  | 10 | 3 |
| ed6d47696b5b | cora | pairnorm |  | 10 | 5 |
| 8883899ec14c | cora | pairnorm |  | 10 | 7 |
| f9b4f89b85d7 | cora | pairnorm |  | 10 | 8 |
| 772705e7dd0c | cora | pairnorm |  | 10 | 9 |
| 119e745abf4e | cora | pairnorm |  | 20 | 0 |
| c49eeefc7457 | cora | pairnorm |  | 20 | 1 |
| 4dab1770a1e4 | cora | pairnorm |  | 20 | 3 |
| 4cce2ac1db13 | cora | pairnorm |  | 20 | 4 |
| 1f2e31001f58 | cora | pairnorm |  | 20 | 6 |
| f0d8fe1fd241 | cora | pairnorm |  | 30 | 0 |
| 79e46cc7fe22 | cora | pairnorm |  | 30 | 2 |
| e1b07b77470f | cora | pairnorm |  | 30 | 3 |
| 3b7b12b3596d | cora | pairnorm |  | 30 | 4 |
| 429a6747b5fb | cora | pairnorm |  | 30 | 5 |
| c61fb0a3a227 | cora | pairnorm |  | 30 | 6 |
| 89dc1378f192 | cora | pairnorm |  | 30 | 8 |
| 803b5d21f64e | cora | pairnorm |  | 30 | 9 |
| bdb6a2cf0dbe | cora | dropedge |  | 2 | 4 |
| 5ba91580f671 | cora | dropedge |  | 2 | 5 |
| 0b0e1c6673cf | cora | dropedge |  | 2 | 7 |
| bdabe84ee5b8 | cora | dropedge |  | 20 | 0 |
| 282069f77d7b | cora | dropedge |  | 20 | 1 |
| 6f0b53bde632 | cora | dropedge |  | 20 | 2 |
| 4a7a9a9f29b9 | cora | dropedge |  | 20 | 3 |
| b80918d40fb3 | cora | dropedge |  | 20 | 4 |
| 639b37318a94 | cora | dropedge |  | 20 | 5 |
| b756998c93fa | cora | dropedge |  | 20 | 6 |
| 1848d74d8178 | cora | dropedge |  | 20 | 7 |
| a3f5f13f1a60 | cora | dropedge |  | 20 | 8 |
| ba620f26c535 | cora | dropedge |  | 20 | 9 |
| a81731312304 | cora | dropedge |  | 30 | 0 |
| 13943d5ee404 | cora | dropedge |  | 30 | 1 |
| c7c7560fd6ec | cora | dropedge |  | 30 | 6 |
| 77d1e8569b8a | cora | dropedge |  | 30 | 7 |
| 288e841cd28c | cora | dropedge |  | 30 | 8 |
| c7233448814f | cora | dropedge |  | 30 | 9 |
| 5df29d8304fa | cora | graphunet |  | 2 | 0 |
| 6fe7389b0674 | cora | graphunet |  | 2 | 1 |
| 679bc2271ec0 | cora | graphunet |  | 2 | 2 |
| e30f149a3cac | cora | graphunet |  | 2 | 3 |
| 85dbf033ae63 | cora | graphunet |  | 2 | 4 |
| f5990c89b460 | cora | graphunet |  | 2 | 5 |
| c305fa1e150a | cora | graphunet |  | 2 | 6 |
| f7f518bce1d2 | cora | graphunet |  | 2 | 7 |
| 6a29cb145a72 | cora | graphunet |  | 4 | 2 |
| a95623fc0568 | cora | graphunet |  | 4 | 3 |
| 3ce493163335 | cora | graphunet |  | 4 | 4 |
| 6938f4c4903e | cora | graphunet |  | 4 | 5 |
| 8467db801c40 | cora | graphunet |  | 4 | 6 |
| c006753bd874 | cora | graphunet |  | 4 | 7 |
| 61e3695ca2b5 | cora | graphunet |  | 4 | 8 |
| fda2665cd0e7 | cora | graphunet |  | 4 | 9 |
| 1687bb40e8ba | cora | graphunet |  | 8 | 0 |
| 005e92a8d5a2 | cora | graphunet |  | 8 | 1 |
| fec61e5c6831 | cora | graphunet |  | 8 | 2 |
| 143892024c6b | cora | graphunet |  | 8 | 3 |
| bd88b03dd9cc | cora | graphunet |  | 8 | 4 |
| ffb5879a5010 | cora | graphunet |  | 8 | 5 |
| 39ee7bc6e0bb | cora | graphunet |  | 8 | 6 |
| 1f140e790eb9 | cora | graphunet |  | 8 | 7 |
| ce0576c771c7 | cora | graphunet |  | 8 | 8 |
| 227dfb6a4825 | cora | graphunet |  | 8 | 9 |
| 2228c3db6693 | cora | graphunet |  | 10 | 0 |
| 60ca0de8ebb6 | cora | graphunet |  | 10 | 1 |
| 3a4a6b7807cf | cora | graphunet |  | 10 | 2 |
| a65cfd10aeca | cora | graphunet |  | 10 | 3 |
| 93337c57e27f | cora | graphunet |  | 10 | 4 |
| 961b427fe960 | cora | graphunet |  | 10 | 5 |
| 1e87d51a8ff1 | cora | graphunet |  | 20 | 0 |
| 39976a8cc5de | cora | graphunet |  | 20 | 1 |
| ccf67e6a7be5 | cora | graphunet |  | 20 | 2 |
| 8caecbe126f6 | cora | graphunet |  | 20 | 3 |
| 22c0b64f8786 | cora | graphunet |  | 20 | 4 |
| 67a9e5faf577 | cora | graphunet |  | 20 | 5 |
| e09adae22981 | cora | graphunet |  | 20 | 6 |
| cc6a9e47dce3 | cora | graphunet |  | 20 | 7 |
| 7e6b7aa08922 | cora | graphunet |  | 20 | 8 |
| c16d19f89803 | cora | graphunet |  | 20 | 9 |
| a737556d046c | cora | graphunet |  | 30 | 0 |
| 587c216a053c | cora | graphunet |  | 30 | 1 |
| 10b99f90088a | cora | graphunet |  | 30 | 2 |
| 97e2f94702ae | cora | graphunet |  | 30 | 3 |
| ddb4e2beff54 | cora | graphunet |  | 30 | 4 |
| f7d263112bf5 | cora | graphunet |  | 30 | 5 |
| f2aa89e6a75e | cora | graphunet |  | 30 | 6 |
| 15fef5dcca29 | cora | graphunet |  | 30 | 7 |
| 5f5308b294c1 | cora | graphunet |  | 30 | 8 |
| 5b346662a75e | cora | graphunet |  | 30 | 9 |
| 98a9b287f0aa | citeseer | nhdp |  | 2 | 0 |
| 0ba776df8f34 | citeseer | nhdp |  | 2 | 2 |
| a1cf4d456ced | citeseer | nhdp |  | 2 | 3 |
| 33868e445575 | citeseer | nhdp |  | 2 | 5 |
| 8b80b7780364 | citeseer | nhdp |  | 2 | 7 |
| 91c6d898ed9e | citeseer | nhdp |  | 2 | 8 |
| b9743a9384a4 | citeseer | nhdp |  | 2 | 9 |
| f4e1999df691 | citeseer | nhdp |  | 4 | 0 |
| 100d48a9a1c1 | citeseer | nhdp |  | 4 | 1 |
| 314cf299c6ee | citeseer | nhdp |  | 4 | 3 |
| 5a57feaf783f | citeseer | nhdp |  | 4 | 4 |
| 93a41aebc4fe | citeseer | nhdp |  | 8 | 4 |
| caad9e3bb99f | citeseer | nhdp |  | 8 | 5 |
| f4a7825c8a4d | citeseer | nhdp |  | 8 | 6 |
| dffc7ea1cf35 | citeseer | nhdp |  | 8 | 8 |
| 5b332aa5c7d5 | citeseer | nhdp |  | 8 | 9 |
| dae5190b075c | citeseer | nhdp |  | 10 | 1 |
| 83163b371a3b | citeseer | nhdp |  | 10 | 2 |
| b360e0737ca5 | citeseer | nhdp |  | 10 | 4 |
| 38f96a12672a | citeseer | nhdp |  | 10 | 5 |
| 3b342543747f | citeseer | nhdp |  | 10 | 7 |
| c15333ef0b64 | citeseer | nhdp |  | 20 | 3 |
| 3e458598f0bc | citeseer | nhdp |  | 20 | 4 |
| 6e5407307551 | citeseer | nhdp |  | 20 | 5 |
| e834cbb76490 | citeseer | nhdp |  | 20 | 6 |
| b681b28bb011 | citeseer | nhdp |  | 20 | 7 |
| b7356a22038e | citeseer | nhdp |  | 30 | 0 |
| 4ecd91e78455 | citeseer | nhdp |  | 30 | 2 |
| 638a71e9d7fb | citeseer | nhdp |  | 30 | 3 |
| f0cf868e7f64 | citeseer | nhdp |  | 30 | 4 |
| 85329a8156cb | citeseer | nhdp |  | 30 | 5 |
| 3a0f9615ecbc | citeseer | gcn |  | 2 | 0 |
| d4d369d107de | citeseer | gcn |  | 2 | 1 |
| 47c85feca64e | citeseer | gcn |  | 2 | 2 |
| fa05773fa246 | citeseer | gcn |  | 2 | 3 |
| 182153d56b78 | citeseer | gcn |  | 2 | 4 |
| e386ea7e8a75 | citeseer | gcn |  | 2 | 5 |
| 14f5b1e40a27 | citeseer | gcn |  | 2 | 6 |
| 7b4541b0692f | citeseer | gcn |  | 2 | 7 |
| df2d9a5fc684 | citeseer | gcn |  | 2 | 8 |
| 84a9a9267097 | citeseer | gcn |  | 2 | 9 |
| 0c6c382f8499 | citeseer | gcn |  | 4 | 0 |
| a66908afb73d | citeseer | gcn |  | 4 | 1 |
| ab902d8e86a5 | citeseer | gcn |  | 4 | 2 |
| 15b2de42b90d | citeseer | gcn |  | 4 | 3 |
| 600472ccc9c0 | citeseer | gcn |  | 4 | 4 |
| 5abe0d471ed1 | citeseer | gcn |  | 4 | 5 |
| 10dd02fdd066 | citeseer | gcn |  | 4 | 6 |
| f72e5a0b33ea | citeseer | gcn |  | 4 | 7 |
| bcf22757a2e4 | citeseer | gcn |  | 4 | 8 |
| 3468f01e6135 | citeseer | gcn |  | 4 | 9 |
| 54603bc579f3 | citeseer | gcn |  | 8 | 0 |
| 23475b5c3491 | citeseer | gcn |  | 8 | 1 |
| 23ab0f3fbb05 | citeseer | gcn |  | 8 | 2 |
| 378f64694425 | citeseer | gcn |  | 8 | 3 |
| ae610896b48b | citeseer | gcn |  | 8 | 4 |
| ef8639310afa | citeseer | gcn |  | 8 | 5 |
| d4b4f791ef63 | citeseer | gcn |  | 8 | 6 |
| e5bb69b1ae0f | citeseer | gcn |  | 8 | 7 |
| 2263260ac51d | citeseer | gcn |  | 10 | 2 |
| 750648323e44 | citeseer | gcn |  | 10 | 3 |
| 7b22ca7c0cd0 | citeseer | gcn |  | 10 | 4 |
| 1d35dc204a69 | citeseer | gcn |  | 10 | 5 |
| 8e7dfacde6f0 | citeseer | gcn |  | 10 | 6 |
| d15f79824360 | citeseer | gcn |  | 10 | 7 |
| c68a1f4fa7ea | citeseer | gcn |  | 10 | 8 |
| 408c30e2782e | citeseer | gcn |  | 10 | 9 |
| 0c5e83253035 | citeseer | gcn |  | 20 | 0 |
| f819ffd42471 | citeseer | gcn |  | 20 | 1 |
| 98c06cfc2343 | citeseer | gcn |  | 20 | 2 |
| be49d832d815 | citeseer | gcn |  | 20 | 3 |
| 85f40a0ef436 | citeseer | gcn |  | 20 | 4 |
| 2f7955e281e0 | citeseer | gcn |  | 20 | 5 |
| 269ceb0d69f6 | citeseer | gcn |  | 20 | 6 |
| 845f8e108200 | citeseer | gcn |  | 20 | 7 |
| e961760f9178 | citeseer | gcn |  | 20 | 8 |
| 1c08500ac36a | citeseer | gcn |  | 20 | 9 |
| 6897b11177d4 | citeseer | gcn |  | 30 | 0 |
| 6339a714a2af | citeseer | gcn |  | 30 | 1 |
| 7e9810b276de | citeseer | gcn |  | 30 | 2 |
| 09d52601aa70 | citeseer | gcn |  | 30 | 3 |
| 7a737a09f278 | citeseer | gcn |  | 30 | 4 |
| 6a0fbb4e4a87 | citeseer | gcn |  | 30 | 5 |
| 58febc5400b7 | citeseer | gcn |  | 30 | 6 |
| badd8d242aee | citeseer | gcn |  | 30 | 7 |
| ac78b3ea2ed4 | citeseer | gcn |  | 30 | 8 |
| 1edaa2d2711b | citeseer | gcn |  | 30 | 9 |
| 75678bb8e57c | citeseer | gat |  | 2 | 0 |
| 2aaa9f0dc610 | citeseer | gat |  | 2 | 1 |
| 2abc03e77bb9 | citeseer | gat |  | 2 | 2 |
| 8bd588531944 | citeseer | gat |  | 2 | 3 |
| 040214bf75fb | citeseer | gat |  | 2 | 4 |
| 1ae215da0ade | citeseer | gat |  | 2 | 5 |
| 8a8fcb26a81c | citeseer | gat |  | 2 | 6 |
| 681443c7dcf2 | citeseer | gat |  | 2 | 7 |
| 389929897554 | citeseer | gat |  | 2 | 8 |
| 57b1c084dd2c | citeseer | gat |  | 2 | 9 |
| d1f73a11ada4 | citeseer | gat |  | 4 | 0 |
| d3f069b68d97 | citeseer | gat |  | 4 | 1 |
| ec3e21fd109a | citeseer | gat |  | 4 | 2 |
| 74b7bdc55a10 | citeseer | gat |  | 4 | 3 |
| bb87d5587cdd | citeseer | gat |  | 4 | 4 |
| 71c51fd6e00f | citeseer | gat |  | 4 | 5 |
| 48305cb18d35 | citeseer | gat |  | 4 | 6 |
| 763c2f7314e1 | citeseer | gat |  | 4 | 7 |
| ec6fb655729f | citeseer | gat |  | 4 | 8 |
| 98c243e97ad5 | citeseer | gat |  | 4 | 9 |
| a3abaa6fd484 | citeseer | gat |  | 8 | 0 |
| 1a4173e71ab8 | citeseer | gat |  | 8 | 1 |
| 7cd256d2dddd | citeseer | gat |  | 8 | 2 |
| 9d5c8dfb114e | citeseer | gat |  | 8 | 3 |
| 6e59368ca66d | citeseer | gat |  | 8 | 4 |
| 0cda3fe4bd95 | citeseer | gat |  | 8 | 5 |
| 911651b566ae | citeseer | gat |  | 8 | 6 |
| accfbf1f5c15 | citeseer | gat |  | 8 | 7 |
| db7f03e17437 | citeseer | gat |  | 8 | 8 |
| 771114d810d0 | citeseer | gat |  | 8 | 9 |
| 679326d26b23 | citeseer | gat |  | 10 | 0 |
| e7d9eeb82b77 | citeseer | gat |  | 10 | 2 |
| 0bccea6288dc | citeseer | gat |  | 10 | 3 |
| 1fdf1aec4816 | citeseer | gat |  | 10 | 5 |
| 68f10115e7ce | citeseer | gat |  | 10 | 7 |
| 4656e762aaa7 | citeseer | gat |  | 10 | 8 |
| 3c2ad1e83039 | citeseer | gat |  | 10 | 9 |
| ecf6d229c282 | citeseer | gat |  | 20 | 0 |
| 91a8a6f433ee | citeseer | gat |  | 20 | 1 |
| 672057f775a0 | citeseer | gat |  | 20 | 3 |
| 40e6847facf8 | citeseer | gat |  | 20 | 4 |
| 45824c20ce4a | citeseer | gat |  | 20 | 6 |
| e798d6343b12 | citeseer | gat |  | 30 | 0 |
| 4706a3b0fedc | citeseer | gat |  | 30 | 2 |
| d761f34eebe4 | citeseer | gat |  | 30 | 3 |
| 9707c51911ee | citeseer | gat |  | 30 | 4 |
| c5c6b371b7cf | citeseer | gat |  | 30 | 5 |
| f883dc13f369 | citeseer | gat |  | 30 | 6 |
| 16dee0c37b4a | citeseer | gat |  | 30 | 8 |
| 599b4f071c28 | citeseer | gat |  | 30 | 9 |
| 16cad20a878b | citeseer | sgc |  | 2 | 1 |
| 55742ee42c66 | citeseer | sgc |  | 2 | 2 |
| 34f41adf6b29 | citeseer | sgc |  | 2 | 4 |
| 5855a9897693 | citeseer | sgc |  | 2 | 5 |
| 866106fa8d80 | citeseer | sgc |  | 2 | 7 |
| d997d1ab7236 | citeseer | sgc |  | 4 | 3 |
| ff3351e24e72 | citeseer | sgc |  | 4 | 4 |
| 0e9fcd780752 | citeseer | sgc |  | 4 | 5 |
| a06db445f29d | citeseer | sgc |  | 4 | 6 |
| 330b4da9e5f5 | citeseer | sgc |  | 4 | 7 |
| 1f82d4eb4eb2 | citeseer | sgc |  | 4 | 8 |
| 52e466e2a41c | citeseer | sgc |  | 4 | 9 |
| 58a2210a09ae | citeseer | sgc |  | 8 | 0 |
| db4b5bb21766 | citeseer | sgc |  | 8 | 1 |
| 9ef49a77ae3f | citeseer | sgc |  | 8 | 2 |
| ece42aee3085 | citeseer | sgc |  | 8 | 3 |
| 424cf30e688c | citeseer | sgc |  | 8 | 4 |
| 1165bef26e6c | citeseer | sgc |  | 8 | 5 |
| 811d14dd44d6 | citeseer | sgc |  | 8 | 6 |
| 529b7af6d779 | citeseer | sgc |  | 8 | 7 |
| e472ef61cd67 | citeseer | sgc |  | 8 | 8 |
| cbc69291d655 | citeseer | sgc |  | 8 | 9 |
| 089ec5f5b920 | citeseer | sgc |  | 10 | 0 |
| 6d504948d271 | citeseer | sgc |  | 10 | 1 |
| 6cbd931cf02d | citeseer | sgc |  | 10 | 2 |
| 4468418ab462 | citeseer | sgc |  | 10 | 3 |
| 17a11aa92b75 | citeseer | sgc |  | 10 | 4 |
| bd97eb39e43e | citeseer | sgc |  | 10 | 5 |
| b01b24606216 | citeseer | sgc |  | 10 | 6 |
| 3caf3b93508e | citeseer | sgc |  | 10 | 7 |
| 3e5e9285f8e2 | citeseer | sgc |  | 10 | 8 |
| 6c77558ee807 | citeseer | sgc |  | 10 | 9 |
| 295a5fa055ea | citeseer | sgc |  | 20 | 0 |
| dea9dba4feb0 | citeseer | sgc |  | 20 | 1 |
| 66695a5f6f22 | citeseer | sgc |  | 20 | 2 |
| 8858699c1a3c | citeseer | sgc |  | 20 | 3 |
| bc9804f73c9d | citeseer | sgc |  | 20 | 4 |
| cbd0983badd9 | citeseer | sgc |  | 20 | 5 |
| 139f6967ec2e | citeseer | sgc |  | 20 | 6 |
| 299e3a644731 | citeseer | sgc |  | 20 | 7 |
| 33e25f35eb99 | citeseer | sgc |  | 20 | 8 |
| 755678023ced | citeseer | sgc |  | 20 | 9 |
| 5dbc7f16c2a5 | citeseer | sgc |  | 30 | 0 |
| 59e695335a4d | citeseer | sgc |  | 30 | 1 |
| 42a1c9b950e4 | citeseer | sgc |  | 30 | 2 |
| 85c3270d0c29 | citeseer | sgc |  | 30 | 3 |
| b2f898ff2914 | citeseer | sgc |  | 30 | 4 |
| 756d278f1db9 | citeseer | sgc |  | 30 | 5 |
| fa1f83a5fb57 | citeseer | sgc |  | 30 | 6 |
| 8138d69c7494 | citeseer | sgc |  | 30 | 7 |
| 3a37b445dfcd | citeseer | sgc |  | 30 | 8 |
| ab344642f167 | citeseer | sgc |  | 30 | 9 |
| ba60b6cf3de9 | citeseer | appnp |  | 2 | 0 |
| df19b6dfc163 | citeseer | appnp |  | 2 | 1 |
| 57f4f7106086 | citeseer | appnp |  | 2 | 2 |
| a02fde92e8e6 | citeseer | appnp |  | 2 | 3 |
| 2ca97b70e207 | citeseer | appnp |  | 2 | 4 |
| f3e9e47a9ee8 | citeseer | appnp |  | 2 | 5 |
| f9f0566c2be6 | citeseer | appnp |  | 2 | 6 |
| f9f4ef0a3959 | citeseer | appnp |  | 2 | 7 |
| c52f6c7c979a | citeseer | appnp |  | 2 | 8 |
| 7a5ffa822afd | citeseer | appnp |  | 2 | 9 |
| c3e644c9963a | citeseer | appnp |  | 4 | 0 |
| 3f15bf13586d | citeseer | appnp |  | 4 | 1 |
| f1eff4405f42 | citeseer | appnp |  | 4 | 2 |
| 2aeb53ff6618 | citeseer | appnp |  | 4 | 3 |
| 812dea39ee94 | citeseer | appnp |  | 4 | 4 |
| d86cd6c48412 | citeseer | appnp |  | 4 | 5 |
| e767d8b88352 | citeseer | appnp |  | 4 | 6 |
| 2064279ddbf2 | citeseer | appnp |  | 4 | 7 |
| 5c13720baa6e | citeseer | appnp |  | 4 | 8 |
| 562e03c24457 | citeseer | appnp |  | 4 | 9 |
| e2e1ec123378 | citeseer | appnp |  | 8 | 0 |
| 12416dc92b1f | citeseer | appnp |  | 8 | 1 |
| 47691d907f98 | citeseer | appnp |  | 8 | 2 |
| 1ec0e493b202 | citeseer | appnp |  | 8 | 3 |
| 321f04280714 | citeseer | appnp |  | 8 | 4 |
| 44e5139126d0 | citeseer | appnp |  | 8 | 5 |
| a62b73052dd3 | citeseer | appnp |  | 8 | 6 |
| 529df367c64e | citeseer | appnp |  | 8 | 7 |
| f40b100540e8 | citeseer | appnp |  | 8 | 8 |
| 648ee576a03b | citeseer | appnp |  | 8 | 9 |
| 854359544582 | citeseer | appnp |  | 10 | 0 |
| 93dd64056473 | citeseer | appnp |  | 10 | 1 |
| 312dc97b6bcc | citeseer | appnp |  | 10 | 2 |
| 39e41c4a9530 | citeseer | appnp |  | 10 | 3 |
| 59befe5daa0a | citeseer | appnp |  | 10 | 4 |
| 4ed4f5dfb990 | citeseer | appnp |  | 10 | 5 |
| 4446e2db3074 | citeseer | appnp |  | 10 | 6 |
| 702ba5b0c300 | citeseer | appnp |  | 10 | 7 |
| 6e9a8eb66a31 | citeseer | appnp |  | 10 | 8 |
| d43464d8d74c | citeseer | appnp |  | 10 | 9 |
| ea51c83f98e8 | citeseer | appnp |  | 20 | 0 |
| 21e68f3c2039 | citeseer | appnp |  | 20 | 1 |
| 3525cffaf748 | citeseer | appnp |  | 20 | 2 |
| b03244b11fb9 | citeseer | appnp |  | 20 | 3 |
| e70012c0a46c | citeseer | appnp |  | 20 | 4 |
| f956d0a8b084 | citeseer | appnp |  | 20 | 5 |
| 3f634aa0966d | citeseer | appnp |  | 20 | 6 |
| fd9ca23b5100 | citeseer | appnp |  | 20 | 7 |
| 18bd24b756ee | citeseer | appnp |  | 20 | 8 |
| 450e0672366c | citeseer | appnp |  | 20 | 9 |
| f63dbddf44d2 | citeseer | appnp |  | 30 | 0 |
| db6bd9367e37 | citeseer | appnp |  | 30 | 1 |
| dbe4e8f6ff08 | citeseer | appnp |  | 30 | 2 |
| 8978ae7e2c16 | citeseer | appnp |  | 30 | 3 |
| f4a4141a38a0 | citeseer | appnp |  | 30 | 4 |
| 2755b012e505 | citeseer | appnp |  | 30 | 5 |
| 005bf6f2f299 | citeseer | appnp |  | 30 | 6 |
| 17b337ed1ec9 | citeseer | appnp |  | 30 | 7 |
| d52606107b5c | citeseer | appnp |  | 30 | 8 |
| 955d5d64e8be | citeseer | appnp |  | 30 | 9 |
| b808d6d77e45 | citeseer | arma |  | 2 | 0 |
| bc635326afce | citeseer | arma |  | 2 | 2 |
| b06eb96ea707 | citeseer | arma |  | 2 | 3 |
| 5556f0c183a1 | citeseer | arma |  | 2 | 5 |
| cea7ccfb1c3c | citeseer | arma |  | 2 | 7 |
| 4ea5e0e03ef2 | citeseer | arma |  | 2 | 8 |
| b52ed1a81f7b | citeseer | arma |  | 2 | 9 |
| 5e3a4a29665b | citeseer | arma |  | 4 | 0 |
| f0535a9835f9 | citeseer | arma |  | 4 | 1 |
| 7835fb044c50 | citeseer | arma |  | 4 | 3 |
| 87f5a2a05316 | citeseer | arma |  | 4 | 4 |
| cf55a795f220 | citeseer | arma |  | 4 | 6 |
| fc296380d024 | citeseer | arma |  | 8 | 0 |
| 8fcf944c8c61 | citeseer | arma |  | 8 | 2 |
| edd809b68c9d | citeseer | arma |  | 8 | 3 |
| caf01d07b609 | citeseer | arma |  | 8 | 4 |
| 2c6b48d184ad | citeseer | arma |  | 8 | 5 |
| de1d93152949 | citeseer | arma |  | 8 | 6 |
| f1e7290fb0b4 | citeseer | arma |  | 8 | 8 |
| 2f75ab2ee1bf | citeseer | arma |  | 8 | 9 |
| 287c985996c5 | citeseer | arma |  | 10 | 1 |
| b1bfeaa4e67f | citeseer | arma |  | 10 | 2 |
| e4fea69f4084 | citeseer | arma |  | 10 | 4 |
| a2f2e2f9be60 | citeseer | arma |  | 10 | 5 |
| ddb44ab982d2 | citeseer | arma |  | 10 | 7 |
| d3fdd1d52e9b | citeseer | arma |  | 20 | 3 |
| 81c01401fc89 | citeseer | arma |  | 20 | 4 |
| 61ec364c33b4 | citeseer | arma |  | 20 | 5 |
| 1c51c7a74532 | citeseer | arma |  | 20 | 6 |
| 09b4e4eed537 | citeseer | arma |  | 20 | 7 |
| 0b3f8aaf4e39 | citeseer | arma |  | 20 | 8 |
| f65d300f94b2 | citeseer | arma |  | 20 | 9 |
| a2b6327fdd41 | citeseer | arma |  | 30 | 0 |
| 09d7700cf123 | citeseer | arma |  | 30 | 1 |
| adcbcb4f99f9 | citeseer | arma |  | 30 | 2 |
| d30c386118ba | citeseer | arma |  | 30 | 3 |
| 5621ab646a21 | citeseer | arma |  | 30 | 4 |
| d7015ccb5000 | citeseer | arma |  | 30 | 5 |
| c50b79609774 | citeseer | arma |  | 30 | 6 |
| 6ca561b279b0 | citeseer | arma |  | 30 | 7 |
| e4e3d297056c | citeseer | arma |  | 30 | 8 |
| f47cc7aba86f | citeseer | arma |  | 30 | 9 |
| 1dbd54d7613b | citeseer | gcnii |  | 2 | 4 |
| be8c6e6a75a4 | citeseer | gcnii |  | 2 | 5 |
| 36f885a6b112 | citeseer | gcnii |  | 2 | 6 |
| 07703c00e7e8 | citeseer | gcnii |  | 2 | 7 |
| 80c6359e472d | citeseer | gcnii |  | 20 | 0 |
| 6d2e93c53e83 | citeseer | gcnii |  | 20 | 1 |
| a573eb50892c | citeseer | gcnii |  | 20 | 2 |
| 068b9508e75d | citeseer | gcnii |  | 20 | 3 |
| 42eac02c1007 | citeseer | gcnii |  | 20 | 4 |
| cde0c432f73f | citeseer | gcnii |  | 20 | 5 |
| a9f370939e88 | citeseer | gcnii |  | 20 | 6 |
| 8e6b40c15818 | citeseer | gcnii |  | 20 | 7 |
| 447cb853208d | citeseer | gcnii |  | 20 | 8 |
| 61d9cd8b599d | citeseer | gcnii |  | 20 | 9 |
| 15a182d92f81 | citeseer | gcnii |  | 30 | 0 |
| f751901d29ea | citeseer | gcnii |  | 30 | 1 |
| 14825de32feb | citeseer | gcnii |  | 30 | 6 |
| 1a9acf24c8e9 | citeseer | gcnii |  | 30 | 7 |
| 7479c0ff92af | citeseer | gcnii |  | 30 | 8 |
| aab42246f54d | citeseer | gcnii |  | 30 | 9 |
| cb1d37a52bd5 | citeseer | pairnorm |  | 2 | 0 |
| 7ae806c7e12f | citeseer | pairnorm |  | 2 | 1 |
| e6888e6b0fec | citeseer | pairnorm |  | 2 | 2 |
| 8d96f5bef9b4 | citeseer | pairnorm |  | 2 | 3 |
| 9d6d66ea0021 | citeseer | pairnorm |  | 2 | 4 |
| 1f6c73207468 | citeseer | pairnorm |  | 2 | 5 |
| 12520729cabf | citeseer | pairnorm |  | 2 | 6 |
| 6b08f2d569a5 | citeseer | pairnorm |  | 2 | 7 |
| 4792c90e15f0 | citeseer | pairnorm |  | 4 | 2 |
| 4a30f49eb380 | citeseer | pairnorm |  | 4 | 3 |
| bec8d1c4e1e6 | citeseer | pairnorm |  | 4 | 4 |
| 6595a81db482 | citeseer | pairnorm |  | 4 | 5 |
| d6e37b881ab6 | citeseer | pairnorm |  | 4 | 6 |
| 90d6ed0accfd | citeseer | pairnorm |  | 4 | 7 |
| ddc2a7b768a3 | citeseer | pairnorm |  | 4 | 8 |
| a3559fe7f34e | citeseer | pairnorm |  | 4 | 9 |
| 0023af819d1a | citeseer | pairnorm |  | 8 | 0 |
| c719b5af8118 | citeseer | pairnorm |  | 8 | 1 |
| 83fd8e15fbdd | citeseer | pairnorm |  | 8 | 2 |
| 3e7a8b634d12 | citeseer | pairnorm |  | 8 | 3 |
| 2f283cea1c59 | citeseer | pairnorm |  | 8 | 4 |
| 83499fae1bd6 | citeseer | pairnorm |  | 8 | 5 |
| 09ea3ba22329 | citeseer | pairnorm |  | 8 | 6 |
| 0658eacea352 | citeseer | pairnorm |  | 8 | 7 |
| da01f425ce40 | citeseer | pairnorm |  | 8 | 8 |
| ee7f3a2f82cd | citeseer | pairnorm |  | 8 | 9 |
| 5d76f0913884 | citeseer | pairnorm |  | 10 | 0 |
| 4aa4020fdb04 | citeseer | pairnorm |  | 10 | 2 |
| 6760ad228692 | citeseer | pairnorm |  | 10 | 3 |
| 43b0fbbcd389 | citeseer | pairnorm |  | 10 | 5 |
| 6ba62342e893 | citeseer | pairnorm |  | 20 | 0 |
| 8b1176daf6c3 | citeseer | pairnorm |  | 20 | 1 |
| c7f974ae70ff | citeseer | pairnorm |  | 20 | 3 |
| cd43bf570367 | citeseer | pairnorm |  | 20 | 4 |
| eb7d0f07edf6 | citeseer | pairnorm |  | 20 | 6 |
| 5b39149ea54b | citeseer | pairnorm |  | 30 | 0 |
| 3e8d3a26781e | citeseer | pairnorm |  | 30 | 2 |
| 0738831d235f | citeseer | pairnorm |  | 30 | 3 |
| 0a00b573de2d | citeseer | pairnorm |  | 30 | 4 |
| e45f0774d4cc | citeseer | pairnorm |  | 30 | 5 |
| c1020c0f9fe6 | citeseer | pairnorm |  | 30 | 6 |
| aacfd6480506 | citeseer | pairnorm |  | 30 | 8 |
| a4a4162b1ba7 | citeseer | pairnorm |  | 30 | 9 |
| 9392e2ccb173 | citeseer | dropedge |  | 2 | 1 |
| b05ea997c564 | citeseer | dropedge |  | 2 | 2 |
| d7e82394fb58 | citeseer | dropedge |  | 2 | 4 |
| 9060138ca79b | citeseer | dropedge |  | 2 | 5 |
| cfb9bcef0616 | citeseer | dropedge |  | 2 | 7 |
| 49ed5c1fc826 | citeseer | dropedge |  | 4 | 3 |
| 6179b14a055e | citeseer | dropedge |  | 4 | 4 |
| 690ba9b3aff7 | citeseer | dropedge |  | 4 | 5 |
| 3c6d71d55819 | citeseer | dropedge |  | 8 | 4 |
| 988280161aff | citeseer | dropedge |  | 8 | 5 |
| 662f9d381c37 | citeseer | dropedge |  | 8 | 6 |
| 5f7b0e2a3438 | citeseer | dropedge |  | 8 | 7 |
| 567a8a76ca17 | citeseer | dropedge |  | 8 | 8 |
| 85e30645acc3 | citeseer | dropedge |  | 8 | 9 |
| ecc6c929330a | citeseer | dropedge |  | 10 | 0 |
| 23795c8686cb | citeseer | dropedge |  | 10 | 1 |
| 1d69eef880de | citeseer | dropedge |  | 10 | 2 |
| 7bbff7425f0d | citeseer | dropedge |  | 10 | 3 |
| fce22596e4d4 | citeseer | dropedge |  | 10 | 4 |
| 6114ffc09ef8 | citeseer | dropedge |  | 10 | 5 |
| 1c6d4a573489 | citeseer | dropedge |  | 10 | 6 |
| cd3838adc93c | citeseer | dropedge |  | 10 | 7 |
| 1ef85eeba0c0 | citeseer | dropedge |  | 10 | 8 |
| 3d2087c5282c | citeseer | dropedge |  | 10 | 9 |
| 0f582399d661 | citeseer | dropedge |  | 20 | 0 |
| 4e9f82c43459 | citeseer | dropedge |  | 20 | 1 |
| a49479127e35 | citeseer | dropedge |  | 20 | 2 |
| e77ec0fadf55 | citeseer | dropedge |  | 20 | 3 |
| dd0aa10c3eb8 | citeseer | dropedge |  | 20 | 4 |
| 74d5a6556647 | citeseer | dropedge |  | 20 | 5 |
| 1f967b3a4dff | citeseer | dropedge |  | 20 | 6 |
| bcb61d2cf9cb | citeseer | dropedge |  | 20 | 7 |
| e03907ea3c15 | citeseer | dropedge |  | 30 | 0 |
| 16e03d91391b | citeseer | dropedge |  | 30 | 1 |
| 6273f4da13d7 | citeseer | dropedge |  | 30 | 2 |
| e810497c61d4 | citeseer | dropedge |  | 30 | 3 |
| b248a69a0cec | citeseer | dropedge |  | 30 | 4 |
| e9e84bf056c4 | citeseer | dropedge |  | 30 | 5 |
| 4a137fbfed59 | citeseer | graphunet |  | 2 | 0 |
| ef658839c25a | citeseer | graphunet |  | 2 | 1 |
| bb1704e2994b | citeseer | graphunet |  | 2 | 2 |
| 771f86952397 | citeseer | graphunet |  | 2 | 3 |
| 5d0e3847d539 | citeseer | graphunet |  | 2 | 4 |
| e60b59ff11f2 | citeseer | graphunet |  | 2 | 5 |
| 75e1340b330d | citeseer | graphunet |  | 2 | 6 |
| 22db9623b102 | citeseer | graphunet |  | 2 | 7 |
| acc02fc89c33 | citeseer | graphunet |  | 2 | 8 |
| a7cc795b2697 | citeseer | graphunet |  | 2 | 9 |
| bbb1e1f8444e | citeseer | graphunet |  | 4 | 0 |
| 07e90862aa18 | citeseer | graphunet |  | 4 | 1 |
| 609fd1d7114d | citeseer | graphunet |  | 4 | 2 |
| 0ff79a674854 | citeseer | graphunet |  | 4 | 3 |
| 8edbeeb04ee1 | citeseer | graphunet |  | 4 | 4 |
| 9353d5873da0 | citeseer | graphunet |  | 4 | 5 |
| 9a8a7b69a6aa | citeseer | graphunet |  | 4 | 6 |
| 159a21a43447 | citeseer | graphunet |  | 4 | 7 |
| 95b40c39be70 | citeseer | graphunet |  | 4 | 8 |
| e0bbf75a9b21 | citeseer | graphunet |  | 4 | 9 |
| 2f25183bbcfa | citeseer | graphunet |  | 8 | 0 |
| 326b80aa46d9 | citeseer | graphunet |  | 8 | 1 |
| 60cff601e6e1 | citeseer | graphunet |  | 8 | 2 |
| 62d30c2a6c5c | citeseer | graphunet |  | 8 | 3 |
| e6ecfe7e0c79 | citeseer | graphunet |  | 8 | 4 |
| 232ab6e4c993 | citeseer | graphunet |  | 8 | 5 |
| 2a48e4f40698 | citeseer | graphunet |  | 8 | 6 |
| 3291c0f8a725 | citeseer | graphunet |  | 8 | 7 |
| e4096d22035f | citeseer | graphunet |  | 10 | 2 |
| be66d2c151a5 | citeseer | graphunet |  | 10 | 3 |
| 41ecf46c2a79 | citeseer | graphunet |  | 10 | 4 |
| f41ba08e4573 | citeseer | graphunet |  | 10 | 5 |
| 46fcb06e1166 | citeseer | graphunet |  | 10 | 6 |
| ac4fc0c52039 | citeseer | graphunet |  | 10 | 7 |
| 51ff88e55311 | citeseer | graphunet |  | 10 | 8 |
| 6fbbb768aec0 | citeseer | graphunet |  | 10 | 9 |
| 67b9b23cb63b | citeseer | graphunet |  | 20 | 0 |
| e6d93047929c | citeseer | graphunet |  | 20 | 1 |
| aeecb8ad4632 | citeseer | graphunet |  | 20 | 2 |
| e8d85074c56a | citeseer | graphunet |  | 20 | 3 |
| e713695ca28a | citeseer | graphunet |  | 20 | 4 |
| 512cca38443e | citeseer | graphunet |  | 20 | 5 |
| dbb23619e970 | citeseer | graphunet |  | 20 | 6 |
| 2a5e1a954d64 | citeseer | graphunet |  | 20 | 7 |
| bafb17ddaee1 | citeseer | graphunet |  | 20 | 8 |
| 9a70ea65fa9b | citeseer | graphunet |  | 20 | 9 |
| 45695928c1e6 | citeseer | graphunet |  | 30 | 0 |
| 3595215385eb | citeseer | graphunet |  | 30 | 1 |
| 361b832584ab | citeseer | graphunet |  | 30 | 2 |
| ff651d8f5059 | citeseer | graphunet |  | 30 | 3 |
| a732a61cd23a | citeseer | graphunet |  | 30 | 4 |
| 0fd26434f07d | citeseer | graphunet |  | 30 | 5 |
| 9ebce3ac91d3 | citeseer | graphunet |  | 30 | 6 |
| 95070903a3a7 | citeseer | graphunet |  | 30 | 7 |
| 2fb67fcd2af7 | citeseer | graphunet |  | 30 | 8 |
| 3ec6857c1b3d | citeseer | graphunet |  | 30 | 9 |
| 41579d462866 | pubmed | nhdp |  | 2 | 0 |
| 6e1dbbd2b1f5 | pubmed | nhdp |  | 2 | 2 |
| 06ab561f961e | pubmed | nhdp |  | 2 | 3 |
| c3cbbc9b879c | pubmed | nhdp |  | 2 | 5 |
| cc6d7358da09 | pubmed | nhdp |  | 2 | 7 |
| 9514d0c0d5e9 | pubmed | nhdp |  | 2 | 8 |
| 842eedef0a7a | pubmed | nhdp |  | 2 | 9 |
| d54a12c06c1d | pubmed | nhdp |  | 4 | 0 |
| 394b607a57f5 | pubmed | nhdp |  | 4 | 1 |
| 334214b6b6d4 | pubmed | nhdp |  | 4 | 3 |
| 1a23c395ed3a | pubmed | nhdp |  | 4 | 4 |
| 58b6608972f9 | pubmed | nhdp |  | 4 | 6 |
| e77589861116 | pubmed | nhdp |  | 8 | 0 |
| 15a70d1f5385 | pubmed | nhdp |  | 8 | 2 |
| 566f85585adc | pubmed | nhdp |  | 8 | 3 |
| 187ce0879555 | pubmed | nhdp |  | 8 | 4 |
| 221fc3101351 | pubmed | nhdp |  | 8 | 5 |
| 752d200bf081 | pubmed | nhdp |  | 8 | 6 |
| 83d0db8e1c85 | pubmed | nhdp |  | 8 | 8 |
| f706defc78f3 | pubmed | nhdp |  | 8 | 9 |
| 3c8886d8f58d | pubmed | nhdp |  | 10 | 1 |
| f8634c1b6505 | pubmed | nhdp |  | 10 | 2 |
| 978d6c93734b | pubmed | nhdp |  | 10 | 4 |
| 907f4d91a79b | pubmed | nhdp |  | 10 | 5 |
| 0e5a32112df6 | pubmed | nhdp |  | 10 | 7 |
| 27a09f0c3b35 | pubmed | nhdp |  | 20 | 3 |
| 59f0820121fd | pubmed | nhdp |  | 20 | 4 |
| 591621224edc | pubmed | nhdp |  | 20 | 5 |
| 0b4504a5e775 | pubmed | nhdp |  | 20 | 6 |
| d8d622383544 | pubmed | nhdp |  | 20 | 7 |
| ba70eb4a812d | pubmed | nhdp |  | 20 | 8 |
| 320df835429d | pubmed | nhdp |  | 20 | 9 |
| 9aff7eb48033 | pubmed | nhdp |  | 30 | 0 |
| 2ef0283ff91d | pubmed | nhdp |  | 30 | 1 |
| c10c617d868b | pubmed | nhdp |  | 30 | 2 |
| 817c249becd5 | pubmed | nhdp |  | 30 | 3 |
| 966213c5e907 | pubmed | nhdp |  | 30 | 4 |
| b638d091bb79 | pubmed | nhdp |  | 30 | 5 |
| 4d533d5c39d1 | pubmed | nhdp |  | 30 | 6 |
| ae0d7f4230d2 | pubmed | nhdp |  | 30 | 7 |
| 900facdd5583 | pubmed | nhdp |  | 30 | 8 |
| be58da0b93fc | pubmed | nhdp |  | 30 | 9 |
| 283a6b8b796b | pubmed | gcn |  | 2 | 0 |
| df48fa8386dc | pubmed | gcn |  | 2 | 1 |
| 6e3ab579f5a2 | pubmed | gcn |  | 2 | 2 |
| 8b68b98f6412 | pubmed | gcn |  | 2 | 3 |
| 25f7ed773488 | pubmed | gcn |  | 2 | 4 |
| caa6ee37374a | pubmed | gcn |  | 2 | 5 |
| 2497ef965a79 | pubmed | gcn |  | 2 | 6 |
| 0e83f19be73b | pubmed | gcn |  | 2 | 7 |
| abf50ee1d2dd | pubmed | gcn |  | 2 | 8 |
| d138a4a75b3a | pubmed | gcn |  | 2 | 9 |
| cfb1da58725a | pubmed | gcn |  | 4 | 0 |
| 5f05bd01f650 | pubmed | gcn |  | 4 | 1 |
| ef744dccdd39 | pubmed | gcn |  | 4 | 2 |
| 7fe3d2f19aab | pubmed | gcn |  | 4 | 3 |
| 86202cc0f937 | pubmed | gcn |  | 4 | 4 |
| 9ccffff3617d | pubmed | gcn |  | 4 | 5 |
| 6cb4ef04af3f | pubmed | gcn |  | 4 | 6 |
| 9da587863967 | pubmed | gcn |  | 4 | 7 |
| eb9ba0ef38cc | pubmed | gcn |  | 4 | 8 |
| db21822a3263 | pubmed | gcn |  | 4 | 9 |
| b28a17ef2ca8 | pubmed | gcn |  | 8 | 0 |
| 5d6f486d09e4 | pubmed | gcn |  | 8 | 1 |
| ec2fd4f10247 | pubmed | gcn |  | 8 | 2 |
| cca173013171 | pubmed | gcn |  | 8 | 3 |
| 6ac70b881686 | pubmed | gcn |  | 8 | 4 |
| c51562678a41 | pubmed | gcn |  | 8 | 5 |
| e90728293b1e | pubmed | gcn |  | 8 | 6 |
| 81728fd9d841 | pubmed | gcn |  | 8 | 7 |
| fbc2c320c6b0 | pubmed | gcn |  | 8 | 8 |
| 1bd2926f4173 | pubmed | gcn |  | 8 | 9 |
| fa048ea6c614 | pubmed | gcn |  | 10 | 0 |
| e5d8b24b6088 | pubmed | gcn |  | 10 | 1 |
| 612953d2eb16 | pubmed | gcn |  | 10 | 2 |
| 019f97b8afe4 | pubmed | gcn |  | 10 | 3 |
| 1a2d2dcf88be | pubmed | gcn |  | 10 | 4 |
| bccd7d3235f6 | pubmed | gcn |  | 10 | 5 |
| 5c12ec1e4188 | pubmed | gcn |  | 10 | 6 |
| 4656c563c072 | pubmed | gcn |  | 10 | 7 |
| 5bb441fe9947 | pubmed | gcn |  | 10 | 8 |
| 6cdb37c7bbb4 | pubmed | gcn |  | 10 | 9 |
| 5a641ed3542b | pubmed | gcn |  | 20 | 0 |
| dc73f2a7a041 | pubmed | gcn |  | 20 | 1 |
| aaa0d9de4b62 | pubmed | gcn |  | 20 | 2 |
| 4aa508e4c0cb | pubmed | gcn |  | 20 | 3 |
| 35ed0bc86e1d | pubmed | gcn |  | 20 | 4 |
| 65622e9642e6 | pubmed | gcn |  | 20 | 5 |
| 49a6fbc6f530 | pubmed | gcn |  | 20 | 6 |
| 8fe591cb5f4c | pubmed | gcn |  | 20 | 7 |
| 348b14a28bbe | pubmed | gcn |  | 20 | 8 |
| 75e19169bcc6 | pubmed | gcn |  | 20 | 9 |
| 5d0d49cb237a | pubmed | gcn |  | 30 | 0 |
| 59ef6078dc2e | pubmed | gcn |  | 30 | 1 |
| e86efcfda1d3 | pubmed | gcn |  | 30 | 2 |
| 0e187e530396 | pubmed | gcn |  | 30 | 3 |
| d558275db743 | pubmed | gcn |  | 30 | 4 |
| 0c6406f919ee | pubmed | gcn |  | 30 | 5 |
| 6dfc957917b2 | pubmed | gcn |  | 30 | 6 |
| 88dedea79f0f | pubmed | gcn |  | 30 | 7 |
| e3cab8490c99 | pubmed | gcn |  | 30 | 8 |
| e5a1975dee2e | pubmed | gcn |  | 30 | 9 |
| 47d14d627bb4 | pubmed | gat |  | 2 | 0 |
| 64f461d9a1a7 | pubmed | gat |  | 2 | 1 |
| 75394a05da1a | pubmed | gat |  | 2 | 2 |
| f9a7b35d6068 | pubmed | gat |  | 2 | 3 |
| fe32efd92629 | pubmed | gat |  | 2 | 4 |
| 3647f33d862d | pubmed | gat |  | 2 | 5 |
| ad2ef86fecf5 | pubmed | gat |  | 2 | 6 |
| 67a3189ba35d | pubmed | gat |  | 2 | 7 |
| 21a0e927f7ae | pubmed | gat |  | 2 | 8 |
| 974f3d6341e6 | pubmed | gat |  | 2 | 9 |
| e3b375ba1799 | pubmed | gat |  | 4 | 0 |
| 404da6ced935 | pubmed | gat |  | 4 | 1 |
| ae290ec32d54 | pubmed | gat |  | 4 | 2 |
| d01bcc478763 | pubmed | gat |  | 4 | 3 |
| 0c045933eb71 | pubmed | gat |  | 4 | 4 |
| b2a038ce937b | pubmed | gat |  | 4 | 5 |
| 3290c2eb7c0d | pubmed | gat |  | 4 | 6 |
| c60879d93f3d | pubmed | gat |  | 4 | 7 |
| 9b5811b6718e | pubmed | gat |  | 4 | 8 |
| fcce2db5e0bf | pubmed | gat |  | 4 | 9 |
| dc80d416dd77 | pubmed | gat |  | 8 | 0 |
| f6de858bd81b | pubmed | gat |  | 8 | 1 |
| c9b5842d4bcd | pubmed | gat |  | 8 | 2 |
| 3435b748b8a3 | pubmed | gat |  | 8 | 3 |
| 777cbceb6a87 | pubmed | gat |  | 8 | 4 |
| 24573786550a | pubmed | gat |  | 8 | 5 |
| b00c3e6a0cea | pubmed | gat |  | 8 | 6 |
| 5958c4c52689 | pubmed | gat |  | 8 | 7 |
| 4fb60d708d32 | pubmed | gat |  | 8 | 8 |
| 62b81884fe4a | pubmed | gat |  | 8 | 9 |
| 9c91c832cad5 | pubmed | gat |  | 10 | 0 |
| d648e3ce393b | pubmed | gat |  | 10 | 2 |
| c7e514ad9d87 | pubmed | gat |  | 10 | 3 |
| b53773fb80da | pubmed | gat |  | 10 | 5 |
| f156698c4302 | pubmed | gat |  | 10 | 7 |
| a7dc2d98c16e | pubmed | gat |  | 10 | 8 |
| 42d8c825225d | pubmed | gat |  | 10 | 9 |
| 8721f7e8c4f1 | pubmed | gat |  | 20 | 0 |
| 07844cb7e324 | pubmed | gat |  | 20 | 1 |
| aeb1bd0084b6 | pubmed | gat |  | 20 | 3 |
| ebd2dda99736 | pubmed | gat |  | 20 | 4 |
| ac28d8dbcc68 | pubmed | gat |  | 20 | 6 |
| 6a12ae0d9309 | pubmed | gat |  | 30 | 0 |
| 2bac255fcf5c | pubmed | gat |  | 30 | 2 |
| 80bb0d24fc57 | pubmed | gat |  | 30 | 3 |
| 7132594f5db9 | pubmed | gat |  | 30 | 4 |
| f510e4936be2 | pubmed | gat |  | 30 | 5 |
| bfb2b0926448 | pubmed | gat |  | 30 | 6 |
| 8d17a72b663a | pubmed | gat |  | 30 | 8 |
| 61bf0932a986 | pubmed | gat |  | 30 | 9 |
| 992621ba41d9 | pubmed | sgc |  | 2 | 1 |
| 3ac048abf544 | pubmed | sgc |  | 2 | 2 |
| fa1f9457a026 | pubmed | sgc |  | 2 | 4 |
| f667fb0fde7b | pubmed | sgc |  | 2 | 5 |
| b0808dfefd60 | pubmed | sgc |  | 2 | 7 |
| 721e2d5fc4de | pubmed | sgc |  | 4 | 3 |
| 52f6f87611c8 | pubmed | sgc |  | 4 | 4 |
| efea92771638 | pubmed | sgc |  | 4 | 5 |
| dad251073a56 | pubmed | sgc |  | 4 | 6 |
| 39d017c56f4d | pubmed | sgc |  | 4 | 7 |
| a00acc805ad8 | pubmed | sgc |  | 4 | 8 |
| 9bdad31015c2 | pubmed | sgc |  | 4 | 9 |
| 8d6ffa3594be | pubmed | sgc |  | 8 | 0 |
| 3d6e2a058801 | pubmed | sgc |  | 8 | 1 |
| 24393b5a5487 | pubmed | sgc |  | 8 | 2 |
| 2209754173fe | pubmed | sgc |  | 8 | 3 |
| 96090a312896 | pubmed | sgc |  | 8 | 4 |
| 7310fd34f465 | pubmed | sgc |  | 8 | 5 |
| b9a3967586bc | pubmed | sgc |  | 8 | 6 |
| 468363362f9e | pubmed | sgc |  | 8 | 7 |
| 9b561815b810 | pubmed | sgc |  | 8 | 8 |
| 92c7b29dc4e4 | pubmed | sgc |  | 8 | 9 |
| 56553d059d00 | pubmed | sgc |  | 10 | 0 |
| 1914b1cb3768 | pubmed | sgc |  | 10 | 1 |
| 3f983e0fc695 | pubmed | sgc |  | 10 | 2 |
| 65e426bdc1b4 | pubmed | sgc |  | 10 | 3 |
| 9a1b9ab66144 | pubmed | sgc |  | 10 | 4 |
| a4a896a080f7 | pubmed | sgc |  | 10 | 5 |
| 435908469273 | pubmed | sgc |  | 10 | 6 |
| af9a2be04d40 | pubmed | sgc |  | 10 | 7 |
| e85de808b2d5 | pubmed | sgc |  | 10 | 8 |
| f0e82f54333b | pubmed | sgc |  | 10 | 9 |
| c5dc22ad318e | pubmed | sgc |  | 20 | 0 |
| 45ccde2f39e5 | pubmed | sgc |  | 20 | 1 |
| af6b133976b7 | pubmed | sgc |  | 20 | 2 |
| dbb4f2d7867e | pubmed | sgc |  | 20 | 3 |
| 06a85f847d9a | pubmed | sgc |  | 20 | 4 |
| 3723be09f4c6 | pubmed | sgc |  | 20 | 5 |
| 235863815403 | pubmed | sgc |  | 20 | 6 |
| 526a2667766c | pubmed | sgc |  | 20 | 7 |
| 1b9ed4dbfc0c | pubmed | sgc |  | 20 | 8 |
| 894830d95a35 | pubmed | sgc |  | 20 | 9 |
| 5195839b6f6e | pubmed | sgc |  | 30 | 0 |
| c781c978c3e5 | pubmed | sgc |  | 30 | 1 |
| 0130c159642e | pubmed | sgc |  | 30 | 2 |
| 41bf863a6108 | pubmed | sgc |  | 30 | 3 |
| c359e73c01e3 | pubmed | sgc |  | 30 | 4 |
| 81ae3f353f7d | pubmed | sgc |  | 30 | 5 |
| 8ccf338a29fb | pubmed | sgc |  | 30 | 6 |
| b451913becc7 | pubmed | sgc |  | 30 | 7 |
| 1f72bd776efe | pubmed | sgc |  | 30 | 8 |
| d491e55c6bd3 | pubmed | sgc |  | 30 | 9 |
| 0bb3063ddcea | pubmed | appnp |  | 2 | 4 |
| 96bf6f130b11 | pubmed | appnp |  | 2 | 5 |
| e088db814b87 | pubmed | appnp |  | 2 | 6 |
| 1e4c4a1fe40b | pubmed | appnp |  | 2 | 7 |
| f536e874132c | pubmed | appnp |  | 20 | 0 |
| 051014067f62 | pubmed | appnp |  | 20 | 1 |
| 6064d7d66874 | pubmed | appnp |  | 20 | 2 |
| b53242f20757 | pubmed | appnp |  | 20 | 3 |
| 02998de16313 | pubmed | appnp |  | 20 | 4 |
| 624634973902 | pubmed | appnp |  | 20 | 5 |
| 315b19229b61 | pubmed | appnp |  | 20 | 6 |
| 004108885d6a | pubmed | appnp |  | 20 | 7 |
| c9f8a6c734b6 | pubmed | appnp |  | 20 | 8 |
| b7346591c428 | pubmed | appnp |  | 20 | 9 |
| d4cbd923081f | pubmed | appnp |  | 30 | 0 |
| a9b4ab67f5f3 | pubmed | appnp |  | 30 | 1 |
| ed33ec61ccb4 | pubmed | appnp |  | 30 | 6 |
| 914035b68fc7 | pubmed | appnp |  | 30 | 7 |
| 09cd468f7adc | pubmed | appnp |  | 30 | 8 |
| 7f6e9d0f53a4 | pubmed | appnp |  | 30 | 9 |
| 0b4b66b80ffb | pubmed | arma |  | 2 | 0 |
| 9cf39874673b | pubmed | arma |  | 2 | 2 |
| a0444c780f34 | pubmed | arma |  | 2 | 3 |
| 7f577d4bd5d9 | pubmed | arma |  | 2 | 5 |
| 56bb56148ed5 | pubmed | arma |  | 2 | 7 |
| 75a8f9f64f54 | pubmed | arma |  | 4 | 3 |
| 5f88bd3f8557 | pubmed | arma |  | 4 | 4 |
| b4e5df1ece04 | pubmed | arma |  | 4 | 6 |
| 047b26e30ad8 | pubmed | arma |  | 8 | 0 |
| 1fe637491d6c | pubmed | arma |  | 8 | 2 |
| 325358c59a2b | pubmed | arma |  | 8 | 3 |
| b5d2df785152 | pubmed | arma |  | 8 | 4 |
| 1e83204a74dc | pubmed | arma |  | 8 | 5 |
| e1e7fecbf598 | pubmed | arma |  | 8 | 6 |
| 9d2824236f19 | pubmed | arma |  | 8 | 8 |
| df796791d7f5 | pubmed | arma |  | 8 | 9 |
| d690ae091766 | pubmed | arma |  | 10 | 1 |
| d4b7c0be6ab0 | pubmed | arma |  | 10 | 2 |
| 82b425a664b6 | pubmed | arma |  | 10 | 4 |
| 31d308c70f09 | pubmed | arma |  | 10 | 5 |
| 5bb6d20a2421 | pubmed | arma |  | 10 | 7 |
| 18e50b4e2a86 | pubmed | arma |  | 20 | 3 |
| 9e9ff95e8801 | pubmed | arma |  | 20 | 4 |
| 880deec08c23 | pubmed | arma |  | 20 | 5 |
| 6b7a3864bc50 | pubmed | arma |  | 20 | 6 |
| 185c504dfb85 | pubmed | arma |  | 20 | 7 |
| 635119821c9e | pubmed | arma |  | 20 | 8 |
| a54797220108 | pubmed | arma |  | 20 | 9 |
| 7eb74ad473ab | pubmed | arma |  | 30 | 0 |
| 41e8ae5392a6 | pubmed | arma |  | 30 | 1 |
| 413b63310819 | pubmed | arma |  | 30 | 2 |
| 6de9f64566f0 | pubmed | arma |  | 30 | 3 |
| 66df19b16cf1 | pubmed | arma |  | 30 | 4 |
| 559284119b00 | pubmed | arma |  | 30 | 5 |
| 921634755aeb | pubmed | arma |  | 30 | 6 |
| 3cc6da568483 | pubmed | arma |  | 30 | 7 |
| 2fa5db2ad729 | pubmed | arma |  | 30 | 8 |
| ae3bf332bee0 | pubmed | arma |  | 30 | 9 |
| 51857a2034d3 | pubmed | gcnii |  | 2 | 0 |
| 6f8fafb87ba8 | pubmed | gcnii |  | 2 | 1 |
| 6861f8afc94d | pubmed | gcnii |  | 2 | 2 |
| 62728711be8d | pubmed | gcnii |  | 2 | 3 |
| a02d96f2b1a8 | pubmed | gcnii |  | 2 | 4 |
| 2c78ce3a72d8 | pubmed | gcnii |  | 2 | 5 |
| 34fa406b92b6 | pubmed | gcnii |  | 2 | 6 |
| 1e4c11659339 | pubmed | gcnii |  | 2 | 7 |
| 6165b695ef62 | pubmed | gcnii |  | 2 | 8 |
| fc2c7a667ffa | pubmed | gcnii |  | 2 | 9 |
| 850267d31b7f | pubmed | gcnii |  | 4 | 0 |
| e4bacf73a21e | pubmed | gcnii |  | 4 | 1 |
| 3b7dbfd4e765 | pubmed | gcnii |  | 4 | 2 |
| 897b43f68e2b | pubmed | gcnii |  | 4 | 3 |
| 8edcbeddc405 | pubmed | gcnii |  | 4 | 4 |
| 752fe5e8411c | pubmed | gcnii |  | 4 | 5 |
| 821b7c438e9b | pubmed | gcnii |  | 8 | 4 |
| f1658698eb31 | pubmed | gcnii |  | 8 | 5 |
| 6f1809bc882e | pubmed | gcnii |  | 8 | 6 |
| dd998d8dfed7 | pubmed | gcnii |  | 8 | 7 |
| 55f5b5cdbf93 | pubmed | gcnii |  | 8 | 8 |
| cf693bc84bde | pubmed | gcnii |  | 8 | 9 |
| 7f2f6e81b839 | pubmed | gcnii |  | 10 | 0 |
| bb5e9e97e18d | pubmed | gcnii |  | 10 | 1 |
| 3215809154d7 | pubmed | gcnii |  | 10 | 2 |
| cd2cd290f0d7 | pubmed | gcnii |  | 10 | 3 |
| 33dd195f6389 | pubmed | gcnii |  | 10 | 4 |
| 74dd347ce24e | pubmed | gcnii |  | 10 | 5 |
| 2dee531e4831 | pubmed | gcnii |  | 10 | 6 |
| 6060d86ab84f | pubmed | gcnii |  | 10 | 7 |
| 4ab5ac4008ce | pubmed | gcnii |  | 10 | 8 |
| 6317efc372ae | pubmed | gcnii |  | 10 | 9 |
| e06589f5c407 | pubmed | gcnii |  | 20 | 0 |
| 1d645e519d4b | pubmed | gcnii |  | 20 | 1 |
| 82980caf09e4 | pubmed | gcnii |  | 20 | 2 |
| 9098928c86b4 | pubmed | gcnii |  | 20 | 3 |
| 412405391ce4 | pubmed | gcnii |  | 20 | 4 |
| 331443ae68c3 | pubmed | gcnii |  | 20 | 5 |
| 8075dee5c8db | pubmed | gcnii |  | 20 | 6 |
| 0699ddab6855 | pubmed | gcnii |  | 20 | 7 |
| 43aa68f25f53 | pubmed | gcnii |  | 20 | 8 |
| 01d18159a46b | pubmed | gcnii |  | 20 | 9 |
| 1ac3d3542e24 | pubmed | gcnii |  | 30 | 0 |
| efd97a1ea71b | pubmed | gcnii |  | 30 | 1 |
| 7a4d4e1166c3 | pubmed | gcnii |  | 30 | 2 |
| 9404073a1c51 | pubmed | gcnii |  | 30 | 3 |
| 84c3100fb197 | pubmed | gcnii |  | 30 | 4 |
| b0080a6aa409 | pubmed | gcnii |  | 30 | 5 |
| d4862c13c5db | pubmed | pairnorm |  | 2 | 0 |
| d3415a247047 | pubmed | pairnorm |  | 2 | 1 |
| c9f5a62221c6 | pubmed | pairnorm |  | 2 | 2 |
| 3ca9407b2ca0 | pubmed | pairnorm |  | 2 | 3 |
| 600a99f436f1 | pubmed | pairnorm |  | 2 | 4 |
| c5f93282c62c | pubmed | pairnorm |  | 2 | 5 |
| 06706db8b65c | pubmed | pairnorm |  | 2 | 6 |
| 067e19bf2b37 | pubmed | pairnorm |  | 2 | 7 |
| 4f0fc5fbc11d | pubmed | pairnorm |  | 2 | 8 |
| 9d73e0e9d718 | pubmed | pairnorm |  | 2 | 9 |
| 4756b350b271 | pubmed | pairnorm |  | 4 | 0 |
| a802ab35e252 | pubmed | pairnorm |  | 4 | 1 |
| 105bff41e6a6 | pubmed | pairnorm |  | 4 | 2 |
| 293623293177 | pubmed | pairnorm |  | 4 | 3 |
| e4e4dc36a4e9 | pubmed | pairnorm |  | 4 | 4 |
| 0941dab7d11f | pubmed | pairnorm |  | 4 | 5 |
| 06ffae6fce4f | pubmed | pairnorm |  | 4 | 6 |
| 049020ea7132 | pubmed | pairnorm |  | 4 | 7 |
| e502d6abbc7d | pubmed | pairnorm |  | 4 | 8 |
| 6aec96dc8534 | pubmed | pairnorm |  | 4 | 9 |
| 61a5ec2b5e57 | pubmed | pairnorm |  | 8 | 0 |
| 3f36efbd5874 | pubmed | pairnorm |  | 8 | 1 |
| a404af58e77d | pubmed | pairnorm |  | 8 | 2 |
| b3abbd7f8207 | pubmed | pairnorm |  | 8 | 3 |
| b5fd01ed0a22 | pubmed | pairnorm |  | 8 | 4 |
| eb9dfacd937d | pubmed | pairnorm |  | 8 | 5 |
| 9fbeac485bfd | pubmed | pairnorm |  | 8 | 6 |
| 5127e3fe6aa1 | pubmed | pairnorm |  | 8 | 7 |
| a906c726bcd8 | pubmed | pairnorm |  | 10 | 2 |
| 03af8ccfddf6 | pubmed | pairnorm |  | 10 | 3 |
| 2bb7c34efb3d | pubmed | pairnorm |  | 10 | 5 |
| 3301dd9bdfa8 | pubmed | pairnorm |  | 10 | 7 |
| ba4f26d25a6d | pubmed | pairnorm |  | 10 | 8 |
| 10a2ea3fc82b | pubmed | pairnorm |  | 10 | 9 |
| 90aa06b33dd9 | pubmed | pairnorm |  | 20 | 0 |
| 9ba92ebe2b5a | pubmed | pairnorm |  | 20 | 1 |
| 66742bf055a1 | pubmed | pairnorm |  | 20 | 3 |
| deae341f1d1e | pubmed | pairnorm |  | 20 | 4 |
| 9e4cbf21ad5d | pubmed | pairnorm |  | 20 | 6 |
| c991e876e1bb | pubmed | pairnorm |  | 30 | 0 |
| 5ff1f455d051 | pubmed | pairnorm |  | 30 | 2 |
| f55396b56475 | pubmed | pairnorm |  | 30 | 3 |
| 81c75e171b95 | pubmed | pairnorm |  | 30 | 4 |
| c35345bfa798 | pubmed | pairnorm |  | 30 | 5 |
| 66ac9936d07b | pubmed | pairnorm |  | 30 | 6 |
| 8772835071e6 | pubmed | pairnorm |  | 30 | 8 |
| 19e336ab0216 | pubmed | pairnorm |  | 30 | 9 |
| cfdce60cb83d | pubmed | dropedge |  | 2 | 1 |
| f3530999ea03 | pubmed | dropedge |  | 2 | 2 |
| 0847ee544a98 | pubmed | dropedge |  | 2 | 4 |
| 8cff92d47740 | pubmed | dropedge |  | 2 | 5 |
| 9a92c1f1d721 | pubmed | dropedge |  | 2 | 7 |
| 179c3f71083a | pubmed | dropedge |  | 4 | 3 |
| 08d4c39a8b36 | pubmed | dropedge |  | 4 | 4 |
| 5203764f1788 | pubmed | dropedge |  | 4 | 5 |
| 6501822a24a7 | pubmed | dropedge |  | 4 | 6 |
| 9ea385d9a553 | pubmed | dropedge |  | 4 | 7 |
| 4d561756c0ef | pubmed | dropedge |  | 4 | 8 |
| 7cf6c095eafc | pubmed | dropedge |  | 4 | 9 |
| b3b2a576156c | pubmed | dropedge |  | 8 | 0 |
| ec6eb0708c30 | pubmed | dropedge |  | 8 | 1 |
| c64d05f6143e | pubmed | dropedge |  | 8 | 2 |
| ef90f6bcef3a | pubmed | dropedge |  | 8 | 3 |
| 41f87dfb227e | pubmed | dropedge |  | 8 | 4 |
| c6693a8c40fb | pubmed | dropedge |  | 8 | 5 |
| a699b70fd219 | pubmed | dropedge |  | 8 | 6 |
| 5805d1b3f9a0 | pubmed | dropedge |  | 8 | 7 |
| a59c4a2ad97f | pubmed | dropedge |  | 8 | 8 |
| 4fde7637796b | pubmed | dropedge |  | 8 | 9 |
| 1528ad0bb4a2 | pubmed | dropedge |  | 10 | 0 |
| 789886093304 | pubmed | dropedge |  | 10 | 1 |
| 4cd728f34d32 | pubmed | dropedge |  | 10 | 2 |
| 7853420f8530 | pubmed | dropedge |  | 10 | 3 |
| eadcc31d8fac | pubmed | dropedge |  | 10 | 4 |
| c578e948cea2 | pubmed | dropedge |  | 10 | 5 |
| 2def8c641e07 | pubmed | dropedge |  | 10 | 6 |
| 13b212794e66 | pubmed | dropedge |  | 10 | 7 |
| 08def51a7208 | pubmed | dropedge |  | 10 | 8 |
| dcf592decdd4 | pubmed | dropedge |  | 10 | 9 |
| 47bd3da94887 | pubmed | dropedge |  | 20 | 0 |
| 1c59dce77abe | pubmed | dropedge |  | 20 | 1 |
| 1ff218b79bcd | pubmed | dropedge |  | 20 | 2 |
| 7a3382a1b3cc | pubmed | dropedge |  | 20 | 3 |
| ae58c5450074 | pubmed | dropedge |  | 20 | 4 |
| 05ce297c2314 | pubmed | dropedge |  | 20 | 5 |
| f7d7584c8f18 | pubmed | dropedge |  | 20 | 6 |
| 223d79fe1a81 | pubmed | dropedge |  | 20 | 7 |
| 6ee21bc011f6 | pubmed | dropedge |  | 20 | 8 |
| a9aaf67dfe84 | pubmed | dropedge |  | 20 | 9 |
| ea61716fcd95 | pubmed | dropedge |  | 30 | 0 |
| 8727717b0127 | pubmed | dropedge |  | 30 | 1 |
| 76452aa76239 | pubmed | dropedge |  | 30 | 2 |
| 3cb117cc458a | pubmed | dropedge |  | 30 | 3 |
| 760cd59eb345 | pubmed | dropedge |  | 30 | 4 |
| a41cdbe45a99 | pubmed | dropedge |  | 30 | 5 |
| 45fc75be8c15 | pubmed | dropedge |  | 30 | 6 |
| 4b55201cfea4 | pubmed | dropedge |  | 30 | 7 |
| 71f71608f64d | pubmed | dropedge |  | 30 | 8 |
| 5cb8ade48296 | pubmed | dropedge |  | 30 | 9 |
| 25f1e7a9316f | pubmed | graphunet |  | 2 | 0 |
| ae0305265227 | pubmed | graphunet |  | 2 | 1 |
| 8a1d92ef09e2 | pubmed | graphunet |  | 2 | 2 |
| 8b25b27b3bae | pubmed | graphunet |  | 2 | 3 |
| 7e7451546b40 | pubmed | graphunet |  | 2 | 4 |
| b4e6c107a4b9 | pubmed | graphunet |  | 2 | 5 |
| 7f65fa257c4d | pubmed | graphunet |  | 2 | 6 |
| 73f27f7b406a | pubmed | graphunet |  | 2 | 7 |
| 2ae719899a73 | pubmed | graphunet |  | 2 | 8 |
| 8deb1f7b0692 | pubmed | graphunet |  | 2 | 9 |
| dfe00f2bf837 | pubmed | graphunet |  | 4 | 0 |
| 3d22d8108331 | pubmed | graphunet |  | 4 | 1 |
| 58f87af038ab | pubmed | graphunet |  | 4 | 2 |
| 85efface4871 | pubmed | graphunet |  | 4 | 3 |
| 6970c620f8c8 | pubmed | graphunet |  | 4 | 4 |
| 75ba402004a3 | pubmed | graphunet |  | 4 | 5 |
| 503cda045500 | pubmed | graphunet |  | 4 | 6 |
| 3ddf184a0076 | pubmed | graphunet |  | 4 | 7 |
| 6f391e298a37 | pubmed | graphunet |  | 4 | 8 |
| 6259e42615b2 | pubmed | graphunet |  | 4 | 9 |
| 9e1e993e8fbb | pubmed | graphunet |  | 8 | 0 |
| 9b821711c7e6 | pubmed | graphunet |  | 8 | 1 |
| 87594ab7cf4e | pubmed | graphunet |  | 8 | 2 |
| 6f20b76bf6b3 | pubmed | graphunet |  | 8 | 3 |
| f4a3f410673b | pubmed | graphunet |  | 8 | 4 |
| c3ef9c46b0d6 | pubmed | graphunet |  | 8 | 5 |
| 0fae6b26da90 | pubmed | graphunet |  | 8 | 6 |
| cf0097633e06 | pubmed | graphunet |  | 8 | 7 |
| feaaec25dcca | pubmed | graphunet |  | 8 | 8 |
| ec57510bff4b | pubmed | graphunet |  | 8 | 9 |
| bfeaf0365c8c | pubmed | graphunet |  | 10 | 0 |
| 8286999d6b90 | pubmed | graphunet |  | 10 | 1 |
| ceac2cd5bcc5 | pubmed | graphunet |  | 10 | 2 |
| c1d0d0f19171 | pubmed | graphunet |  | 10 | 3 |
| 9e58dc9e84f1 | pubmed | graphunet |  | 10 | 4 |
| 745e270ce4d5 | pubmed | graphunet |  | 10 | 5 |
| e59f8c0c4761 | pubmed | graphunet |  | 10 | 6 |
| 3f668ac70627 | pubmed | graphunet |  | 10 | 7 |
| 97baabf4d143 | pubmed | graphunet |  | 10 | 8 |
| c4c709fe493c | pubmed | graphunet |  | 10 | 9 |
| b6b0bd2f64c0 | pubmed | graphunet |  | 20 | 0 |
| f1389f56705c | pubmed | graphunet |  | 20 | 1 |
| ce1edbad01b6 | pubmed | graphunet |  | 20 | 2 |
| eed9539f34f7 | pubmed | graphunet |  | 20 | 3 |
| d4981b81da2a | pubmed | graphunet |  | 20 | 4 |
| f2b69f65e6a4 | pubmed | graphunet |  | 20 | 5 |
| 802de9827d57 | pubmed | graphunet |  | 20 | 6 |
| b29f976fd124 | pubmed | graphunet |  | 20 | 7 |
| b37f18e3df6e | pubmed | graphunet |  | 20 | 8 |
| 015cd49a1e89 | pubmed | graphunet |  | 20 | 9 |
| 846237986923 | pubmed | graphunet |  | 30 | 0 |
| 090e1d5e9a0e | pubmed | graphunet |  | 30 | 1 |
| 341a25d34a85 | pubmed | graphunet |  | 30 | 2 |
| 26c36e734e8d | pubmed | graphunet |  | 30 | 3 |
| a00e4776bce1 | pubmed | graphunet |  | 30 | 4 |
| 8a074ef7b859 | pubmed | graphunet |  | 30 | 5 |
| 6358bb4f592e | pubmed | graphunet |  | 30 | 6 |
| e651bf0c040b | pubmed | graphunet |  | 30 | 7 |
| 5a715c1bc5f8 | pubmed | graphunet |  | 30 | 8 |
| 60af45ec70f0 | pubmed | graphunet |  | 30 | 9 |
| 34e6c4df2778 | photo | nhdp |  | 2 | 0 |
| 267067174015 | photo | nhdp |  | 2 | 2 |
| f233044be060 | photo | nhdp |  | 2 | 3 |
| 0a09d596e389 | photo | nhdp |  | 2 | 5 |
| 5361be5ac6c7 | photo | nhdp |  | 2 | 7 |
| 10e37557162c | photo | nhdp |  | 2 | 8 |
| 0ab303494bc7 | photo | nhdp |  | 2 | 9 |
| 677f582da8e1 | photo | nhdp |  | 4 | 0 |
| 9555597eeef2 | photo | nhdp |  | 4 | 1 |
| b7ca57019763 | photo | nhdp |  | 4 | 3 |
| 971c1f8ac259 | photo | nhdp |  | 4 | 4 |
| c79a59d3aade | photo | nhdp |  | 4 | 6 |
| 672b6bfb6741 | photo | nhdp |  | 8 | 0 |
| f397667bf362 | photo | nhdp |  | 8 | 2 |
| ed98b2dab8d1 | photo | nhdp |  | 8 | 3 |
| fadcf3b6286b | photo | nhdp |  | 8 | 4 |
| 70673a97e59a | photo | nhdp |  | 8 | 5 |
| 17d111fd9b6a | photo | nhdp |  | 8 | 6 |
| 6ef6509a0768 | photo | nhdp |  | 8 | 8 |
| 88ca4b0a468e | photo | nhdp |  | 8 | 9 |
| 2dea1fd8fa1b | photo | nhdp |  | 10 | 1 |
| 0477e58d512a | photo | nhdp |  | 10 | 2 |
| aa8f8640fe5a | photo | nhdp |  | 10 | 4 |
| d501e0260838 | photo | nhdp |  | 10 | 5 |
| 9aa60c8f41ea | photo | nhdp |  | 10 | 7 |
| b5503e416b6c | photo | nhdp |  | 20 | 3 |
| 2801c3cd8d37 | photo | nhdp |  | 20 | 4 |
| 29f655a5ea6e | photo | nhdp |  | 20 | 5 |
| 286b023e9413 | photo | nhdp |  | 20 | 6 |
| 6f108637dc53 | photo | nhdp |  | 20 | 7 |
| 1fe55b982cf2 | photo | nhdp |  | 20 | 8 |
| 93225576c236 | photo | nhdp |  | 20 | 9 |
| c40e3abf015e | photo | nhdp |  | 30 | 0 |
| 476205847047 | photo | nhdp |  | 30 | 1 |
| de07bc8ca254 | photo | nhdp |  | 30 | 2 |
| 52960f8678e9 | photo | nhdp |  | 30 | 3 |
| 1c71ee2cc244 | photo | nhdp |  | 30 | 4 |
| c6bf844d5635 | photo | nhdp |  | 30 | 5 |
| ba75ce3db3a2 | photo | nhdp |  | 30 | 6 |
| 6cad82bf33e3 | photo | nhdp |  | 30 | 7 |
| 11a6b968e516 | photo | nhdp |  | 30 | 8 |
| 37369f0c17f0 | photo | nhdp |  | 30 | 9 |
| c72656796af1 | photo | gcn |  | 2 | 0 |
| 94f68d237c71 | photo | gcn |  | 2 | 1 |
| e54f70d53529 | photo | gcn |  | 2 | 2 |
| 94010c14e13d | photo | gcn |  | 2 | 3 |
| 3d4812a2a3f9 | photo | gcn |  | 2 | 4 |
| de5e120971a7 | photo | gcn |  | 2 | 5 |
| 94c72c225f98 | photo | gcn |  | 2 | 6 |
| 4ca48f5d0618 | photo | gcn |  | 2 | 7 |
| b7288044334d | photo | gcn |  | 2 | 8 |
| 0ba863e9ef73 | photo | gcn |  | 2 | 9 |
| c768f4e50527 | photo | gcn |  | 4 | 0 |
| 92338c692ead | photo | gcn |  | 4 | 1 |
| a2a2de72e093 | photo | gcn |  | 4 | 2 |
| d23f24896574 | photo | gcn |  | 4 | 3 |
| 3b74d3b9a56e | photo | gcn |  | 4 | 4 |
| 622a864f94fd | photo | gcn |  | 4 | 5 |
| ebb830ce3a61 | photo | gcn |  | 4 | 6 |
| 13e902ef448b | photo | gcn |  | 4 | 7 |
| d69d74f432b6 | photo | gcn |  | 4 | 8 |
| 6c0cea8f4a76 | photo | gcn |  | 4 | 9 |
| e10d6c02694b | photo | gcn |  | 8 | 0 |
| d9d150636c7b | photo | gcn |  | 8 | 1 |
| fc38c828f051 | photo | gcn |  | 8 | 2 |
| 28198e08ffdb | photo | gcn |  | 8 | 3 |
| 11d907907131 | photo | gcn |  | 8 | 4 |
| a603230bffce | photo | gcn |  | 8 | 5 |
| a0f1e14e9828 | photo | gcn |  | 8 | 6 |
| ce64858c2ee0 | photo | gcn |  | 8 | 7 |
| e3cea01b105d | photo | gcn |  | 8 | 8 |
| ebaf7fff34e5 | photo | gcn |  | 8 | 9 |
| f56c9e05f6f3 | photo | gcn |  | 10 | 0 |
| 2592cf4b84e5 | photo | gcn |  | 10 | 1 |
| 15d2e2de2bab | photo | gcn |  | 10 | 2 |
| d8f7e0b0019f | photo | gcn |  | 10 | 3 |
| 45fef410e4d1 | photo | gcn |  | 10 | 4 |
| 88ce7e19aab8 | photo | gcn |  | 10 | 5 |
| 862b0465b2d5 | photo | gcn |  | 10 | 6 |
| 5addb41940fc | photo | gcn |  | 10 | 7 |
| f7498d065933 | photo | gcn |  | 10 | 8 |
| e72727cfb478 | photo | gcn |  | 10 | 9 |
| 42bed4d2a040 | photo | gcn |  | 20 | 0 |
| 5faa3a244da4 | photo | gcn |  | 20 | 1 |
| 20a2888c0c7b | photo | gcn |  | 20 | 2 |
| d87eccea1ac4 | photo | gcn |  | 20 | 3 |
| e641bf2ed693 | photo | gcn |  | 20 | 4 |
| de1c3cd5af51 | photo | gcn |  | 20 | 5 |
| 4da5919828bf | photo | gcn |  | 20 | 6 |
| a97e40d6a19c | photo | gcn |  | 20 | 7 |
| 83e1371af9f8 | photo | gcn |  | 20 | 8 |
| f3233f466848 | photo | gcn |  | 20 | 9 |
| 58e17f0c6a61 | photo | gcn |  | 30 | 0 |
| 3f62acd33118 | photo | gcn |  | 30 | 1 |
| e2b09fdebe26 | photo | gcn |  | 30 | 2 |
| 8d88f8abf940 | photo | gcn |  | 30 | 3 |
| bab4c754503b | photo | gcn |  | 30 | 4 |
| 2f26ec00775c | photo | gcn |  | 30 | 5 |
| d4004ff7d81d | photo | gcn |  | 30 | 6 |
| 34ead5cf6400 | photo | gcn |  | 30 | 7 |
| 38df2f90eaf3 | photo | gcn |  | 30 | 8 |
| 62dfd156f7e2 | photo | gcn |  | 30 | 9 |
| 8c7f9160bc39 | photo | gat |  | 2 | 4 |
| 58259815a067 | photo | gat |  | 2 | 5 |
| 501d378f0677 | photo | gat |  | 2 | 6 |
| 9b082e1ae467 | photo | gat |  | 2 | 7 |
| 4b89368169cf | photo | gat |  | 4 | 3 |
| 6f7d3e2e8b5e | photo | gat |  | 4 | 4 |
| 28c333436e8f | photo | gat |  | 4 | 5 |
| c2d91cbfea76 | photo | gat |  | 4 | 6 |
| 3bed7505ab6c | photo | gat |  | 4 | 7 |
| a70438f1e84d | photo | gat |  | 4 | 8 |
| 3c7229d4febe | photo | gat |  | 4 | 9 |
| cc06a8b9695f | photo | gat |  | 8 | 8 |
| 0a2eeea8dfe6 | photo | gat |  | 8 | 9 |
| e5ac49772796 | photo | gat |  | 10 | 0 |
| 9705d8a6b92d | photo | gat |  | 20 | 0 |
| 036f29c01609 | photo | gat |  | 20 | 1 |
| 83b7ee2d95b8 | photo | gat |  | 20 | 3 |
| 19e88e6f13b0 | photo | gat |  | 20 | 4 |
| 338e87ac2800 | photo | gat |  | 20 | 6 |
| f9e426587454 | photo | gat |  | 30 | 0 |
| 93f5b633a03c | photo | gat |  | 30 | 6 |
| 16029aa55778 | photo | gat |  | 30 | 8 |
| a82cd36b0afd | photo | gat |  | 30 | 9 |
| 974e13353547 | photo | sgc |  | 2 | 1 |
| 7705b6d170be | photo | sgc |  | 2 | 2 |
| baeefef91284 | photo | sgc |  | 2 | 4 |
| 2e5fd9b660c9 | photo | sgc |  | 2 | 5 |
| debf4d009835 | photo | sgc |  | 2 | 7 |
| 7e8ad89ef71c | photo | sgc |  | 4 | 3 |
| 746662c841ef | photo | sgc |  | 4 | 4 |
| a2a815fea584 | photo | sgc |  | 4 | 5 |
| be1af7772da1 | photo | sgc |  | 4 | 6 |
| 21237591b8ad | photo | sgc |  | 4 | 7 |
| 72b6615cb333 | photo | sgc |  | 4 | 8 |
| 5bf8c43a07ad | photo | sgc |  | 4 | 9 |
| 40b10ba62876 | photo | sgc |  | 8 | 0 |
| dfe9ed806222 | photo | sgc |  | 8 | 1 |
| 4d569762893a | photo | sgc |  | 8 | 2 |
| 0fae68304595 | photo | sgc |  | 8 | 3 |
| 1001a0b6fb85 | photo | sgc |  | 8 | 4 |
| 9a673a179b87 | photo | sgc |  | 8 | 5 |
| d35eacf88b07 | photo | sgc |  | 8 | 6 |
| 6c3f1128a341 | photo | sgc |  | 8 | 7 |
| 9234eaca4a46 | photo | sgc |  | 8 | 8 |
| e2c06c2cdf46 | photo | sgc |  | 8 | 9 |
| 0282b29e288b | photo | sgc |  | 10 | 0 |
| fef35d9fa743 | photo | sgc |  | 10 | 1 |
| acaf84028fe4 | photo | sgc |  | 10 | 2 |
| 06f423325402 | photo | sgc |  | 10 | 3 |
| 42de9ebe1db6 | photo | sgc |  | 10 | 4 |
| fe938f5ddc70 | photo | sgc |  | 10 | 5 |
| 657114c78d8e | photo | sgc |  | 10 | 6 |
| 678c7e80203f | photo | sgc |  | 10 | 7 |
| 77738238a12e | photo | sgc |  | 10 | 8 |
| c8ae151a406c | photo | sgc |  | 10 | 9 |
| 306c938ae685 | photo | sgc |  | 20 | 0 |
| 11a4ff7c8f18 | photo | sgc |  | 20 | 1 |
| 6702ce2ada29 | photo | sgc |  | 20 | 2 |
| d5a97b775564 | photo | sgc |  | 20 | 3 |
| b4b79dbf6bf2 | photo | sgc |  | 20 | 4 |
| 6a8dcb2405de | photo | sgc |  | 20 | 5 |
| be0d0d7d14cc | photo | sgc |  | 20 | 6 |
| b667c1310df7 | photo | sgc |  | 20 | 7 |
| ff0189b91f4e | photo | sgc |  | 20 | 8 |
| 31cdfa8f24c4 | photo | sgc |  | 20 | 9 |
| c19e00b0a2b5 | photo | sgc |  | 30 | 0 |
| 1e543065939a | photo | sgc |  | 30 | 1 |
| f8feb327cee1 | photo | sgc |  | 30 | 2 |
| d9d9b753dfaf | photo | sgc |  | 30 | 3 |
| b32fb2a6e1f0 | photo | sgc |  | 30 | 4 |
| e97f91b9d076 | photo | sgc |  | 30 | 5 |
| eca08e02522f | photo | sgc |  | 30 | 6 |
| e927f455e399 | photo | sgc |  | 30 | 7 |
| 7513c4fd8eef | photo | sgc |  | 30 | 8 |
| 8562edace994 | photo | sgc |  | 30 | 9 |
| ac8d1cbb1059 | photo | appnp |  | 2 | 0 |
| 87a2bf8aab6b | photo | appnp |  | 2 | 1 |
| c2c84efbfc60 | photo | appnp |  | 2 | 2 |
| 655c08e75dd9 | photo | appnp |  | 2 | 3 |
| a072a4a5b3f3 | photo | appnp |  | 2 | 4 |
| 8090c5611268 | photo | appnp |  | 2 | 5 |
| 7ffc326b448e | photo | appnp |  | 2 | 6 |
| 1166798326d8 | photo | appnp |  | 2 | 7 |
| e0576cd87a1f | photo | appnp |  | 2 | 8 |
| e95261d56a4f | photo | appnp |  | 2 | 9 |
| 1ae836a447a1 | photo | appnp |  | 4 | 0 |
| e75c452f3726 | photo | appnp |  | 4 | 1 |
| 0e7479c73b78 | photo | appnp |  | 4 | 2 |
| 02b1b2af90ce | photo | appnp |  | 4 | 3 |
| 5724573ffc42 | photo | appnp |  | 4 | 4 |
| c1164e04bdb5 | photo | appnp |  | 4 | 5 |
| 38d902bc6d2b | photo | appnp |  | 8 | 0 |
| 7dac2c81292e | photo | appnp |  | 8 | 1 |
| 5908dd5b3b60 | photo | appnp |  | 8 | 2 |
| 301cae827127 | photo | appnp |  | 8 | 3 |
| 6cb9f787c227 | photo | appnp |  | 8 | 4 |
| 21f8b92b0dac | photo | appnp |  | 8 | 5 |
| 6fa3faa594bb | photo | appnp |  | 8 | 6 |
| 507cb3b58037 | photo | appnp |  | 8 | 7 |
| 80c49fe78f1b | photo | appnp |  | 8 | 8 |
| 8bf4e5d1b195 | photo | appnp |  | 8 | 9 |
| a7d3ce5b22c7 | photo | appnp |  | 10 | 0 |
| 622aba99924c | photo | appnp |  | 10 | 1 |
| e0dcfb836a74 | photo | appnp |  | 10 | 2 |
| cc07b3d9bd15 | photo | appnp |  | 10 | 3 |
| 09b6f7ea959e | photo | appnp |  | 10 | 4 |
| 79a5955eb313 | photo | appnp |  | 10 | 5 |
| 09a569f21804 | photo | appnp |  | 10 | 6 |
| 22a8b0f1cbb3 | photo | appnp |  | 10 | 7 |
| b4d645442dec | photo | appnp |  | 10 | 8 |
| 7e173f2c2ab6 | photo | appnp |  | 10 | 9 |
| 7a6449bf0df6 | photo | appnp |  | 20 | 0 |
| 917b6b1b36e3 | photo | appnp |  | 20 | 1 |
| a0e47ead6121 | photo | appnp |  | 20 | 2 |
| 1ec4919a5834 | photo | appnp |  | 20 | 3 |
| 529995928c5f | photo | appnp |  | 20 | 4 |
| 1ee8c709fd6c | photo | appnp |  | 20 | 5 |
| 63554fe15700 | photo | appnp |  | 20 | 6 |
| 62f39674fd9a | photo | appnp |  | 20 | 7 |
| eca14425c2d6 | photo | appnp |  | 20 | 8 |
| 8f52b831f7f8 | photo | appnp |  | 20 | 9 |
| 5f18a35da84b | photo | appnp |  | 30 | 0 |
| 53dea97ee641 | photo | appnp |  | 30 | 1 |
| 9d4daa506015 | photo | appnp |  | 30 | 2 |
| 2629824bcb9d | photo | appnp |  | 30 | 3 |
| fb16f8f14090 | photo | appnp |  | 30 | 4 |
| 75701bb62aaa | photo | appnp |  | 30 | 5 |
| 69fcdf6e7984 | photo | appnp |  | 30 | 6 |
| 7f4476c22e41 | photo | appnp |  | 30 | 7 |
| 461ad17e8f15 | photo | appnp |  | 30 | 8 |
| e3d8f5f2c09e | photo | appnp |  | 30 | 9 |
| 8932b677ee4d | photo | arma |  | 2 | 0 |
| 63bde3990dfd | photo | arma |  | 2 | 2 |
| 7ace58d3e0f6 | photo | arma |  | 2 | 3 |
| fb5497509ff7 | photo | arma |  | 2 | 5 |
| 996db9e219a7 | photo | arma |  | 2 | 7 |
| b896352ffa29 | photo | arma |  | 2 | 8 |
| 05a8def9882b | photo | arma |  | 2 | 9 |
| 6346357acd86 | photo | arma |  | 4 | 0 |
| b34a75629179 | photo | arma |  | 4 | 1 |
| df9675389bf0 | photo | arma |  | 4 | 3 |
| 86e3f2de0242 | photo | arma |  | 4 | 4 |
| b9fb04f701dd | photo | arma |  | 4 | 6 |
| 230a7fbc279a | photo | arma |  | 8 | 0 |
| 6c528c02a954 | photo | arma |  | 8 | 2 |
| f8d64fcf3f95 | photo | arma |  | 8 | 3 |
| 3e77f3f25a11 | photo | arma |  | 8 | 4 |
| 9d2778e2730f | photo | arma |  | 8 | 5 |
| 901074825bb4 | photo | arma |  | 8 | 6 |
| 15e1364f25a4 | photo | arma |  | 10 | 2 |
| 8ec8c57001b8 | photo | arma |  | 10 | 4 |
| 10b9e585c8af | photo | arma |  | 10 | 5 |
| a1e2686db25c | photo | arma |  | 10 | 7 |
| 070c1366ec38 | photo | arma |  | 20 | 3 |
| e328e5101de7 | photo | arma |  | 20 | 4 |
| ed9dfb02fa7c | photo | arma |  | 20 | 5 |
| 7ff21b5f9969 | photo | arma |  | 20 | 6 |
| d6aa562efa9b | photo | arma |  | 20 | 7 |
| 15475ba0291b | photo | arma |  | 20 | 8 |
| 4e5b86c3e238 | photo | arma |  | 20 | 9 |
| 0e6be295824d | photo | arma |  | 30 | 0 |
| 095731d6ed9b | photo | arma |  | 30 | 1 |
| 2ad7becf6faa | photo | arma |  | 30 | 2 |
| ffdd2e4af55a | photo | arma |  | 30 | 3 |
| 538123a57e27 | photo | arma |  | 30 | 4 |
| 79c513876a87 | photo | arma |  | 30 | 5 |
| e8fb1ccec3fd | photo | arma |  | 30 | 6 |
| 5823345810e4 | photo | arma |  | 30 | 7 |
| dc7177f4bd31 | photo | arma |  | 30 | 8 |
| 4f7e2dda35e5 | photo | arma |  | 30 | 9 |
| ad2990aa52c0 | photo | gcnii |  | 2 | 0 |
| 6d72c616c485 | photo | gcnii |  | 2 | 1 |
| 32d7d4cfd34a | photo | gcnii |  | 2 | 2 |
| d89325a59ca9 | photo | gcnii |  | 2 | 3 |
| 407ca1d67ca6 | photo | gcnii |  | 2 | 4 |
| a98748705058 | photo | gcnii |  | 2 | 5 |
| 7fbad925d60d | photo | gcnii |  | 2 | 6 |
| 2b9beb1f6584 | photo | gcnii |  | 2 | 7 |
| f42392933b32 | photo | gcnii |  | 2 | 8 |
| 528643b2ac36 | photo | gcnii |  | 2 | 9 |
| 75fbd60c1499 | photo | gcnii |  | 4 | 0 |
| 2e5d19b93c09 | photo | gcnii |  | 4 | 1 |
| e21355ff7517 | photo | gcnii |  | 4 | 2 |
| 5302e3ba65c5 | photo | gcnii |  | 4 | 3 |
| e84dab87512b | photo | gcnii |  | 4 | 4 |
| ed05be6575f9 | photo | gcnii |  | 4 | 5 |
| 870071e7dec6 | photo | gcnii |  | 4 | 6 |
| 555002260ddd | photo | gcnii |  | 4 | 7 |
| 0bdc27a558e5 | photo | gcnii |  | 4 | 8 |
| 91bbaafa37dc | photo | gcnii |  | 4 | 9 |
| 1767e20c04c5 | photo | gcnii |  | 8 | 0 |
| acb9e8253c98 | photo | gcnii |  | 8 | 1 |
| 295c659b6ea8 | photo | gcnii |  | 8 | 2 |
| f2698fb5f7c3 | photo | gcnii |  | 8 | 3 |
| 0eb8ef9956ca | photo | gcnii |  | 8 | 4 |
| 1ee4b8892fc4 | photo | gcnii |  | 8 | 5 |
| bff7ba1776be | photo | gcnii |  | 8 | 6 |
| f1b82a7c65ee | photo | gcnii |  | 8 | 7 |
| c9b436c666c7 | photo | gcnii |  | 8 | 8 |
| 5d8ce63c47dd | photo | gcnii |  | 8 | 9 |
| 409291b31e8f | photo | gcnii |  | 10 | 0 |
| 19ac2cc44615 | photo | gcnii |  | 10 | 1 |
| 39e59177a494 | photo | gcnii |  | 10 | 2 |
| 4dcfbd6327a6 | photo | gcnii |  | 10 | 3 |
| a391dd53951a | photo | gcnii |  | 10 | 4 |
| b615b5a6a9ac | photo | gcnii |  | 10 | 5 |
| b9e5cdfa6dae | photo | gcnii |  | 10 | 6 |
| 8cffe2beaefb | photo | gcnii |  | 10 | 7 |
| 655bd3690576 | photo | gcnii |  | 10 | 8 |
| b52e558f9d51 | photo | gcnii |  | 10 | 9 |
| ff12377ed5ac | photo | gcnii |  | 20 | 0 |
| 7ce969e2cf78 | photo | gcnii |  | 20 | 1 |
| 43374d99dcda | photo | gcnii |  | 20 | 2 |
| a43d98bd1d73 | photo | gcnii |  | 20 | 3 |
| 76bbea5eb9af | photo | gcnii |  | 20 | 4 |
| e4f8fe9e2851 | photo | gcnii |  | 20 | 5 |
| 1bbad3cb1afd | photo | gcnii |  | 20 | 6 |
| b988821be4ac | photo | gcnii |  | 20 | 7 |
| 8589b613fcc7 | photo | gcnii |  | 20 | 8 |
| f9de24c274eb | photo | gcnii |  | 20 | 9 |
| 3ca4048e12c8 | photo | gcnii |  | 30 | 0 |
| 1ac5803b758f | photo | gcnii |  | 30 | 1 |
| df570ceb4903 | photo | gcnii |  | 30 | 2 |
| 5e20862af2f4 | photo | gcnii |  | 30 | 3 |
| 9d7264df5a68 | photo | gcnii |  | 30 | 4 |
| de47f8bf85aa | photo | gcnii |  | 30 | 5 |
| d2ea668e4d2f | photo | gcnii |  | 30 | 6 |
| a4f4ba6347b7 | photo | gcnii |  | 30 | 7 |
| eb8c7dbdfb20 | photo | gcnii |  | 30 | 8 |
| f72acbd122e7 | photo | gcnii |  | 30 | 9 |
| 5682472f811d | photo | pairnorm |  | 2 | 0 |
| f7a7b4fe868e | photo | pairnorm |  | 2 | 1 |
| cf87f4b48402 | photo | pairnorm |  | 2 | 2 |
| 5fb345f67c06 | photo | pairnorm |  | 2 | 3 |
| 7a9a7bcae20e | photo | pairnorm |  | 2 | 4 |
| 9ed1d5bf4998 | photo | pairnorm |  | 2 | 5 |
| 93a79dcc6c52 | photo | pairnorm |  | 2 | 6 |
| 219abbc69114 | photo | pairnorm |  | 2 | 7 |
| ee640b1bcca3 | photo | pairnorm |  | 2 | 8 |
| ffac7c541a4d | photo | pairnorm |  | 2 | 9 |
| a3739ee1ab41 | photo | pairnorm |  | 4 | 0 |
| f2ae8cdcb353 | photo | pairnorm |  | 4 | 1 |
| fcbd0f4456e1 | photo | pairnorm |  | 4 | 2 |
| 12f99e8dd265 | photo | pairnorm |  | 4 | 3 |
| b547179661cb | photo | pairnorm |  | 4 | 4 |
| e471c2ac859a | photo | pairnorm |  | 4 | 5 |
| 4683c713645d | photo | pairnorm |  | 4 | 6 |
| 262b3a9b3cca | photo | pairnorm |  | 4 | 7 |
| c91fba0eaf31 | photo | pairnorm |  | 4 | 8 |
| 54be54e3fef7 | photo | pairnorm |  | 4 | 9 |
| e54317c3a5af | photo | pairnorm |  | 8 | 0 |
| 1f009937b396 | photo | pairnorm |  | 8 | 1 |
| 79ca649805b0 | photo | pairnorm |  | 8 | 2 |
| bbcfec765839 | photo | pairnorm |  | 8 | 3 |
| 5ebc1b2829c7 | photo | pairnorm |  | 8 | 4 |
| ff557a98e792 | photo | pairnorm |  | 8 | 5 |
| 13f1c84a698e | photo | pairnorm |  | 8 | 6 |
| d7c74261812f | photo | pairnorm |  | 8 | 7 |
| 4aacaea09732 | photo | pairnorm |  | 8 | 8 |
| 9b830b848208 | photo | pairnorm |  | 8 | 9 |
| 9c38444cc22f | photo | pairnorm |  | 10 | 0 |
| 75cb7248ee53 | photo | pairnorm |  | 10 | 2 |
| 7556c099d982 | photo | pairnorm |  | 10 | 3 |
| 2a53129fd330 | photo | pairnorm |  | 10 | 5 |
| 20b527846991 | photo | pairnorm |  | 10 | 7 |
| 731864fcc207 | photo | pairnorm |  | 10 | 8 |
| d93a46e4b850 | photo | pairnorm |  | 10 | 9 |
| 01bed1c403b0 | photo | pairnorm |  | 20 | 0 |
| 763ac2eb223b | photo | pairnorm |  | 20 | 1 |
| df7809e794d5 | photo | pairnorm |  | 20 | 3 |
| 5208df044205 | photo | pairnorm |  | 20 | 4 |
| 6ced1487ae6b | photo | pairnorm |  | 20 | 6 |
| 1e2ec937fb7b | photo | pairnorm |  | 30 | 0 |
| 054d64f5ecc4 | photo | pairnorm |  | 30 | 2 |
| 1ba08b2ec22e | photo | pairnorm |  | 30 | 3 |
| 250aa8aa0869 | photo | pairnorm |  | 30 | 4 |
| 6169f0fea8cc | photo | pairnorm |  | 30 | 5 |
| 59245040a967 | photo | pairnorm |  | 30 | 6 |
| c7c04210ed66 | photo | pairnorm |  | 30 | 8 |
| 385fae6d1cb7 | photo | pairnorm |  | 30 | 9 |
| 1b38e09777c3 | photo | dropedge |  | 2 | 1 |
| 570a8d9a5b19 | photo | dropedge |  | 2 | 2 |
| d88c4e374d1c | photo | dropedge |  | 2 | 4 |
| 6dffc81513a3 | photo | dropedge |  | 2 | 5 |
| b2adcdc59557 | photo | dropedge |  | 2 | 7 |
| 6922ba3758c8 | photo | dropedge |  | 4 | 3 |
| f4234659e7f2 | photo | dropedge |  | 4 | 4 |
| 20613b5ed1a1 | photo | dropedge |  | 4 | 5 |
| c1adf29670f1 | photo | dropedge |  | 4 | 6 |
| 2be58f18af48 | photo | dropedge |  | 4 | 7 |
| 63de89c4b74f | photo | dropedge |  | 4 | 8 |
| f26c877b657b | photo | dropedge |  | 4 | 9 |
| f3e163663a55 | photo | dropedge |  | 8 | 0 |
| 81d2dfc6e26d | photo | dropedge |  | 8 | 1 |
| 2886ce7c23c1 | photo | dropedge |  | 8 | 2 |
| 577c2b7d3e6e | photo | dropedge |  | 8 | 3 |
| 04cc96437929 | photo | dropedge |  | 8 | 4 |
| 41604701da4f | photo | dropedge |  | 8 | 5 |
| 4f70afcef6f8 | photo | dropedge |  | 8 | 6 |
| ba0e605de570 | photo | dropedge |  | 8 | 7 |
| 9c01f8295cc0 | photo | dropedge |  | 8 | 8 |
| af28cc65a148 | photo | dropedge |  | 8 | 9 |
| 0075de990e71 | photo | dropedge |  | 10 | 0 |
| be272913ecf5 | photo | dropedge |  | 10 | 1 |
| 57d2fa75a0a4 | photo | dropedge |  | 10 | 2 |
| 2b24d7cdf145 | photo | dropedge |  | 10 | 3 |
| 75e1660d37e0 | photo | dropedge |  | 10 | 4 |
| fa31f5c2798e | photo | dropedge |  | 10 | 5 |
| 6f71c3c43020 | photo | dropedge |  | 10 | 6 |
| 16d7d5c948b4 | photo | dropedge |  | 10 | 7 |
| c46e302d2e21 | photo | dropedge |  | 10 | 8 |
| a63da812a52e | photo | dropedge |  | 10 | 9 |
| 860e3f9868a4 | photo | dropedge |  | 20 | 0 |
| dd7563a755f6 | photo | dropedge |  | 20 | 1 |
| 2fd0a2de4abb | photo | dropedge |  | 20 | 2 |
| a087b4894784 | photo | dropedge |  | 20 | 3 |
| 5b9a71c8fda9 | photo | dropedge |  | 20 | 4 |
| 3c8c3b19861f | photo | dropedge |  | 20 | 5 |
| 705e32e8fc55 | photo | dropedge |  | 20 | 6 |
| e78991d75279 | photo | dropedge |  | 20 | 7 |
| 6ed80099f850 | photo | dropedge |  | 20 | 8 |
| d3a13886c1be | photo | dropedge |  | 20 | 9 |
| 24858820ec2a | photo | dropedge |  | 30 | 0 |
| fd4e7a31cc41 | photo | dropedge |  | 30 | 1 |
| dd31d89b216e | photo | dropedge |  | 30 | 2 |
| 327904e834a9 | photo | dropedge |  | 30 | 3 |
| 17fe74a0f6b3 | photo | dropedge |  | 30 | 4 |
| d8511300f3c3 | photo | dropedge |  | 30 | 5 |
| 8e82e7df020a | photo | dropedge |  | 30 | 6 |
| a0473b288d4a | photo | dropedge |  | 30 | 7 |
| ed681020558e | photo | dropedge |  | 30 | 8 |
| 463e6991a66d | photo | dropedge |  | 30 | 9 |
| 9618f2b515a8 | photo | graphunet |  | 2 | 0 |
| 706e9de33509 | photo | graphunet |  | 2 | 1 |
| bc57a6e84c0b | photo | graphunet |  | 2 | 2 |
| f9c8e9838c04 | photo | graphunet |  | 2 | 3 |
| 8563ac8e99cc | photo | graphunet |  | 2 | 4 |
| f2241dc14846 | photo | graphunet |  | 2 | 5 |
| 0d647822255a | photo | graphunet |  | 2 | 6 |
| 1cae3c2f6761 | photo | graphunet |  | 2 | 7 |
| 3daa66f12c3e | photo | graphunet |  | 2 | 8 |
| 5d05e94a1f3a | photo | graphunet |  | 2 | 9 |
| c911ff952952 | photo | graphunet |  | 4 | 0 |
| a9ef2698b9c5 | photo | graphunet |  | 4 | 1 |
| 08ed689057e1 | photo | graphunet |  | 4 | 2 |
| 7df490b88962 | photo | graphunet |  | 4 | 3 |
| 4e4392b894ba | photo | graphunet |  | 4 | 4 |
| 7dde72b9503f | photo | graphunet |  | 4 | 5 |
| 8cb7c8ae3ba4 | photo | graphunet |  | 4 | 6 |
| 09815abb1215 | photo | graphunet |  | 4 | 7 |
| b2549905c109 | photo | graphunet |  | 4 | 8 |
| 458bbfd990c4 | photo | graphunet |  | 4 | 9 |
| 9faedfb98000 | photo | graphunet |  | 8 | 0 |
| e99802f67fbb | photo | graphunet |  | 8 | 1 |
| d3c00e5c329e | photo | graphunet |  | 8 | 2 |
| 7fd311b76dad | photo | graphunet |  | 8 | 3 |
| ba305f372a0b | photo | graphunet |  | 8 | 4 |
| ab72553c930e | photo | graphunet |  | 8 | 5 |
| 52a707be62ac | photo | graphunet |  | 8 | 6 |
| 4b5ac74196f7 | photo | graphunet |  | 8 | 7 |
| 7d60b0e38669 | photo | graphunet |  | 8 | 8 |
| 7d49168e0c75 | photo | graphunet |  | 8 | 9 |
| 5bd6411d6db0 | photo | graphunet |  | 10 | 0 |
| 9795f69efe0f | photo | graphunet |  | 10 | 1 |
| c0c08da6e490 | photo | graphunet |  | 10 | 2 |
| e721c4bbab8d | photo | graphunet |  | 10 | 3 |
| f43f4736ce9d | photo | graphunet |  | 10 | 4 |
| aee97312c1b4 | photo | graphunet |  | 10 | 5 |
| b51310427087 | photo | graphunet |  | 10 | 6 |
| ab8dc5b832cd | photo | graphunet |  | 10 | 7 |
| 3f60513050db | photo | graphunet |  | 10 | 8 |
| c54a92c46b2b | photo | graphunet |  | 10 | 9 |
| 6aa2a01e01aa | photo | graphunet |  | 20 | 0 |
| 6cc83aaea1cc | photo | graphunet |  | 20 | 1 |
| d4a0f7dd229f | photo | graphunet |  | 20 | 2 |
| 1211aae959ba | photo | graphunet |  | 20 | 3 |
| 2e17a917f2a8 | photo | graphunet |  | 20 | 4 |
| e3f2df227a7e | photo | graphunet |  | 20 | 5 |
| 36ba9c71f997 | photo | graphunet |  | 20 | 6 |
| 2e4aa732ac01 | photo | graphunet |  | 20 | 7 |
| a573c5444867 | photo | graphunet |  | 20 | 8 |
| 7e11ffe80612 | photo | graphunet |  | 20 | 9 |
| 98d25d46cfe9 | photo | graphunet |  | 30 | 0 |
| 1f70d7daca8b | photo | graphunet |  | 30 | 1 |
| b580f7c7d4b1 | photo | graphunet |  | 30 | 2 |
| f68a73cadfec | photo | graphunet |  | 30 | 3 |
| 48afd35514a9 | photo | graphunet |  | 30 | 4 |
| fac83f3b461f | photo | graphunet |  | 30 | 5 |
| 06a1e657794d | photo | graphunet |  | 30 | 6 |
| ddfb905b5b46 | photo | graphunet |  | 30 | 7 |
| 237da9c1076d | photo | graphunet |  | 30 | 8 |
| 65ca0c46dd2f | photo | graphunet |  | 30 | 9 |
