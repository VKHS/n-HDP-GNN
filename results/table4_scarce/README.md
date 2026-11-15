# table4_scarce

Sweep `configs/table4_scarce.toml`: 2400 runs expected, 224 completed, 0 failed, 2176 missing.

Values are mean ± 95% t-interval over seeds; n is the number of completed seeds in the cell.

Code commit(s) of the runs: 0f7045972e60a86d7966841bc3e5c4bfe0caa016

## test (accuracy in %)

**citeseer** -- test

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 69.26 ± 1.34 (n=9) | 65.31 ± 3.75 (n=3) | 21.08 (n=1) | 24.25 (n=1) |
| dropedge | 68.22 (n=1) | 71.69 (n=1) |  |  |
| gat |  |  | 63.00 ± 1.41 (n=9) | 23.54 ± 5.32 (n=3) |
| nhdp | 62.27 ± 2.68 (n=9) | 61.30 ± 3.96 (n=3) | 55.87 (n=1) | 63.70 (n=1) |
| pairnorm |  |  | 46.80 ± 1.86 (n=9) | 37.80 ± 19.16 (n=3) |
| sgc | 64.76 (n=1) | 68.98 (n=1) |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.00391 | 0.25 |  |  |
| dropedge |  |  |  |  |
| gat |  |  |  |  |
| pairnorm |  |  |  |  |
| sgc |  |  |  |  |

**cora** -- test

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 79.16 ± 1.96 (n=9) | 77.72 ± 6.21 (n=3) | 28.52 (n=1) | 30.19 (n=1) |
| dropedge | 79.07 (n=1) | 79.63 (n=1) |  |  |
| gat |  |  | 76.93 ± 2.08 (n=9) | 31.48 ± 5.58 (n=3) |
| nhdp | 64.61 ± 3.71 (n=9) | 63.58 ± 6.07 (n=3) | 62.41 (n=1) | 69.44 (n=1) |
| pairnorm |  |  | 61.67 ± 1.96 (n=9) | 66.05 ± 4.83 (n=3) |
| sgc | 36.85 (n=1) | 34.26 (n=1) |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.00391 | 0.25 |  |  |
| dropedge |  |  |  |  |
| gat |  |  |  |  |
| pairnorm |  |  |  |  |
| sgc |  |  |  |  |

**photo** -- test

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 92.86 ± 0.39 (n=9) | 92.46 ± 2.76 (n=3) | 34.79 (n=1) | 25.38 (n=1) |
| dropedge | 91.50 (n=1) | 90.84 (n=1) |  |  |
| gat |  |  | 87.44 ± 0.77 (n=9) | 28.34 ± 12.76 (n=3) |
| nhdp | 78.96 ± 4.10 (n=9) | 71.27 ± 33.13 (n=3) | 74.23 (n=1) | 25.38 (n=1) |
| pairnorm |  |  | 90.19 ± 0.74 (n=9) | 63.37 ± 79.97 (n=3) |
| sgc | 44.15 (n=1) | 25.38 (n=1) |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.00391 | 0.25 |  |  |
| dropedge |  |  |  |  |
| gat |  |  |  |  |
| pairnorm |  |  |  |  |
| sgc |  |  |  |  |

**pubmed** -- test

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 85.32 ± 0.21 (n=9) | 83.24 ± 0.98 (n=3) | 48.72 (n=1) | 42.33 (n=1) |
| dropedge | 85.06 (n=1) | 83.90 (n=1) |  |  |
| gat |  |  | 82.15 ± 0.31 (n=9) | 50.18 ± 24.70 (n=3) |
| nhdp | 82.02 ± 1.30 (n=9) | 82.05 ± 2.93 (n=3) | 82.02 (n=1) | 82.93 (n=1) |
| pairnorm |  |  | 82.29 ± 0.39 (n=9) | 81.87 ± 1.11 (n=3) |
| sgc | 81.31 (n=1) | 79.41 (n=1) |  |  |

Wilcoxon p-value vs nhdp (paired by seed)

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.00391 | 0.25 |  |  |
| dropedge |  |  |  |  |
| gat |  |  |  |  |
| pairnorm |  |  |  |  |
| sgc |  |  |  |  |

## embedding.centroid_gap

**citeseer** -- embedding.centroid_gap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.43 ± 0.07 (n=9) | 0.59 ± 0.08 (n=3) | -0.00 (n=1) | -0.00 (n=1) |
| dropedge | 0.33 (n=1) | 0.75 (n=1) |  |  |
| gat |  |  | 0.68 ± 0.05 (n=9) | -0.00 ± 0.00 (n=3) |
| nhdp | 0.53 ± 0.13 (n=9) | 0.59 ± 0.45 (n=3) | 0.28 (n=1) | 0.66 (n=1) |
| pairnorm |  |  | 0.03 ± 0.05 (n=9) | 0.18 ± 0.70 (n=3) |
| sgc | 0.11 (n=1) | 0.19 (n=1) |  |  |

**cora** -- embedding.centroid_gap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.55 ± 0.04 (n=9) | 0.73 ± 0.09 (n=3) | -0.01 (n=1) | -0.00 (n=1) |
| dropedge | 0.83 (n=1) | 0.78 (n=1) |  |  |
| gat |  |  | 0.88 ± 0.04 (n=9) | -0.00 ± 0.00 (n=3) |
| nhdp | 0.59 ± 0.16 (n=9) | 0.71 ± 0.19 (n=3) | 0.68 (n=1) | 0.81 (n=1) |
| pairnorm |  |  | 0.32 ± 0.05 (n=9) | 0.59 ± 0.19 (n=3) |
| sgc | 0.01 (n=1) | -0.00 (n=1) |  |  |

**photo** -- embedding.centroid_gap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.88 ± 0.02 (n=9) | 0.90 ± 0.12 (n=3) | 0.23 (n=1) | -0.00 (n=1) |
| dropedge | 0.86 (n=1) | 0.78 (n=1) |  |  |
| gat |  |  | 0.80 ± 0.05 (n=9) | 0.00 ± 0.00 (n=3) |
| nhdp | 0.46 ± 0.13 (n=9) | 0.39 ± 0.85 (n=3) | 0.32 (n=1) | 0.00 (n=1) |
| pairnorm |  |  | 0.63 ± 0.03 (n=9) | 0.51 ± 1.17 (n=3) |
| sgc | 0.00 (n=1) | -0.00 (n=1) |  |  |

**pubmed** -- embedding.centroid_gap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 1.22 ± 0.02 (n=9) | 1.14 ± 0.14 (n=3) | 0.45 (n=1) | -0.00 (n=1) |
| dropedge | 1.16 (n=1) | 1.18 (n=1) |  |  |
| gat |  |  | 1.15 ± 0.04 (n=9) | 0.00 ± 0.00 (n=3) |
| nhdp | 1.07 ± 0.08 (n=9) | 1.17 ± 0.19 (n=3) | 1.22 (n=1) | 0.65 (n=1) |
| pairnorm |  |  | 1.19 ± 0.03 (n=9) | 1.09 ± 0.13 (n=3) |
| sgc | 1.15 (n=1) | 1.14 (n=1) |  |  |

## embedding.madgap

**citeseer** -- embedding.madgap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.64 ± 0.05 (n=9) | 0.85 ± 0.03 (n=3) | 0.00 (n=1) | -0.00 (n=1) |
| dropedge | 0.57 (n=1) | 0.87 (n=1) |  |  |
| gat |  |  | 0.92 ± 0.03 (n=9) | 0.00 ± 0.00 (n=3) |
| nhdp | 0.61 ± 0.08 (n=9) | 0.63 ± 0.17 (n=3) | 0.48 (n=1) | 0.64 (n=1) |
| pairnorm |  |  | 0.58 ± 0.04 (n=9) | 0.56 ± 0.60 (n=3) |
| sgc | 0.36 (n=1) | 0.44 (n=1) |  |  |

**cora** -- embedding.madgap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.45 ± 0.02 (n=9) | 0.66 ± 0.09 (n=3) | 0.00 (n=1) | -0.00 (n=1) |
| dropedge | 0.60 (n=1) | 0.66 (n=1) |  |  |
| gat |  |  | 0.76 ± 0.05 (n=9) | 0.00 ± 0.00 (n=3) |
| nhdp | 0.39 ± 0.07 (n=9) | 0.41 ± 0.08 (n=3) | 0.44 (n=1) | 0.50 (n=1) |
| pairnorm |  |  | 0.46 ± 0.04 (n=9) | 0.64 ± 0.30 (n=3) |
| sgc | 0.01 (n=1) | 0.01 (n=1) |  |  |

**photo** -- embedding.madgap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.31 ± 0.03 (n=9) | 0.39 ± 0.17 (n=3) | 0.06 (n=1) | 0.00 (n=1) |
| dropedge | 0.44 (n=1) | 0.48 (n=1) |  |  |
| gat |  |  | 0.45 ± 0.07 (n=9) | 0.00 ± 0.01 (n=3) |
| nhdp | 0.21 ± 0.05 (n=9) | 0.21 ± 0.38 (n=3) | 0.19 (n=1) | 0.00 (n=1) |
| pairnorm |  |  | 0.25 ± 0.02 (n=9) | 0.65 ± 0.93 (n=3) |
| sgc | 0.00 (n=1) | -0.00 (n=1) |  |  |

**pubmed** -- embedding.madgap

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.85 ± 0.01 (n=9) | 0.87 ± 0.08 (n=3) | 0.62 (n=1) | -0.00 (n=1) |
| dropedge | 0.82 (n=1) | 0.92 (n=1) |  |  |
| gat |  |  | 0.94 ± 0.03 (n=9) | 0.00 ± 0.00 (n=3) |
| nhdp | 0.65 ± 0.03 (n=9) | 0.69 ± 0.16 (n=3) | 0.72 (n=1) | 0.55 (n=1) |
| pairnorm |  |  | 0.90 ± 0.02 (n=9) | 0.85 ± 0.08 (n=3) |
| sgc | 0.78 (n=1) | 0.83 (n=1) |  |  |

## timings.epoch_s_mean (seconds per training epoch (one thread))

**citeseer** -- timings.epoch_s_mean

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.62 ± 0.06 (n=9) | 0.71 ± 0.28 (n=3) | 0.60 (n=1) | 0.71 (n=1) |
| dropedge | 0.25 (n=1) | 0.27 (n=1) |  |  |
| gat |  |  | 0.42 ± 0.04 (n=9) | 0.63 ± 0.21 (n=3) |
| nhdp | 0.35 ± 0.04 (n=9) | 0.43 ± 0.19 (n=3) | 0.40 (n=1) | 0.55 (n=1) |
| pairnorm |  |  | 0.39 ± 0.04 (n=9) | 0.53 ± 0.20 (n=3) |
| sgc | 0.48 (n=1) | 0.94 (n=1) |  |  |

**cora** -- timings.epoch_s_mean

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.19 ± 0.02 (n=9) | 0.25 ± 0.12 (n=3) | 0.25 (n=1) | 0.35 (n=1) |
| dropedge | 0.07 (n=1) | 0.09 (n=1) |  |  |
| gat |  |  | 0.20 ± 0.02 (n=9) | 0.38 ± 0.08 (n=3) |
| nhdp | 0.12 ± 0.01 (n=9) | 0.17 ± 0.06 (n=3) | 0.20 (n=1) | 0.31 (n=1) |
| pairnorm |  |  | 0.16 ± 0.01 (n=9) | 0.26 ± 0.06 (n=3) |
| sgc | 0.18 (n=1) | 0.37 (n=1) |  |  |

**photo** -- timings.epoch_s_mean

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.54 ± 0.07 (n=9) | 1.19 ± 0.77 (n=3) | 1.94 (n=1) | 3.75 (n=1) |
| dropedge | 0.25 (n=1) | 0.38 (n=1) |  |  |
| gat |  |  | 2.22 ± 0.25 (n=9) | 5.86 ± 1.56 (n=3) |
| nhdp | 0.75 ± 0.07 (n=9) | 1.50 ± 0.54 (n=3) | 2.36 (n=1) | 4.86 (n=1) |
| pairnorm |  |  | 1.51 ± 0.16 (n=9) | 3.28 ± 0.85 (n=3) |
| sgc | 1.70 (n=1) | 3.26 (n=1) |  |  |

**pubmed** -- timings.epoch_s_mean

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 0.64 ± 0.06 (n=9) | 1.07 ± 0.48 (n=3) | 1.22 (n=1) | 2.04 (n=1) |
| dropedge | 0.24 (n=1) | 0.34 (n=1) |  |  |
| gat |  |  | 1.31 ± 0.10 (n=9) | 2.50 ± 0.31 (n=3) |
| nhdp | 0.58 ± 0.05 (n=9) | 0.95 ± 0.29 (n=3) | 1.10 (n=1) | 1.94 (n=1) |
| pairnorm |  |  | 1.25 ± 0.12 (n=9) | 2.42 ± 0.65 (n=3) |
| sgc | 0.56 (n=1) | 1.09 (n=1) |  |  |

## peak_rss_mb (peak resident memory of the run's process in MB)

**citeseer** -- peak_rss_mb

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 803.89 ± 0.93 (n=9) | 813.52 ± 3.01 (n=3) | 859.03 (n=1) | 942.98 (n=1) |
| dropedge | 753.79 (n=1) | 758.49 (n=1) |  |  |
| gat |  |  | 863.06 ± 2.67 (n=9) | 989.18 ± 9.00 (n=3) |
| nhdp | 823.77 ± 1.99 (n=9) | 852.86 ± 15.41 (n=3) | 959.48 (n=1) | 1102.36 (n=1) |
| pairnorm |  |  | 809.88 ± 2.38 (n=9) | 883.80 ± 15.63 (n=3) |
| sgc | 1088.08 (n=1) | 1088.18 (n=1) |  |  |

**cora** -- peak_rss_mb

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 670.81 ± 3.79 (n=9) | 675.77 ± 23.95 (n=3) | 712.52 (n=1) | 767.93 (n=1) |
| dropedge | 633.70 (n=1) | 638.17 (n=1) |  |  |
| gat |  |  | 722.02 ± 4.37 (n=9) | 847.40 ± 30.28 (n=3) |
| nhdp | 669.08 ± 3.73 (n=9) | 689.15 ± 10.75 (n=3) | 784.89 (n=1) | 939.55 (n=1) |
| pairnorm |  |  | 670.97 ± 4.78 (n=9) | 728.90 ± 19.20 (n=3) |
| sgc | 750.74 (n=1) | 780.42 (n=1) |  |  |

**photo** -- peak_rss_mb

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 826.93 ± 7.03 (n=9) | 864.43 ± 56.85 (n=3) | 1037.25 (n=1) | 1178.29 (n=1) |
| dropedge | 785.21 (n=1) | 804.59 (n=1) |  |  |
| gat |  |  | 1870.52 ± 13.95 (n=9) | 3107.01 ± 82.91 (n=3) |
| nhdp | 1105.25 ± 4.99 (n=9) | 1322.19 ± 36.68 (n=3) | 2063.33 (n=1) | 3222.01 (n=1) |
| pairnorm |  |  | 961.71 ± 5.49 (n=9) | 1177.11 ± 7.33 (n=3) |
| sgc | 2056.48 (n=1) | 2074.26 (n=1) |  |  |

**pubmed** -- peak_rss_mb

| model | 2 | 4 | 10 | 20 |
|---|---|---|---|---|
| arma | 1004.41 ± 3.29 (n=9) | 1023.61 ± 4.74 (n=3) | 1358.20 (n=1) | 1860.35 (n=1) |
| dropedge | 1002.18 (n=1) | 1024.09 (n=1) |  |  |
| gat |  |  | 1610.20 ± 27.37 (n=9) | 2586.96 ± 102.58 (n=3) |
| nhdp | 1138.30 ± 17.18 (n=9) | 1278.10 ± 39.35 (n=3) | 1916.45 (n=1) | 2780.52 (n=1) |
| pairnorm |  |  | 1179.27 ± 26.89 (n=9) | 1540.02 ± 119.46 (n=3) |
| sgc | 1121.03 (n=1) | 1121.82 (n=1) |  |  |

## Missing runs

Not attempted, or stopped by the job's deadline.

| key | dataset | model | tag | depth | seed |
|---|---|---|---|---|---|
| 62d3bd26eb91 | cora | nhdp |  | 2 | 7 |
| a37a2df0a41f | cora | nhdp |  | 4 | 1 |
| acc7bd307c40 | cora | nhdp |  | 4 | 3 |
| 086365ba4605 | cora | nhdp |  | 4 | 4 |
| 8c2d75992a6d | cora | nhdp |  | 4 | 5 |
| 22bbc8be9289 | cora | nhdp |  | 4 | 6 |
| f9b524ab80df | cora | nhdp |  | 4 | 7 |
| fa79581d0c5b | cora | nhdp |  | 4 | 9 |
| 79c2c7a5fa6a | cora | nhdp |  | 8 | 0 |
| 1ad15eed8069 | cora | nhdp |  | 8 | 1 |
| d4cab6cb2e27 | cora | nhdp |  | 8 | 2 |
| 1e04a64dbf63 | cora | nhdp |  | 8 | 3 |
| d533ea86294c | cora | nhdp |  | 8 | 4 |
| 737daf6720fd | cora | nhdp |  | 8 | 5 |
| 7ec60fcb2f64 | cora | nhdp |  | 8 | 6 |
| da90f52f2f1f | cora | nhdp |  | 8 | 7 |
| 963f58d4b323 | cora | nhdp |  | 8 | 8 |
| 08c64757fe53 | cora | nhdp |  | 8 | 9 |
| b3d8963d4fdb | cora | nhdp |  | 10 | 0 |
| 0c28e08dc5a4 | cora | nhdp |  | 10 | 1 |
| 9b5402f8793b | cora | nhdp |  | 10 | 3 |
| 109a05aa0a32 | cora | nhdp |  | 10 | 4 |
| 37157e64020c | cora | nhdp |  | 10 | 5 |
| 67487f5febe6 | cora | nhdp |  | 10 | 6 |
| 53ad0e7a4873 | cora | nhdp |  | 10 | 7 |
| 75fdb5319064 | cora | nhdp |  | 10 | 8 |
| 0aac2c85de2d | cora | nhdp |  | 10 | 9 |
| 42fdffd8e140 | cora | nhdp |  | 20 | 0 |
| 600008579d74 | cora | nhdp |  | 20 | 1 |
| e2a82434d56c | cora | nhdp |  | 20 | 2 |
| c8f5b7d380b3 | cora | nhdp |  | 20 | 3 |
| 051852add2d0 | cora | nhdp |  | 20 | 4 |
| 9407c6939271 | cora | nhdp |  | 20 | 6 |
| 1447953c9efb | cora | nhdp |  | 20 | 7 |
| ba94b6718d99 | cora | nhdp |  | 20 | 8 |
| 9938b537a627 | cora | nhdp |  | 20 | 9 |
| a1c05f8f240a | cora | nhdp |  | 30 | 0 |
| 2ef4fc177c5f | cora | nhdp |  | 30 | 1 |
| d16f370ef091 | cora | nhdp |  | 30 | 2 |
| 9fab89438b6c | cora | nhdp |  | 30 | 3 |
| 73f30c841d9c | cora | nhdp |  | 30 | 4 |
| 2955bcb902f6 | cora | nhdp |  | 30 | 5 |
| 6eda447db521 | cora | nhdp |  | 30 | 6 |
| eaf0c90c7758 | cora | nhdp |  | 30 | 7 |
| 5bd96983565d | cora | nhdp |  | 30 | 8 |
| 60d402c3586d | cora | nhdp |  | 30 | 9 |
| b3f20541e820 | cora | gcn |  | 2 | 0 |
| ce03838c1fab | cora | gcn |  | 2 | 1 |
| e05b84cacd09 | cora | gcn |  | 2 | 2 |
| 4b499505d738 | cora | gcn |  | 2 | 3 |
| f2dffc2e9596 | cora | gcn |  | 2 | 4 |
| f64e1e470c8a | cora | gcn |  | 2 | 5 |
| c9c14e137dd3 | cora | gcn |  | 2 | 6 |
| d69fdb1d7020 | cora | gcn |  | 2 | 7 |
| cff2c23137be | cora | gcn |  | 2 | 8 |
| 870c659a1f5b | cora | gcn |  | 2 | 9 |
| 1386dcd48f05 | cora | gcn |  | 4 | 0 |
| 15c00c2d4e89 | cora | gcn |  | 4 | 1 |
| a1fef21d46f9 | cora | gcn |  | 4 | 2 |
| 083e5520b414 | cora | gcn |  | 4 | 3 |
| d0dd92811703 | cora | gcn |  | 4 | 4 |
| 9fb18e5d7ffa | cora | gcn |  | 4 | 5 |
| 3be56c1e586a | cora | gcn |  | 4 | 6 |
| 4b66a0b7dc4e | cora | gcn |  | 4 | 7 |
| e5f4707c43b6 | cora | gcn |  | 4 | 8 |
| 0b463ad257a5 | cora | gcn |  | 4 | 9 |
| 8bb6b23864de | cora | gcn |  | 8 | 0 |
| 24a4037b1c27 | cora | gcn |  | 8 | 1 |
| 415dbc0130ff | cora | gcn |  | 8 | 2 |
| f0a61e824566 | cora | gcn |  | 8 | 3 |
| 96dda86dd20e | cora | gcn |  | 8 | 4 |
| cf4d08b7b4eb | cora | gcn |  | 8 | 5 |
| 05976f789015 | cora | gcn |  | 8 | 6 |
| 47bd3b211a53 | cora | gcn |  | 8 | 7 |
| 38b95ae09cc8 | cora | gcn |  | 8 | 8 |
| 59610f14bce8 | cora | gcn |  | 8 | 9 |
| f7ee7c077a8e | cora | gcn |  | 10 | 0 |
| ce0064ae9a42 | cora | gcn |  | 10 | 1 |
| b35d4bb22337 | cora | gcn |  | 10 | 2 |
| 8c50e48e9c11 | cora | gcn |  | 10 | 3 |
| 08cd8a50d475 | cora | gcn |  | 10 | 4 |
| 0bed2e04ff46 | cora | gcn |  | 10 | 5 |
| d58607a46efd | cora | gcn |  | 10 | 6 |
| 73b7c89f3340 | cora | gcn |  | 10 | 7 |
| 7873908a3f08 | cora | gcn |  | 10 | 8 |
| 55ff6779e2a0 | cora | gcn |  | 10 | 9 |
| 3a5096fe1439 | cora | gcn |  | 20 | 0 |
| 3740c383e728 | cora | gcn |  | 20 | 1 |
| 1bc0c5873534 | cora | gcn |  | 20 | 2 |
| 764bbd0dc6e9 | cora | gcn |  | 20 | 3 |
| 83f0f2907782 | cora | gcn |  | 20 | 4 |
| 49b3a2924169 | cora | gcn |  | 20 | 5 |
| 66f3e3f75efb | cora | gcn |  | 20 | 6 |
| 0cb1b0071b5e | cora | gcn |  | 20 | 7 |
| 30b8e6770fb4 | cora | gcn |  | 20 | 8 |
| ba944dbfe589 | cora | gcn |  | 20 | 9 |
| 1cee2856d1e2 | cora | gcn |  | 30 | 0 |
| 416629008173 | cora | gcn |  | 30 | 1 |
| 11c14fb04ced | cora | gcn |  | 30 | 2 |
| 63a64f9e8eea | cora | gcn |  | 30 | 3 |
| 9e0d02d0c3be | cora | gcn |  | 30 | 4 |
| 0ab1f1c03a6b | cora | gcn |  | 30 | 5 |
| dd2cb5b2af78 | cora | gcn |  | 30 | 6 |
| 7f06dc1a9b3b | cora | gcn |  | 30 | 7 |
| ffe39b8cbb84 | cora | gcn |  | 30 | 8 |
| f0b0c886b48d | cora | gcn |  | 30 | 9 |
| 68dc4c0bfd3c | cora | gat |  | 2 | 0 |
| f32d14a6ccac | cora | gat |  | 2 | 1 |
| f4ac32b9e8eb | cora | gat |  | 2 | 2 |
| c835db1d8e54 | cora | gat |  | 2 | 3 |
| 82601d39d9de | cora | gat |  | 2 | 4 |
| 9d6b8ec06924 | cora | gat |  | 2 | 5 |
| 0b2d6ad5ab58 | cora | gat |  | 2 | 6 |
| 0453f148d6db | cora | gat |  | 2 | 7 |
| 13f8c449f603 | cora | gat |  | 2 | 8 |
| 5d6184d9070f | cora | gat |  | 2 | 9 |
| 8bc55fe99191 | cora | gat |  | 4 | 0 |
| 0bfdb07941b9 | cora | gat |  | 4 | 1 |
| 065cd16d1c0e | cora | gat |  | 4 | 2 |
| c72d2a5fe767 | cora | gat |  | 4 | 3 |
| 54b98d738237 | cora | gat |  | 4 | 4 |
| 606eb13cb4de | cora | gat |  | 4 | 5 |
| 3a3bf7ebf159 | cora | gat |  | 4 | 6 |
| 7a5bc8b9411a | cora | gat |  | 4 | 7 |
| 31396dc5378e | cora | gat |  | 4 | 8 |
| 283b3d60eb90 | cora | gat |  | 4 | 9 |
| ee07e5f014b4 | cora | gat |  | 8 | 0 |
| b1054a359055 | cora | gat |  | 8 | 1 |
| 8fe04a7170fe | cora | gat |  | 8 | 2 |
| f9ea57fcb9ca | cora | gat |  | 8 | 3 |
| 143aa86939dd | cora | gat |  | 8 | 4 |
| 2461853e3ae8 | cora | gat |  | 8 | 5 |
| 327a66cc0c93 | cora | gat |  | 8 | 6 |
| 640f97ad055f | cora | gat |  | 8 | 7 |
| c01e04eedec7 | cora | gat |  | 8 | 8 |
| 3e64ffda2686 | cora | gat |  | 8 | 9 |
| 4ca0cb0187c0 | cora | gat |  | 10 | 7 |
| e7e64b637595 | cora | gat |  | 20 | 1 |
| eac443b0a978 | cora | gat |  | 20 | 3 |
| c0d2f58a678a | cora | gat |  | 20 | 4 |
| 9bb239082272 | cora | gat |  | 20 | 5 |
| 1f5c316896b1 | cora | gat |  | 20 | 6 |
| 999b39b1cf6d | cora | gat |  | 20 | 7 |
| d5444506b5f7 | cora | gat |  | 20 | 9 |
| 19a34f71c3db | cora | gat |  | 30 | 0 |
| eb59a60c7bb2 | cora | gat |  | 30 | 1 |
| 64599a6e6be9 | cora | gat |  | 30 | 2 |
| d8368dd2ea93 | cora | gat |  | 30 | 3 |
| 61a9375ae517 | cora | gat |  | 30 | 4 |
| f0c12cda7c9c | cora | gat |  | 30 | 5 |
| a4dfaccdd36c | cora | gat |  | 30 | 6 |
| 52588dc1a0d1 | cora | gat |  | 30 | 7 |
| 1e51f93c8893 | cora | gat |  | 30 | 8 |
| c4c6aac4f06a | cora | gat |  | 30 | 9 |
| 63732830b70d | cora | sgc |  | 2 | 0 |
| 6026f8db0e54 | cora | sgc |  | 2 | 1 |
| 9ca4f578ab4a | cora | sgc |  | 2 | 3 |
| 0034b2895cd6 | cora | sgc |  | 2 | 4 |
| 45d19485199c | cora | sgc |  | 2 | 5 |
| 318237de4c08 | cora | sgc |  | 2 | 6 |
| 31cf15e47e4d | cora | sgc |  | 2 | 7 |
| c89e6e643526 | cora | sgc |  | 2 | 8 |
| c8094bbffcb4 | cora | sgc |  | 2 | 9 |
| 1533afad00ef | cora | sgc |  | 4 | 0 |
| a1f85e2b1b45 | cora | sgc |  | 4 | 1 |
| 112d91582c3c | cora | sgc |  | 4 | 2 |
| 35e3aedbb2ff | cora | sgc |  | 4 | 3 |
| e880107ea2ad | cora | sgc |  | 4 | 4 |
| eef8972817d1 | cora | sgc |  | 4 | 6 |
| bd7d66de3fc3 | cora | sgc |  | 4 | 7 |
| 52a4fe299003 | cora | sgc |  | 4 | 8 |
| 3fcab847666a | cora | sgc |  | 4 | 9 |
| ffd854138b65 | cora | sgc |  | 8 | 0 |
| c37255bd6129 | cora | sgc |  | 8 | 1 |
| 6d62c91242d2 | cora | sgc |  | 8 | 2 |
| 859435cc2586 | cora | sgc |  | 8 | 3 |
| 371044b312e6 | cora | sgc |  | 8 | 4 |
| 2a1d7993c863 | cora | sgc |  | 8 | 5 |
| 95e08e9c873d | cora | sgc |  | 8 | 6 |
| 83f0d924db9a | cora | sgc |  | 8 | 7 |
| 679ddbcd4e9f | cora | sgc |  | 8 | 8 |
| 2fa39c55b0c0 | cora | sgc |  | 8 | 9 |
| bd6bf218d414 | cora | sgc |  | 10 | 0 |
| 486c6849aa22 | cora | sgc |  | 10 | 1 |
| 2b6ae66ec4b8 | cora | sgc |  | 10 | 2 |
| b73cb3472fb5 | cora | sgc |  | 10 | 3 |
| cd0f3c3a1858 | cora | sgc |  | 10 | 4 |
| 18ced1f0218c | cora | sgc |  | 10 | 5 |
| c902c1302801 | cora | sgc |  | 10 | 6 |
| 5fa4a0aa6ee2 | cora | sgc |  | 10 | 7 |
| d35a65fac5a9 | cora | sgc |  | 10 | 8 |
| 6896ffac0dc0 | cora | sgc |  | 10 | 9 |
| d85bb2a845c6 | cora | sgc |  | 20 | 0 |
| 049687fbfbac | cora | sgc |  | 20 | 1 |
| 79d358c213b7 | cora | sgc |  | 20 | 2 |
| 046f0b0e239f | cora | sgc |  | 20 | 3 |
| d8e7fa29c424 | cora | sgc |  | 20 | 4 |
| ffdb41c44dda | cora | sgc |  | 20 | 5 |
| 3de2d09cd558 | cora | sgc |  | 20 | 6 |
| 21a3bca0b44d | cora | sgc |  | 20 | 7 |
| fcf8092547f3 | cora | sgc |  | 20 | 8 |
| 70b291a8bbd8 | cora | sgc |  | 20 | 9 |
| e1881c48488a | cora | sgc |  | 30 | 0 |
| 7de4f74ec19c | cora | sgc |  | 30 | 1 |
| 64c877a7815e | cora | sgc |  | 30 | 2 |
| 445aaaa99a33 | cora | sgc |  | 30 | 3 |
| 5f7867452167 | cora | sgc |  | 30 | 4 |
| 81405d2c2f56 | cora | sgc |  | 30 | 5 |
| 712a38465787 | cora | sgc |  | 30 | 6 |
| 5790be21ae71 | cora | sgc |  | 30 | 7 |
| b69f968747d2 | cora | sgc |  | 30 | 8 |
| 62b0466b952a | cora | sgc |  | 30 | 9 |
| 4eca401a6f1f | cora | appnp |  | 2 | 0 |
| 133c0328018a | cora | appnp |  | 2 | 1 |
| f37a06d2e498 | cora | appnp |  | 2 | 2 |
| ca2a203c03a9 | cora | appnp |  | 2 | 3 |
| 89fcb19a469f | cora | appnp |  | 2 | 4 |
| e9b1996b650b | cora | appnp |  | 2 | 5 |
| a6dd016ca806 | cora | appnp |  | 2 | 6 |
| 37736fffa226 | cora | appnp |  | 2 | 7 |
| ca91fab11034 | cora | appnp |  | 2 | 8 |
| bc0539860543 | cora | appnp |  | 2 | 9 |
| e94c62df47ed | cora | appnp |  | 4 | 0 |
| 65b7b443911e | cora | appnp |  | 4 | 1 |
| 9a6eeb7d3c9f | cora | appnp |  | 4 | 2 |
| 9d0f774ba963 | cora | appnp |  | 4 | 3 |
| 982eb99da881 | cora | appnp |  | 4 | 4 |
| e2518a5c9693 | cora | appnp |  | 4 | 5 |
| 17d60e3193d6 | cora | appnp |  | 4 | 6 |
| 45fdc139bc8b | cora | appnp |  | 4 | 7 |
| bd01230effdf | cora | appnp |  | 4 | 8 |
| c4f70470d767 | cora | appnp |  | 4 | 9 |
| 10f7fdc3b424 | cora | appnp |  | 8 | 0 |
| e92e63d4cb32 | cora | appnp |  | 8 | 1 |
| 33dbcb0dd6ed | cora | appnp |  | 8 | 2 |
| 5bd0e8b32a61 | cora | appnp |  | 8 | 3 |
| ba19c8998e0c | cora | appnp |  | 8 | 4 |
| 7b19276d0fa4 | cora | appnp |  | 8 | 5 |
| 817cc6d42efd | cora | appnp |  | 8 | 6 |
| f07750e2a70c | cora | appnp |  | 8 | 7 |
| e53ba7392394 | cora | appnp |  | 8 | 8 |
| 8c9b5c17c395 | cora | appnp |  | 8 | 9 |
| d90252dc1b09 | cora | appnp |  | 10 | 0 |
| 10fbf8007707 | cora | appnp |  | 10 | 1 |
| 44e0a2dff9de | cora | appnp |  | 10 | 2 |
| 3be52c374610 | cora | appnp |  | 10 | 3 |
| c769f87a8ba4 | cora | appnp |  | 10 | 4 |
| 1bb3fb407a7b | cora | appnp |  | 10 | 5 |
| c5bd636116f2 | cora | appnp |  | 10 | 6 |
| 9d0a11627470 | cora | appnp |  | 10 | 7 |
| 3e9ce08c53b9 | cora | appnp |  | 10 | 8 |
| 73f79eba6718 | cora | appnp |  | 10 | 9 |
| 15c359b46128 | cora | appnp |  | 20 | 0 |
| 07dfac20450d | cora | appnp |  | 20 | 1 |
| 90f4223ec03e | cora | appnp |  | 20 | 2 |
| 70231bec1ada | cora | appnp |  | 20 | 3 |
| c30739060503 | cora | appnp |  | 20 | 4 |
| 00f9a5858054 | cora | appnp |  | 20 | 5 |
| 5ed38e126d70 | cora | appnp |  | 20 | 6 |
| 905c4a89d73a | cora | appnp |  | 20 | 7 |
| 00552edcd63d | cora | appnp |  | 20 | 8 |
| 2e020996fb17 | cora | appnp |  | 20 | 9 |
| c59c523d5748 | cora | appnp |  | 30 | 0 |
| b9eb90e4ae23 | cora | appnp |  | 30 | 1 |
| 75048ae59293 | cora | appnp |  | 30 | 2 |
| 53ad24aac63c | cora | appnp |  | 30 | 3 |
| 4602af34a6ed | cora | appnp |  | 30 | 4 |
| 28d159f884a4 | cora | appnp |  | 30 | 5 |
| 3df69c54f9a5 | cora | appnp |  | 30 | 6 |
| 367a91decb54 | cora | appnp |  | 30 | 7 |
| 2099e61bbc3d | cora | appnp |  | 30 | 8 |
| 3ef6b3a8041c | cora | appnp |  | 30 | 9 |
| 3525c6f695ba | cora | arma |  | 2 | 7 |
| fc42cc00bb4d | cora | arma |  | 4 | 1 |
| bf5bd7bf2981 | cora | arma |  | 4 | 3 |
| 25c5a2d8f15c | cora | arma |  | 4 | 4 |
| dbe3444a3411 | cora | arma |  | 4 | 5 |
| dbbff6da1747 | cora | arma |  | 4 | 6 |
| 603413a4a53a | cora | arma |  | 4 | 7 |
| ae9c22d772a7 | cora | arma |  | 4 | 9 |
| 044c30a80f1c | cora | arma |  | 8 | 0 |
| 570fa6918d20 | cora | arma |  | 8 | 1 |
| fa6d622ed536 | cora | arma |  | 8 | 2 |
| 99a5dfd822bd | cora | arma |  | 8 | 3 |
| 0099c06b3729 | cora | arma |  | 8 | 4 |
| e30ae07fbb65 | cora | arma |  | 8 | 5 |
| bcf9fc12e9bd | cora | arma |  | 8 | 6 |
| fead91e44a58 | cora | arma |  | 8 | 7 |
| a5f04c29f902 | cora | arma |  | 8 | 8 |
| c1b321d42229 | cora | arma |  | 8 | 9 |
| 84c2d160b269 | cora | arma |  | 10 | 0 |
| e87a2d1b8df9 | cora | arma |  | 10 | 1 |
| 3f16c4d1078b | cora | arma |  | 10 | 3 |
| 9eafd1404701 | cora | arma |  | 10 | 4 |
| 0bbd1e8aefa3 | cora | arma |  | 10 | 5 |
| 0d9ddef4a2d2 | cora | arma |  | 10 | 6 |
| 8e0f8f6f4440 | cora | arma |  | 10 | 7 |
| a7a0d632fd59 | cora | arma |  | 10 | 8 |
| 771233136a0b | cora | arma |  | 10 | 9 |
| a1b286f91d14 | cora | arma |  | 20 | 0 |
| 76f9eab899dc | cora | arma |  | 20 | 1 |
| 77a9bef2235b | cora | arma |  | 20 | 2 |
| 6e3cad0ee543 | cora | arma |  | 20 | 3 |
| eff3574fad37 | cora | arma |  | 20 | 4 |
| 5fa9cee18bb1 | cora | arma |  | 20 | 6 |
| 3909b6749d4d | cora | arma |  | 20 | 7 |
| 286e9a592ba6 | cora | arma |  | 20 | 8 |
| f48092dd1640 | cora | arma |  | 20 | 9 |
| 3570ec1b71c4 | cora | arma |  | 30 | 0 |
| 8632e081280d | cora | arma |  | 30 | 1 |
| 05cb8fbf30fb | cora | arma |  | 30 | 2 |
| 8c637f293a51 | cora | arma |  | 30 | 3 |
| 1b24ff5ff12c | cora | arma |  | 30 | 4 |
| 5021b81dcad2 | cora | arma |  | 30 | 5 |
| 0df6f75c44f8 | cora | arma |  | 30 | 6 |
| 3a1b7aba1cc3 | cora | arma |  | 30 | 7 |
| 7a0b7d90ebab | cora | arma |  | 30 | 8 |
| 17b5a96c392e | cora | arma |  | 30 | 9 |
| 5e72832b4968 | cora | gcnii |  | 2 | 0 |
| 55585dcb6d9b | cora | gcnii |  | 2 | 1 |
| 5c9854f301e9 | cora | gcnii |  | 2 | 2 |
| 82aabda6ceed | cora | gcnii |  | 2 | 3 |
| 381f8f2844d4 | cora | gcnii |  | 2 | 4 |
| 6b2d7dcfd6eb | cora | gcnii |  | 2 | 5 |
| ed2e68c044b9 | cora | gcnii |  | 2 | 6 |
| 68feca6daa52 | cora | gcnii |  | 2 | 7 |
| 2fd479fe7e85 | cora | gcnii |  | 2 | 8 |
| 4a586af607cc | cora | gcnii |  | 2 | 9 |
| 1de7813eb820 | cora | gcnii |  | 4 | 0 |
| 40dc2bfe9f5c | cora | gcnii |  | 4 | 1 |
| 2df4f91ba539 | cora | gcnii |  | 4 | 2 |
| 06a0ffbd67f4 | cora | gcnii |  | 4 | 3 |
| 6935832f59c8 | cora | gcnii |  | 4 | 4 |
| e3fc1e8c0508 | cora | gcnii |  | 4 | 5 |
| a4bb23b32889 | cora | gcnii |  | 4 | 6 |
| ce918e9d1bd9 | cora | gcnii |  | 4 | 7 |
| f1d16d08ed65 | cora | gcnii |  | 4 | 8 |
| e273132cf697 | cora | gcnii |  | 4 | 9 |
| bb95be297055 | cora | gcnii |  | 8 | 0 |
| dbf9b232081e | cora | gcnii |  | 8 | 1 |
| 833b39c2cc31 | cora | gcnii |  | 8 | 2 |
| b7272e2c6da8 | cora | gcnii |  | 8 | 3 |
| ea6ca1c85ea4 | cora | gcnii |  | 8 | 4 |
| 49457d037bdf | cora | gcnii |  | 8 | 5 |
| 04671a3d3b05 | cora | gcnii |  | 8 | 6 |
| dd07fd4a278f | cora | gcnii |  | 8 | 7 |
| 0b59c270cae7 | cora | gcnii |  | 8 | 8 |
| e3024199389f | cora | gcnii |  | 8 | 9 |
| 3c94b4d00431 | cora | gcnii |  | 10 | 0 |
| ec413767eaee | cora | gcnii |  | 10 | 1 |
| 2d0d71260841 | cora | gcnii |  | 10 | 2 |
| af6a7f2ef137 | cora | gcnii |  | 10 | 3 |
| a16cae92c9ca | cora | gcnii |  | 10 | 4 |
| c8a02f125d79 | cora | gcnii |  | 10 | 5 |
| 79a96bb73036 | cora | gcnii |  | 10 | 6 |
| 894b0f7594a2 | cora | gcnii |  | 10 | 7 |
| ced6c358c802 | cora | gcnii |  | 10 | 8 |
| c712edd97992 | cora | gcnii |  | 10 | 9 |
| 5ffaeffdf8a6 | cora | gcnii |  | 20 | 0 |
| a56dc64b1acf | cora | gcnii |  | 20 | 1 |
| c07849113bd8 | cora | gcnii |  | 20 | 2 |
| 5a6d041be9b5 | cora | gcnii |  | 20 | 3 |
| 14c5a2463b67 | cora | gcnii |  | 20 | 4 |
| d8f772070a49 | cora | gcnii |  | 20 | 5 |
| 3c7e96194755 | cora | gcnii |  | 20 | 6 |
| aa075735814d | cora | gcnii |  | 20 | 7 |
| 8666146da14b | cora | gcnii |  | 20 | 8 |
| 85cdeb82b0f0 | cora | gcnii |  | 20 | 9 |
| 617fc411f954 | cora | gcnii |  | 30 | 0 |
| cd104a793a0a | cora | gcnii |  | 30 | 1 |
| 4678c5faa557 | cora | gcnii |  | 30 | 2 |
| f887fb19f98e | cora | gcnii |  | 30 | 3 |
| 09ffc26c1bec | cora | gcnii |  | 30 | 4 |
| c238d897a4e2 | cora | gcnii |  | 30 | 5 |
| 553b3ba5549e | cora | gcnii |  | 30 | 6 |
| 53230a6bd736 | cora | gcnii |  | 30 | 7 |
| bfb2fcceb764 | cora | gcnii |  | 30 | 8 |
| 40b15587603a | cora | gcnii |  | 30 | 9 |
| f56b383c461e | cora | pairnorm |  | 2 | 0 |
| bc3a7a9537d6 | cora | pairnorm |  | 2 | 1 |
| 1246e09b1fe5 | cora | pairnorm |  | 2 | 2 |
| b3392789a36a | cora | pairnorm |  | 2 | 3 |
| 98688d572f49 | cora | pairnorm |  | 2 | 4 |
| c8cd9ef3fb3e | cora | pairnorm |  | 2 | 5 |
| 5dc5e2ffebc5 | cora | pairnorm |  | 2 | 6 |
| 964eb5dfd543 | cora | pairnorm |  | 2 | 7 |
| 71e2a311da71 | cora | pairnorm |  | 2 | 8 |
| 2e5dd31a288f | cora | pairnorm |  | 2 | 9 |
| 0595750f301c | cora | pairnorm |  | 4 | 0 |
| 965f5786945b | cora | pairnorm |  | 4 | 1 |
| 792b0c847fd2 | cora | pairnorm |  | 4 | 2 |
| 4dc4fdb9cb6b | cora | pairnorm |  | 4 | 3 |
| 9fc0a917d781 | cora | pairnorm |  | 4 | 4 |
| 9cc3b1a3ef5b | cora | pairnorm |  | 4 | 5 |
| 1d45e2fd2662 | cora | pairnorm |  | 4 | 6 |
| 71359bad7c41 | cora | pairnorm |  | 4 | 7 |
| cd98f4f32610 | cora | pairnorm |  | 4 | 8 |
| 1ff775668402 | cora | pairnorm |  | 4 | 9 |
| 54f7e269cd84 | cora | pairnorm |  | 8 | 0 |
| 17da809ae321 | cora | pairnorm |  | 8 | 1 |
| bc4b1c98f73c | cora | pairnorm |  | 8 | 2 |
| 17ec8df43098 | cora | pairnorm |  | 8 | 3 |
| 867394ec8a49 | cora | pairnorm |  | 8 | 4 |
| b458b2b82a1e | cora | pairnorm |  | 8 | 5 |
| 8e74ad0ed36a | cora | pairnorm |  | 8 | 6 |
| b721d33e19ce | cora | pairnorm |  | 8 | 7 |
| ad9c4b6ec1b7 | cora | pairnorm |  | 8 | 8 |
| 9d8e9cb323f5 | cora | pairnorm |  | 8 | 9 |
| 7d851a6594d5 | cora | pairnorm |  | 10 | 7 |
| 71c99ccb33d7 | cora | pairnorm |  | 20 | 1 |
| 8077a9d7d038 | cora | pairnorm |  | 20 | 3 |
| 3480afca5a5d | cora | pairnorm |  | 20 | 4 |
| 2f1fc1325450 | cora | pairnorm |  | 20 | 5 |
| 25f1662c9d9d | cora | pairnorm |  | 20 | 6 |
| 49e8de9d05a4 | cora | pairnorm |  | 20 | 7 |
| d38ced0d54e0 | cora | pairnorm |  | 20 | 9 |
| 3ae02b4ec392 | cora | pairnorm |  | 30 | 0 |
| 690f61fde928 | cora | pairnorm |  | 30 | 1 |
| 47607ffd9d23 | cora | pairnorm |  | 30 | 2 |
| 7c28ad9ff1c6 | cora | pairnorm |  | 30 | 3 |
| 935442599925 | cora | pairnorm |  | 30 | 4 |
| 87270082a3a0 | cora | pairnorm |  | 30 | 5 |
| 06d71330f283 | cora | pairnorm |  | 30 | 6 |
| fa2984d2aaf2 | cora | pairnorm |  | 30 | 7 |
| 73338934475d | cora | pairnorm |  | 30 | 8 |
| ac5945978b76 | cora | pairnorm |  | 30 | 9 |
| aaeee4e2dce5 | cora | dropedge |  | 2 | 0 |
| f8c65a1631b5 | cora | dropedge |  | 2 | 1 |
| db3a6b35e905 | cora | dropedge |  | 2 | 3 |
| ee87cb9cae12 | cora | dropedge |  | 2 | 4 |
| 91f789e819af | cora | dropedge |  | 2 | 5 |
| b7c7cfad135a | cora | dropedge |  | 2 | 6 |
| 1dcf2ee5423d | cora | dropedge |  | 2 | 7 |
| a38e9514dc9b | cora | dropedge |  | 2 | 8 |
| bd507be14fe3 | cora | dropedge |  | 2 | 9 |
| 62634b8e1b24 | cora | dropedge |  | 4 | 0 |
| 5ded8152d7db | cora | dropedge |  | 4 | 1 |
| 2ec0e33324c6 | cora | dropedge |  | 4 | 2 |
| 4b477736aa88 | cora | dropedge |  | 4 | 3 |
| d05d1ab15308 | cora | dropedge |  | 4 | 4 |
| 8b1066cd29ec | cora | dropedge |  | 4 | 6 |
| 88c0db0c85b4 | cora | dropedge |  | 4 | 7 |
| 085d4704e2e8 | cora | dropedge |  | 4 | 8 |
| 80cfaf450285 | cora | dropedge |  | 4 | 9 |
| 42fa95e88758 | cora | dropedge |  | 8 | 0 |
| 3e66604932e6 | cora | dropedge |  | 8 | 1 |
| 3157a1e3e5e5 | cora | dropedge |  | 8 | 2 |
| 7424e2e80d88 | cora | dropedge |  | 8 | 3 |
| b10ba88425d9 | cora | dropedge |  | 8 | 4 |
| ba0149c65770 | cora | dropedge |  | 8 | 5 |
| 82106652365a | cora | dropedge |  | 8 | 6 |
| 96ab046ec093 | cora | dropedge |  | 8 | 7 |
| c64559bb3466 | cora | dropedge |  | 8 | 8 |
| a28b173e5f9e | cora | dropedge |  | 8 | 9 |
| 573f642936cf | cora | dropedge |  | 10 | 0 |
| 226d6ce24085 | cora | dropedge |  | 10 | 1 |
| f609299f6e4e | cora | dropedge |  | 10 | 2 |
| 848816e58743 | cora | dropedge |  | 10 | 3 |
| 8eed5b14f776 | cora | dropedge |  | 10 | 4 |
| f4ba66f9e01e | cora | dropedge |  | 10 | 5 |
| 0922be4fa5f2 | cora | dropedge |  | 10 | 6 |
| 7ea7a6614c11 | cora | dropedge |  | 10 | 7 |
| 71c3bfa07464 | cora | dropedge |  | 10 | 8 |
| 241b1b290414 | cora | dropedge |  | 10 | 9 |
| 3bfb9086f01b | cora | dropedge |  | 20 | 0 |
| bfe49955748f | cora | dropedge |  | 20 | 1 |
| bebea22fe142 | cora | dropedge |  | 20 | 2 |
| 8398ec114840 | cora | dropedge |  | 20 | 3 |
| 59b92e91e10c | cora | dropedge |  | 20 | 4 |
| 3669bbcee218 | cora | dropedge |  | 20 | 5 |
| be100ec5e35d | cora | dropedge |  | 20 | 6 |
| d34e56e0f4a6 | cora | dropedge |  | 20 | 7 |
| 6966210f190b | cora | dropedge |  | 20 | 8 |
| 8465d47d7756 | cora | dropedge |  | 20 | 9 |
| f121ba088b52 | cora | dropedge |  | 30 | 0 |
| 229a76f78a92 | cora | dropedge |  | 30 | 1 |
| df6cdc5dfedd | cora | dropedge |  | 30 | 2 |
| 61386f72af78 | cora | dropedge |  | 30 | 3 |
| 85af2d019edd | cora | dropedge |  | 30 | 4 |
| e3a30c6ca327 | cora | dropedge |  | 30 | 5 |
| ff4aa7ce7b2c | cora | dropedge |  | 30 | 6 |
| 05d7bdf96ea4 | cora | dropedge |  | 30 | 7 |
| 344ef8f10f5d | cora | dropedge |  | 30 | 8 |
| dcbe132eb240 | cora | dropedge |  | 30 | 9 |
| 429aa88d4a27 | cora | graphunet |  | 2 | 0 |
| 9d3a9989e709 | cora | graphunet |  | 2 | 1 |
| 41e0678d3519 | cora | graphunet |  | 2 | 2 |
| d56141ac4e64 | cora | graphunet |  | 2 | 3 |
| 68f1df37dc32 | cora | graphunet |  | 2 | 4 |
| 51feba633110 | cora | graphunet |  | 2 | 5 |
| 69c515693d3c | cora | graphunet |  | 2 | 6 |
| b84f62547936 | cora | graphunet |  | 2 | 7 |
| 8cacda340501 | cora | graphunet |  | 2 | 8 |
| d9fa59b9f0a8 | cora | graphunet |  | 2 | 9 |
| 5beb94b5443c | cora | graphunet |  | 4 | 0 |
| d06e30b67409 | cora | graphunet |  | 4 | 1 |
| 8b149c7adf78 | cora | graphunet |  | 4 | 2 |
| 055ee1d472e6 | cora | graphunet |  | 4 | 3 |
| 1ae5d7479452 | cora | graphunet |  | 4 | 4 |
| 7af66f6451db | cora | graphunet |  | 4 | 5 |
| 25c118785dc1 | cora | graphunet |  | 4 | 6 |
| 894388b62042 | cora | graphunet |  | 4 | 7 |
| decc604dfec4 | cora | graphunet |  | 4 | 8 |
| c4f59364e239 | cora | graphunet |  | 4 | 9 |
| 961a83559ef0 | cora | graphunet |  | 8 | 0 |
| 865f4fb5b054 | cora | graphunet |  | 8 | 1 |
| 85f3b0963b1e | cora | graphunet |  | 8 | 2 |
| 0f21e5291ace | cora | graphunet |  | 8 | 3 |
| 084ba219028e | cora | graphunet |  | 8 | 4 |
| 658bdb78df7d | cora | graphunet |  | 8 | 5 |
| a9c18a97c41a | cora | graphunet |  | 8 | 6 |
| 66e66789dffe | cora | graphunet |  | 8 | 7 |
| b68cfb12563c | cora | graphunet |  | 8 | 8 |
| 212d177b3e07 | cora | graphunet |  | 8 | 9 |
| 601105d8c21c | cora | graphunet |  | 10 | 0 |
| 0f75187f1b89 | cora | graphunet |  | 10 | 1 |
| 90a6f2f4db14 | cora | graphunet |  | 10 | 2 |
| cf4de5b03259 | cora | graphunet |  | 10 | 3 |
| bdf305c10a57 | cora | graphunet |  | 10 | 4 |
| bdd71d8aafd1 | cora | graphunet |  | 10 | 5 |
| d9f91951f4f2 | cora | graphunet |  | 10 | 6 |
| e04cf3d5dc10 | cora | graphunet |  | 10 | 7 |
| 4dad8c3fdfca | cora | graphunet |  | 10 | 8 |
| 8e2f4d962f19 | cora | graphunet |  | 10 | 9 |
| f422022ab8ac | cora | graphunet |  | 20 | 0 |
| 663c20ecda70 | cora | graphunet |  | 20 | 1 |
| f9f065cdcaea | cora | graphunet |  | 20 | 2 |
| 7075276b123d | cora | graphunet |  | 20 | 3 |
| 281582448fdb | cora | graphunet |  | 20 | 4 |
| 05ad78741a88 | cora | graphunet |  | 20 | 5 |
| 163a2f83b1fb | cora | graphunet |  | 20 | 6 |
| 150bd589f637 | cora | graphunet |  | 20 | 7 |
| 5964465fbe9b | cora | graphunet |  | 20 | 8 |
| 69989af4f2cc | cora | graphunet |  | 20 | 9 |
| 83d22ca535be | cora | graphunet |  | 30 | 0 |
| 4a467382f451 | cora | graphunet |  | 30 | 1 |
| a9c16c81523c | cora | graphunet |  | 30 | 2 |
| 7c754f5d34b1 | cora | graphunet |  | 30 | 3 |
| 279a72002740 | cora | graphunet |  | 30 | 4 |
| b6b90269cd99 | cora | graphunet |  | 30 | 5 |
| 452efc2851e7 | cora | graphunet |  | 30 | 6 |
| 60a86e7450bd | cora | graphunet |  | 30 | 7 |
| 3cccc6b4b851 | cora | graphunet |  | 30 | 8 |
| dd040967fa3d | cora | graphunet |  | 30 | 9 |
| d2fa84e2b6fc | citeseer | nhdp |  | 2 | 7 |
| b96e6e1c8127 | citeseer | nhdp |  | 4 | 1 |
| 8f54dfe81155 | citeseer | nhdp |  | 4 | 3 |
| 2b662d65562f | citeseer | nhdp |  | 4 | 4 |
| b69b64778624 | citeseer | nhdp |  | 4 | 5 |
| 2cce6df509a7 | citeseer | nhdp |  | 4 | 6 |
| e01dc6e5617a | citeseer | nhdp |  | 4 | 7 |
| 927be7adf277 | citeseer | nhdp |  | 4 | 9 |
| f5f436c22644 | citeseer | nhdp |  | 8 | 0 |
| 9f0890d5ba84 | citeseer | nhdp |  | 8 | 1 |
| fe590bc6879e | citeseer | nhdp |  | 8 | 2 |
| 9c335cbed74f | citeseer | nhdp |  | 8 | 3 |
| d921e373a99b | citeseer | nhdp |  | 8 | 4 |
| 38401c280b38 | citeseer | nhdp |  | 8 | 5 |
| b3d1cfe54556 | citeseer | nhdp |  | 8 | 6 |
| 61e9268b4bc2 | citeseer | nhdp |  | 8 | 7 |
| b71635eb61b6 | citeseer | nhdp |  | 8 | 8 |
| 9b15f9a6db24 | citeseer | nhdp |  | 8 | 9 |
| 196839355073 | citeseer | nhdp |  | 10 | 0 |
| a7e6ac291d48 | citeseer | nhdp |  | 10 | 1 |
| 55fc219353c5 | citeseer | nhdp |  | 10 | 3 |
| 956a5c508d0f | citeseer | nhdp |  | 10 | 4 |
| a4b029f90eb4 | citeseer | nhdp |  | 10 | 5 |
| 765a05107e22 | citeseer | nhdp |  | 10 | 6 |
| 1f603c0ffca5 | citeseer | nhdp |  | 10 | 7 |
| 24826de83f96 | citeseer | nhdp |  | 10 | 8 |
| a93f544452c8 | citeseer | nhdp |  | 10 | 9 |
| 1fb7c2723c2d | citeseer | nhdp |  | 20 | 0 |
| 17cb2ec3de70 | citeseer | nhdp |  | 20 | 1 |
| 21d9a1aa92d7 | citeseer | nhdp |  | 20 | 2 |
| 18f1038f9506 | citeseer | nhdp |  | 20 | 3 |
| 4e30e7eb9d23 | citeseer | nhdp |  | 20 | 4 |
| ee210259f4be | citeseer | nhdp |  | 20 | 6 |
| 3cf69a972c3b | citeseer | nhdp |  | 20 | 7 |
| 71c26712b2e1 | citeseer | nhdp |  | 20 | 8 |
| 640a677315f8 | citeseer | nhdp |  | 20 | 9 |
| 542f2ea9d636 | citeseer | nhdp |  | 30 | 0 |
| 4869ffaae960 | citeseer | nhdp |  | 30 | 1 |
| 182bfc3d924a | citeseer | nhdp |  | 30 | 2 |
| 9335c7e97e06 | citeseer | nhdp |  | 30 | 3 |
| 9b44ab33a87f | citeseer | nhdp |  | 30 | 4 |
| 7d0eb3efe939 | citeseer | nhdp |  | 30 | 5 |
| b2857825bb9a | citeseer | nhdp |  | 30 | 6 |
| a91bf4e45d3b | citeseer | nhdp |  | 30 | 7 |
| 7d9d7242bf9b | citeseer | nhdp |  | 30 | 8 |
| 6872599b52f1 | citeseer | nhdp |  | 30 | 9 |
| 820b2efd973f | citeseer | gcn |  | 2 | 0 |
| 1de6cb4cd50f | citeseer | gcn |  | 2 | 1 |
| b26a449ec86b | citeseer | gcn |  | 2 | 2 |
| d495fe764c3f | citeseer | gcn |  | 2 | 3 |
| 7db0e061c738 | citeseer | gcn |  | 2 | 4 |
| 95ef0f38fe14 | citeseer | gcn |  | 2 | 5 |
| cb0faf8453ee | citeseer | gcn |  | 2 | 6 |
| e78a043118b6 | citeseer | gcn |  | 2 | 7 |
| f97cfae58839 | citeseer | gcn |  | 2 | 8 |
| b0ff1cef88f9 | citeseer | gcn |  | 2 | 9 |
| 28ec36b815b9 | citeseer | gcn |  | 4 | 0 |
| 3b5c9c769147 | citeseer | gcn |  | 4 | 1 |
| cb7ea681cd40 | citeseer | gcn |  | 4 | 2 |
| af0c9235cea4 | citeseer | gcn |  | 4 | 3 |
| 2d3313d5afac | citeseer | gcn |  | 4 | 4 |
| d1eb3951122e | citeseer | gcn |  | 4 | 5 |
| ab8eac71e646 | citeseer | gcn |  | 4 | 6 |
| a725e25c6d8a | citeseer | gcn |  | 4 | 7 |
| c3160f7a92f6 | citeseer | gcn |  | 4 | 8 |
| cddb84edc103 | citeseer | gcn |  | 4 | 9 |
| 006ca981c627 | citeseer | gcn |  | 8 | 0 |
| d1d31e4757e7 | citeseer | gcn |  | 8 | 1 |
| 44d3e1b94a35 | citeseer | gcn |  | 8 | 2 |
| b957c5fb81bb | citeseer | gcn |  | 8 | 3 |
| 99e281fdfa73 | citeseer | gcn |  | 8 | 4 |
| 55dc661c3f9d | citeseer | gcn |  | 8 | 5 |
| 955641cb6b70 | citeseer | gcn |  | 8 | 6 |
| cf9987035c49 | citeseer | gcn |  | 8 | 7 |
| 7144855d0cdd | citeseer | gcn |  | 8 | 8 |
| 0e491be3e85d | citeseer | gcn |  | 8 | 9 |
| 933c992ecec5 | citeseer | gcn |  | 10 | 0 |
| 06324cb3cfeb | citeseer | gcn |  | 10 | 1 |
| 94d6fd95b26a | citeseer | gcn |  | 10 | 2 |
| 232e26fda4e3 | citeseer | gcn |  | 10 | 3 |
| da785ebdb0fe | citeseer | gcn |  | 10 | 4 |
| 5b086467c869 | citeseer | gcn |  | 10 | 5 |
| b492e8c81de6 | citeseer | gcn |  | 10 | 6 |
| a6d5ae76230c | citeseer | gcn |  | 10 | 7 |
| 02316788f665 | citeseer | gcn |  | 10 | 8 |
| 48eb553fa9f5 | citeseer | gcn |  | 10 | 9 |
| dbafb600792c | citeseer | gcn |  | 20 | 0 |
| 3d06e79e9131 | citeseer | gcn |  | 20 | 1 |
| 9b66f8e32c51 | citeseer | gcn |  | 20 | 2 |
| fcaf09e5d73b | citeseer | gcn |  | 20 | 3 |
| 5f6272c62aa0 | citeseer | gcn |  | 20 | 4 |
| 0ce611351580 | citeseer | gcn |  | 20 | 5 |
| bac4ed9f7f68 | citeseer | gcn |  | 20 | 6 |
| 65cddd00bb2e | citeseer | gcn |  | 20 | 7 |
| 956fb84bde53 | citeseer | gcn |  | 20 | 8 |
| c3a9f177bbf6 | citeseer | gcn |  | 20 | 9 |
| e3e6bdd2ba19 | citeseer | gcn |  | 30 | 0 |
| ef406d7e990f | citeseer | gcn |  | 30 | 1 |
| e41e314cf5f7 | citeseer | gcn |  | 30 | 2 |
| ddea44430aa1 | citeseer | gcn |  | 30 | 3 |
| f98dbe960ec4 | citeseer | gcn |  | 30 | 4 |
| 1dd3d3e69878 | citeseer | gcn |  | 30 | 5 |
| 7389c7e5e473 | citeseer | gcn |  | 30 | 6 |
| 3ca32e50e070 | citeseer | gcn |  | 30 | 7 |
| 25ea5e78c441 | citeseer | gcn |  | 30 | 8 |
| 28ac4673d968 | citeseer | gcn |  | 30 | 9 |
| fa80ca54d2ea | citeseer | gat |  | 2 | 0 |
| 2fbd146f5a4e | citeseer | gat |  | 2 | 1 |
| d5a21f444044 | citeseer | gat |  | 2 | 2 |
| ce83e448d090 | citeseer | gat |  | 2 | 3 |
| ea0148b0de20 | citeseer | gat |  | 2 | 4 |
| e057f10eaa5d | citeseer | gat |  | 2 | 5 |
| ac64cb4ea304 | citeseer | gat |  | 2 | 6 |
| ce3ef4cacc6d | citeseer | gat |  | 2 | 7 |
| 300a3e053844 | citeseer | gat |  | 2 | 8 |
| 27a2f488be3a | citeseer | gat |  | 2 | 9 |
| 2199b0e418a9 | citeseer | gat |  | 4 | 0 |
| 90606f074569 | citeseer | gat |  | 4 | 1 |
| 6d93fff95965 | citeseer | gat |  | 4 | 2 |
| 2d16b3f751e0 | citeseer | gat |  | 4 | 3 |
| e3036f3a4955 | citeseer | gat |  | 4 | 4 |
| dcc7fb0d1807 | citeseer | gat |  | 4 | 5 |
| ae1c229937d1 | citeseer | gat |  | 4 | 6 |
| 22a7d138bf65 | citeseer | gat |  | 4 | 7 |
| b9a397d41159 | citeseer | gat |  | 4 | 8 |
| c62f3c470093 | citeseer | gat |  | 4 | 9 |
| 4ac75f0caf98 | citeseer | gat |  | 8 | 0 |
| 00f7331863f4 | citeseer | gat |  | 8 | 1 |
| ab0018d5ad8b | citeseer | gat |  | 8 | 2 |
| 1b41e77747b3 | citeseer | gat |  | 8 | 3 |
| 31f0731c7fbe | citeseer | gat |  | 8 | 4 |
| e9c8e61a329f | citeseer | gat |  | 8 | 5 |
| e1382273d1dc | citeseer | gat |  | 8 | 6 |
| 3fcf711abc78 | citeseer | gat |  | 8 | 7 |
| fa5ce1e4334a | citeseer | gat |  | 8 | 8 |
| 25bb2be3b429 | citeseer | gat |  | 8 | 9 |
| 94842d3dc061 | citeseer | gat |  | 10 | 7 |
| 02c7e78a6fb0 | citeseer | gat |  | 20 | 1 |
| a09849d87f49 | citeseer | gat |  | 20 | 3 |
| 555d5dc290ab | citeseer | gat |  | 20 | 4 |
| 642886e6d1f5 | citeseer | gat |  | 20 | 5 |
| 944420ea7c49 | citeseer | gat |  | 20 | 6 |
| bf71490ea051 | citeseer | gat |  | 20 | 7 |
| b67d93811b89 | citeseer | gat |  | 20 | 9 |
| bb6bb9546fd9 | citeseer | gat |  | 30 | 0 |
| cd2bcdabc0d3 | citeseer | gat |  | 30 | 1 |
| bb86ace483a6 | citeseer | gat |  | 30 | 2 |
| ebf7187e29d4 | citeseer | gat |  | 30 | 3 |
| d415d1811854 | citeseer | gat |  | 30 | 4 |
| 3ebe28a3fdfc | citeseer | gat |  | 30 | 5 |
| 5ba1242e59ac | citeseer | gat |  | 30 | 6 |
| dca861885fcf | citeseer | gat |  | 30 | 7 |
| 3ffd64d3a2c6 | citeseer | gat |  | 30 | 8 |
| 74930e9eb7d2 | citeseer | gat |  | 30 | 9 |
| 5143743b92df | citeseer | sgc |  | 2 | 0 |
| a79990be3ee9 | citeseer | sgc |  | 2 | 1 |
| 4e3d0603678c | citeseer | sgc |  | 2 | 3 |
| 0a5d74402524 | citeseer | sgc |  | 2 | 4 |
| 916f6ad03d82 | citeseer | sgc |  | 2 | 5 |
| d40345319c8b | citeseer | sgc |  | 2 | 6 |
| 20fceac714eb | citeseer | sgc |  | 2 | 7 |
| 43fb1837f106 | citeseer | sgc |  | 2 | 8 |
| 403c996f47f9 | citeseer | sgc |  | 2 | 9 |
| 4d590556737c | citeseer | sgc |  | 4 | 0 |
| 5c69b65f2509 | citeseer | sgc |  | 4 | 1 |
| aa51b81a93a5 | citeseer | sgc |  | 4 | 2 |
| 885759b67c26 | citeseer | sgc |  | 4 | 3 |
| 68221cb3176d | citeseer | sgc |  | 4 | 4 |
| 719ecc9e3a81 | citeseer | sgc |  | 4 | 6 |
| c8b675c382da | citeseer | sgc |  | 4 | 7 |
| 37019ce34635 | citeseer | sgc |  | 4 | 8 |
| 26d30999c35f | citeseer | sgc |  | 4 | 9 |
| 7583c1906dab | citeseer | sgc |  | 8 | 0 |
| 879aa3406314 | citeseer | sgc |  | 8 | 1 |
| e94c6477b5a4 | citeseer | sgc |  | 8 | 2 |
| 32e9cf21b983 | citeseer | sgc |  | 8 | 3 |
| 0996bf910010 | citeseer | sgc |  | 8 | 4 |
| 7099037c2e38 | citeseer | sgc |  | 8 | 5 |
| 61809fceb21c | citeseer | sgc |  | 8 | 6 |
| 1f30946a3dc7 | citeseer | sgc |  | 8 | 7 |
| 95a1d15255bd | citeseer | sgc |  | 8 | 8 |
| 1c9ce2f020f5 | citeseer | sgc |  | 8 | 9 |
| ef7e55630a04 | citeseer | sgc |  | 10 | 0 |
| dd47ff5c5d39 | citeseer | sgc |  | 10 | 1 |
| aa67821c1784 | citeseer | sgc |  | 10 | 2 |
| 9b975a17c692 | citeseer | sgc |  | 10 | 3 |
| 0224d2266c9b | citeseer | sgc |  | 10 | 4 |
| 267420abad4e | citeseer | sgc |  | 10 | 5 |
| dfc879525e28 | citeseer | sgc |  | 10 | 6 |
| 263873120e5f | citeseer | sgc |  | 10 | 7 |
| b73ad2983837 | citeseer | sgc |  | 10 | 8 |
| a2d21f11fe3f | citeseer | sgc |  | 10 | 9 |
| 897ce1bc0e8d | citeseer | sgc |  | 20 | 0 |
| 8bd01899af36 | citeseer | sgc |  | 20 | 1 |
| 6812ffd81b6a | citeseer | sgc |  | 20 | 2 |
| 6418c866c80f | citeseer | sgc |  | 20 | 3 |
| c194284ca866 | citeseer | sgc |  | 20 | 4 |
| 7c592c38b16c | citeseer | sgc |  | 20 | 5 |
| 92cd226c137b | citeseer | sgc |  | 20 | 6 |
| c37d0f6ec814 | citeseer | sgc |  | 20 | 7 |
| c1a58e11f32e | citeseer | sgc |  | 20 | 8 |
| 3d13eb35627a | citeseer | sgc |  | 20 | 9 |
| 6adc0f643a5e | citeseer | sgc |  | 30 | 0 |
| 0a5febf65885 | citeseer | sgc |  | 30 | 1 |
| 8adf7457994d | citeseer | sgc |  | 30 | 2 |
| b7eae4bbec1f | citeseer | sgc |  | 30 | 3 |
| cb3afb52b57b | citeseer | sgc |  | 30 | 4 |
| bac537fb8a44 | citeseer | sgc |  | 30 | 5 |
| 1d6429cdf63b | citeseer | sgc |  | 30 | 6 |
| 143cacc3e51d | citeseer | sgc |  | 30 | 7 |
| 5c338e055e89 | citeseer | sgc |  | 30 | 8 |
| 02375602aed7 | citeseer | sgc |  | 30 | 9 |
| cc9165e1c6e2 | citeseer | appnp |  | 2 | 0 |
| 221406d6bfc9 | citeseer | appnp |  | 2 | 1 |
| e689ea4ee0c8 | citeseer | appnp |  | 2 | 2 |
| b17b03939154 | citeseer | appnp |  | 2 | 3 |
| 12ca86038535 | citeseer | appnp |  | 2 | 4 |
| e70aa1033807 | citeseer | appnp |  | 2 | 5 |
| a2ca423bf171 | citeseer | appnp |  | 2 | 6 |
| 1d13ec09cd91 | citeseer | appnp |  | 2 | 7 |
| 404f8b502ee5 | citeseer | appnp |  | 2 | 8 |
| 7437e1aff7de | citeseer | appnp |  | 2 | 9 |
| 21ea4e0f49df | citeseer | appnp |  | 4 | 0 |
| ea0a33e5ee2f | citeseer | appnp |  | 4 | 1 |
| 3b5f012e0087 | citeseer | appnp |  | 4 | 2 |
| 284b2cd2864b | citeseer | appnp |  | 4 | 3 |
| e40b8f375cb1 | citeseer | appnp |  | 4 | 4 |
| b5b48d4c9f86 | citeseer | appnp |  | 4 | 5 |
| 6fd6b26882f0 | citeseer | appnp |  | 4 | 6 |
| 83fc5cf5263b | citeseer | appnp |  | 4 | 7 |
| 1d47d77337e1 | citeseer | appnp |  | 4 | 8 |
| 0b64f3b21e6f | citeseer | appnp |  | 4 | 9 |
| 91827f0d45fb | citeseer | appnp |  | 8 | 0 |
| dcbc6f829e92 | citeseer | appnp |  | 8 | 1 |
| 38c819fbe80c | citeseer | appnp |  | 8 | 2 |
| 4ff40635b81b | citeseer | appnp |  | 8 | 3 |
| 15cc6aac917c | citeseer | appnp |  | 8 | 4 |
| 8bf6d679ede6 | citeseer | appnp |  | 8 | 5 |
| f82c899bf995 | citeseer | appnp |  | 8 | 6 |
| 278e12695697 | citeseer | appnp |  | 8 | 7 |
| a890eedd6960 | citeseer | appnp |  | 8 | 8 |
| 9d24fb5988b8 | citeseer | appnp |  | 8 | 9 |
| 4e3759f46c01 | citeseer | appnp |  | 10 | 0 |
| 2234f0835c4f | citeseer | appnp |  | 10 | 1 |
| 34de433b091c | citeseer | appnp |  | 10 | 2 |
| 605ae32377c9 | citeseer | appnp |  | 10 | 3 |
| 6e99aba702da | citeseer | appnp |  | 10 | 4 |
| bba039a113bb | citeseer | appnp |  | 10 | 5 |
| 1d37bb8c53fc | citeseer | appnp |  | 10 | 6 |
| f1c592c98e79 | citeseer | appnp |  | 10 | 7 |
| 7cb4b7dfcd36 | citeseer | appnp |  | 10 | 8 |
| 27a96061fcf7 | citeseer | appnp |  | 10 | 9 |
| 3576e5d2127f | citeseer | appnp |  | 20 | 0 |
| 330edc76cb92 | citeseer | appnp |  | 20 | 1 |
| b2b29caae24b | citeseer | appnp |  | 20 | 2 |
| e9e4888343d5 | citeseer | appnp |  | 20 | 3 |
| 375f9609fd0c | citeseer | appnp |  | 20 | 4 |
| 8b81594cb630 | citeseer | appnp |  | 20 | 5 |
| 41c7aea54dd6 | citeseer | appnp |  | 20 | 6 |
| 6572ceccdd8c | citeseer | appnp |  | 20 | 7 |
| 5001495c1894 | citeseer | appnp |  | 20 | 8 |
| 6dfae3ba0a56 | citeseer | appnp |  | 20 | 9 |
| 641c5a3fd8cb | citeseer | appnp |  | 30 | 0 |
| ea953959eb65 | citeseer | appnp |  | 30 | 1 |
| d612d5d9b72a | citeseer | appnp |  | 30 | 2 |
| e654b0acc142 | citeseer | appnp |  | 30 | 3 |
| a5bee9ff7871 | citeseer | appnp |  | 30 | 4 |
| b441a67c444c | citeseer | appnp |  | 30 | 5 |
| 5fa4e18e4df0 | citeseer | appnp |  | 30 | 6 |
| be0ff9110279 | citeseer | appnp |  | 30 | 7 |
| 654170e92693 | citeseer | appnp |  | 30 | 8 |
| 0d33905cb865 | citeseer | appnp |  | 30 | 9 |
| bf8ed4975839 | citeseer | arma |  | 2 | 7 |
| 519e10d22a02 | citeseer | arma |  | 4 | 1 |
| db548a2f2e75 | citeseer | arma |  | 4 | 3 |
| 26b42aa986d8 | citeseer | arma |  | 4 | 4 |
| 23ec3cf9508c | citeseer | arma |  | 4 | 5 |
| fd60b0b2a05a | citeseer | arma |  | 4 | 6 |
| f8023e47cac1 | citeseer | arma |  | 4 | 7 |
| 56074c604129 | citeseer | arma |  | 4 | 9 |
| 2c037eeb278b | citeseer | arma |  | 8 | 0 |
| b179fd873db5 | citeseer | arma |  | 8 | 1 |
| 2cf7b5539c7d | citeseer | arma |  | 8 | 2 |
| 52ee9ea2bdc7 | citeseer | arma |  | 8 | 3 |
| d9b87e4485c4 | citeseer | arma |  | 8 | 4 |
| 8fdd4946dd34 | citeseer | arma |  | 8 | 5 |
| 88ad1a0b53d4 | citeseer | arma |  | 8 | 6 |
| fbe2ce276bb5 | citeseer | arma |  | 8 | 7 |
| 49bfdd6ab51c | citeseer | arma |  | 8 | 8 |
| 956c953c8727 | citeseer | arma |  | 8 | 9 |
| 2abc8fec6e63 | citeseer | arma |  | 10 | 0 |
| 0061879dbc80 | citeseer | arma |  | 10 | 1 |
| cd751870eae0 | citeseer | arma |  | 10 | 3 |
| 607dc2521f7a | citeseer | arma |  | 10 | 4 |
| 480891716845 | citeseer | arma |  | 10 | 5 |
| a7266f19f55c | citeseer | arma |  | 10 | 6 |
| 8c70effbb108 | citeseer | arma |  | 10 | 7 |
| 2a1200b5e007 | citeseer | arma |  | 10 | 8 |
| f87d301c69a6 | citeseer | arma |  | 10 | 9 |
| 1553b137909f | citeseer | arma |  | 20 | 0 |
| e7d3afef6933 | citeseer | arma |  | 20 | 1 |
| 086ab48c1141 | citeseer | arma |  | 20 | 2 |
| d98660e24717 | citeseer | arma |  | 20 | 3 |
| b6f4e106f4e0 | citeseer | arma |  | 20 | 4 |
| 2eab12142564 | citeseer | arma |  | 20 | 6 |
| b12afce5a825 | citeseer | arma |  | 20 | 7 |
| cfd71df8418a | citeseer | arma |  | 20 | 8 |
| 80237fdd704d | citeseer | arma |  | 20 | 9 |
| 176c994b4c21 | citeseer | arma |  | 30 | 0 |
| 99f76da42c00 | citeseer | arma |  | 30 | 1 |
| aaa11118d141 | citeseer | arma |  | 30 | 2 |
| a7f890c77e9b | citeseer | arma |  | 30 | 3 |
| b000fe10fd00 | citeseer | arma |  | 30 | 4 |
| d939c9863ba8 | citeseer | arma |  | 30 | 5 |
| e625cacf7a77 | citeseer | arma |  | 30 | 6 |
| 653253b19006 | citeseer | arma |  | 30 | 7 |
| 241ae422bfff | citeseer | arma |  | 30 | 8 |
| e26efd9b827a | citeseer | arma |  | 30 | 9 |
| 0c02f1d54dd1 | citeseer | gcnii |  | 2 | 0 |
| cea1ca3c246a | citeseer | gcnii |  | 2 | 1 |
| 83777fc28602 | citeseer | gcnii |  | 2 | 2 |
| 32ae47c3a5bb | citeseer | gcnii |  | 2 | 3 |
| bfdd7a95d1b7 | citeseer | gcnii |  | 2 | 4 |
| 35fef2bcd90f | citeseer | gcnii |  | 2 | 5 |
| 4eb989e25651 | citeseer | gcnii |  | 2 | 6 |
| a094e18363b7 | citeseer | gcnii |  | 2 | 7 |
| bef046a9fdd8 | citeseer | gcnii |  | 2 | 8 |
| 20b8f078499e | citeseer | gcnii |  | 2 | 9 |
| 3f2489fe1358 | citeseer | gcnii |  | 4 | 0 |
| 3172a2f93057 | citeseer | gcnii |  | 4 | 1 |
| c1b4e6acf2d4 | citeseer | gcnii |  | 4 | 2 |
| 56d70ead1aa9 | citeseer | gcnii |  | 4 | 3 |
| 4bae2547508c | citeseer | gcnii |  | 4 | 4 |
| 4521e2554b50 | citeseer | gcnii |  | 4 | 5 |
| d79ce1aee805 | citeseer | gcnii |  | 4 | 6 |
| be737c2cd3a0 | citeseer | gcnii |  | 4 | 7 |
| e9e5a0e1c41b | citeseer | gcnii |  | 4 | 8 |
| bd81301a4267 | citeseer | gcnii |  | 4 | 9 |
| 1aee2d27ff0d | citeseer | gcnii |  | 8 | 0 |
| 99e5b791799e | citeseer | gcnii |  | 8 | 1 |
| cc0bd3b0f613 | citeseer | gcnii |  | 8 | 2 |
| f7646a8068b6 | citeseer | gcnii |  | 8 | 3 |
| 453dbc67832c | citeseer | gcnii |  | 8 | 4 |
| 6534e33baa63 | citeseer | gcnii |  | 8 | 5 |
| 9c5bfc9307d9 | citeseer | gcnii |  | 8 | 6 |
| ecc1af07b620 | citeseer | gcnii |  | 8 | 7 |
| 28326b6807fe | citeseer | gcnii |  | 8 | 8 |
| 6861a0cae17e | citeseer | gcnii |  | 8 | 9 |
| 6b3ca734caf1 | citeseer | gcnii |  | 10 | 0 |
| 7ef90512cdd4 | citeseer | gcnii |  | 10 | 1 |
| 6772f9fd6695 | citeseer | gcnii |  | 10 | 2 |
| 5d8350b59320 | citeseer | gcnii |  | 10 | 3 |
| 081298dd77ec | citeseer | gcnii |  | 10 | 4 |
| 7abc7c0a2e3e | citeseer | gcnii |  | 10 | 5 |
| 4f13f506842e | citeseer | gcnii |  | 10 | 6 |
| 9187a60b8a92 | citeseer | gcnii |  | 10 | 7 |
| d55f4a80928f | citeseer | gcnii |  | 10 | 8 |
| 0d64af8bd60f | citeseer | gcnii |  | 10 | 9 |
| a8c095ea29df | citeseer | gcnii |  | 20 | 0 |
| ecd14a6949c5 | citeseer | gcnii |  | 20 | 1 |
| a7263414a69b | citeseer | gcnii |  | 20 | 2 |
| c75a2526050a | citeseer | gcnii |  | 20 | 3 |
| 18a65021b518 | citeseer | gcnii |  | 20 | 4 |
| 7c1e29ff4a89 | citeseer | gcnii |  | 20 | 5 |
| 3c836462805d | citeseer | gcnii |  | 20 | 6 |
| e2257ce16dd5 | citeseer | gcnii |  | 20 | 7 |
| 1d22aaf26a3f | citeseer | gcnii |  | 20 | 8 |
| 90a5a6e19b55 | citeseer | gcnii |  | 20 | 9 |
| 2f7d5070b68b | citeseer | gcnii |  | 30 | 0 |
| fe8ed5f1aa7a | citeseer | gcnii |  | 30 | 1 |
| 21806f400f0b | citeseer | gcnii |  | 30 | 2 |
| a2a383cbbcee | citeseer | gcnii |  | 30 | 3 |
| 997d5d79c9ca | citeseer | gcnii |  | 30 | 4 |
| eba94708d534 | citeseer | gcnii |  | 30 | 5 |
| b2068b419dac | citeseer | gcnii |  | 30 | 6 |
| 50471c2b57dd | citeseer | gcnii |  | 30 | 7 |
| 3a0e1703faf8 | citeseer | gcnii |  | 30 | 8 |
| 7f398a23c9f6 | citeseer | gcnii |  | 30 | 9 |
| 291392714221 | citeseer | pairnorm |  | 2 | 0 |
| 564ad90dbb8a | citeseer | pairnorm |  | 2 | 1 |
| 497800ab225c | citeseer | pairnorm |  | 2 | 2 |
| e7fe071c7a23 | citeseer | pairnorm |  | 2 | 3 |
| df3178c13728 | citeseer | pairnorm |  | 2 | 4 |
| c0c48b0b3c90 | citeseer | pairnorm |  | 2 | 5 |
| 0c8e2f469e0a | citeseer | pairnorm |  | 2 | 6 |
| 5080f08b5e2f | citeseer | pairnorm |  | 2 | 7 |
| 0b9eaa3728fa | citeseer | pairnorm |  | 2 | 8 |
| 5f350d14d5a1 | citeseer | pairnorm |  | 2 | 9 |
| e50d82109db6 | citeseer | pairnorm |  | 4 | 0 |
| e3338c206211 | citeseer | pairnorm |  | 4 | 1 |
| ff19f5369a31 | citeseer | pairnorm |  | 4 | 2 |
| 784fe9cc1ed6 | citeseer | pairnorm |  | 4 | 3 |
| 7de995ffda09 | citeseer | pairnorm |  | 4 | 4 |
| 620a16936ff6 | citeseer | pairnorm |  | 4 | 5 |
| f4694c88de9a | citeseer | pairnorm |  | 4 | 6 |
| c40859d8d7a4 | citeseer | pairnorm |  | 4 | 7 |
| cd79ac66e55b | citeseer | pairnorm |  | 4 | 8 |
| 23ff2573df4c | citeseer | pairnorm |  | 4 | 9 |
| 122932add20a | citeseer | pairnorm |  | 8 | 0 |
| 05248f35cacf | citeseer | pairnorm |  | 8 | 1 |
| b4802a3d7e9b | citeseer | pairnorm |  | 8 | 2 |
| ee89b5a8722e | citeseer | pairnorm |  | 8 | 3 |
| 3e91410370b1 | citeseer | pairnorm |  | 8 | 4 |
| 29a12cb39e04 | citeseer | pairnorm |  | 8 | 5 |
| b9c7058e6b7a | citeseer | pairnorm |  | 8 | 6 |
| 097b594ca4c7 | citeseer | pairnorm |  | 8 | 7 |
| a8b6f84b2c98 | citeseer | pairnorm |  | 8 | 8 |
| 70c82bccc72b | citeseer | pairnorm |  | 8 | 9 |
| f3192f6d9b19 | citeseer | pairnorm |  | 10 | 7 |
| 179103b7bab4 | citeseer | pairnorm |  | 20 | 1 |
| 28171c986496 | citeseer | pairnorm |  | 20 | 3 |
| c4419a9da1bc | citeseer | pairnorm |  | 20 | 4 |
| dcda37377578 | citeseer | pairnorm |  | 20 | 5 |
| 693a90fb241d | citeseer | pairnorm |  | 20 | 6 |
| 79abc1e7c6ae | citeseer | pairnorm |  | 20 | 7 |
| 4e757039b8c1 | citeseer | pairnorm |  | 20 | 9 |
| 91492f8c145d | citeseer | pairnorm |  | 30 | 0 |
| 068cc216895b | citeseer | pairnorm |  | 30 | 1 |
| ec4b6ec285f0 | citeseer | pairnorm |  | 30 | 2 |
| 0075e3fdf2fa | citeseer | pairnorm |  | 30 | 3 |
| ec4e389e9275 | citeseer | pairnorm |  | 30 | 4 |
| 275ea1288c43 | citeseer | pairnorm |  | 30 | 5 |
| f6e6fc53e93b | citeseer | pairnorm |  | 30 | 6 |
| ec8a601a10b2 | citeseer | pairnorm |  | 30 | 7 |
| 2df1c872126d | citeseer | pairnorm |  | 30 | 8 |
| 18babeaa3200 | citeseer | pairnorm |  | 30 | 9 |
| 879c5dbce5ae | citeseer | dropedge |  | 2 | 0 |
| 2e31b66c8201 | citeseer | dropedge |  | 2 | 1 |
| b4c1146eed57 | citeseer | dropedge |  | 2 | 3 |
| df463253e5cf | citeseer | dropedge |  | 2 | 4 |
| 2bdc8c923de3 | citeseer | dropedge |  | 2 | 5 |
| 42b21a75578d | citeseer | dropedge |  | 2 | 6 |
| f5e670517b2f | citeseer | dropedge |  | 2 | 7 |
| 5be669831d33 | citeseer | dropedge |  | 2 | 8 |
| 5ad6e86b4d78 | citeseer | dropedge |  | 2 | 9 |
| 62a7797dac89 | citeseer | dropedge |  | 4 | 0 |
| 33122ec397b7 | citeseer | dropedge |  | 4 | 1 |
| 8fefd0672b63 | citeseer | dropedge |  | 4 | 2 |
| 8a41f096f042 | citeseer | dropedge |  | 4 | 3 |
| 77abd6645e8c | citeseer | dropedge |  | 4 | 4 |
| 270a604ecbdf | citeseer | dropedge |  | 4 | 6 |
| 4ed5f6c2fda5 | citeseer | dropedge |  | 4 | 7 |
| a6b441112e60 | citeseer | dropedge |  | 4 | 8 |
| 6dfb307a9f50 | citeseer | dropedge |  | 4 | 9 |
| 511337992f6e | citeseer | dropedge |  | 8 | 0 |
| 96db12a210c0 | citeseer | dropedge |  | 8 | 1 |
| a54b4457012d | citeseer | dropedge |  | 8 | 2 |
| 6ce9c15b977c | citeseer | dropedge |  | 8 | 3 |
| d93b17e38e8f | citeseer | dropedge |  | 8 | 4 |
| d9f5326db859 | citeseer | dropedge |  | 8 | 5 |
| 9c61cb066f00 | citeseer | dropedge |  | 8 | 6 |
| 56bfeafa2834 | citeseer | dropedge |  | 8 | 7 |
| f69c4f7305b3 | citeseer | dropedge |  | 8 | 8 |
| 486a63d666e0 | citeseer | dropedge |  | 8 | 9 |
| ba364986f64d | citeseer | dropedge |  | 10 | 0 |
| 19d6268fc401 | citeseer | dropedge |  | 10 | 1 |
| d8ffa3d58abb | citeseer | dropedge |  | 10 | 2 |
| 68647aa7edfb | citeseer | dropedge |  | 10 | 3 |
| 8ecf161d61d3 | citeseer | dropedge |  | 10 | 4 |
| 0107a25dd805 | citeseer | dropedge |  | 10 | 5 |
| 2eb3187eec1a | citeseer | dropedge |  | 10 | 6 |
| 69a419440ad3 | citeseer | dropedge |  | 10 | 7 |
| 9ebc9208b2ec | citeseer | dropedge |  | 10 | 8 |
| baed01fc656a | citeseer | dropedge |  | 10 | 9 |
| ea94185945c6 | citeseer | dropedge |  | 20 | 0 |
| e3d92806a3ba | citeseer | dropedge |  | 20 | 1 |
| bb5da7f0d065 | citeseer | dropedge |  | 20 | 2 |
| 5c13e8291db1 | citeseer | dropedge |  | 20 | 3 |
| 447607a86f05 | citeseer | dropedge |  | 20 | 4 |
| 89bf081008bc | citeseer | dropedge |  | 20 | 5 |
| 4f2ed1668a87 | citeseer | dropedge |  | 20 | 6 |
| 44a96d9ab932 | citeseer | dropedge |  | 20 | 7 |
| 677ed403fe21 | citeseer | dropedge |  | 20 | 8 |
| 29e72248cbf0 | citeseer | dropedge |  | 20 | 9 |
| a064f149b770 | citeseer | dropedge |  | 30 | 0 |
| a3ed7abefd63 | citeseer | dropedge |  | 30 | 1 |
| e7701955f0a9 | citeseer | dropedge |  | 30 | 2 |
| 7bccc0b3296a | citeseer | dropedge |  | 30 | 3 |
| 07f2d16e68cc | citeseer | dropedge |  | 30 | 4 |
| 74e9a7168802 | citeseer | dropedge |  | 30 | 5 |
| fc22a4bbafe2 | citeseer | dropedge |  | 30 | 6 |
| 7de6350a2ff0 | citeseer | dropedge |  | 30 | 7 |
| 632ac89e0094 | citeseer | dropedge |  | 30 | 8 |
| ce96217dba21 | citeseer | dropedge |  | 30 | 9 |
| 719f06d8ccff | citeseer | graphunet |  | 2 | 0 |
| 5458eed350d1 | citeseer | graphunet |  | 2 | 1 |
| afe6a9cba22f | citeseer | graphunet |  | 2 | 2 |
| 4c46aea00e21 | citeseer | graphunet |  | 2 | 3 |
| ea6e27787c58 | citeseer | graphunet |  | 2 | 4 |
| 8b36c086295b | citeseer | graphunet |  | 2 | 5 |
| b0ade9ceae1d | citeseer | graphunet |  | 2 | 6 |
| fbcb0860b141 | citeseer | graphunet |  | 2 | 7 |
| f66f3d359ec9 | citeseer | graphunet |  | 2 | 8 |
| 805f6f6b0ca5 | citeseer | graphunet |  | 2 | 9 |
| e90a856ab904 | citeseer | graphunet |  | 4 | 0 |
| 90b99061e1d9 | citeseer | graphunet |  | 4 | 1 |
| 2c63f29bf6fe | citeseer | graphunet |  | 4 | 2 |
| c879faaad9d1 | citeseer | graphunet |  | 4 | 3 |
| 05f9681a497a | citeseer | graphunet |  | 4 | 4 |
| 081aabb56bda | citeseer | graphunet |  | 4 | 5 |
| 03d83726aacc | citeseer | graphunet |  | 4 | 6 |
| 0743ed65977a | citeseer | graphunet |  | 4 | 7 |
| 63749753ced1 | citeseer | graphunet |  | 4 | 8 |
| 4ca43c463563 | citeseer | graphunet |  | 4 | 9 |
| d4b3910a6dfc | citeseer | graphunet |  | 8 | 0 |
| 7eed0794cc7c | citeseer | graphunet |  | 8 | 1 |
| dc848b40e5b6 | citeseer | graphunet |  | 8 | 2 |
| e06f3c619b44 | citeseer | graphunet |  | 8 | 3 |
| 2784d90b210a | citeseer | graphunet |  | 8 | 4 |
| 54a21c47da3a | citeseer | graphunet |  | 8 | 5 |
| 1a2d7729327b | citeseer | graphunet |  | 8 | 6 |
| efe6b2ed9ee0 | citeseer | graphunet |  | 8 | 7 |
| ccc62a319a19 | citeseer | graphunet |  | 8 | 8 |
| 9289d0cea593 | citeseer | graphunet |  | 8 | 9 |
| ca2de2dbfbcd | citeseer | graphunet |  | 10 | 0 |
| 996907b8ef99 | citeseer | graphunet |  | 10 | 1 |
| 5b086c9adcfc | citeseer | graphunet |  | 10 | 2 |
| 6eb3349ee8b1 | citeseer | graphunet |  | 10 | 3 |
| 3ad49f7a4021 | citeseer | graphunet |  | 10 | 4 |
| 6fb5199fcd47 | citeseer | graphunet |  | 10 | 5 |
| 5104165efefd | citeseer | graphunet |  | 10 | 6 |
| 10e4814e2d76 | citeseer | graphunet |  | 10 | 7 |
| afc2697a1d0e | citeseer | graphunet |  | 10 | 8 |
| 9ea75f34a10d | citeseer | graphunet |  | 10 | 9 |
| 9afb4d88e1f8 | citeseer | graphunet |  | 20 | 0 |
| 61d2f10f27a8 | citeseer | graphunet |  | 20 | 1 |
| 119c53013d8d | citeseer | graphunet |  | 20 | 2 |
| cc5330546b40 | citeseer | graphunet |  | 20 | 3 |
| 97d0cb4ed6e6 | citeseer | graphunet |  | 20 | 4 |
| 79a258847720 | citeseer | graphunet |  | 20 | 5 |
| 303735998583 | citeseer | graphunet |  | 20 | 6 |
| 42bd90026045 | citeseer | graphunet |  | 20 | 7 |
| cac851259e70 | citeseer | graphunet |  | 20 | 8 |
| fc957cca9a74 | citeseer | graphunet |  | 20 | 9 |
| 8313d7680997 | citeseer | graphunet |  | 30 | 0 |
| c9b73f81fcca | citeseer | graphunet |  | 30 | 1 |
| e61484ee4a97 | citeseer | graphunet |  | 30 | 2 |
| a862b84b1ded | citeseer | graphunet |  | 30 | 3 |
| 3642b47f414f | citeseer | graphunet |  | 30 | 4 |
| c05f1ca1f54e | citeseer | graphunet |  | 30 | 5 |
| e7f1a174af77 | citeseer | graphunet |  | 30 | 6 |
| 2de49d58e088 | citeseer | graphunet |  | 30 | 7 |
| e5b17293a87d | citeseer | graphunet |  | 30 | 8 |
| 6be0f80a9640 | citeseer | graphunet |  | 30 | 9 |
| 24b4ee68c52d | pubmed | nhdp |  | 2 | 7 |
| ee362a664f82 | pubmed | nhdp |  | 4 | 1 |
| 8f8033ffb874 | pubmed | nhdp |  | 4 | 3 |
| c547baf1bfcb | pubmed | nhdp |  | 4 | 4 |
| a93a6578da08 | pubmed | nhdp |  | 4 | 5 |
| acfda53c2d21 | pubmed | nhdp |  | 4 | 6 |
| c24efc432f96 | pubmed | nhdp |  | 4 | 7 |
| 648c02e2c0de | pubmed | nhdp |  | 4 | 9 |
| 8400b8cd98df | pubmed | nhdp |  | 8 | 0 |
| 63b487559848 | pubmed | nhdp |  | 8 | 1 |
| 84d481732224 | pubmed | nhdp |  | 8 | 2 |
| 83eb3addc31a | pubmed | nhdp |  | 8 | 3 |
| 9c826389d617 | pubmed | nhdp |  | 8 | 4 |
| 6251ba46f923 | pubmed | nhdp |  | 8 | 5 |
| ceedc5e4e050 | pubmed | nhdp |  | 8 | 6 |
| 6eda7cd91974 | pubmed | nhdp |  | 8 | 7 |
| c1e1449169d3 | pubmed | nhdp |  | 8 | 8 |
| 0d29d33200b1 | pubmed | nhdp |  | 8 | 9 |
| 13ccc718f7ca | pubmed | nhdp |  | 10 | 0 |
| 53130c0b2368 | pubmed | nhdp |  | 10 | 1 |
| 343133b30aeb | pubmed | nhdp |  | 10 | 3 |
| 516e928de067 | pubmed | nhdp |  | 10 | 4 |
| 36478f52fe2f | pubmed | nhdp |  | 10 | 5 |
| 9e8b246cdde4 | pubmed | nhdp |  | 10 | 6 |
| a48ef64350c2 | pubmed | nhdp |  | 10 | 7 |
| ffca30fd5b82 | pubmed | nhdp |  | 10 | 8 |
| c0e5cd0c3e89 | pubmed | nhdp |  | 10 | 9 |
| 1237b88dbdc5 | pubmed | nhdp |  | 20 | 0 |
| 3a7a298dd5b2 | pubmed | nhdp |  | 20 | 1 |
| d6b0e377c361 | pubmed | nhdp |  | 20 | 2 |
| 55311453553b | pubmed | nhdp |  | 20 | 3 |
| 8820f2679365 | pubmed | nhdp |  | 20 | 4 |
| 05312a6f5001 | pubmed | nhdp |  | 20 | 6 |
| 8c890e9c3860 | pubmed | nhdp |  | 20 | 7 |
| 77c184be6d05 | pubmed | nhdp |  | 20 | 8 |
| 38410436a157 | pubmed | nhdp |  | 20 | 9 |
| 26e6613b8031 | pubmed | nhdp |  | 30 | 0 |
| 2ab0f8c77242 | pubmed | nhdp |  | 30 | 1 |
| aed402d4192e | pubmed | nhdp |  | 30 | 2 |
| 8404040c331f | pubmed | nhdp |  | 30 | 3 |
| ff2e8729e839 | pubmed | nhdp |  | 30 | 4 |
| 011bb5e3aa0d | pubmed | nhdp |  | 30 | 5 |
| e3fd5014afa9 | pubmed | nhdp |  | 30 | 6 |
| e4cd8a9aa761 | pubmed | nhdp |  | 30 | 7 |
| 9bfce1a57988 | pubmed | nhdp |  | 30 | 8 |
| cf780d1e0cc1 | pubmed | nhdp |  | 30 | 9 |
| 404aee5eb78b | pubmed | gcn |  | 2 | 0 |
| 5bab88138665 | pubmed | gcn |  | 2 | 1 |
| 618824d822d7 | pubmed | gcn |  | 2 | 2 |
| 1fb73e281553 | pubmed | gcn |  | 2 | 3 |
| 1587e46ae0ab | pubmed | gcn |  | 2 | 4 |
| e9b5638069ce | pubmed | gcn |  | 2 | 5 |
| 043fce4d2f40 | pubmed | gcn |  | 2 | 6 |
| dd6473c3813d | pubmed | gcn |  | 2 | 7 |
| 54b8ff3fbeb4 | pubmed | gcn |  | 2 | 8 |
| cb5b338a80b9 | pubmed | gcn |  | 2 | 9 |
| 346d0351f04b | pubmed | gcn |  | 4 | 0 |
| a561006b7a41 | pubmed | gcn |  | 4 | 1 |
| d29f64fb7aa6 | pubmed | gcn |  | 4 | 2 |
| 94ab9b53d76a | pubmed | gcn |  | 4 | 3 |
| d8e68e3d6c8d | pubmed | gcn |  | 4 | 4 |
| 96925c6ed1b5 | pubmed | gcn |  | 4 | 5 |
| e23ef4c1516e | pubmed | gcn |  | 4 | 6 |
| 6b7502820fa4 | pubmed | gcn |  | 4 | 7 |
| d1cce53e04c6 | pubmed | gcn |  | 4 | 8 |
| 99dcf7fbc8b3 | pubmed | gcn |  | 4 | 9 |
| 05a071fa4fde | pubmed | gcn |  | 8 | 0 |
| 187d09224e33 | pubmed | gcn |  | 8 | 1 |
| c9bc8f752dad | pubmed | gcn |  | 8 | 2 |
| 17bee3465d7b | pubmed | gcn |  | 8 | 3 |
| 97d70ddca436 | pubmed | gcn |  | 8 | 4 |
| e43244d4f3cd | pubmed | gcn |  | 8 | 5 |
| 225a0e4538a1 | pubmed | gcn |  | 8 | 6 |
| 6617951150e2 | pubmed | gcn |  | 8 | 7 |
| 9664f276b217 | pubmed | gcn |  | 8 | 8 |
| 255e31d5449d | pubmed | gcn |  | 8 | 9 |
| a0ef681c8238 | pubmed | gcn |  | 10 | 0 |
| 9e565c409e3a | pubmed | gcn |  | 10 | 1 |
| 82ceb6ac21bd | pubmed | gcn |  | 10 | 2 |
| 830d8c3c0938 | pubmed | gcn |  | 10 | 3 |
| cb80f8822935 | pubmed | gcn |  | 10 | 4 |
| 3489eab0423a | pubmed | gcn |  | 10 | 5 |
| 5f7d28dab2ec | pubmed | gcn |  | 10 | 6 |
| b3f4efa689fa | pubmed | gcn |  | 10 | 7 |
| e4b92a50cc3f | pubmed | gcn |  | 10 | 8 |
| 3fcb3bc7ccb0 | pubmed | gcn |  | 10 | 9 |
| 0e18a0f905bc | pubmed | gcn |  | 20 | 0 |
| 20417c59d5c0 | pubmed | gcn |  | 20 | 1 |
| eb81e6504b38 | pubmed | gcn |  | 20 | 2 |
| c6d4e35d2fd5 | pubmed | gcn |  | 20 | 3 |
| 756c5e3d0020 | pubmed | gcn |  | 20 | 4 |
| 4ae6ec45c17c | pubmed | gcn |  | 20 | 5 |
| 723ec5043e4f | pubmed | gcn |  | 20 | 6 |
| ded2bc165302 | pubmed | gcn |  | 20 | 7 |
| 71a989cc13ba | pubmed | gcn |  | 20 | 8 |
| 47ff8cac82b3 | pubmed | gcn |  | 20 | 9 |
| 1b5b947bde82 | pubmed | gcn |  | 30 | 0 |
| 727290d4080a | pubmed | gcn |  | 30 | 1 |
| 9a4af3392aa4 | pubmed | gcn |  | 30 | 2 |
| 37e2be8badc3 | pubmed | gcn |  | 30 | 3 |
| 5306e63307c2 | pubmed | gcn |  | 30 | 4 |
| 8be7b9124cee | pubmed | gcn |  | 30 | 5 |
| e6c4d636603f | pubmed | gcn |  | 30 | 6 |
| e6c49c9b73a5 | pubmed | gcn |  | 30 | 7 |
| 1532b7a4b6c8 | pubmed | gcn |  | 30 | 8 |
| b71f128f55e8 | pubmed | gcn |  | 30 | 9 |
| b7cc97557fa1 | pubmed | gat |  | 2 | 0 |
| f60c787ffef1 | pubmed | gat |  | 2 | 1 |
| 7421d094a0b8 | pubmed | gat |  | 2 | 2 |
| c4833d4f53b0 | pubmed | gat |  | 2 | 3 |
| 229efd2e3918 | pubmed | gat |  | 2 | 4 |
| 5c8f6e92b979 | pubmed | gat |  | 2 | 5 |
| 1aca3e3c53db | pubmed | gat |  | 2 | 6 |
| c5beb01d36a3 | pubmed | gat |  | 2 | 7 |
| 3d7b946aed6e | pubmed | gat |  | 2 | 8 |
| d742511ee8ec | pubmed | gat |  | 2 | 9 |
| fc1f362c1f9f | pubmed | gat |  | 4 | 0 |
| 56a6594cba70 | pubmed | gat |  | 4 | 1 |
| 163e7b9724af | pubmed | gat |  | 4 | 2 |
| 6c5fa7587fef | pubmed | gat |  | 4 | 3 |
| 5c25139b9ba3 | pubmed | gat |  | 4 | 4 |
| d5ab05d1d308 | pubmed | gat |  | 4 | 5 |
| 8d72ff1fc183 | pubmed | gat |  | 4 | 6 |
| 2f44b7969a2b | pubmed | gat |  | 4 | 7 |
| e3f282fbe16b | pubmed | gat |  | 4 | 8 |
| eab9045d9f0d | pubmed | gat |  | 4 | 9 |
| 821a9df36ada | pubmed | gat |  | 8 | 0 |
| ea23056e5b1a | pubmed | gat |  | 8 | 1 |
| 9645e2b95e1d | pubmed | gat |  | 8 | 2 |
| 88899428151e | pubmed | gat |  | 8 | 3 |
| 5f5cf8f70402 | pubmed | gat |  | 8 | 4 |
| 5c6bcb1f55a2 | pubmed | gat |  | 8 | 5 |
| 054db20f96b8 | pubmed | gat |  | 8 | 6 |
| 98f6f95e12f1 | pubmed | gat |  | 8 | 7 |
| b6ff1f1b0cf2 | pubmed | gat |  | 8 | 8 |
| 003cfc72c50b | pubmed | gat |  | 8 | 9 |
| 350676b425ab | pubmed | gat |  | 10 | 7 |
| 02a516e5fe2e | pubmed | gat |  | 20 | 1 |
| 638e2ca61ecc | pubmed | gat |  | 20 | 3 |
| d095e415630f | pubmed | gat |  | 20 | 4 |
| 04afee847532 | pubmed | gat |  | 20 | 5 |
| 42fb5a391897 | pubmed | gat |  | 20 | 6 |
| cc6061fed87f | pubmed | gat |  | 20 | 7 |
| 261795f916bf | pubmed | gat |  | 20 | 9 |
| c65386cb48a6 | pubmed | gat |  | 30 | 0 |
| 06bc5dead792 | pubmed | gat |  | 30 | 1 |
| ab4a0bc6fd31 | pubmed | gat |  | 30 | 2 |
| a0340a42f781 | pubmed | gat |  | 30 | 3 |
| ef92febc9f80 | pubmed | gat |  | 30 | 4 |
| fcd83ddd0e74 | pubmed | gat |  | 30 | 5 |
| 4af961cf8132 | pubmed | gat |  | 30 | 6 |
| 98c5bd1e3148 | pubmed | gat |  | 30 | 7 |
| 07b86c5208c1 | pubmed | gat |  | 30 | 8 |
| 8121c5b8f018 | pubmed | gat |  | 30 | 9 |
| 93ec40c99572 | pubmed | sgc |  | 2 | 0 |
| 1bdc709b8199 | pubmed | sgc |  | 2 | 1 |
| 4f4b80f07647 | pubmed | sgc |  | 2 | 3 |
| b83d212cd50a | pubmed | sgc |  | 2 | 4 |
| 58074ffa9cb1 | pubmed | sgc |  | 2 | 5 |
| e8620bb8ba83 | pubmed | sgc |  | 2 | 6 |
| 35f609df3700 | pubmed | sgc |  | 2 | 7 |
| 3bde24fbd088 | pubmed | sgc |  | 2 | 8 |
| e9e1a231e86f | pubmed | sgc |  | 2 | 9 |
| ba0195b7f075 | pubmed | sgc |  | 4 | 0 |
| f62751cb56fa | pubmed | sgc |  | 4 | 1 |
| 9ef6373abb2a | pubmed | sgc |  | 4 | 2 |
| cfcba0db51ee | pubmed | sgc |  | 4 | 3 |
| 40329b5c8f8a | pubmed | sgc |  | 4 | 4 |
| aac021f97441 | pubmed | sgc |  | 4 | 6 |
| 4014e3b2458e | pubmed | sgc |  | 4 | 7 |
| 8d81fc7d7b26 | pubmed | sgc |  | 4 | 8 |
| 0a33f42dc58c | pubmed | sgc |  | 4 | 9 |
| 16384a1137c2 | pubmed | sgc |  | 8 | 0 |
| 25804df69984 | pubmed | sgc |  | 8 | 1 |
| bc5bce4425ae | pubmed | sgc |  | 8 | 2 |
| 0ea6610a7e9d | pubmed | sgc |  | 8 | 3 |
| 21200b92f65b | pubmed | sgc |  | 8 | 4 |
| 573836a93759 | pubmed | sgc |  | 8 | 5 |
| 017067c6396d | pubmed | sgc |  | 8 | 6 |
| 019832456737 | pubmed | sgc |  | 8 | 7 |
| 67bfe6870a8d | pubmed | sgc |  | 8 | 8 |
| 5b668e750d36 | pubmed | sgc |  | 8 | 9 |
| c9a412bc6870 | pubmed | sgc |  | 10 | 0 |
| 2ec5f30c81e3 | pubmed | sgc |  | 10 | 1 |
| 65cf98aefd9d | pubmed | sgc |  | 10 | 2 |
| e673f5ae17c4 | pubmed | sgc |  | 10 | 3 |
| 7a21e44dd19a | pubmed | sgc |  | 10 | 4 |
| 141965eed60a | pubmed | sgc |  | 10 | 5 |
| 80fcb36438a0 | pubmed | sgc |  | 10 | 6 |
| 0ef784f26ea3 | pubmed | sgc |  | 10 | 7 |
| 539a475b668b | pubmed | sgc |  | 10 | 8 |
| 6684a12127fa | pubmed | sgc |  | 10 | 9 |
| 9e5208fca0d5 | pubmed | sgc |  | 20 | 0 |
| dc4e702bcaf6 | pubmed | sgc |  | 20 | 1 |
| ab872305cf6c | pubmed | sgc |  | 20 | 2 |
| 51fcfd45ca30 | pubmed | sgc |  | 20 | 3 |
| ffb4e0fb10a7 | pubmed | sgc |  | 20 | 4 |
| 6c32f4ed5c3b | pubmed | sgc |  | 20 | 5 |
| deb318a06c29 | pubmed | sgc |  | 20 | 6 |
| 5af0630ef9e1 | pubmed | sgc |  | 20 | 7 |
| 7ed2259d1ad7 | pubmed | sgc |  | 20 | 8 |
| 316e653acb42 | pubmed | sgc |  | 20 | 9 |
| b305e2c0c2ba | pubmed | sgc |  | 30 | 0 |
| fcf7d7873fba | pubmed | sgc |  | 30 | 1 |
| 0f9eb694e0ed | pubmed | sgc |  | 30 | 2 |
| 14684d03372d | pubmed | sgc |  | 30 | 3 |
| bb32075c5bb7 | pubmed | sgc |  | 30 | 4 |
| f30b9c1e1677 | pubmed | sgc |  | 30 | 5 |
| 90908f469f29 | pubmed | sgc |  | 30 | 6 |
| f36d56f7b0ce | pubmed | sgc |  | 30 | 7 |
| 95b092b002d5 | pubmed | sgc |  | 30 | 8 |
| 96150b664802 | pubmed | sgc |  | 30 | 9 |
| 9b7f6e281716 | pubmed | appnp |  | 2 | 0 |
| f2c84dac6338 | pubmed | appnp |  | 2 | 1 |
| 6cf3f02c09a6 | pubmed | appnp |  | 2 | 2 |
| e04511c584c2 | pubmed | appnp |  | 2 | 3 |
| 41ed9900ab78 | pubmed | appnp |  | 2 | 4 |
| b6847c4023fc | pubmed | appnp |  | 2 | 5 |
| 89754946c1c3 | pubmed | appnp |  | 2 | 6 |
| 5bf75bcd8b7c | pubmed | appnp |  | 2 | 7 |
| 7aaf12aa7cda | pubmed | appnp |  | 2 | 8 |
| 5c9da79133a4 | pubmed | appnp |  | 2 | 9 |
| 7a763d111c9e | pubmed | appnp |  | 4 | 0 |
| 6c89141c6317 | pubmed | appnp |  | 4 | 1 |
| 0e2e5f0333c0 | pubmed | appnp |  | 4 | 2 |
| cd9428432968 | pubmed | appnp |  | 4 | 3 |
| b4d143c99ae1 | pubmed | appnp |  | 4 | 4 |
| 657889ef1afd | pubmed | appnp |  | 4 | 5 |
| 9841b0b1c42a | pubmed | appnp |  | 4 | 6 |
| 379340a4ed44 | pubmed | appnp |  | 4 | 7 |
| 52805ab5c39c | pubmed | appnp |  | 4 | 8 |
| 835d11fa2296 | pubmed | appnp |  | 4 | 9 |
| a10d442e01b9 | pubmed | appnp |  | 8 | 0 |
| ecc9e9a52c3c | pubmed | appnp |  | 8 | 1 |
| 1513d102c8d0 | pubmed | appnp |  | 8 | 2 |
| 86f3767c128b | pubmed | appnp |  | 8 | 3 |
| 1af62f134363 | pubmed | appnp |  | 8 | 4 |
| 635bddb16462 | pubmed | appnp |  | 8 | 5 |
| df8d079e4358 | pubmed | appnp |  | 8 | 6 |
| 02757256b559 | pubmed | appnp |  | 8 | 7 |
| c28d224f19b6 | pubmed | appnp |  | 8 | 8 |
| cd53ce411a57 | pubmed | appnp |  | 8 | 9 |
| 4a31395b8dce | pubmed | appnp |  | 10 | 0 |
| 71b24bb9185a | pubmed | appnp |  | 10 | 1 |
| 6430421ecf9a | pubmed | appnp |  | 10 | 2 |
| 059084bf2fea | pubmed | appnp |  | 10 | 3 |
| fe2cdf63939e | pubmed | appnp |  | 10 | 4 |
| e868e5b77348 | pubmed | appnp |  | 10 | 5 |
| 59b0aba16eef | pubmed | appnp |  | 10 | 6 |
| 3d1ce677fcaf | pubmed | appnp |  | 10 | 7 |
| 68dc16171396 | pubmed | appnp |  | 10 | 8 |
| a65105f1901d | pubmed | appnp |  | 10 | 9 |
| 000cb238307c | pubmed | appnp |  | 20 | 0 |
| 37126d713f57 | pubmed | appnp |  | 20 | 1 |
| 40bb953aea0a | pubmed | appnp |  | 20 | 2 |
| a47dee0fe716 | pubmed | appnp |  | 20 | 3 |
| 6ec2099c4765 | pubmed | appnp |  | 20 | 4 |
| f2d968d87be8 | pubmed | appnp |  | 20 | 5 |
| e09b9cc54e12 | pubmed | appnp |  | 20 | 6 |
| 463a080dfe3e | pubmed | appnp |  | 20 | 7 |
| c0ea2b00bab7 | pubmed | appnp |  | 20 | 8 |
| ca3acd5707ca | pubmed | appnp |  | 20 | 9 |
| 3b47c5242286 | pubmed | appnp |  | 30 | 0 |
| 4885fb1acbab | pubmed | appnp |  | 30 | 1 |
| 2716131de2ae | pubmed | appnp |  | 30 | 2 |
| 1805ff54ff51 | pubmed | appnp |  | 30 | 3 |
| 222dd9e29736 | pubmed | appnp |  | 30 | 4 |
| e0e2d1b934d5 | pubmed | appnp |  | 30 | 5 |
| 4c57284a4b89 | pubmed | appnp |  | 30 | 6 |
| 69b8e5c03f52 | pubmed | appnp |  | 30 | 7 |
| e588f4acad88 | pubmed | appnp |  | 30 | 8 |
| 5e9ebeaa0e49 | pubmed | appnp |  | 30 | 9 |
| 8b956fba7488 | pubmed | arma |  | 2 | 7 |
| d37ab7d5c45e | pubmed | arma |  | 4 | 1 |
| db543cdb4140 | pubmed | arma |  | 4 | 3 |
| 292e91f5c597 | pubmed | arma |  | 4 | 4 |
| 8c4f4dfc7eeb | pubmed | arma |  | 4 | 5 |
| ff74e245576e | pubmed | arma |  | 4 | 6 |
| c74a27a8dec0 | pubmed | arma |  | 4 | 7 |
| 6112a3de8bec | pubmed | arma |  | 4 | 9 |
| 0b02f0307af1 | pubmed | arma |  | 8 | 0 |
| 238ffe8dd07a | pubmed | arma |  | 8 | 1 |
| 42c0346edb0f | pubmed | arma |  | 8 | 2 |
| 965c7e984bab | pubmed | arma |  | 8 | 3 |
| dc733e809978 | pubmed | arma |  | 8 | 4 |
| e14206206b86 | pubmed | arma |  | 8 | 5 |
| 74dd06a5e9ee | pubmed | arma |  | 8 | 6 |
| 1bc75f3a6123 | pubmed | arma |  | 8 | 7 |
| e09d55d0e758 | pubmed | arma |  | 8 | 8 |
| 5cdd0c43ef60 | pubmed | arma |  | 8 | 9 |
| f512fa017bd0 | pubmed | arma |  | 10 | 0 |
| 80747d54493e | pubmed | arma |  | 10 | 1 |
| 744c04108f12 | pubmed | arma |  | 10 | 3 |
| befa2f82d4fc | pubmed | arma |  | 10 | 4 |
| d7e479a333c0 | pubmed | arma |  | 10 | 5 |
| bb47eb56c0f0 | pubmed | arma |  | 10 | 6 |
| 5803d400244d | pubmed | arma |  | 10 | 7 |
| 548f6e844c3c | pubmed | arma |  | 10 | 8 |
| 3c40f6c21c41 | pubmed | arma |  | 10 | 9 |
| 5ef29705d88b | pubmed | arma |  | 20 | 0 |
| 58989798a340 | pubmed | arma |  | 20 | 1 |
| 7bcd7e5b7a28 | pubmed | arma |  | 20 | 2 |
| 8aa5200cbdbb | pubmed | arma |  | 20 | 3 |
| 922491fc99fa | pubmed | arma |  | 20 | 4 |
| 0d9230f0321e | pubmed | arma |  | 20 | 6 |
| e9e8f8283a36 | pubmed | arma |  | 20 | 7 |
| fbfc171b5b55 | pubmed | arma |  | 20 | 8 |
| 692043e34816 | pubmed | arma |  | 20 | 9 |
| 9e55a9985207 | pubmed | arma |  | 30 | 0 |
| ad6f74321e0c | pubmed | arma |  | 30 | 1 |
| b7f7c9c7dd45 | pubmed | arma |  | 30 | 2 |
| 07651a838abf | pubmed | arma |  | 30 | 3 |
| 214c0f4332df | pubmed | arma |  | 30 | 4 |
| 6c0b7a99cda4 | pubmed | arma |  | 30 | 5 |
| 2299c819b069 | pubmed | arma |  | 30 | 6 |
| b1013868aaad | pubmed | arma |  | 30 | 7 |
| 804c5b5e9755 | pubmed | arma |  | 30 | 8 |
| 28372e94c823 | pubmed | arma |  | 30 | 9 |
| 9724f9027d42 | pubmed | gcnii |  | 2 | 0 |
| 4d0a1c348f49 | pubmed | gcnii |  | 2 | 1 |
| 73be0628c916 | pubmed | gcnii |  | 2 | 2 |
| cb500a798309 | pubmed | gcnii |  | 2 | 3 |
| 1b374d2862fa | pubmed | gcnii |  | 2 | 4 |
| e1eb902c72cc | pubmed | gcnii |  | 2 | 5 |
| a21fb46ccecc | pubmed | gcnii |  | 2 | 6 |
| a249e5fc0d85 | pubmed | gcnii |  | 2 | 7 |
| 5e36997ab6da | pubmed | gcnii |  | 2 | 8 |
| f5f9259e3037 | pubmed | gcnii |  | 2 | 9 |
| 90f572d24a96 | pubmed | gcnii |  | 4 | 0 |
| cecf20aaee81 | pubmed | gcnii |  | 4 | 1 |
| 8bbe2c754c14 | pubmed | gcnii |  | 4 | 2 |
| 44cdb0ec1898 | pubmed | gcnii |  | 4 | 3 |
| 2f572a070b6c | pubmed | gcnii |  | 4 | 4 |
| dc8c9ce43766 | pubmed | gcnii |  | 4 | 5 |
| f20d107f3518 | pubmed | gcnii |  | 4 | 6 |
| 61c2ed4f9519 | pubmed | gcnii |  | 4 | 7 |
| 048ac012b4c1 | pubmed | gcnii |  | 4 | 8 |
| 3f00909ba946 | pubmed | gcnii |  | 4 | 9 |
| 4191f0fff9dd | pubmed | gcnii |  | 8 | 0 |
| 47efece5d1f8 | pubmed | gcnii |  | 8 | 1 |
| 30b55e945f24 | pubmed | gcnii |  | 8 | 2 |
| 4c8fa24bf7c0 | pubmed | gcnii |  | 8 | 3 |
| 8fa2c80de0c6 | pubmed | gcnii |  | 8 | 4 |
| 5880c835efc6 | pubmed | gcnii |  | 8 | 5 |
| ba3b01e31ccc | pubmed | gcnii |  | 8 | 6 |
| dd6b9409a42d | pubmed | gcnii |  | 8 | 7 |
| 51c423604d84 | pubmed | gcnii |  | 8 | 8 |
| 2a10f6ddbcca | pubmed | gcnii |  | 8 | 9 |
| f994544e696d | pubmed | gcnii |  | 10 | 0 |
| 5789e152fbef | pubmed | gcnii |  | 10 | 1 |
| 269271f55d1d | pubmed | gcnii |  | 10 | 2 |
| 4e3e9e161886 | pubmed | gcnii |  | 10 | 3 |
| 56e35fb7fed9 | pubmed | gcnii |  | 10 | 4 |
| b25993424faa | pubmed | gcnii |  | 10 | 5 |
| d835d0016010 | pubmed | gcnii |  | 10 | 6 |
| 652eab22df38 | pubmed | gcnii |  | 10 | 7 |
| 5a08ca4adf5a | pubmed | gcnii |  | 10 | 8 |
| 378278971428 | pubmed | gcnii |  | 10 | 9 |
| f6cd221dd572 | pubmed | gcnii |  | 20 | 0 |
| 5d01ab42e8ab | pubmed | gcnii |  | 20 | 1 |
| 71337951ee60 | pubmed | gcnii |  | 20 | 2 |
| 8de6d7f80d8a | pubmed | gcnii |  | 20 | 3 |
| 60654c10c158 | pubmed | gcnii |  | 20 | 4 |
| 44eefdde4ca8 | pubmed | gcnii |  | 20 | 5 |
| cc791e2cc3a6 | pubmed | gcnii |  | 20 | 6 |
| e89ec012e475 | pubmed | gcnii |  | 20 | 7 |
| c2ea114f3b5b | pubmed | gcnii |  | 20 | 8 |
| 791e45d4b1e4 | pubmed | gcnii |  | 20 | 9 |
| 1c938906bb1b | pubmed | gcnii |  | 30 | 0 |
| 23cfee5bac61 | pubmed | gcnii |  | 30 | 1 |
| 60599438e65c | pubmed | gcnii |  | 30 | 2 |
| 108eae651c0d | pubmed | gcnii |  | 30 | 3 |
| 8fae10d5c40f | pubmed | gcnii |  | 30 | 4 |
| a37b3d38487e | pubmed | gcnii |  | 30 | 5 |
| 7acbf151685f | pubmed | gcnii |  | 30 | 6 |
| 13b1e90a871e | pubmed | gcnii |  | 30 | 7 |
| b42f293fdc7e | pubmed | gcnii |  | 30 | 8 |
| e6571f740df4 | pubmed | gcnii |  | 30 | 9 |
| dcb86f90c5f2 | pubmed | pairnorm |  | 2 | 0 |
| a38a09fb8b02 | pubmed | pairnorm |  | 2 | 1 |
| 8fae077c3ec7 | pubmed | pairnorm |  | 2 | 2 |
| c30313130041 | pubmed | pairnorm |  | 2 | 3 |
| 2873f1df4b56 | pubmed | pairnorm |  | 2 | 4 |
| 5991da50f910 | pubmed | pairnorm |  | 2 | 5 |
| 9fcaf90cf206 | pubmed | pairnorm |  | 2 | 6 |
| 23d3e54959a0 | pubmed | pairnorm |  | 2 | 7 |
| e74b4062f308 | pubmed | pairnorm |  | 2 | 8 |
| 3140ece40d83 | pubmed | pairnorm |  | 2 | 9 |
| 98f58fcec121 | pubmed | pairnorm |  | 4 | 0 |
| 689033015132 | pubmed | pairnorm |  | 4 | 1 |
| 49f0d7065275 | pubmed | pairnorm |  | 4 | 2 |
| 742a082193d2 | pubmed | pairnorm |  | 4 | 3 |
| 9c48a8a3bd1f | pubmed | pairnorm |  | 4 | 4 |
| bc0c37feeb42 | pubmed | pairnorm |  | 4 | 5 |
| 381b58b67267 | pubmed | pairnorm |  | 4 | 6 |
| 9cdf65c2ae03 | pubmed | pairnorm |  | 4 | 7 |
| 9cec974c3da2 | pubmed | pairnorm |  | 4 | 8 |
| a2ce76317220 | pubmed | pairnorm |  | 4 | 9 |
| eca1cf8effe7 | pubmed | pairnorm |  | 8 | 0 |
| 4182e39a2f4d | pubmed | pairnorm |  | 8 | 1 |
| 394e954fb245 | pubmed | pairnorm |  | 8 | 2 |
| a6fe2bbc01ac | pubmed | pairnorm |  | 8 | 3 |
| a007928d363f | pubmed | pairnorm |  | 8 | 4 |
| 303b6dd9eb0b | pubmed | pairnorm |  | 8 | 5 |
| c53631f4d8da | pubmed | pairnorm |  | 8 | 6 |
| e586dece61f3 | pubmed | pairnorm |  | 8 | 7 |
| 55126efd648c | pubmed | pairnorm |  | 8 | 8 |
| 2758ce97dcff | pubmed | pairnorm |  | 8 | 9 |
| 5b3933d02c80 | pubmed | pairnorm |  | 10 | 7 |
| 528372c15488 | pubmed | pairnorm |  | 20 | 1 |
| baea1437e9a3 | pubmed | pairnorm |  | 20 | 3 |
| 6ab060262cb3 | pubmed | pairnorm |  | 20 | 4 |
| a9245041c054 | pubmed | pairnorm |  | 20 | 5 |
| 58a42deb321c | pubmed | pairnorm |  | 20 | 6 |
| 17b91f4456ce | pubmed | pairnorm |  | 20 | 7 |
| 65ddb75e0373 | pubmed | pairnorm |  | 20 | 9 |
| d9e1c70315cb | pubmed | pairnorm |  | 30 | 0 |
| 3013b318921a | pubmed | pairnorm |  | 30 | 1 |
| cea67b64be94 | pubmed | pairnorm |  | 30 | 2 |
| 8fc1b86a09b3 | pubmed | pairnorm |  | 30 | 3 |
| 4fd239558f77 | pubmed | pairnorm |  | 30 | 4 |
| 0d2d6221c13e | pubmed | pairnorm |  | 30 | 5 |
| 5df94cc02155 | pubmed | pairnorm |  | 30 | 6 |
| 840560d30299 | pubmed | pairnorm |  | 30 | 7 |
| 670f005e3c00 | pubmed | pairnorm |  | 30 | 8 |
| 58b00b816565 | pubmed | pairnorm |  | 30 | 9 |
| 1aa5bc12f989 | pubmed | dropedge |  | 2 | 0 |
| 9f9fa1c8c68a | pubmed | dropedge |  | 2 | 1 |
| 22154119c5af | pubmed | dropedge |  | 2 | 3 |
| 06207cfa621c | pubmed | dropedge |  | 2 | 4 |
| f29f689dba26 | pubmed | dropedge |  | 2 | 5 |
| 958cfe28499b | pubmed | dropedge |  | 2 | 6 |
| dd96a6a4f694 | pubmed | dropedge |  | 2 | 7 |
| 7a25a144a87e | pubmed | dropedge |  | 2 | 8 |
| de85e93c260f | pubmed | dropedge |  | 2 | 9 |
| c7bbf5093106 | pubmed | dropedge |  | 4 | 0 |
| b5a49ce1b668 | pubmed | dropedge |  | 4 | 1 |
| 26ec79fb85f0 | pubmed | dropedge |  | 4 | 2 |
| c49e67b16f73 | pubmed | dropedge |  | 4 | 3 |
| cb3357feacd8 | pubmed | dropedge |  | 4 | 4 |
| 87cb38336ba0 | pubmed | dropedge |  | 4 | 6 |
| a1f609a74892 | pubmed | dropedge |  | 4 | 7 |
| 174de6ee2b76 | pubmed | dropedge |  | 4 | 8 |
| fa0398b9e03e | pubmed | dropedge |  | 4 | 9 |
| d53b7312f748 | pubmed | dropedge |  | 8 | 0 |
| ad1a4488d471 | pubmed | dropedge |  | 8 | 1 |
| 6c04fe7995e3 | pubmed | dropedge |  | 8 | 2 |
| dd45f578d8e6 | pubmed | dropedge |  | 8 | 3 |
| 9c6d045eec15 | pubmed | dropedge |  | 8 | 4 |
| 5a086747be0a | pubmed | dropedge |  | 8 | 5 |
| 5a1f9d7d48a7 | pubmed | dropedge |  | 8 | 6 |
| ae129f8b7423 | pubmed | dropedge |  | 8 | 7 |
| 5fa7e0a8df61 | pubmed | dropedge |  | 8 | 8 |
| 87aab7f29879 | pubmed | dropedge |  | 8 | 9 |
| c25363c85877 | pubmed | dropedge |  | 10 | 0 |
| 0f7a7d5f99be | pubmed | dropedge |  | 10 | 1 |
| d120dd533b1b | pubmed | dropedge |  | 10 | 2 |
| 2f6bc865bf60 | pubmed | dropedge |  | 10 | 3 |
| 0f1d80a49438 | pubmed | dropedge |  | 10 | 4 |
| 4dc0f15034b4 | pubmed | dropedge |  | 10 | 5 |
| 01222b6021a8 | pubmed | dropedge |  | 10 | 6 |
| 95f6ef17452c | pubmed | dropedge |  | 10 | 7 |
| 9bb7c0584e1a | pubmed | dropedge |  | 10 | 8 |
| d8e4be08337c | pubmed | dropedge |  | 10 | 9 |
| b3db4ad784ce | pubmed | dropedge |  | 20 | 0 |
| 05ca4f1bc1c4 | pubmed | dropedge |  | 20 | 1 |
| e893cb2de64c | pubmed | dropedge |  | 20 | 2 |
| afdf65efe586 | pubmed | dropedge |  | 20 | 3 |
| 444347a975f6 | pubmed | dropedge |  | 20 | 4 |
| ed03dc50eeba | pubmed | dropedge |  | 20 | 5 |
| ca63f8c12302 | pubmed | dropedge |  | 20 | 6 |
| 36abcf24ae40 | pubmed | dropedge |  | 20 | 7 |
| 9d024eb026e3 | pubmed | dropedge |  | 20 | 8 |
| 59d372c4387f | pubmed | dropedge |  | 20 | 9 |
| 3a83b17b86aa | pubmed | dropedge |  | 30 | 0 |
| 5a4eb0b89287 | pubmed | dropedge |  | 30 | 1 |
| cea4a962a62e | pubmed | dropedge |  | 30 | 2 |
| 54a8edd4a512 | pubmed | dropedge |  | 30 | 3 |
| 37cc5fe93807 | pubmed | dropedge |  | 30 | 4 |
| 81973645b50c | pubmed | dropedge |  | 30 | 5 |
| 35d2ae3f5d1e | pubmed | dropedge |  | 30 | 6 |
| cb37ee4356fc | pubmed | dropedge |  | 30 | 7 |
| b539a34ff1bb | pubmed | dropedge |  | 30 | 8 |
| 9dce96a61a7e | pubmed | dropedge |  | 30 | 9 |
| e6d509d910b0 | pubmed | graphunet |  | 2 | 0 |
| 682dac060c9b | pubmed | graphunet |  | 2 | 1 |
| b0bbc5d41ab4 | pubmed | graphunet |  | 2 | 2 |
| 102a3751837d | pubmed | graphunet |  | 2 | 3 |
| e54567169f7f | pubmed | graphunet |  | 2 | 4 |
| 8cd593482636 | pubmed | graphunet |  | 2 | 5 |
| d9fecf07d632 | pubmed | graphunet |  | 2 | 6 |
| 5fccc8c44396 | pubmed | graphunet |  | 2 | 7 |
| 9f82c8bf9b21 | pubmed | graphunet |  | 2 | 8 |
| e38ba077c462 | pubmed | graphunet |  | 2 | 9 |
| e29eb932a6d6 | pubmed | graphunet |  | 4 | 0 |
| 4d162c74ca31 | pubmed | graphunet |  | 4 | 1 |
| 990113606501 | pubmed | graphunet |  | 4 | 2 |
| 348e73c1ea69 | pubmed | graphunet |  | 4 | 3 |
| 892e6157de3d | pubmed | graphunet |  | 4 | 4 |
| ba12fe1aad35 | pubmed | graphunet |  | 4 | 5 |
| f85a05ebb65b | pubmed | graphunet |  | 4 | 6 |
| e71aa101f5b2 | pubmed | graphunet |  | 4 | 7 |
| 9d03d3deb166 | pubmed | graphunet |  | 4 | 8 |
| d3948ee1b0ff | pubmed | graphunet |  | 4 | 9 |
| 6458ec66dac0 | pubmed | graphunet |  | 8 | 0 |
| a8d784b73439 | pubmed | graphunet |  | 8 | 1 |
| 74c7ca376b48 | pubmed | graphunet |  | 8 | 2 |
| 64e0e8534e87 | pubmed | graphunet |  | 8 | 3 |
| 534399b9e644 | pubmed | graphunet |  | 8 | 4 |
| 1a0d88af08ca | pubmed | graphunet |  | 8 | 5 |
| 3ccdbf6de313 | pubmed | graphunet |  | 8 | 6 |
| 85b4f8fc67b7 | pubmed | graphunet |  | 8 | 7 |
| 2d6d866a716d | pubmed | graphunet |  | 8 | 8 |
| 58552034902c | pubmed | graphunet |  | 8 | 9 |
| 1b64b86e716a | pubmed | graphunet |  | 10 | 0 |
| 29854d8d5724 | pubmed | graphunet |  | 10 | 1 |
| 9986c4f1f163 | pubmed | graphunet |  | 10 | 2 |
| 8ce16876b15d | pubmed | graphunet |  | 10 | 3 |
| 1cc4115e84e4 | pubmed | graphunet |  | 10 | 4 |
| 0bd3cac4a989 | pubmed | graphunet |  | 10 | 5 |
| 0fd8a101be6a | pubmed | graphunet |  | 10 | 6 |
| 8c0b3c064c86 | pubmed | graphunet |  | 10 | 7 |
| b8a310b1751f | pubmed | graphunet |  | 10 | 8 |
| d0c4aac4af29 | pubmed | graphunet |  | 10 | 9 |
| 6bb48c23e400 | pubmed | graphunet |  | 20 | 0 |
| 096eea9dcb52 | pubmed | graphunet |  | 20 | 1 |
| d16c0275fc73 | pubmed | graphunet |  | 20 | 2 |
| a83aad6caf8a | pubmed | graphunet |  | 20 | 3 |
| 0e88a9bae8eb | pubmed | graphunet |  | 20 | 4 |
| c2f5af83e5cd | pubmed | graphunet |  | 20 | 5 |
| 78dfa4a77ecb | pubmed | graphunet |  | 20 | 6 |
| de5de70cb3a9 | pubmed | graphunet |  | 20 | 7 |
| 304a43142cda | pubmed | graphunet |  | 20 | 8 |
| 656e8fe84dbe | pubmed | graphunet |  | 20 | 9 |
| a099db6081f3 | pubmed | graphunet |  | 30 | 0 |
| fc2d7a5922c7 | pubmed | graphunet |  | 30 | 1 |
| c11b1cd5dbb7 | pubmed | graphunet |  | 30 | 2 |
| a9243ab4f94c | pubmed | graphunet |  | 30 | 3 |
| fb8d971a94d5 | pubmed | graphunet |  | 30 | 4 |
| 8be6e849bb07 | pubmed | graphunet |  | 30 | 5 |
| dd6e3e769a75 | pubmed | graphunet |  | 30 | 6 |
| da02582dd96a | pubmed | graphunet |  | 30 | 7 |
| d910d0969c96 | pubmed | graphunet |  | 30 | 8 |
| fc17276f3e6f | pubmed | graphunet |  | 30 | 9 |
| 4a0e1903e24d | photo | nhdp |  | 2 | 7 |
| 81587eac21ba | photo | nhdp |  | 4 | 1 |
| b027e092d73a | photo | nhdp |  | 4 | 3 |
| df162fd8061f | photo | nhdp |  | 4 | 4 |
| 2fc072903601 | photo | nhdp |  | 4 | 5 |
| e1de35475986 | photo | nhdp |  | 4 | 6 |
| 00cb043e4df2 | photo | nhdp |  | 4 | 7 |
| 3d9fcee7082f | photo | nhdp |  | 4 | 9 |
| a02eaf3e5ae5 | photo | nhdp |  | 8 | 0 |
| 3f6557a1b701 | photo | nhdp |  | 8 | 1 |
| c57518208958 | photo | nhdp |  | 8 | 2 |
| 4df6bc6c7c06 | photo | nhdp |  | 8 | 3 |
| be649ab8dbd3 | photo | nhdp |  | 8 | 4 |
| 949c80116426 | photo | nhdp |  | 8 | 5 |
| 8379f586f3d6 | photo | nhdp |  | 8 | 6 |
| 0fc708155447 | photo | nhdp |  | 8 | 7 |
| 136b597c50c7 | photo | nhdp |  | 8 | 8 |
| cb8c9de1b9b8 | photo | nhdp |  | 8 | 9 |
| 8f8b92a35c17 | photo | nhdp |  | 10 | 0 |
| 3d7dec6ba2fc | photo | nhdp |  | 10 | 1 |
| e2f5b874b8dd | photo | nhdp |  | 10 | 3 |
| 20ff5e0f56a0 | photo | nhdp |  | 10 | 4 |
| 84347f85d980 | photo | nhdp |  | 10 | 5 |
| c923977d4bdb | photo | nhdp |  | 10 | 6 |
| 8e4a53ae4252 | photo | nhdp |  | 10 | 7 |
| 99556ef25df9 | photo | nhdp |  | 10 | 8 |
| c8bca4ce315f | photo | nhdp |  | 10 | 9 |
| 50b479398bb4 | photo | nhdp |  | 20 | 0 |
| 15a422f45ab0 | photo | nhdp |  | 20 | 1 |
| 9d90bfd8ea1e | photo | nhdp |  | 20 | 2 |
| 26b4a5e795a7 | photo | nhdp |  | 20 | 3 |
| 2ea9e7adecb0 | photo | nhdp |  | 20 | 4 |
| e4f271d54f9c | photo | nhdp |  | 20 | 6 |
| 60795bda7694 | photo | nhdp |  | 20 | 7 |
| 2bd0d7cb5540 | photo | nhdp |  | 20 | 8 |
| 8232b09df15a | photo | nhdp |  | 20 | 9 |
| c9a783066aef | photo | nhdp |  | 30 | 0 |
| 55e3649c9ade | photo | nhdp |  | 30 | 1 |
| f5864ca17703 | photo | nhdp |  | 30 | 2 |
| 9553e792ac2d | photo | nhdp |  | 30 | 3 |
| 23adebaf42e3 | photo | nhdp |  | 30 | 4 |
| d836e560c7b1 | photo | nhdp |  | 30 | 5 |
| fdd994089968 | photo | nhdp |  | 30 | 6 |
| b475e178fd0d | photo | nhdp |  | 30 | 7 |
| 3edb35621686 | photo | nhdp |  | 30 | 8 |
| 3c89f47cb957 | photo | nhdp |  | 30 | 9 |
| b6bba8d4c8e5 | photo | gcn |  | 2 | 0 |
| cebceae3280f | photo | gcn |  | 2 | 1 |
| 873d4df97f9c | photo | gcn |  | 2 | 2 |
| 240127dc6aae | photo | gcn |  | 2 | 3 |
| 49f28d1ff2d8 | photo | gcn |  | 2 | 4 |
| 8151372dd8ae | photo | gcn |  | 2 | 5 |
| 882b96e7bab4 | photo | gcn |  | 2 | 6 |
| c99c05441b10 | photo | gcn |  | 2 | 7 |
| ca7616a63864 | photo | gcn |  | 2 | 8 |
| 9d05dba5b8a2 | photo | gcn |  | 2 | 9 |
| a1fbd3f6a9e8 | photo | gcn |  | 4 | 0 |
| 2a8e1faca095 | photo | gcn |  | 4 | 1 |
| f00d72a31477 | photo | gcn |  | 4 | 2 |
| 672e447089a1 | photo | gcn |  | 4 | 3 |
| 52d32905dbb9 | photo | gcn |  | 4 | 4 |
| f203b0cb98a5 | photo | gcn |  | 4 | 5 |
| 177523baa7a8 | photo | gcn |  | 4 | 6 |
| 7b06e99372d5 | photo | gcn |  | 4 | 7 |
| 1a00028c3682 | photo | gcn |  | 4 | 8 |
| 2410d2ea1205 | photo | gcn |  | 4 | 9 |
| 34531bce3e3c | photo | gcn |  | 8 | 0 |
| a47ad73e6a94 | photo | gcn |  | 8 | 1 |
| d1c2037e17f6 | photo | gcn |  | 8 | 2 |
| e314c1a903db | photo | gcn |  | 8 | 3 |
| eb97eb7c8500 | photo | gcn |  | 8 | 4 |
| e703a2fa9556 | photo | gcn |  | 8 | 5 |
| 24674247bcc3 | photo | gcn |  | 8 | 6 |
| e3f9555f095a | photo | gcn |  | 8 | 7 |
| 89e7f79c8774 | photo | gcn |  | 8 | 8 |
| c2a3bd99a570 | photo | gcn |  | 8 | 9 |
| 8e0deb840d58 | photo | gcn |  | 10 | 0 |
| cdfe06af96e4 | photo | gcn |  | 10 | 1 |
| 3e7b3c6345e6 | photo | gcn |  | 10 | 2 |
| e60fee773a5d | photo | gcn |  | 10 | 3 |
| 78ecc4c4a971 | photo | gcn |  | 10 | 4 |
| 38b299edabb9 | photo | gcn |  | 10 | 5 |
| d57d4485397e | photo | gcn |  | 10 | 6 |
| 7d1fb78a96a2 | photo | gcn |  | 10 | 7 |
| afb1d5b700be | photo | gcn |  | 10 | 8 |
| 865c6ed4b83b | photo | gcn |  | 10 | 9 |
| cf506a403202 | photo | gcn |  | 20 | 0 |
| 59192172e700 | photo | gcn |  | 20 | 1 |
| 0e567af19047 | photo | gcn |  | 20 | 2 |
| 1f1962bc7218 | photo | gcn |  | 20 | 3 |
| 55f0a7f4b24f | photo | gcn |  | 20 | 4 |
| e98b7c508ea4 | photo | gcn |  | 20 | 5 |
| 7d9eecba5517 | photo | gcn |  | 20 | 6 |
| d7202601df6b | photo | gcn |  | 20 | 7 |
| d9e7d6356da9 | photo | gcn |  | 20 | 8 |
| 42fc4fbfc499 | photo | gcn |  | 20 | 9 |
| f182fba21a1f | photo | gcn |  | 30 | 0 |
| de8ed8ffa11a | photo | gcn |  | 30 | 1 |
| 8ab7022808a3 | photo | gcn |  | 30 | 2 |
| 2cd3af2023de | photo | gcn |  | 30 | 3 |
| 5c2ab3caef8e | photo | gcn |  | 30 | 4 |
| c47ace137fac | photo | gcn |  | 30 | 5 |
| 8a2a7fc7275a | photo | gcn |  | 30 | 6 |
| 6309f40cd8a3 | photo | gcn |  | 30 | 7 |
| 7987d4dfde1e | photo | gcn |  | 30 | 8 |
| 0fb36eb2cacf | photo | gcn |  | 30 | 9 |
| e0e5c95012fd | photo | gat |  | 2 | 0 |
| a16b891066c6 | photo | gat |  | 2 | 1 |
| acc20474d629 | photo | gat |  | 2 | 2 |
| b8759c5f655b | photo | gat |  | 2 | 3 |
| 5016e4c28428 | photo | gat |  | 2 | 4 |
| 72cd4f1b14bc | photo | gat |  | 2 | 5 |
| 2cfdc0286470 | photo | gat |  | 2 | 6 |
| 2ce808eb4b4d | photo | gat |  | 2 | 7 |
| 9b31b70e618b | photo | gat |  | 2 | 8 |
| fbd1e61e30a4 | photo | gat |  | 2 | 9 |
| 96bfb3bb68c5 | photo | gat |  | 4 | 0 |
| 72e7d6323ec7 | photo | gat |  | 4 | 1 |
| 982de29c97ad | photo | gat |  | 4 | 2 |
| 658fefaef658 | photo | gat |  | 4 | 3 |
| 698bc04b6bb1 | photo | gat |  | 4 | 4 |
| 8c1f0409a6e9 | photo | gat |  | 4 | 5 |
| 8b731acded93 | photo | gat |  | 4 | 6 |
| c6a49ee642e4 | photo | gat |  | 4 | 7 |
| ef09fa21ac70 | photo | gat |  | 4 | 8 |
| d7f081e09197 | photo | gat |  | 4 | 9 |
| 816a9cc76b2d | photo | gat |  | 8 | 0 |
| 5277f0a33d58 | photo | gat |  | 8 | 1 |
| e85c19ce202b | photo | gat |  | 8 | 2 |
| 1bd0734d3599 | photo | gat |  | 8 | 3 |
| 3a12a295932e | photo | gat |  | 8 | 4 |
| c1565fd16bf3 | photo | gat |  | 8 | 5 |
| f349caccb469 | photo | gat |  | 8 | 6 |
| 21d4f9fbc33b | photo | gat |  | 8 | 7 |
| a690badeb412 | photo | gat |  | 8 | 8 |
| 8fa312e67b49 | photo | gat |  | 8 | 9 |
| 60c7b70cd5ce | photo | gat |  | 10 | 7 |
| 9d82288f9be9 | photo | gat |  | 20 | 1 |
| bd4f55510b94 | photo | gat |  | 20 | 3 |
| a43dc7f7b2c6 | photo | gat |  | 20 | 4 |
| 43e98676ad1e | photo | gat |  | 20 | 5 |
| bfe62be05765 | photo | gat |  | 20 | 6 |
| 2c37cddfb3f9 | photo | gat |  | 20 | 7 |
| 0a4eead11477 | photo | gat |  | 20 | 9 |
| 2f3c766c7c68 | photo | gat |  | 30 | 0 |
| bc6a413971b9 | photo | gat |  | 30 | 1 |
| b50503f1d6ac | photo | gat |  | 30 | 2 |
| fe6e5e48dd3a | photo | gat |  | 30 | 3 |
| 5df527818fc2 | photo | gat |  | 30 | 4 |
| c23a5dc6426e | photo | gat |  | 30 | 5 |
| 7f78d2493f5e | photo | gat |  | 30 | 6 |
| 0396aec88a95 | photo | gat |  | 30 | 7 |
| 737da571127b | photo | gat |  | 30 | 8 |
| 1980a25999ec | photo | gat |  | 30 | 9 |
| 4a967f18a5f9 | photo | sgc |  | 2 | 0 |
| b3202f79a8b7 | photo | sgc |  | 2 | 1 |
| 5aa3bc1b401a | photo | sgc |  | 2 | 3 |
| caeda75ce361 | photo | sgc |  | 2 | 4 |
| a6cfcd28a32e | photo | sgc |  | 2 | 5 |
| 0dc4d93fa96f | photo | sgc |  | 2 | 6 |
| f47b528ecada | photo | sgc |  | 2 | 7 |
| 9d0439089b5c | photo | sgc |  | 2 | 8 |
| fcf7bf5d39df | photo | sgc |  | 2 | 9 |
| beb015bcbbe3 | photo | sgc |  | 4 | 0 |
| 5e8fb9a552d1 | photo | sgc |  | 4 | 1 |
| 6d3c4704600d | photo | sgc |  | 4 | 2 |
| 748024a156ff | photo | sgc |  | 4 | 3 |
| 2bcb50786d9c | photo | sgc |  | 4 | 4 |
| 2c9323b5fa7f | photo | sgc |  | 4 | 6 |
| 46c46c3494dc | photo | sgc |  | 4 | 7 |
| 6f1940804703 | photo | sgc |  | 4 | 8 |
| 215fb5ca3e03 | photo | sgc |  | 4 | 9 |
| 633b89fa3fe6 | photo | sgc |  | 8 | 0 |
| c3d2d2862659 | photo | sgc |  | 8 | 1 |
| 850017b56425 | photo | sgc |  | 8 | 2 |
| 8ed011ecaa18 | photo | sgc |  | 8 | 3 |
| 8fd542102b66 | photo | sgc |  | 8 | 4 |
| 51ed96ac48a5 | photo | sgc |  | 8 | 5 |
| 0bfb39a9c4fb | photo | sgc |  | 8 | 6 |
| 25019836b740 | photo | sgc |  | 8 | 7 |
| 156534be2312 | photo | sgc |  | 8 | 8 |
| 195c3245a515 | photo | sgc |  | 8 | 9 |
| 926c23847702 | photo | sgc |  | 10 | 0 |
| 699a044b562d | photo | sgc |  | 10 | 1 |
| e69edd1c8f4b | photo | sgc |  | 10 | 2 |
| b56e8f7b0c94 | photo | sgc |  | 10 | 3 |
| c600d2eb4ddd | photo | sgc |  | 10 | 4 |
| 5fdd6db1d0e8 | photo | sgc |  | 10 | 5 |
| 6d657e774bb8 | photo | sgc |  | 10 | 6 |
| 77d1f976e8f8 | photo | sgc |  | 10 | 7 |
| a9f60c0abf84 | photo | sgc |  | 10 | 8 |
| d9ad2518a9ff | photo | sgc |  | 10 | 9 |
| d466e7c523d6 | photo | sgc |  | 20 | 0 |
| 8b42751279de | photo | sgc |  | 20 | 1 |
| 87af77506e3f | photo | sgc |  | 20 | 2 |
| e024384951d0 | photo | sgc |  | 20 | 3 |
| 029d6ffc34cf | photo | sgc |  | 20 | 4 |
| 1de4fb213fac | photo | sgc |  | 20 | 5 |
| 11cf8fc19108 | photo | sgc |  | 20 | 6 |
| ca83de211ea4 | photo | sgc |  | 20 | 7 |
| 887364db4818 | photo | sgc |  | 20 | 8 |
| 5543411b15e5 | photo | sgc |  | 20 | 9 |
| 6a7e36e585ef | photo | sgc |  | 30 | 0 |
| 21c190f859a4 | photo | sgc |  | 30 | 1 |
| 4202df7be36e | photo | sgc |  | 30 | 2 |
| 51753b3a4108 | photo | sgc |  | 30 | 3 |
| 71a91c312b2e | photo | sgc |  | 30 | 4 |
| 5b2817de6e85 | photo | sgc |  | 30 | 5 |
| 7392c1b163aa | photo | sgc |  | 30 | 6 |
| 460b20ff38c5 | photo | sgc |  | 30 | 7 |
| 9d981cbcfae3 | photo | sgc |  | 30 | 8 |
| 80cbc693b4fe | photo | sgc |  | 30 | 9 |
| 34c9619a0588 | photo | appnp |  | 2 | 0 |
| 56d8df4f3846 | photo | appnp |  | 2 | 1 |
| 7d782d5de4b3 | photo | appnp |  | 2 | 2 |
| 3d376e9b68ab | photo | appnp |  | 2 | 3 |
| 3d90fb9777a3 | photo | appnp |  | 2 | 4 |
| cb9a11c2eb8f | photo | appnp |  | 2 | 5 |
| a8716a23a61e | photo | appnp |  | 2 | 6 |
| 6f86922edec2 | photo | appnp |  | 2 | 7 |
| 81bd93bd6ffc | photo | appnp |  | 2 | 8 |
| d88a62a596f3 | photo | appnp |  | 2 | 9 |
| 9939dc041145 | photo | appnp |  | 4 | 0 |
| 479937063b44 | photo | appnp |  | 4 | 1 |
| c2dd7589da35 | photo | appnp |  | 4 | 2 |
| 1e77550019f0 | photo | appnp |  | 4 | 3 |
| ac9035fa543b | photo | appnp |  | 4 | 4 |
| 6d754dc9fdbd | photo | appnp |  | 4 | 5 |
| 7a05717650f4 | photo | appnp |  | 4 | 6 |
| 07a2525f2a59 | photo | appnp |  | 4 | 7 |
| cc5d73dabe06 | photo | appnp |  | 4 | 8 |
| 5eeda504b7b4 | photo | appnp |  | 4 | 9 |
| a749e28d25fc | photo | appnp |  | 8 | 0 |
| db727aac851a | photo | appnp |  | 8 | 1 |
| a3141380bead | photo | appnp |  | 8 | 2 |
| 01e82048d5f2 | photo | appnp |  | 8 | 3 |
| b19ce990b54a | photo | appnp |  | 8 | 4 |
| 1c43ec5150b7 | photo | appnp |  | 8 | 5 |
| babeb350c8cd | photo | appnp |  | 8 | 6 |
| c60943c56e9d | photo | appnp |  | 8 | 7 |
| fc8275abd831 | photo | appnp |  | 8 | 8 |
| 663a972ede74 | photo | appnp |  | 8 | 9 |
| 9e33e54cf20f | photo | appnp |  | 10 | 0 |
| d1f85ad4a3d8 | photo | appnp |  | 10 | 1 |
| 11c3fc18c3a9 | photo | appnp |  | 10 | 2 |
| 1db90f496fcc | photo | appnp |  | 10 | 3 |
| 5f6528b20b18 | photo | appnp |  | 10 | 4 |
| bc1f4e30dc33 | photo | appnp |  | 10 | 5 |
| a65ee85a62b5 | photo | appnp |  | 10 | 6 |
| 092aed01d9b7 | photo | appnp |  | 10 | 7 |
| 105e2042fd5a | photo | appnp |  | 10 | 8 |
| e9ccecd3b204 | photo | appnp |  | 10 | 9 |
| caf53dacf3cc | photo | appnp |  | 20 | 0 |
| 4f86bc1fba5f | photo | appnp |  | 20 | 1 |
| 48828a3e3ec6 | photo | appnp |  | 20 | 2 |
| 2605e9c2cd50 | photo | appnp |  | 20 | 3 |
| e6d7a91d6a6c | photo | appnp |  | 20 | 4 |
| c8d4d7ca94d8 | photo | appnp |  | 20 | 5 |
| 3f8241ec546d | photo | appnp |  | 20 | 6 |
| c07b5d1f8413 | photo | appnp |  | 20 | 7 |
| 23045093a4ea | photo | appnp |  | 20 | 8 |
| 5219951887ef | photo | appnp |  | 20 | 9 |
| 02ee6020c293 | photo | appnp |  | 30 | 0 |
| 8cb4e2d4846d | photo | appnp |  | 30 | 1 |
| b543cc75302a | photo | appnp |  | 30 | 2 |
| ef59a51675aa | photo | appnp |  | 30 | 3 |
| 3eb6594d52d6 | photo | appnp |  | 30 | 4 |
| 35bb351cfd9a | photo | appnp |  | 30 | 5 |
| 515b9c376615 | photo | appnp |  | 30 | 6 |
| a6bf2f374334 | photo | appnp |  | 30 | 7 |
| 347c16f9838f | photo | appnp |  | 30 | 8 |
| 8094ad0b2ae6 | photo | appnp |  | 30 | 9 |
| e493bacb622a | photo | arma |  | 2 | 7 |
| f9b02b050ebb | photo | arma |  | 4 | 1 |
| 35ca78956b92 | photo | arma |  | 4 | 3 |
| 541826eb6078 | photo | arma |  | 4 | 4 |
| 8405ad6074f5 | photo | arma |  | 4 | 5 |
| 3740baed29c8 | photo | arma |  | 4 | 6 |
| 744e5f6e4a8f | photo | arma |  | 4 | 7 |
| 42c82cfb5f07 | photo | arma |  | 4 | 9 |
| 1e781af297c7 | photo | arma |  | 8 | 0 |
| 0b3d3457ec57 | photo | arma |  | 8 | 1 |
| 3045c194bdfe | photo | arma |  | 8 | 2 |
| 65ffb044fdbe | photo | arma |  | 8 | 3 |
| 34aa61c71504 | photo | arma |  | 8 | 4 |
| 2b3a174d6cd3 | photo | arma |  | 8 | 5 |
| 2caedd47000b | photo | arma |  | 8 | 6 |
| d8def0d26372 | photo | arma |  | 8 | 7 |
| f45f8c948977 | photo | arma |  | 8 | 8 |
| 92c3311ded6f | photo | arma |  | 8 | 9 |
| 4f74b2954d4f | photo | arma |  | 10 | 0 |
| 2e69d852ee51 | photo | arma |  | 10 | 1 |
| 170859d1b8a6 | photo | arma |  | 10 | 3 |
| bd6e54679543 | photo | arma |  | 10 | 4 |
| f77b9405c85b | photo | arma |  | 10 | 5 |
| d30b0fa5bdb0 | photo | arma |  | 10 | 6 |
| f6427257122c | photo | arma |  | 10 | 7 |
| 1e763e1722ee | photo | arma |  | 10 | 8 |
| e02f4d871b19 | photo | arma |  | 10 | 9 |
| a13d03b4b752 | photo | arma |  | 20 | 0 |
| f2ec72cc2c08 | photo | arma |  | 20 | 1 |
| b3d0aef7b2e6 | photo | arma |  | 20 | 2 |
| 690b4a16202f | photo | arma |  | 20 | 3 |
| 193e965f9abb | photo | arma |  | 20 | 4 |
| 6a96caa267f8 | photo | arma |  | 20 | 6 |
| f9c8605ceffb | photo | arma |  | 20 | 7 |
| 6d904f9d7c00 | photo | arma |  | 20 | 8 |
| 660f8a4a19ea | photo | arma |  | 20 | 9 |
| 21c96e4e2a7d | photo | arma |  | 30 | 0 |
| 522ab59439b8 | photo | arma |  | 30 | 1 |
| fb83469078d8 | photo | arma |  | 30 | 2 |
| b5ea70dad93e | photo | arma |  | 30 | 3 |
| 18a9505095ef | photo | arma |  | 30 | 4 |
| 7063207cea2b | photo | arma |  | 30 | 5 |
| da601ca8ff8e | photo | arma |  | 30 | 6 |
| 6a0985b45094 | photo | arma |  | 30 | 7 |
| 639b4881d231 | photo | arma |  | 30 | 8 |
| 71468536edd0 | photo | arma |  | 30 | 9 |
| e1c3e5ff6e9f | photo | gcnii |  | 2 | 0 |
| 7f26ed8a72c9 | photo | gcnii |  | 2 | 1 |
| 98dd4ee461cd | photo | gcnii |  | 2 | 2 |
| a92cd07373d9 | photo | gcnii |  | 2 | 3 |
| 5078a324e20d | photo | gcnii |  | 2 | 4 |
| 4baf7df6e9db | photo | gcnii |  | 2 | 5 |
| c5570a9f32f4 | photo | gcnii |  | 2 | 6 |
| b60d008ff029 | photo | gcnii |  | 2 | 7 |
| 51edc52e27e8 | photo | gcnii |  | 2 | 8 |
| 53997a70a6f1 | photo | gcnii |  | 2 | 9 |
| ec8b56372cad | photo | gcnii |  | 4 | 0 |
| f4b4d62a5d69 | photo | gcnii |  | 4 | 1 |
| 4c17652a3c84 | photo | gcnii |  | 4 | 2 |
| 7ef343d866b2 | photo | gcnii |  | 4 | 3 |
| b6fc2f042c5b | photo | gcnii |  | 4 | 4 |
| aac41dc19dff | photo | gcnii |  | 4 | 5 |
| 53919273319a | photo | gcnii |  | 4 | 6 |
| c38565c7b37c | photo | gcnii |  | 4 | 7 |
| 3d94727db7fe | photo | gcnii |  | 4 | 8 |
| 0f05fd04f35b | photo | gcnii |  | 4 | 9 |
| 72cf198ddc76 | photo | gcnii |  | 8 | 0 |
| fb8debc11a86 | photo | gcnii |  | 8 | 1 |
| d6526b03f687 | photo | gcnii |  | 8 | 2 |
| 527184c9d81a | photo | gcnii |  | 8 | 3 |
| 0239090882d7 | photo | gcnii |  | 8 | 4 |
| d458f1565323 | photo | gcnii |  | 8 | 5 |
| 725a154a722b | photo | gcnii |  | 8 | 6 |
| bb0dedadade7 | photo | gcnii |  | 8 | 7 |
| 486d1c0d95fc | photo | gcnii |  | 8 | 8 |
| 1b9bde7e04db | photo | gcnii |  | 8 | 9 |
| 6942eb96d5b8 | photo | gcnii |  | 10 | 0 |
| 5a7c026be0d4 | photo | gcnii |  | 10 | 1 |
| 6b3c6f4c2480 | photo | gcnii |  | 10 | 2 |
| 36121cbcf873 | photo | gcnii |  | 10 | 3 |
| 1481f45fe7cd | photo | gcnii |  | 10 | 4 |
| 38b68f3df77d | photo | gcnii |  | 10 | 5 |
| f50c73f4efc7 | photo | gcnii |  | 10 | 6 |
| e0e1f9ea90df | photo | gcnii |  | 10 | 7 |
| 18ce94d13e74 | photo | gcnii |  | 10 | 8 |
| 309a86fa0c9c | photo | gcnii |  | 10 | 9 |
| a171f09bd07f | photo | gcnii |  | 20 | 0 |
| 6bab8ac15736 | photo | gcnii |  | 20 | 1 |
| c731e9831689 | photo | gcnii |  | 20 | 2 |
| 2695141adf5d | photo | gcnii |  | 20 | 3 |
| 2c4a2991fe5a | photo | gcnii |  | 20 | 4 |
| fdf4306ed31c | photo | gcnii |  | 20 | 5 |
| 953d0f98350d | photo | gcnii |  | 20 | 6 |
| 235240c9e004 | photo | gcnii |  | 20 | 7 |
| a691fea847a9 | photo | gcnii |  | 20 | 8 |
| b8e45e798ea7 | photo | gcnii |  | 20 | 9 |
| 3ce08f95b005 | photo | gcnii |  | 30 | 0 |
| 51f4ea8990da | photo | gcnii |  | 30 | 1 |
| 2ec7513039bb | photo | gcnii |  | 30 | 2 |
| 0824ff63d472 | photo | gcnii |  | 30 | 3 |
| 197008985d81 | photo | gcnii |  | 30 | 4 |
| 75a8530c87a3 | photo | gcnii |  | 30 | 5 |
| 2fc1cd529b32 | photo | gcnii |  | 30 | 6 |
| af2d76148a35 | photo | gcnii |  | 30 | 7 |
| 96c22a7cb55d | photo | gcnii |  | 30 | 8 |
| 4036ce94d415 | photo | gcnii |  | 30 | 9 |
| 068725ca07ba | photo | pairnorm |  | 2 | 0 |
| f9ae40951d0f | photo | pairnorm |  | 2 | 1 |
| 48e60aa3d7d9 | photo | pairnorm |  | 2 | 2 |
| 9ee90b44687a | photo | pairnorm |  | 2 | 3 |
| 6107c09a280a | photo | pairnorm |  | 2 | 4 |
| 4666e028f295 | photo | pairnorm |  | 2 | 5 |
| cb0930c0d1e1 | photo | pairnorm |  | 2 | 6 |
| 9c34bca5452b | photo | pairnorm |  | 2 | 7 |
| 528a7e785a51 | photo | pairnorm |  | 2 | 8 |
| 8b940763b73c | photo | pairnorm |  | 2 | 9 |
| fa4828df53e3 | photo | pairnorm |  | 4 | 0 |
| 466b2d9a6b91 | photo | pairnorm |  | 4 | 1 |
| 3b67a12078cb | photo | pairnorm |  | 4 | 2 |
| 3e264b9fdd5f | photo | pairnorm |  | 4 | 3 |
| 7bf921a47df1 | photo | pairnorm |  | 4 | 4 |
| 2f7cf5a335ee | photo | pairnorm |  | 4 | 5 |
| e4d4b2ae7dac | photo | pairnorm |  | 4 | 6 |
| 3e4e94f486dd | photo | pairnorm |  | 4 | 7 |
| 1f455006039c | photo | pairnorm |  | 4 | 8 |
| 53510f9b1524 | photo | pairnorm |  | 4 | 9 |
| 27942e3fdd96 | photo | pairnorm |  | 8 | 0 |
| e0e95b39e971 | photo | pairnorm |  | 8 | 1 |
| c6f6b7d7e66b | photo | pairnorm |  | 8 | 2 |
| 0d1e4036b524 | photo | pairnorm |  | 8 | 3 |
| bcb8a82fc551 | photo | pairnorm |  | 8 | 4 |
| 8f3e43a24f0d | photo | pairnorm |  | 8 | 5 |
| a1f50a258773 | photo | pairnorm |  | 8 | 6 |
| 8ee073ffca89 | photo | pairnorm |  | 8 | 7 |
| 1b21c8dde594 | photo | pairnorm |  | 8 | 8 |
| d7349d6d6c0f | photo | pairnorm |  | 8 | 9 |
| 2768c75e5ae8 | photo | pairnorm |  | 10 | 7 |
| 10a4c8082105 | photo | pairnorm |  | 20 | 1 |
| bfd8586b617c | photo | pairnorm |  | 20 | 3 |
| 2cc53a613066 | photo | pairnorm |  | 20 | 4 |
| 0422fb5b7cd6 | photo | pairnorm |  | 20 | 5 |
| df51af544851 | photo | pairnorm |  | 20 | 6 |
| 886751409ee7 | photo | pairnorm |  | 20 | 7 |
| 2dc812596576 | photo | pairnorm |  | 20 | 9 |
| 93643b7e959a | photo | pairnorm |  | 30 | 0 |
| 91d02207a423 | photo | pairnorm |  | 30 | 1 |
| 0af4cc62a280 | photo | pairnorm |  | 30 | 2 |
| 982381f889f9 | photo | pairnorm |  | 30 | 3 |
| f4bc95e831a7 | photo | pairnorm |  | 30 | 4 |
| 77a3bcd68396 | photo | pairnorm |  | 30 | 5 |
| 57381215a50c | photo | pairnorm |  | 30 | 6 |
| 7080fc210a7e | photo | pairnorm |  | 30 | 7 |
| a5b550422057 | photo | pairnorm |  | 30 | 8 |
| 93ec5976383c | photo | pairnorm |  | 30 | 9 |
| 6d02ffd0535b | photo | dropedge |  | 2 | 0 |
| 6b22eea8efd4 | photo | dropedge |  | 2 | 1 |
| 1b64461c3635 | photo | dropedge |  | 2 | 3 |
| 5a50b7e42536 | photo | dropedge |  | 2 | 4 |
| 6527c39fdfdf | photo | dropedge |  | 2 | 5 |
| d19a11cfd1dd | photo | dropedge |  | 2 | 6 |
| 97953a36e793 | photo | dropedge |  | 2 | 7 |
| eac30467a640 | photo | dropedge |  | 2 | 8 |
| 34f723749037 | photo | dropedge |  | 2 | 9 |
| 6ce4e80a4f7e | photo | dropedge |  | 4 | 0 |
| 1f30650b8295 | photo | dropedge |  | 4 | 1 |
| fdac04e66c6f | photo | dropedge |  | 4 | 2 |
| 088ed8f61b50 | photo | dropedge |  | 4 | 3 |
| bdb9a2e324da | photo | dropedge |  | 4 | 4 |
| f5dac16ea533 | photo | dropedge |  | 4 | 6 |
| 5c16f51b6b94 | photo | dropedge |  | 4 | 7 |
| f4fc897be1ba | photo | dropedge |  | 4 | 8 |
| a4fa00a7adad | photo | dropedge |  | 4 | 9 |
| a075f0e8781a | photo | dropedge |  | 8 | 0 |
| 27976d452510 | photo | dropedge |  | 8 | 1 |
| 01b96b17c954 | photo | dropedge |  | 8 | 2 |
| d9954102eaa0 | photo | dropedge |  | 8 | 3 |
| 00ac4911febf | photo | dropedge |  | 8 | 4 |
| 2aa0cc7c4608 | photo | dropedge |  | 8 | 5 |
| 38399a7624f4 | photo | dropedge |  | 8 | 6 |
| 9fb1fedb68b6 | photo | dropedge |  | 8 | 7 |
| faa71b2b0bfa | photo | dropedge |  | 8 | 8 |
| 0192979ebc9a | photo | dropedge |  | 8 | 9 |
| 8b5386e703d2 | photo | dropedge |  | 10 | 0 |
| 26f355920582 | photo | dropedge |  | 10 | 1 |
| d58a5b374187 | photo | dropedge |  | 10 | 2 |
| c5741a1e1a84 | photo | dropedge |  | 10 | 3 |
| 31881eaf058c | photo | dropedge |  | 10 | 4 |
| f9edaf825e04 | photo | dropedge |  | 10 | 5 |
| 17c1f2eeb007 | photo | dropedge |  | 10 | 6 |
| 44bb1706c9f0 | photo | dropedge |  | 10 | 7 |
| 78125205da1b | photo | dropedge |  | 10 | 8 |
| 83ed5c5dd93e | photo | dropedge |  | 10 | 9 |
| 1e9dfaabaa0b | photo | dropedge |  | 20 | 0 |
| aefb706eff24 | photo | dropedge |  | 20 | 1 |
| 2311a0b94dc5 | photo | dropedge |  | 20 | 2 |
| 4ec835ebd73d | photo | dropedge |  | 20 | 3 |
| 3f56cdaf6f51 | photo | dropedge |  | 20 | 4 |
| f531459940a9 | photo | dropedge |  | 20 | 5 |
| eb41eb1c28db | photo | dropedge |  | 20 | 6 |
| aa24aa4d24ac | photo | dropedge |  | 20 | 7 |
| e36bf1666c1d | photo | dropedge |  | 20 | 8 |
| f803b2a0d04d | photo | dropedge |  | 20 | 9 |
| d25039f731f1 | photo | dropedge |  | 30 | 0 |
| 1d0522b9f9bc | photo | dropedge |  | 30 | 1 |
| 0839533768b6 | photo | dropedge |  | 30 | 2 |
| 57e40f08850b | photo | dropedge |  | 30 | 3 |
| d8ae3a4aec5b | photo | dropedge |  | 30 | 4 |
| 873e422f4319 | photo | dropedge |  | 30 | 5 |
| 8c0e9442b570 | photo | dropedge |  | 30 | 6 |
| 5be7c297c9b9 | photo | dropedge |  | 30 | 7 |
| e6dfacb9f126 | photo | dropedge |  | 30 | 8 |
| 1d99500f1413 | photo | dropedge |  | 30 | 9 |
| e9a1fdf91068 | photo | graphunet |  | 2 | 0 |
| 6568052f061a | photo | graphunet |  | 2 | 1 |
| 964e86901a53 | photo | graphunet |  | 2 | 2 |
| 39401717b867 | photo | graphunet |  | 2 | 3 |
| efe8ca80f154 | photo | graphunet |  | 2 | 4 |
| ccacbed8420e | photo | graphunet |  | 2 | 5 |
| 99078cf07473 | photo | graphunet |  | 2 | 6 |
| f0cd087090b5 | photo | graphunet |  | 2 | 7 |
| 8cd755f798bd | photo | graphunet |  | 2 | 8 |
| 6f87c0bdd144 | photo | graphunet |  | 2 | 9 |
| c8fe0fc081d7 | photo | graphunet |  | 4 | 0 |
| a721452fde5c | photo | graphunet |  | 4 | 1 |
| efe7f36e2a41 | photo | graphunet |  | 4 | 2 |
| c3f2e943936e | photo | graphunet |  | 4 | 3 |
| d449ae9d940d | photo | graphunet |  | 4 | 4 |
| 36725b8e1e25 | photo | graphunet |  | 4 | 5 |
| 32486e27b3cf | photo | graphunet |  | 4 | 6 |
| d13025353dbe | photo | graphunet |  | 4 | 7 |
| c02b737a8211 | photo | graphunet |  | 4 | 8 |
| 4b9ea2fe3776 | photo | graphunet |  | 4 | 9 |
| 1858301e4d53 | photo | graphunet |  | 8 | 0 |
| e07b8a071ec4 | photo | graphunet |  | 8 | 1 |
| 77431ba4291a | photo | graphunet |  | 8 | 2 |
| 580eabbd1013 | photo | graphunet |  | 8 | 3 |
| c8146e807127 | photo | graphunet |  | 8 | 4 |
| 1d95be8be02b | photo | graphunet |  | 8 | 5 |
| cae75815dbcd | photo | graphunet |  | 8 | 6 |
| e50761cc8718 | photo | graphunet |  | 8 | 7 |
| c1ac188fcfec | photo | graphunet |  | 8 | 8 |
| ca250e209681 | photo | graphunet |  | 8 | 9 |
| 4875277f5368 | photo | graphunet |  | 10 | 0 |
| b32bbc830606 | photo | graphunet |  | 10 | 1 |
| 459ccb5d2f25 | photo | graphunet |  | 10 | 2 |
| 510329879152 | photo | graphunet |  | 10 | 3 |
| c38f39ec91dd | photo | graphunet |  | 10 | 4 |
| 566d5a85dd58 | photo | graphunet |  | 10 | 5 |
| 0992350f58d3 | photo | graphunet |  | 10 | 6 |
| 5ef9d23e296c | photo | graphunet |  | 10 | 7 |
| 3dc9aa6bb8e8 | photo | graphunet |  | 10 | 8 |
| 1af3862077fb | photo | graphunet |  | 10 | 9 |
| fbb9babfbd72 | photo | graphunet |  | 20 | 0 |
| e16f8a0c193e | photo | graphunet |  | 20 | 1 |
| 556fc635f77a | photo | graphunet |  | 20 | 2 |
| b86f112a6408 | photo | graphunet |  | 20 | 3 |
| 0f8ee95cf40b | photo | graphunet |  | 20 | 4 |
| 12b4022140fd | photo | graphunet |  | 20 | 5 |
| f8e3f091be7b | photo | graphunet |  | 20 | 6 |
| f59f7b1188eb | photo | graphunet |  | 20 | 7 |
| ab68881798e7 | photo | graphunet |  | 20 | 8 |
| 813fa1525966 | photo | graphunet |  | 20 | 9 |
| 3437215e8894 | photo | graphunet |  | 30 | 0 |
| db72fb7ca641 | photo | graphunet |  | 30 | 1 |
| 6d7c699dbc42 | photo | graphunet |  | 30 | 2 |
| ae9b2d2860a8 | photo | graphunet |  | 30 | 3 |
| 24b32085c3b2 | photo | graphunet |  | 30 | 4 |
| 354676e5b8f8 | photo | graphunet |  | 30 | 5 |
| f6741117da2d | photo | graphunet |  | 30 | 6 |
| 90743cfc3cdf | photo | graphunet |  | 30 | 7 |
| 7a76a0cf23cb | photo | graphunet |  | 30 | 8 |
| 8675eb4c1f5a | photo | graphunet |  | 30 | 9 |
