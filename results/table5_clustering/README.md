# table5_clustering

Sweep `configs/table5_clustering.toml --only dataset=cora|citeseer|pubmed|photo|ogbn-arxiv`: 200 runs expected, 190 completed, 10 failed (10 memory), 0 missing.

Selection: ogbn-products and ogbn-proteins are not part of the runs on GitHub-hosted runners (4 vCPUs and 16 GB of memory per job); run them on a cluster with scripts/launch_cluster.sh.

Values are mean ± 95% t-interval over seeds; n is the number of completed seeds in the cell.

Failed runs are counted per cell by reason: *memory*: the run needed more memory than the budget while running alone.

Code commit(s) of the runs: 20d9d0fac98f9a83bb60e54bf36ddf1c98869b65, 4f00db1371643ba63fd5a37ce4d9e24d08bf1f24

## test (accuracy in %)

**test**

| model | citeseer | cora | ogbn-arxiv | photo | pubmed |
|---|---|---|---|---|---|
| appnp | 76.99 ± 3.57 (n=5) | 88.59 ± 0.91 (n=5) | 65.66 ± 0.29 (n=5) | 92.91 ± 0.55 (n=5) | 88.70 ± 0.35 (n=5) |
| arma | 78.04 ± 1.68 (n=5) | 88.00 ± 1.02 (n=5) | 66.47 ± 0.45 (n=5) | 94.19 ± 0.72 (n=5) | 88.05 ± 0.33 (n=5) |
| gat | 77.02 ± 3.16 (n=5) | 88.00 ± 1.70 (n=5) | 66.42 ± 0.06 (n=5) | 92.37 ± 0.65 (n=5) | 86.81 ± 0.47 (n=5) |
| gcn | 76.78 ± 3.58 (n=5) | 88.04 ± 1.68 (n=5) | 67.02 ± 0.31 (n=5) | 92.11 ± 0.70 (n=5) | 87.94 ± 0.67 (n=5) |
| gcnii | 76.02 ± 2.01 (n=5) | 87.44 ± 1.11 (n=5) | 64.98 ± 0.31 (n=5) | 92.70 ± 0.79 (n=5) | 89.07 ± 0.55 (n=5) |
| graphunet | 76.39 ± 2.89 (n=5) | 88.33 ± 1.40 (n=5) | failed: 5 memory | failed: 5 memory | 88.58 ± 0.34 (n=5) |
| nhdp | 71.42 ± 2.09 (n=5) | 80.93 ± 2.49 (n=5) | 61.64 ± 0.23 (n=5) | 85.04 ± 6.48 (n=5) | 88.06 ± 0.59 (n=5) |
| sgc | 75.57 ± 2.16 (n=5) | 51.04 ± 22.88 (n=5) | 63.11 ± 0.18 (n=5) | 40.89 ± 12.11 (n=5) | 79.03 ± 6.94 (n=5) |

## embedding.nmi

**embedding.nmi**

| model | citeseer | cora | ogbn-arxiv | photo | pubmed |
|---|---|---|---|---|---|
| appnp | 0.59 ± 0.03 (n=5) | 0.79 ± 0.01 (n=5) | 0.44 ± 0.00 (n=5) | 0.78 ± 0.00 (n=5) | 0.59 ± 0.01 (n=5) |
| arma | 0.61 ± 0.02 (n=5) | 0.80 ± 0.03 (n=5) | 0.46 ± 0.00 (n=5) | 0.85 ± 0.01 (n=5) | 0.58 ± 0.00 (n=5) |
| gat | 0.59 ± 0.03 (n=5) | 0.77 ± 0.03 (n=5) | 0.46 ± 0.00 (n=5) | 0.78 ± 0.01 (n=5) | 0.53 ± 0.01 (n=5) |
| gcn | 0.58 ± 0.03 (n=5) | 0.77 ± 0.02 (n=5) | 0.46 ± 0.00 (n=5) | 0.77 ± 0.00 (n=5) | 0.58 ± 0.01 (n=5) |
| gcnii | 0.61 ± 0.04 (n=5) | 0.79 ± 0.02 (n=5) | 0.47 ± 0.00 (n=5) | 0.82 ± 0.03 (n=5) | 0.62 ± 0.01 (n=5) |
| graphunet | 0.57 ± 0.02 (n=5) | 0.76 ± 0.02 (n=5) | failed: 5 memory | failed: 5 memory | 0.59 ± 0.01 (n=5) |
| nhdp | 0.67 ± 0.02 (n=5) | 0.78 ± 0.02 (n=5) | 0.43 ± 0.00 (n=5) | 0.82 ± 0.05 (n=5) | 0.68 ± 0.05 (n=5) |
| sgc | 0.57 ± 0.00 (n=5) | 0.57 ± 0.12 (n=5) | 0.44 ± 0.00 (n=5) | 0.38 ± 0.27 (n=5) | 0.40 ± 0.08 (n=5) |

## embedding.ari

**embedding.ari**

| model | citeseer | cora | ogbn-arxiv | photo | pubmed |
|---|---|---|---|---|---|
| appnp | 0.62 ± 0.03 (n=5) | 0.83 ± 0.02 (n=5) | 0.24 ± 0.01 (n=5) | 0.74 ± 0.01 (n=5) | 0.64 ± 0.01 (n=5) |
| arma | 0.65 ± 0.03 (n=5) | 0.83 ± 0.04 (n=5) | 0.26 ± 0.01 (n=5) | 0.86 ± 0.01 (n=5) | 0.62 ± 0.01 (n=5) |
| gat | 0.62 ± 0.03 (n=5) | 0.81 ± 0.03 (n=5) | 0.27 ± 0.01 (n=5) | 0.71 ± 0.02 (n=5) | 0.59 ± 0.01 (n=5) |
| gcn | 0.61 ± 0.03 (n=5) | 0.80 ± 0.02 (n=5) | 0.27 ± 0.02 (n=5) | 0.72 ± 0.02 (n=5) | 0.65 ± 0.01 (n=5) |
| gcnii | 0.65 ± 0.04 (n=5) | 0.82 ± 0.03 (n=5) | 0.29 ± 0.01 (n=5) | 0.81 ± 0.07 (n=5) | 0.67 ± 0.02 (n=5) |
| graphunet | 0.61 ± 0.02 (n=5) | 0.79 ± 0.03 (n=5) | failed: 5 memory | failed: 5 memory | 0.63 ± 0.02 (n=5) |
| nhdp | 0.72 ± 0.02 (n=5) | 0.82 ± 0.02 (n=5) | 0.25 ± 0.01 (n=5) | 0.83 ± 0.07 (n=5) | 0.74 ± 0.06 (n=5) |
| sgc | 0.61 ± 0.01 (n=5) | 0.54 ± 0.16 (n=5) | 0.25 ± 0.01 (n=5) | 0.26 ± 0.21 (n=5) | 0.43 ± 0.09 (n=5) |

## Failed runs

| key | dataset | model | tag | depth | seed | reason | detail |
|---|---|---|---|---|---|---|---|
| cf2bc1238537 | ogbn-arxiv | graphunet |  | 2 | 0 | memory | exceeded the 13 GB memory budget while running alone (13656 MB) |
| afdb331e541a | ogbn-arxiv | graphunet |  | 2 | 1 | memory | exceeded the 13 GB memory budget while running alone (13467 MB) |
| 5a8ab3954935 | ogbn-arxiv | graphunet |  | 2 | 2 | memory | exceeded the 13 GB memory budget while running alone (13553 MB) |
| 76db8ef128e9 | ogbn-arxiv | graphunet |  | 2 | 3 | memory | exceeded the 13 GB memory budget while running alone (13942 MB) |
| 072148959da0 | ogbn-arxiv | graphunet |  | 2 | 4 | memory | exceeded the 13 GB memory budget while running alone (13641 MB) |
| d04acbdcf3ae | photo | graphunet |  | 2 | 0 | memory | exceeded the 13 GB memory budget while running alone (13364 MB) |
| ed3c73db5cc1 | photo | graphunet |  | 2 | 1 | memory | exceeded the 13 GB memory budget while running alone (13325 MB) |
| 7c763923d92a | photo | graphunet |  | 2 | 2 | memory | exceeded the 13 GB memory budget while running alone (13353 MB) |
| 1f7c1ef65992 | photo | graphunet |  | 2 | 3 | memory | exceeded the 13 GB memory budget while running alone (13332 MB) |
| cbe73d7635b4 | photo | graphunet |  | 2 | 4 | memory | exceeded the 13 GB memory budget while running alone (13365 MB) |

