# fig11_heads

Sweep `configs/fig11_heads.toml --only dataset=cora|citeseer|pubmed|photo|ogbn-arxiv`: 250 runs expected, 210 completed, 37 failed (28 memory, 9 time), 3 missing.

Selection: ogbn-products and ogbn-proteins are not part of the runs on GitHub-hosted runners (4 vCPUs and 16 GB of memory per job); run them on a cluster with scripts/launch_cluster.sh.

Values are mean ± 95% t-interval over seeds; n is the number of completed seeds in the cell.

Failed runs are counted per cell by reason: *memory*: the run needed more memory than the budget while running alone; *time*: training provably needed more than the time limit (or the run exceeded the wall-clock limit).

Code commit(s) of the runs: 0f7045972e60a86d7966841bc3e5c4bfe0caa016, 20d9d0fac98f9a83bb60e54bf36ddf1c98869b65

## test (accuracy in %)

**test**

| tag | citeseer | cora | ogbn-arxiv | photo | pubmed |
|---|---|---|---|---|---|
| heads-1 | 46.93 ± 22.47 (n=5) | 67.85 ± 9.42 (n=5) | 35.68 ± 11.36 (n=5) | 63.30 ± 21.21 (n=5) | 85.53 ± 2.41 (n=5) |
| heads-10 | 69.04 ± 8.34 (n=5) | 82.67 ± 1.40 (n=5) | failed: 5 memory | 80.25 ± 12.77 (n=5) | 87.72 ± 0.73 (n=5) |
| heads-12 | 71.48 ± 3.41 (n=5) | 82.96 ± 0.91 (n=5) | failed: 4 memory | 64.41 ± 30.06 (n=5) | 87.55 ± 0.51 (n=5) |
| heads-14 | 69.16 ± 1.51 (n=5) | 81.19 ± 0.96 (n=5) | failed: 5 memory | 66.32 ± 32.00 (n=5) | 87.51 ± 0.54 (n=5) |
| heads-16 | 70.12 ± 2.04 (n=5) | 81.81 ± 1.01 (n=5) | failed: 4 memory | 55.34 ± 22.29 (n=5) | 87.51 ± 0.69 (n=5) |
| heads-18 | 67.74 ± 6.56 (n=5) | 82.81 ± 1.53 (n=5) | failed: 5 memory | 62.88 ± 31.03 (n=5) | 87.21 ± 0.95 (n=5) |
| heads-2 | 69.97 ± 3.94 (n=5) | 80.89 ± 2.11 (n=5) | 52.53 ± 5.09 (n=5) | 89.01 ± 2.47 (n=5) | 87.26 ± 1.04 (n=5) |
| heads-4 | 70.33 ± 2.88 (n=5) | 82.04 ± 2.66 (n=5) | failed: 4 time | 88.08 ± 5.06 (n=5) | 87.59 ± 0.89 (n=5) |
| heads-6 | 71.45 ± 2.17 (n=5) | 82.00 ± 2.71 (n=5) | failed: 5 time | 86.17 ± 8.58 (n=5) | 87.51 ± 0.47 (n=5) |
| heads-8 | 71.05 ± 2.41 (n=5) | 82.37 ± 1.12 (n=5) | failed: 5 memory | 81.87 ± 12.36 (n=5) | 87.77 ± 0.67 (n=5) |

## Failed runs

| key | dataset | model | tag | depth | seed | reason | detail |
|---|---|---|---|---|---|---|---|
| 928da1bdd85e | ogbn-arxiv | nhdp | heads-10 | 10 | 0 | memory | exceeded the 13 GB memory budget while running alone (13598 MB) |
| fa9214965a05 | ogbn-arxiv | nhdp | heads-10 | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13482 MB) |
| 7690323b0844 | ogbn-arxiv | nhdp | heads-10 | 10 | 2 | memory | exceeded the 13 GB memory budget while running alone (13607 MB) |
| ec8cf919e072 | ogbn-arxiv | nhdp | heads-10 | 10 | 3 | memory | exceeded the 13 GB memory budget while running alone (13631 MB) |
| 1a3a8eb2fd24 | ogbn-arxiv | nhdp | heads-10 | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13580 MB) |
| a84e79846522 | ogbn-arxiv | nhdp | heads-12 | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13380 MB) |
| a3e8184ff299 | ogbn-arxiv | nhdp | heads-12 | 10 | 2 | memory | exceeded the 13 GB memory budget while running alone (13374 MB) |
| 523bb8d5b1df | ogbn-arxiv | nhdp | heads-12 | 10 | 3 | memory | exceeded the 13 GB memory budget while running alone (13381 MB) |
| 09df01a084d9 | ogbn-arxiv | nhdp | heads-12 | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13373 MB) |
| fcf74f8a2308 | ogbn-arxiv | nhdp | heads-14 | 10 | 0 | memory | exceeded the 13 GB memory budget while running alone (13376 MB) |
| f1ab2f306815 | ogbn-arxiv | nhdp | heads-14 | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13428 MB) |
| 957a0c8064ad | ogbn-arxiv | nhdp | heads-14 | 10 | 2 | memory | exceeded the 13 GB memory budget while running alone (13370 MB) |
| 695fb2e4c18e | ogbn-arxiv | nhdp | heads-14 | 10 | 3 | memory | exceeded the 13 GB memory budget while running alone (13323 MB) |
| 7ebf91429467 | ogbn-arxiv | nhdp | heads-14 | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13448 MB) |
| df5cf610282c | ogbn-arxiv | nhdp | heads-16 | 10 | 0 | memory | exceeded the 13 GB memory budget while running alone (13757 MB) |
| 0b91c4cff5a6 | ogbn-arxiv | nhdp | heads-16 | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13561 MB) |
| 7d0b248fd8fd | ogbn-arxiv | nhdp | heads-16 | 10 | 2 | memory | exceeded the 13 GB memory budget while running alone (13430 MB) |
| f110cf0f6cff | ogbn-arxiv | nhdp | heads-16 | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13477 MB) |
| 0b407194ef0e | ogbn-arxiv | nhdp | heads-18 | 10 | 0 | memory | exceeded the 13 GB memory budget while running alone (13408 MB) |
| 79d0959f9da4 | ogbn-arxiv | nhdp | heads-18 | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13420 MB) |
| 1891e3b75ad0 | ogbn-arxiv | nhdp | heads-18 | 10 | 2 | memory | exceeded the 13 GB memory budget while running alone (13414 MB) |
| 16877863c0d5 | ogbn-arxiv | nhdp | heads-18 | 10 | 3 | memory | exceeded the 13 GB memory budget while running alone (13414 MB) |
| 41e2fe5266f4 | ogbn-arxiv | nhdp | heads-18 | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13387 MB) |
| c9db672fa2d0 | ogbn-arxiv | nhdp | heads-4 | 10 | 0 | time | stopped after 303 epochs (119.9 min): early stopping needs at least 100 more epochs at >= 20.6 s each, i.e. at least 154 min of training, more than the 150-minu... |
| 48da714ab296 | ogbn-arxiv | nhdp | heads-4 | 10 | 1 | time | stopped after 284 epochs (117.2 min): early stopping needs at least 100 more epochs at >= 22.2 s each, i.e. at least 154 min of training, more than the 150-minu... |
| 1176f60165e0 | ogbn-arxiv | nhdp | heads-4 | 10 | 3 | time | stopped after 315 epochs (129.1 min): early stopping needs at least 100 more epochs at >= 22.4 s each, i.e. at least 166 min of training, more than the 150-minu... |
| 88f895bf9710 | ogbn-arxiv | nhdp | heads-4 | 10 | 4 | time | stopped after 420 epochs (128.7 min): early stopping needs at least 80 more epochs at >= 16.0 s each, i.e. at least 150 min of training, more than the 150-minut... |
| a63b0bf6e8a2 | ogbn-arxiv | nhdp | heads-6 | 10 | 0 | time | stopped after 123 epochs (86.0 min): early stopping needs at least 100 more epochs at >= 39.1 s each, i.e. at least 151 min of training, more than the 150-minut... |
| 986d13c24b3e | ogbn-arxiv | nhdp | heads-6 | 10 | 1 | time | stopped after 117 epochs (85.7 min): early stopping needs at least 100 more epochs at >= 38.7 s each, i.e. at least 150 min of training, more than the 150-minut... |
| c93bcdaf147d | ogbn-arxiv | nhdp | heads-6 | 10 | 2 | time | stopped after 119 epochs (85.3 min): early stopping needs at least 99 more epochs at >= 39.2 s each, i.e. at least 150 min of training, more than the 150-minute... |
| d25777f0966a | ogbn-arxiv | nhdp | heads-6 | 10 | 3 | time | stopped after 122 epochs (85.2 min): early stopping needs at least 100 more epochs at >= 39.8 s each, i.e. at least 152 min of training, more than the 150-minut... |
| b3bc919eff5b | ogbn-arxiv | nhdp | heads-6 | 10 | 4 | time | stopped after 81 epochs (71.7 min): early stopping needs at least 100 more epochs at >= 52.1 s each, i.e. at least 159 min of training, more than the 150-minute... |
| e2924ad3d002 | ogbn-arxiv | nhdp | heads-8 | 10 | 0 | memory | exceeded the 13 GB memory budget while running alone (13692 MB) |
| 49ba2fe8f403 | ogbn-arxiv | nhdp | heads-8 | 10 | 1 | memory | exceeded the 13 GB memory budget while running alone (13618 MB) |
| 9224f0f7e26e | ogbn-arxiv | nhdp | heads-8 | 10 | 2 | memory | exceeded the 13 GB memory budget while running alone (13499 MB) |
| a182cb365e48 | ogbn-arxiv | nhdp | heads-8 | 10 | 3 | memory | exceeded the 13 GB memory budget while running alone (13727 MB) |
| 9355be174298 | ogbn-arxiv | nhdp | heads-8 | 10 | 4 | memory | exceeded the 13 GB memory budget while running alone (13601 MB) |

## Missing runs

Not attempted, or stopped by the job's deadline.

| key | dataset | model | tag | depth | seed |
|---|---|---|---|---|---|
| 46f414d71698 | ogbn-arxiv | nhdp | heads-12 | 10 | 0 |
| c9e556968c9b | ogbn-arxiv | nhdp | heads-4 | 10 | 2 |
| 6439436e01a9 | ogbn-arxiv | nhdp | heads-16 | 10 | 3 |
