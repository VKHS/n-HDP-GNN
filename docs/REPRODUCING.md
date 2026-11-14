# Reproducing the article's tables and figures

Every experiment is one `RunConfig` (see `nhdp_gnn/config.py`). A sweep file
expands into many runs. Each run writes `runs/<sweep>/<key>.json`, and runs whose
JSON exists are skipped, so sweeps can be resumed and split across jobs.

The configs ship generic hyper-parameters. Set the `[base]` tables to the values
used for the article before regenerating its numbers.

## Commands

| Article | Sweep | Aggregate |
|---|---|---|
| Table 1 (ablation) | `python -m nhdp_gnn.sweep configs/table1_ablation.toml` | `python -m nhdp_gnn.report runs/table1_ablation --rows tag --cols dataset --by "" --value test` |
| Table 2 (fully supervised, depth) | `python -m nhdp_gnn.sweep configs/table2_full.toml` | `python -m nhdp_gnn.report runs/table2_full --reference nhdp` |
| Table 3 (semi-supervised) | `... configs/table3_semi.toml` | `... runs/table3_semi --reference nhdp` |
| Table 4 (label-scarce) | `... configs/table4_scarce.toml` | `... runs/table4_scarce --reference nhdp` |
| Table 5 (clustering) | `... configs/table5_clustering.toml` | `--value embedding.nmi` and `--value embedding.ari` with `--cols dataset --by ""` |
| Tables 6-8 (MADGap vs depth) | the runs of Tables 2-4 plus `configs/ogb_depth.toml` | `--value embedding.centroid_gap` (Eq. 28) and `--value embedding.madgap` (Chen et al.) with `--scale 1` |
| Tables 9-12 (scaling, communication) | `HOSTFILE=hosts.txt scripts/launch_cluster.sh configs/hpc_products.toml` (and `hpc_proteins.toml`, depth 10 and 30, each competitor) | `python -m nhdp_gnn.report runs/hpc --scaling` |
| Fig. 11 (heads) | `... configs/fig11_heads.toml` | `--rows tag --cols dataset --by ""` |
| Figs. 13-15, released embeddings | any sweep with `--set save_embeddings=true` | per-layer `.npy` files next to each run's JSON |

Useful options:

* `--dry-run` lists what would run.
* `--only model=nhdp,dataset=cora` filters the grid.
* `--set key=value` overrides a field for every run.
* `--shard i/n` runs every n-th configuration. For example, as a SLURM array:

  ```bash
  #SBATCH --array=0-63
  python -m nhdp_gnn.sweep configs/table2_full.toml --shard ${SLURM_ARRAY_TASK_ID}/64
  ```

`scripts/make_tables.py CONFIG --out results/<name>` writes a sweep's tables in the
article's layout (`README.md`), one CSV per reported value, all run results
(`runs.jsonl.gz`) and the number of expected, completed and missing runs. Cells are
computed from completed runs only; missing runs are listed, never imputed.

## Running sweeps on GitHub Actions

`.github/workflows/reproduce.yml` runs one sweep on GitHub-hosted CPU runners
(4 vCPUs each; every job runs four shards with one thread each; by default one
job per 20 runs, at most 256). Start it from the Actions tab ("Run workflow"), or:

```bash
gh workflow run reproduce.yml -f config=configs/table1_ablation.toml
gh workflow run reproduce.yml -f config=configs/table2_full.toml -f jobs=64 -f only="seed=0|1|2"
git push origin HEAD:reproduce/table2_full      # a push to reproduce/<sweep> runs configs/<sweep>.toml
```

The aggregate job writes `results/<name>/` with `scripts/make_tables.py`, shows the
tables in the run summary, uploads them as an artifact and, with `publish` (default),
pushes them to the branch `results/<name>`. A job stops after 6 hours; runs it did
not finish are listed as missing, and re-running the workflow with a narrower
`only` filter completes them. OGB-scale and multi-node runs need a cluster.

## Measured artifacts without training

`python -m nhdp_gnn.artifacts` (numpy only) measures stages A and B, the partition
study and the training-free depth study on the five small graphs; `scripts/make_figures.py`
draws the figures and `artifacts/SUMMARY.md`. See [artifacts/README.md](../artifacts/README.md).

## Protocol

* **Regimes** (`nhdp_gnn/splits.py`):
  * *full*: stratified 60/20/20 per seed; the official split on OGB.
  * *semi*: the public Planetoid split (20 labels per class, 500 validation, 1000 test); 20 labels per class on Amazon Photo.
  * *scarce*: 5% of the nodes, stratified, drawn from the full-regime training pool.
  * Validation and test nodes are those of the full regime for the same seed, except under the public Planetoid split.
* **Model selection**: early stopping on the validation metric. The selected state is restored before the single test evaluation.
* **Statistics**: mean ± 95% t-interval over seeds. `--reference` adds paired two-sided Wilcoxon p-values against the reference row.
* **Same partition for every method** in multi-part and distributed runs: the partition depends only on the data, the method and the seed.
* **Depth** is the number of propagation steps; each JSON records it as `propagation_depth`. Graph U-Net maps depth L to max(1, ⌊L/2⌋) pooling levels, i.e. 2·levels + 1 GCN layers, so every depth row is a different network.
* **Embedding diagnostics** (Tables 5-8: centroid gap, MADGap, k-means NMI/ARI) are computed on the logits, the one representation every model has. `embedding.centroid_gap_per_layer` profiles the hidden layers of the models that have a layer-wise hidden stack (n-HDP-GNN, GCN, GAT, ARMA, GCNII, PairNorm, DropEdge).
* **BatchNorm with several parts**: every part is normalised with its own statistics, in training and in evaluation (running averages would mix the parts). In DDP each rank keeps its own buffers.
* **Baseline settings**: `alpha` is APPNP's teleport probability and GCNII's initial-residual weight; `theta` is GCNII's identity-mapping strength. ARMA uses one ARMA₁ filter per layer with the activation between layers only.

## Baselines

Included (PyTorch Geometric implementations, arbitrary depth): GCN, GAT, SGC,
APPNP, ARMA, GCNII, PairNorm-GCN, DropEdge-GCN, Graph U-Net.

Not included: EM-GCN, MVPool, H-GCN and GRAND. To report them under this
protocol, run their authors' code with the same splits and seeds; the split masks
come from `nhdp_gnn.splits.regime_split`.

## Hardware notes

* `pip install pymetis` enables METIS partitions (otherwise BFS chunks are used).
* `pip install igraph` makes Louvain fast enough for ogbn-products.
* `ogb_depth.toml` trains ogbn-products in 16 parts on one node. The distributed
  configs use one part per worker.
* Datasets are downloaded to `data/` on first use. Stage A/B outputs are cached in
  `cache/` and shared across depths and models; cache files are written
  atomically, so sweep shards can share the directory.
* OGB: `ogb` 1.3.6 (the latest release) loads its processed files with `torch.load`
  defaults that fail on torch ≥ 2.6; `load_dataset` works around this. ogbn-products
  is a 1.4 GB download: without a terminal (batch or torchrun jobs) it proceeds
  without OGB's confirmation prompt. Downloading once before a sweep avoids
  repeating it per job:

  ```bash
  python -c "from nhdp_gnn.data import load_dataset; load_dataset('ogbn-products')"
  ```

  In distributed runs rank 0 loads the dataset first and the other ranks wait.
* Requirements: torch ≥ 2.3 and torch_geometric ≥ 2.6 (older combinations fail to
  load the processed datasets under current torch).
