# Measured artifacts: stages A and B, partitioning, and a training-free depth study

Everything here was measured by

```bash
python -m nhdp_gnn.artifacts --out artifacts      # Cora, CiteSeer, PubMed, Amazon Photo, Amazon Computers
python scripts/make_figures.py --artifacts artifacts
```

at the code commit, on the machine and with the library versions recorded in
[`manifest.json`](manifest.json). No PyTorch is involved: these artifacts cover
stage A (partitioning, nested Louvain), stage B (nested-HDP inference) and the
propagation operator. Results of trained models (Tables 1-12 of the article) are
produced by the sweeps in `configs/` and are not part of this directory.

[`SUMMARY.md`](SUMMARY.md) holds every table below, generated from the JSON files.

## Files

| File | Content |
|---|---|
| `stage_ab/<graph>.json` | stage A/B with one part: community sizes, n-HDP iterations, convergence and ELBO trace, atoms used, purity of Louvain super-communities and of hard n-HDP clusters, gate statistics on same- and cross-class edges, components and purity bound of the infinite-depth limit for each gate, r_E, timings, the stage configuration |
| `stage_ab/partition_study.csv` | P ∈ {2, 4, 8, 14} parts × {community packing, BFS, random}: ρ, ρ_g, r_E, communities, purity, n-HDP iterations, part balance, timings (one process) |
| `operator_depth/<graph>.json` | P^L X for L ∈ {0, 1, 2, 4, 8, 16, 32, 64} and four operators: linear-probe accuracy (per split and mean), centroid gap (Eq. 28), MADGap, mean row norm |
| `figures/fig_stage_ab.png` | ELBO per node, purity, gate selectivity, purity bound of the infinite-depth limit |
| `figures/fig_operator_depth.png` | probe accuracy, centroid gap and MADGap against depth |
| `figures/fig_partition.png` | ρ, ρ_g and wall time against the number of parts |
| `manifest.json` | command, code commit, platform, CPU count, library versions, Louvain backend, total time |

## What each study measures

**Stages A and B.** The stage settings are the defaults of every training
configuration (`RunConfig().pipeline()`): Louvain resolution 1, three-level nested
HDP with K = 64, T₁ = 16, T₂ = 8, features reduced to 32 dimensions, top-4
responsibilities. With one part, Louvain runs on the whole graph.

**Gates.** For every undirected edge (i, j) the n-HDP gate is ⟨π_i, π_j⟩ with the
top-k responsibilities. The hard n-HDP gate keeps an edge when both endpoints have
the same most likely cluster; the Louvain gate keeps it when both lie in the same
super-community. *Gated homophily* is the share of gate mass on same-class edges.

**Purity bound of the infinite-depth limit.** By the Proposition in
[docs/METHOD.md](../docs/METHOD.md), repeated propagation with a gated operator
converges to one representation per connected component of the gate's support,
so a classifier on the limit can reach at most the purity of those components.
The bound depends on the number of components: many small components give a high
bound without giving class information (singletons are pure), so it is an upper
bound only.

**Training-free depth study.** For each operator P (normalised adjacency Â; Â
gated by n-HDP top-k, by hard n-HDP clusters, by Louvain super-communities) the
row-normalised input features are propagated, H_L = P^L X. At each depth a
logistic regression is trained on the L2-normalised rows of H_L with the
semi-supervised split (Planetoid: the public split; Amazon: 20 labels per class,
five seeds), and C ∈ {0.01, 0.1, 1, 10, 100} is chosen on validation accuracy. This
isolates what the operator preserves. It is not the n-HDP-GNN model, which adds
learned attention, residual connections and the community and global levels.
With Â and L = 2 the probe is close to SGC (Wu et al., 2019: logistic regression on
Â²X), which checks the setup.

**Partition study.** ρ is the share of edges cut by the partition, ρ_g the share
of gate mass on cut edges, r_E the edge reduction of the super-node graph. Wall
time is for one process computing all parts in turn; it is not a distributed
measurement. METIS was not available on the machine (see the manifest).

## Findings

Each statement below can be checked in [`SUMMARY.md`](SUMMARY.md) and the JSON files.

1. **Stage B converges with a monotone ELBO on all five graphs**: 40 to 92
   coordinate-ascent iterations (tolerance 10⁻⁶), and stages A and B together take
   1.3 s (Cora) to 9.9 s (Computers) on two CPU cores.

2. **Hard n-HDP clusters agree less with the labels than Louvain super-communities**
   on Cora (purity 0.560 vs 0.763), CiteSeer (0.609 vs 0.750), Photo (0.803 vs 0.859)
   and Computers (0.644 vs 0.744); on PubMed the two are equal (0.743 vs 0.740).

3. **The n-HDP gate separates same-class from cross-class edges weakly.** The mean
   gate on cross-class edges is 0.62 (Photo) to 0.84 (CiteSeer, PubMed) of the mean
   gate on same-class edges, so the class homophily of the gated mass exceeds the
   edge homophily by 0.03 to 0.06 (Cora: 0.810 to 0.869).

4. **The n-HDP top-k gate barely raises the purity bound of the infinite-depth
   limit** (Cora 0.375 against 0.348 without a gate; Photo 0.278 against 0.269): its
   support still connects most of the graph. The Louvain gate raises the bound to
   0.74-0.86. The hard n-HDP gate has a high bound (0.75-0.94) only because it splits
   the graph into thousands of small components (Cora: 1489).

5. **Training-free depth study.** With the normalised adjacency, the probe
   reproduces SGC at L = 2 (Cora 80.5%, CiteSeer 72.4%, PubMed 78.8%), and its best
   depth is 2 or 4 on every graph.
   * The n-HDP top-k gate gives a lower accuracy than the normalised adjacency at
     every depth from 1 to 64 on all five graphs, and it falls far below at depth 64
     (Cora 50.1% vs 76.9%, Photo 23.7% vs 56.7%, Computers 16.4% vs 53.6%). The hard
     n-HDP gate behaves the same way.
   * The Louvain gate stays within 2.6 points of the normalised adjacency up to
     depth 16 on every graph. At depth 64 it is far higher on the Amazon graphs
     (Photo 83.8% vs 56.7%, Computers 66.5% vs 53.6%), about equal on CiteSeer and
     PubMed (within 0.7 points), and lower on Cora (71.4% vs 76.9%).

   This agrees with finding 4: what the operator preserves at depth follows the
   label purity of the gate's components, and the n-HDP responsibilities give
   components that are hardly purer than the graph's own.

6. **Partitioning.** Packing Louvain communities into parts cuts the fewest edges
   at every number of parts on every graph, ahead of BFS chunks and random
   assignment. At P = 14 it cuts 2.1 (Computers) to 8.9 (CiteSeer) times fewer edges
   than BFS chunks (ρ = 0.06-0.36 against 0.55-0.82, and 0.93 for random parts),
   with every part within 5.5% of the mean size.
   * With community packing, the gate puts less mass on cut edges than their share
     of edges in every configuration: ρ_g / ρ is between 0.21 and 0.77. Across all
     methods this holds in 59 of 60 configurations; the exception is Photo with two
     BFS parts (ρ = 0.109, ρ_g = 0.132).
   * The super-node graph has 63% (CiteSeer) to 99.6% (Computers) fewer edges than
     the graph (r_E with one part).
   * In 11 of the 60 partitioned runs, n-HDP inference reached the 100-iteration
     cap before the 10⁻⁶ relative-change tolerance (column `nhdp_converged`); all
     single-part runs converged.

### What this implies for the article's claims

* The cut statistics support the communication argument: community-aligned
  partitions cut far fewer edges than BFS or random partitions, and the gate lowers
  the propagation mass that crosses parts further. Communication time and speedup
  are measured only by the distributed runs (Tables 9-12), which need a cluster.
* Over-smoothing resilience cannot be attributed to the n-HDP-gated operator by
  itself: on its own it preserves less class information than the ungated operator
  at every depth tested. Depth resilience of the trained model, if present, comes
  from the residual and initial connections, the community and global attention,
  or the learned attention weights. The ablations in `configs/table1_ablation.toml`
  (gate off, hard Louvain gate, random partition, no hierarchy) separate these.
