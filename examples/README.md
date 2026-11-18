# Demo

`demo.py` shows what n-HDP-GNN computes, step by step, on one small graph whose structure is known because it was
planted. It needs no download.

```bash
python examples/demo.py --quick      # a small graph, fewer depths, seeds and epochs; writes runs/demo/output/
python examples/demo.py              # the full-size graph; with PyTorch and PyG it also trains six models
python examples/demo.py --train no   # NumPy only: stages A and B and the training-free depth study
```

`make demo-quick` and `make demo` run the first two. A local run writes to `runs/demo/output/`
(`runs/demo/work/` holds the run records and caches of the training), so it never overwrites the committed copy below.

## What the report shows

The graph is a hierarchical stochastic block model: 4 super-communities of 4 communities of 60 nodes at full size,
class labels that follow the super-communities with probability 0.9, and 16-dimensional Gaussian features around a
class mean. Because the planted structure is known, every stage can be compared with it.

| Part of the report | Question it answers | Needs |
|---|---|---|
| 1. The graph | How big is it, how homophilous, how far apart are its nodes? | NumPy |
| 2. Stage A | Does nested Louvain recover the planted communities and super-communities (NMI, ARI)? | NumPy |
| 3. Stage B | Is the n-HDP objective monotone? What does the gate g_ij = ⟨π_i, π_j⟩ give on edges inside a community, between communities of one super-community and between super-communities? | NumPy |
| 4. Depth without training | Applying an operator L times to the features (L up to 64), how do a linear probe and the centroid gap (Eq. 28) of PᴸX change for the normalised adjacency and for the gated operators? | NumPy |
| 5. Depth with training | n-HDP-GNN (gate on and off), GCN, GAT, APPNP and GCNII at several depths: test accuracy, centroid gap and k-means NMI of the logits, mean and 95% interval over seeds, and every failed run. | PyTorch, PyG |

Without PyTorch part 5 is not generated and the report says so.

## Output files

| File | Content |
|---|---|
| `demo.md` | the report: the tables, the figures and the provenance (command, code commit, CPU, versions) |
| `demo.json` | every number of the report, the graph's parameters, the training configuration and the environment |
| `fig_graph.png` | the graph and the same layout with the gate (part 1) |
| `fig_stage_ab.png` | found against planted communities, the objective of stage B, the gate by kind of edge (parts 2 and 3) |
| `fig_depth_free.png` | probe accuracy and centroid gap against depth, without training (part 4) |
| `fig_depth_trained.png` | test accuracy and centroid gap of the logits against depth, with training (part 5) |
| `runs.jsonl.gz` | the result record of every training run, without training curves |

Options: `--train {auto,yes,no}` (`yes` makes a missing PyTorch an error), `--jobs N` (N training runs at a time, one
process each), `--seeds N` (seeds per training cell; 5, or 3 with `--quick`), `--no-figures`, `--out DIR`, `--work DIR`.

## The committed output

[`output/`](output/) is the output of the [`demo` workflow](../.github/workflows/demo.yml): one run of
`python examples/demo.py --train yes` on a GitHub-hosted CPU runner. Nothing in it was typed in or produced on another
machine. The first lines of [`output/demo.md`](output/demo.md) give the command, the code commit, the CPU model, the
run time and the package versions of that run, and its last section lists the command, the platform and the versions again.

To produce it again, start the workflow (Actions, "demo", "Run workflow") or push a branch named `demo/<name>`. The
workflow uploads the output as the artifact `demo-output` and replaces the branch `results/demo` with a commit that holds
it under `examples/output/`; the files are then copied into the repository by a commit of their own, so the history shows
which run they came from.

## How to read it

- It is one synthetic graph from one generator seed. Its numbers describe this graph; they are not evidence about
  accuracy on real graphs, which is what [`results/`](../results/README.md) and [`artifacts/`](../artifacts/README.md) are for.
- The training uses the same generic settings for every model. None was tuned, and none is the article's.
- With a handful of seeds the intervals are wide, and no significance test is reported. A run that fails is listed
  with its reason and is not replaced by a value.
- The training results depend on the CPU model. GitHub assigns the CPU of a runner, and the workflow has run three times
  on three different ones: [44323001112](https://github.com/VKHS/n-HDP-GNN/actions/runs/44323001112) (AMD EPYC 9V45,
  code commit e137276), [44325763931](https://github.com/VKHS/n-HDP-GNN/actions/runs/44325763931) (Intel Xeon 6975P-C,
  0717c4d) and [44330815122](https://github.com/VKHS/n-HDP-GNN/actions/runs/44330815122) (AMD EPYC 9V74, c9540fe). Between
  the commits only `nhdp_gnn/distributed.py`, which the demo does not use, changed. Every number of sections 1 to 4 is
  the same in the three runs to the last digit; only the timings differ. In section 5 the two AMD runs agree in test
  accuracy, validation accuracy and best epoch in all 90 training runs, and the Intel run differs from both: in test
  accuracy in 27 of the 90 runs (4 of 30 at depth 2, 9 of 30 at depth 8, 14 of 30 at depth 32) and in best epoch in 37.
  Of the 27, 24 differ by 4.2 points or less and three by 23 to 25 points: n-HDP-GNN with the gate off at depth 32, seed 0
  (65.1% on AMD, 90.1% on Intel), n-HDP-GNN at depth 32, seed 2 (50.0%, 74.5%) and GAT at depth 8, seed 4 (49.5%, 26.0%).
  The mean over the five seeds differs in 9 of the 18 cells. Three move by 4.6 to 5.3 points: n-HDP-GNN at depth 32 (82.1
  on AMD, 86.7 on Intel), the same with the gate off (77.0, 82.3) and GAT at depth 8 (56.6, 51.5); six move by 0.1 to
  0.3 points; APPNP and GCNII at every depth, GCN at depths 2 and 8 and GAT at depth 2 do not move. The leaders are the
  same on both: n-HDP-GNN has the highest mean at depths 2 and 8, APPNP and GCNII at depth 32, and at depth 32 n-HDP-GNN,
  with or without the gate, stays above GAT and GCN. A single run of an unstable model (deep n-HDP-GNN, GAT at depth 8)
  therefore says little, and the means of those cells in the committed output are those of one CPU type. The values of
  every run are in [`cpu_comparison.json`](cpu_comparison.json), made by
  [`scripts/compare_demo_runs.py`](../scripts/compare_demo_runs.py). The Cora runs gave the same picture
  ([repeatability](../results/README.md)): repeats on one CPU model are bit-identical, different CPU models can differ.
- A probe accuracy that stays flat with depth does not by itself show that the classes stay apart; the centroid gap
  measures how far apart they are. Part 4 of the report shows the measurement behind this statement for the graph it ran on.
