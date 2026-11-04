# legacy/: first-release scripts (May 2025)

These scripts were published with the first version of this repository. They are
kept unchanged for provenance and are not used by the `nhdp_gnn` package.

What they implement:

* **Model.** Three `GATConv` layers on the same graph, named local, community and
  global. Each layer has BatchNorm and dropout, with edge dropout, a linear skip
  connection, a Gaussian latent bottleneck, and an auxiliary head that predicts
  Louvain community ids.
* **n-HDP stand-in.** scikit-learn `AgglomerativeClustering` (Ward linkage, distance
  threshold 0.5) is run three times on the node features. The cluster indices are
  scaled by 0.1 and appended to the features.
* **Evaluation.** Training and evaluation use all nodes; there is no
  train/validation/test split.
* **`MAIN-NETWORKX.py`.** Generates random block graphs whose labels are drawn
  independently of the graph.

An Amazon Photo variant (`MAIN-PHOTOS.py`) is in the git history (commit `f51ab63`).

For the method described in the article, use the package (see the top-level
README and `docs/METHOD.md`).
