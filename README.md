# n-HDP-GNN
This code is related to “Mitigating Over-Smoothing in Graph Neural Networks via Nested Hierarchical Dirichlet Clustering” research paper

<!-- -------------------------------------------------------------------- -->
<!--  README.md for nHDP-GNN                                             -->
<!-- -------------------------------------------------------------------- -->

# nHDP-GNN · Nested Hierarchical Dirichlet Clustering for Over-Smoothing-Resilient Graph Neural Networks

> **TL;DR** A three-stage Graph Attention Network augmented with **Louvain communities**, a **nested (Dirichlet-inspired) hierarchy of clusters**, and a **variational latent bottleneck**.  The model slashes over-smoothing and pushes state-of-the-art accuracy on the Cora benchmark while remaining light-weight and fully reproducible.

<p align="center">
  <img src="docs/architecture.svg" width="600"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-green.svg)](https://www.python.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.x-orange)](https://pytorch-geometric.readthedocs.io/)
&nbsp;

---

## ✨  Highlights

| Feature | Description |
|---------|-------------|
| **Three-stage GAT** | Local → Community → Global attention blocks with residual skip-connection & batch-norm |
| **Nested Hierarchical Clustering** | n-HDP-inspired multi-level node descriptors (fast surrogate using Agglomerative Clustering) |
| **Variational Bottleneck** | μ / σ head + KL regulariser (β-VAE) for robust latent structure |
| **Auxiliary Community Loss** | Extra supervision that further discourages feature collapse |
| **Over-Smoothing Metrics** | Pair-wise, intra/inter class distances, embedding variance, plus t-SNE & confusion-matrix plots |

For full theoretical background and ablation studies, see the accompanying **manuscript** (PDF in [`docs/`](docs/)).  

---

## 🚀  Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<user>/nHDP-GNN.git
cd nHDP-GNN
python -m venv .venv             # optional virtual-env
source .venv/bin/activate
pip install -r requirements.txt  # PyTorch + PyG wheels included
