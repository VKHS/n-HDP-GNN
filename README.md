<!-- -------------------------------------------------------------------- -->

<!--  README.md — nHDP‑GNN                                               -->

<!--  Compatible with the manuscript: “Mitigating Over‑Smoothing in      -->

<!--  Graph Neural Networks via Nested Hierarchical Dirichlet Clustering”-->

<!-- -------------------------------------------------------------------- -->

# nHDP‑GNN

**Nested‑Hierarchical Dirichlet Clustering for Over‑Smoothing‑Resilient Graph Neural Networks**

> A three‑stage Graph Attention Network that fuses **Louvain communities**, a **nested Dirichlet‑inspired hierarchy** and a **variational latent bottleneck**.  The architecture reproduces all experiments and figures reported in the companion paper *Mitigating Over‑Smoothing in Graph Neural Networks via Nested Hierarchical Dirichlet Clustering* and exceeds baseline performance on Cora, Citeseer, and PubMed.

<p align="center">
  <img src="docs/architecture.svg" width="600" alt="Block diagram of the three‑stage nHDP‑GNN"/>
</p>

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](docs/oversmoothing_nhdp_gnn.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-green.svg)](https://www.python.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.x-orange)](https://pytorch-geometric.readthedocs.io/)

---

## 📰  Repository at a Glance

| Folder / file                     | What it contains                                                     | Related paper section |
| --------------------------------- | -------------------------------------------------------------------- | --------------------- |
| `src/main.py`                     | End‑to‑end script to reproduce Tables 2 & 3 (Cora experiments)       | § 5.1, § 6.1          |
| `src/models.py`                   | Implementation of **Enhanced Variational Hierarchical GAT**          | § 4 (Model)           |
| `src/clustering.py`               | Louvain community detection + n‑HDP surrogate clustering             | § 3 (Pre‑processing)  |
| `src/metrics.py`                  | Over‑smoothing / embedding distinctness metrics                      | § 6.3                 |
| `notebooks/`                      | Jupyter notebooks that generate Fig. 7 (t‑SNE) & Fig. 9 (histograms) | Figures 7‑10          |
| `docs/oversmoothing_nhdp_gnn.pdf` | **The manuscript (camera‑ready)**                                    | —                     |

> **Tip:** every figure displayed in the paper can be regenerated with a single make target: `make plots` (see `Makefile`).

---

## ✨  Key Contributions (mirrors the manuscript)

| Contribution                                                                                 | Code component                                   | Where it appears in the PDF |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------- |
| 1. **Hierarchical graph signal**: Louvain→n‑HDP labels concatenated to node features         | `clustering.py › build_hierarchy()`              | Fig. 3, Eq. (5–8)           |
| 2. **Three‑stage multi‑head GAT** (local ▸ community ▸ global) with residual skip‑connection | `models.py › EnhancedVariationalHierarchicalGAT` | Fig. 4, Table 1             |
| 3. **Variational bottleneck** (μ, σ, KL penalty β)                                           | same module                                      | Eq. (12–14)                 |
| 4. **Auxiliary community loss** to keep embeddings discriminative                            | `losses.py`                                      | Eq. (15)                    |
| 5. **Over‑smoothing diagnostics** (pairwise/inter/intra, variance, NMI/ARI)                  | `metrics.py`                                     | Fig. 8, Table 4             |

---

## 🚀  Quick Start

### 1．Clone & install

```bash
# Clone the repo
$ git clone https://github.com/<user>/nHDP‑GNN.git
$ cd nHDP‑GNN

# (Optional) create a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate

# Install dependencies (CPU or CUDA wheel auto‑detected)
$ pip install -r requirements.txt
```

### 2．Reproduce a paper table (Cora example)

```bash
$ python src/main.py \
        --dataset cora \
        --depth 3 \
        --heads 4 \
        --latent-dim 32 \
        --beta 2e-5
```

The script prints accuracy, F1, AUC, and over‑smoothing metrics, then pops up the loss curves, t‑SNE, histograms and confusion‑matrix exactly as shown in Figures 6–10 of the manuscript.

---

## 🏗️  Extending the Work

* **Replace the n‑HDP surrogate** with a true nested Dirichlet Process using `pyro` – see wiki page *Advanced Clustering*.
* **Scale to OGB datasets**: `--dataset ogbn-arxiv` works out‑of‑the‑box; increase `--depth` and adjust GPU memory.
* **Synthetic graphs**: `scripts/gen_synthetic.py` generates the over‑smoothing test suite used in Section 6.4.

---

## 🔒  License

* **Code**: MIT — see [`LICENSE`](LICENSE).
* **Manuscript & figures**: Creative Commons Attribution 4.0 — see [`docs/LICENSE-CC-BY.txt`](docs/LICENSE-CC-BY.txt).

---

## 🙏  Acknowledgements

This project was supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP).
