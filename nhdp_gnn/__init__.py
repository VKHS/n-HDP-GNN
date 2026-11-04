"""n-HDP-GNN: community-aware Bayesian clustering for over-smoothing-resilient,
communication-efficient distributed GNNs.

Reference implementation of Khalili Sadaghiani & Nunez-Yanez,
J. Supercomput. 81, 1560 (2025), https://doi.org/10.1007/s11227-025-08017-9

The numpy modules (communities, nhdp, gating, partition, metrics, splits,
synthetic) have no PyTorch dependency; the model, training and distributed
modules require PyTorch and PyTorch Geometric.
"""

__version__ = "2.0.0"
