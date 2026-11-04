"""Baselines at arbitrary depth, sharing the trainer's interface ``forward(batch) -> (logits, hidden, aux)``.

Provided: GCN, GAT, SGC, APPNP, ARMA, GCNII, PairNorm-GCN, DropEdge-GCN, Graph U-Net
(PyTorch Geometric implementations).  Not provided (no maintained reference
implementation): EM-GCN, MVPool, H-GCN, GRAND -- see docs/REPRODUCING.md.

Conventions shared with ``NHDPGNN``:

* ``hidden`` lists the hidden-width representations a model computes layer by
  layer (it is empty for SGC, APPNP and Graph U-Net, whose propagation does not
  produce a layer-wise hidden stack).  Embedding diagnostics use the logits,
  which every model has.
* ``propagation_depth`` is the number of propagation steps actually performed.
  It equals ``depth`` except for Graph U-Net, where depth L becomes max(1, L // 2)
  pooling levels, i.e. 2 * levels + 1 GCN layers.
* Every trainable parameter receives a gradient in every step, as
  DistributedDataParallel requires.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import APPNP, ARMAConv, GATConv, GCN2Conv, GCNConv, GraphUNet, PairNorm, SGConv
from torch_geometric.utils import dropout_edge


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, dropout=0.5, pairnorm=False, dropedge=0.0, **_):
        super().__init__()
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norm = PairNorm() if pairnorm else None
        self.dropout, self.dropedge = dropout, dropedge
        self.propagation_depth = depth

    def forward(self, b):
        ei = b.edge_index
        if self.training and self.dropedge > 0:
            ei, _ = dropout_edge(ei, p=self.dropedge, force_undirected=True)
        h, hidden = b.x, []
        for i, conv in enumerate(self.convs):
            h = conv(F.dropout(h, self.dropout, self.training), ei)
            if i < len(self.convs) - 1:
                if self.norm is not None:
                    h = self.norm(h)
                h = F.relu(h)
                hidden.append(h)
        return h, hidden, None


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, heads=4, dropout=0.5, **_):
        super().__init__()
        if depth > 1 and hidden % heads:
            raise ValueError(f"hidden size ({hidden}) must be divisible by the number of heads ({heads})")
        convs = [GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)]
        convs += [GATConv(hidden, hidden // heads, heads=heads, dropout=dropout) for _ in range(depth - 2)]
        convs += [GATConv(hidden if depth > 1 else in_dim, out_dim, heads=1, concat=False, dropout=dropout)]
        self.convs = nn.ModuleList(convs if depth > 1 else convs[-1:])
        self.dropout = dropout
        self.propagation_depth = depth

    def forward(self, b):
        h, hidden = b.x, []
        for i, conv in enumerate(self.convs):
            h = conv(F.dropout(h, self.dropout, self.training), b.edge_index)
            if i < len(self.convs) - 1:
                h = F.elu(h)
                hidden.append(h)
        return h, hidden, None


class SGC(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, **_):
        super().__init__()
        self.conv = SGConv(in_dim, out_dim, K=depth, cached=False)
        self.propagation_depth = depth

    def forward(self, b):
        return self.conv(b.x, b.edge_index), [], None


class APPNPNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, dropout=0.5, alpha=0.1, **_):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, out_dim)
        self.prop = APPNP(K=depth, alpha=alpha)
        self.dropout = dropout
        self.propagation_depth = depth

    def forward(self, b):
        h = F.relu(self.lin1(F.dropout(b.x, self.dropout, self.training)))
        return self.prop(self.lin2(F.dropout(h, self.dropout, self.training)), b.edge_index), [], None


class ARMA(nn.Module):
    """One ARMA_1 filter per layer (one stack, one recursion).  ``act=None`` inside the
    convolutions: the activation is applied between layers, never on the logits."""

    def __init__(self, in_dim, hidden, out_dim, depth, dropout=0.5, **_):
        super().__init__()
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        self.convs = nn.ModuleList(
            ARMAConv(a, b, num_stacks=1, num_layers=1, shared_weights=True, act=None, dropout=dropout)
            for a, b in zip(dims[:-1], dims[1:]))
        for conv in self.convs:          # the recursive weight is unused with one recursion step
            conv.weight.requires_grad_(False)
        self.dropout = dropout
        self.propagation_depth = depth

    def forward(self, b):
        h, hidden = b.x, []
        for i, conv in enumerate(self.convs):
            h = conv(F.dropout(h, self.dropout, self.training), b.edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                hidden.append(h)
        return h, hidden, None


class GCNII(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, dropout=0.5, alpha=0.1, theta=0.5, **_):
        super().__init__()
        self.lin_in, self.lin_out = nn.Linear(in_dim, hidden), nn.Linear(hidden, out_dim)
        self.convs = nn.ModuleList(GCN2Conv(hidden, alpha=alpha, theta=theta, layer=l + 1, shared_weights=True)
                                   for l in range(depth))
        self.dropout = dropout
        self.propagation_depth = depth

    def forward(self, b):
        h = x0 = F.relu(self.lin_in(F.dropout(b.x, self.dropout, self.training)))
        hidden = [h]
        for conv in self.convs:
            h = F.relu(conv(F.dropout(h, self.dropout, self.training), x0, b.edge_index))
            hidden.append(h)
        return self.lin_out(F.dropout(h, self.dropout, self.training)), hidden, None


class UNet(nn.Module):
    """Graph U-Net with max(1, depth // 2) pooling levels (pool ratio 0.5), i.e. 2 * levels + 1 GCN layers."""

    def __init__(self, in_dim, hidden, out_dim, depth, dropout=0.5, **_):
        super().__init__()
        levels = max(1, depth // 2)
        self.net = GraphUNet(in_dim, hidden, out_dim, depth=levels, pool_ratios=0.5)
        self.dropout = dropout
        self.propagation_depth = 2 * levels + 1

    def forward(self, b):
        return self.net(F.dropout(b.x, self.dropout, self.training), b.edge_index), [], None


BASELINES = {
    "gcn": GCN, "gat": GAT, "sgc": SGC, "appnp": APPNPNet, "arma": ARMA, "gcnii": GCNII,
    "pairnorm": lambda *a, **k: GCN(*a, pairnorm=True, **k),
    "dropedge": lambda *a, **k: GCN(*a, dropedge=0.3, **k),
    "graphunet": UNet,
}


def build_baseline(name: str, in_dim: int, hidden: int, out_dim: int, depth: int, heads: int = 4,
                   dropout: float = 0.5, alpha: float = 0.1, theta: float = 0.5) -> nn.Module:
    """``alpha`` is APPNP's teleport probability and GCNII's initial-residual weight;
    ``theta`` is GCNII's identity-mapping strength."""
    if name not in BASELINES:
        raise ValueError(f"unknown model {name!r}; choose 'nhdp' or one of {sorted(BASELINES)}")
    return BASELINES[name](in_dim, hidden, out_dim, depth, heads=heads, dropout=dropout, alpha=alpha, theta=theta)
