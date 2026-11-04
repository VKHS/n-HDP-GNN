"""n-HDP-GNN model: encoder, ``depth`` hierarchical layers, decoder.

Each layer (paper Sects. 5-6, 8 and 10):

    m_i   = node-level message over the gated operator     (GatedGATConv, or gated GCN)
          + S_i . CommAttn(S^T H / mass)                     (community level, Eq. 16)
          + GlobalAttn(super-node states)                    (global level, Eq. 17)
    u_i   = Dropout(ELU(Norm(m_i)))                          (normalisation, activation, dropout)
    h_i'  = h_i + u_i            residual="res"
          = (1-a) u_i + a h0_i   residual="initial"  (initial residual)
          = u_i                  residual="none"

S is the top-k responsibility matrix pi from Stage B (or a hard Louvain/random
assignment in the ablations).  An optional auxiliary head predicts Louvain
community ids (Conclusion of the paper).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from .layers import CommunityAttention, GatedGATConv, GlobalAttention


def _norm(kind: str, hidden: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(hidden)
    if kind == "layer":
        return nn.LayerNorm(hidden)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


class NHDPLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float, attn_dropout: float, node_op: str = "gat",
                 community: bool = True, global_attn: bool = True, residual: str = "res",
                 alpha: float = 0.1, norm: str = "batch"):
        super().__init__()
        if node_op == "gat":
            self.node = GatedGATConv(hidden, heads, attn_dropout)
        elif node_op == "gcn":
            self.node = GCNConv(hidden, hidden, normalize=False, add_self_loops=False)
        else:
            raise ValueError(f"unknown node_op {node_op!r}")
        if residual not in ("res", "initial", "none"):
            raise ValueError(f"unknown residual {residual!r}")
        self.node_op, self.residual, self.alpha, self.dropout = node_op, residual, alpha, dropout
        self.comm = CommunityAttention(hidden, heads, attn_dropout) if community else None
        self.glob = GlobalAttention(hidden) if global_attn else None
        self.norm = _norm(norm, hidden)

    def forward(self, h: torch.Tensor, h0: torch.Tensor, b) -> torch.Tensor:
        if self.node_op == "gat":
            m = self.node(h, b.edge_index_sl, b.gate_sl)
        else:
            m = self.node(h, b.edge_index_sl, b.gcn_weight_sl * b.gate_sl)
        if b.S is not None and (self.comm is not None or self.glob is not None):
            mass = b.S.sum(0)
            hc = (b.S.t() @ h) / mass.clamp_min(1e-12).unsqueeze(-1)
            if self.comm is not None:
                hc = self.comm(hc, b.super_edge_index)
                m = m + b.S @ hc
            if self.glob is not None:
                m = m + self.glob(hc, mass)
        u = F.dropout(F.elu(self.norm(m)), self.dropout, self.training)
        if self.residual == "res":
            return h + u
        if self.residual == "initial":
            return (1.0 - self.alpha) * u + self.alpha * h0
        return u


class NHDPGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int, heads: int = 4, dropout: float = 0.5,
                 attn_dropout: float = 0.2, node_op: str = "gat", community: bool = True, global_attn: bool = True,
                 residual: str = "res", alpha: float = 0.1, norm: str = "batch", aux_classes: int = 0):
        super().__init__()
        self.dropout = dropout
        self.encoder = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            NHDPLayer(hidden, heads, dropout, attn_dropout, node_op, community, global_attn, residual, alpha, norm)
            for _ in range(depth)])
        self.decoder = nn.Linear(hidden, out_dim)
        self.aux = nn.Linear(hidden, aux_classes) if aux_classes > 0 else None

    def forward(self, b):
        h = F.relu(self.encoder(F.dropout(b.x, self.dropout, self.training)))
        h0, hidden = h, [h]
        for layer in self.layers:
            h = layer(h, h0, b)
            hidden.append(h)
        logits = self.decoder(F.dropout(h, self.dropout, self.training))
        aux = self.aux(h) if self.aux is not None else None
        return logits, hidden, aux
