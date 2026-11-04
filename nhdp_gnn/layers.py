"""Layers of n-HDP-GNN (paper Sects. 5, 6 and 10).

GatedGATConv        node-level multi-head attention over the responsibility-gated
                    operator: alpha_ij = g_ij exp(e_ij) / sum_k g_ik exp(e_ik).
CommunityAttention  attention among super-nodes on the super-node graph (Eq. 16);
                    super-node states are the responsibility-weighted means of node
                    states (pooling S^T) and are returned to nodes by S (unpooling).
GlobalAttention     attention read-out over super-nodes, broadcast to every node (Eq. 17).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class GatedGATConv(MessagePassing):
    """Multi-head attention normalised over gated neighbourhoods.

    ``edge_index`` must contain self-loops; ``gate`` holds one value in [0, 1]
    per edge.  A zero gate removes the edge; a uniform gate recovers GAT.
    """

    def __init__(self, channels: int, heads: int = 4, attn_dropout: float = 0.0, negative_slope: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        if channels % heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.heads, self.head_dim = heads, channels // heads
        self.attn_dropout, self.negative_slope = attn_dropout, negative_slope
        self.lin = nn.Linear(channels, channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        H, C = self.heads, self.head_dim
        xh = self.lin(x).view(-1, H, C)
        a_src = (xh * self.att_src).sum(-1)
        a_dst = (xh * self.att_dst).sum(-1)
        log_gate = torch.log(gate.clamp_min(1e-12)).unsqueeze(-1)
        out = self.propagate(edge_index, x=xh, a_src=a_src, a_dst=a_dst, log_gate=log_gate)
        return out.reshape(-1, H * C) + self.bias

    def message(self, x_j, a_src_j, a_dst_i, log_gate, index, ptr, size_i):
        e = F.leaky_relu(a_src_j + a_dst_i, self.negative_slope) + log_gate
        alpha = softmax(e, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class CommunityAttention(nn.Module):
    """Multi-head attention among neighbouring super-nodes (self-loops added)."""

    def __init__(self, hidden: int, heads: int = 4, attn_dropout: float = 0.0):
        super().__init__()
        self.conv = GATConv(hidden, hidden // heads, heads=heads, concat=True, dropout=attn_dropout,
                            add_self_loops=True)

    def forward(self, hc: torch.Tensor, super_edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(hc, super_edge_index)


class GlobalAttention(nn.Module):
    """Attention read-out over super-nodes; empty super-nodes get zero weight."""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hc: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        logits = self.score(hc).squeeze(-1) + torch.log(mass.clamp_min(1e-12))
        w = torch.softmax(logits, dim=0)
        return self.proj((w.unsqueeze(-1) * hc).sum(0, keepdim=True))
