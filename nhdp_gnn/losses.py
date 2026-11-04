"""Supervised losses: cross-entropy, focal loss (paper Sect. 7, Citeseer), BCE for multi-label."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """Focal loss; with class weights it is the weighted mean (normalised by the sum
    of the weights, as ``F.cross_entropy``), so gamma = 0 recovers weighted CE."""
    logp_t = F.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1.0 - logp_t.exp()) ** gamma) * logp_t
    if weight is not None:
        w = weight[target]
        return (loss * w).sum() / w.sum().clamp_min(1e-12)
    return loss.mean()


def supervised_loss(logits: torch.Tensor, y: torch.Tensor, multilabel: bool, kind: str = "ce",
                    gamma: float = 2.0, weight: torch.Tensor | None = None) -> torch.Tensor:
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, y)
    if kind == "focal":
        return focal_loss(logits, y, gamma, weight)
    if kind == "ce":
        return F.cross_entropy(logits, y, weight=weight)
    raise ValueError(f"unknown loss {kind!r}")
