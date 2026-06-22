from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class RightDominantFusion(nn.Module):
    """Right-dominant latent fusion of the two post-callosum representations.

    Default: query=z_R (right hemisphere drives the integrated percept).
    Configurable via cfg.fusion.query: "R" | "L" | "symmetric".
    cfg.fusion.ablate=True bypasses attention and uses a concat→project fallback.

    This module is designed to be swappable; all variants live in one code path
    controlled by config — never fork logic for control conditions.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.query_side: str = cfg.fusion.query
        self.ablate: bool = cfg.fusion.ablate
        if not self.ablate:
            self.attn = nn.MultiheadAttention(d, cfg.fusion.heads, batch_first=True)
            self.norm = nn.LayerNorm(d)
        else:
            self.proj = nn.Linear(2 * d, d)

    def forward(self, z_L: Tensor, z_R: Tensor) -> Tensor:
        """Return fused latent z_F [B, d_model]."""
        if self.ablate:
            return self.proj(torch.cat([z_L, z_R], dim=-1))

        if self.query_side == "R":
            query, kv = z_R.unsqueeze(1), torch.stack([z_L, z_R], dim=1)
        elif self.query_side == "L":
            query, kv = z_L.unsqueeze(1), torch.stack([z_L, z_R], dim=1)
        else:  # symmetric
            query = kv = torch.stack([z_L, z_R], dim=1)

        out, _ = self.attn(query, kv, kv)
        return self.norm(out.squeeze(1))
