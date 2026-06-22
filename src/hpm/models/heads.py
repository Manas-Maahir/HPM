from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor


class ProjectionHead(nn.Module):
    """MLP projection head on the fused latent z_F → L2-normalised embedding.

    Trained on by the supervised-contrastive loss; DISCARDED at evaluation, where
    verification uses the pre-head z_F (SimCLR/SupCon convention,
    phaseA_celeba_contrastive.md §3). Dimensions are config-driven.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d = cfg.d_model
        hidden = cfg.contrastive.get("proj_hidden", d)
        out = cfg.contrastive.get("proj_dim", 128)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, z_F: Tensor) -> Tensor:
        """Return an L2-normalised embedding [B, proj_dim]."""
        return F.normalize(self.mlp(z_F), dim=1)


class IdentityHead(nn.Module):
    """MLP identity classifier on the fused latent z_F.

    Outputs class logits (cross-entropy) by default.
    Optional ArcFace-style normalised embedding for contrastive training.
    Wire this head FIRST — establish the core asymmetry before reconstruction
    losses are added (see architecture.md §3.6 and build order §8).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d = cfg.d_model
        n = cfg.data.get("num_identities", 500)
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, n),
        )

    def forward(self, z_F: Tensor) -> Tensor:
        """Return logits [B, num_identities]."""
        return self.mlp(z_F)
