from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class CorpusCallosum(nn.Module):
    """Bidirectional cross-attention between the local (L) and global (R) streams.

    Two passes: L←R and R←L. Each update is scaled by lesion_gain before
    being added back to its stream.

    lesion_gain=1.0 → full interhemispheric transfer
    lesion_gain=0.0 → split-brain condition (no exchange at all)

    lesion_gain is a NON-PERSISTENT buffer (persistent=False): it is not serialised
    into checkpoints. After every load_state_dict the caller must re-apply:
        model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)
    Use set_lesion_gain() during eval to sweep lesion values on one checkpoint.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d = cfg.d_model
        h = cfg.callosum.heads
        self.l_from_r = nn.MultiheadAttention(d, h, batch_first=True)
        self.r_from_l = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm_l = nn.LayerNorm(d)
        self.norm_r = nn.LayerNorm(d)
        self.register_buffer(
            "lesion_gain",
            torch.tensor(float(cfg.callosum.lesion_gain)),
            persistent=False,
        )

    def set_lesion_gain(self, value: float) -> None:
        self.lesion_gain.fill_(value)

    def forward(self, z_L: Tensor, z_R: Tensor) -> tuple[Tensor, Tensor]:
        """Return (z_L' [B,N,d], z_R' [B,N,d]) after bidirectional exchange × lesion_gain."""
        delta_L, _ = self.l_from_r(query=z_L, key=z_R, value=z_R)
        delta_R, _ = self.r_from_l(query=z_R, key=z_L, value=z_L)
        # Norm the attention delta (not the residual sum) so lesion_gain=0 is
        # EXACT identity (split-brain = no exchange, stream untouched).
        z_L_prime = z_L + self.lesion_gain * self.norm_l(delta_L)
        z_R_prime = z_R + self.lesion_gain * self.norm_r(delta_R)
        return z_L_prime, z_R_prime
