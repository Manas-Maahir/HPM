from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from hpm.models.callosum import CorpusCallosum
from hpm.models.decoders import StreamReadoutDecoder, UnifiedPerceptDecoder
from hpm.models.fusion import RightDominantFusion
from hpm.models.heads import IdentityHead, ProjectionHead
from hpm.models.streams import GlobalViTEncoder, LocalCNNEncoder


@dataclass
class HPMOutput:
    identity_logits: Tensor  # [B, num_identities]
    z_L: Tensor  # post-callosum L latent [B,N,d]
    z_R: Tensor  # post-callosum R latent [B,N,d]
    z_F: Tensor  # fused latent [B,d]
    local_percept: Tensor | None = None  # [B,3,H,W] L read-out (None if decoders off)
    global_percept: Tensor | None = None  # [B,3,H,W] R read-out (None if decoders off)
    unified_percept: Tensor | None = None  # [B,3,H,W] integrated face (None if off)
    embedding: Tensor | None = None  # L2-normalised projection of z_F (contrastive)


class HPMModel(nn.Module):
    """Full Hemispheric Perception Model.

    Assembles: LocalCNNEncoder → CorpusCallosum ← GlobalViTEncoder
               → RightDominantFusion → UnifiedPerceptDecoder + IdentityHead
               + two StreamReadoutDecoders (auxiliary).

    x_hi and x_lo are produced upstream by FrequencySplit (data pipeline).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.local_enc = LocalCNNEncoder(cfg)
        self.global_enc = GlobalViTEncoder(cfg)
        self.callosum = CorpusCallosum(cfg)
        self.fusion = RightDominantFusion(cfg)

        # Read-out decoders are diagnostic; skip them when their loss weights are 0
        # (Milestone-1 contrastive) to save memory on a T4. Default on so the full
        # forward contract is unchanged for later milestones / existing tests.
        self.build_decoders = bool(cfg.get("build_decoders", True))
        if self.build_decoders:
            self.l_decoder = StreamReadoutDecoder(cfg)
            self.r_decoder = StreamReadoutDecoder(cfg)
            self.unified_decoder = UnifiedPerceptDecoder(cfg)

        self.identity_head = IdentityHead(cfg)

        # Projection head for supervised-contrastive training (None unless enabled).
        contrastive_cfg = cfg.get("contrastive", None)
        if contrastive_cfg is not None and contrastive_cfg.get("enabled", False):
            self.projection_head: ProjectionHead | None = ProjectionHead(cfg)
        else:
            self.projection_head = None

    def forward(self, x_hi: Tensor, x_lo: Tensor) -> HPMOutput:
        """
        x_hi: high-pass face [B, 3, H, W]  → local / L-FFA stream
        x_lo: low-pass face  [B, 3, H, W]  → global / R-FFA stream
        """
        f_L, z_L = self.local_enc(x_hi)  # f_L [B,C,h,w], z_L [B,d]
        T_R, z_R = self.global_enc(x_lo)  # T_R [B,N+1,d], z_R [B,d]

        # Project CNN spatial map to token sequence for callosum cross-attention.
        # local_enc.proj is nn.Linear(C→d), applied along last dim → [B, h*w, d].
        z_L_seq = self.local_enc.proj(f_L.flatten(2).transpose(1, 2))

        z_L_prime, z_R_prime = self.callosum(z_L_seq, T_R)  # [B,N,d] each

        z_L_pooled = z_L_prime.mean(1)  # mean pool local tokens  [B, d]
        z_R_pooled = z_R_prime[:, 0]  # CLS token from global   [B, d]

        z_F = self.fusion(z_L_pooled, z_R_pooled)  # [B, d]

        if self.build_decoders:
            local_percept = self.l_decoder(z_L_pooled)
            global_percept = self.r_decoder(z_R_pooled)
            unified_percept = self.unified_decoder(z_F)
        else:
            local_percept = global_percept = unified_percept = None

        embedding = self.projection_head(z_F) if self.projection_head is not None else None

        return HPMOutput(
            identity_logits=self.identity_head(z_F),
            z_L=z_L_prime,
            z_R=z_R_prime,
            z_F=z_F,
            local_percept=local_percept,
            global_percept=global_percept,
            unified_percept=unified_percept,
            embedding=embedding,
        )
