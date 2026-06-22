from __future__ import annotations

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class LocalCNNEncoder(nn.Module):
    """High-frequency / featural stream — L-FFA analogue.

    Default backbone: ResNet50 via timm, optionally loaded with InsightFace ArcFace
    pretrained weights. Pretrained mandatory for small-data regime.
    Outputs a spatial feature map and a pooled token, both projected to d_model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.local.backbone,
            pretrained=cfg.local.pretrained,
            num_classes=0,
            global_pool='',
        )
        if getattr(cfg.local, 'pretrained_path', None):
            state = torch.load(cfg.local.pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(state, strict=False)
        self.proj = nn.Linear(cfg.local.out_channels, cfg.d_model)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (f_L [B,C,h,w], z_L [B,d_model])."""
        f_L = self.backbone.forward_features(x)  # [B, 2048, 7, 7] for resnet50 at 224×224
        z_L = self.proj(self.pool(f_L).flatten(1))  # [B, d_model]
        return f_L, z_L


class GlobalViTEncoder(nn.Module):
    """Low-frequency / holistic stream — R-FFA analogue.

    Default backbone: DeiT-Small/ViT-S-16 (timm). Pretrained is a hard constraint
    (see CLAUDE.md §5.1); from-scratch ViTs will fail on small face datasets.
    Outputs patch tokens and a pooled CLS token, both projected to d_model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.global_.backbone,
            pretrained=cfg.global_.pretrained,
        )
        self.proj = nn.Linear(cfg.global_.out_channels, cfg.d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (T_R [B,N,d_model] patch tokens incl. CLS, z_R [B,d_model])."""
        tokens = self.backbone.forward_features(x)  # [B, N+1, embed_dim] incl. CLS at 0
        T_R = self.proj(tokens)                     # [B, N+1, d_model]
        z_R = T_R[:, 0]                             # CLS token → [B, d_model]
        return T_R, z_R
