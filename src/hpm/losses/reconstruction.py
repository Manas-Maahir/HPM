from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReconstructionLoss(nn.Module):
    """L1 + LPIPS perceptual loss for face reconstruction.

    LPIPS network (default: vgg) is frozen and used as a fixed perceptual metric.
    """

    def __init__(self, lpips_net: str = "vgg") -> None:
        super().__init__()
        # Import lazily so the package is optional until reconstruction is enabled.
        import lpips
        self.lpips = lpips.LPIPS(net=lpips_net)
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return scalar L1 + LPIPS loss. pred and target: [B,3,H,W] in [-1,1]."""
        l1 = F.l1_loss(pred, target)
        perceptual = self.lpips(pred, target).mean()
        return l1 + perceptual
