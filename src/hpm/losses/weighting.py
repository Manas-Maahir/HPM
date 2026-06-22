from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from hpm.losses.identity import IdentityLoss
from hpm.losses.reconstruction import ReconstructionLoss


@dataclass
class LossComponents:
    identity: Tensor
    recon_unified: Tensor
    recon_local: Tensor
    recon_global: Tensor
    total: Tensor


class LossWeighter(nn.Module):
    """Combines all loss terms weighted by λ values from config.

    Start with lambda_id=1, lambda_uni=0, lambda_aux=0 (Milestone 1).
    Enable aux at Milestone 2 and unified at Milestone 3.
    """

    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__()
        w = cfg.train.loss_weights
        self.lambda_id: float = w.lambda_id
        self.lambda_uni: float = w.lambda_uni
        self.lambda_aux: float = w.lambda_aux
        self.id_loss = IdentityLoss(num_classes)
        self.recon_loss = ReconstructionLoss()

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        unified_pred: Tensor,
        local_pred: Tensor,
        global_pred: Tensor,
        target: Tensor,
    ) -> LossComponents:
        id_ = self.id_loss(logits, labels)
        uni = self.recon_loss(unified_pred, target) if self.lambda_uni > 0 else torch.zeros_like(id_)
        loc = self.recon_loss(local_pred, target) if self.lambda_aux > 0 else torch.zeros_like(id_)
        glob = self.recon_loss(global_pred, target) if self.lambda_aux > 0 else torch.zeros_like(id_)
        total = self.lambda_id * id_ + self.lambda_uni * uni + self.lambda_aux * (loc + glob)
        return LossComponents(identity=id_, recon_unified=uni, recon_local=loc, recon_global=glob, total=total)
