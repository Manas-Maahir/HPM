from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IdentityLoss(nn.Module):
    """Cross-entropy identity classification loss."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Return scalar cross-entropy loss."""
        return F.cross_entropy(logits, labels)
