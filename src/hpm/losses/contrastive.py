from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls together all embeddings sharing an identity label and pushes apart the
    rest, within a batch. Operates on L2-normalised embeddings; with PK sampling
    every anchor has K-1 (or more, across views) in-batch positives.

    Expects ``features`` shape ``[N, d]`` (already L2-normalised) and integer
    ``labels`` shape ``[N]``. Multiple augmented views are simply concatenated
    along the batch dimension before being passed in.

    Temperature τ comes from config (~0.07–0.1). This is the "SupConOut/SupConIn"
    formulation (loss = mean over anchors of -log of the mean positive likelihood).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        device = features.device
        n = features.shape[0]

        # Cosine-similarity logits (features assumed unit-norm) scaled by τ.
        logits = features @ features.t() / self.temperature
        # Numerical stability: subtract per-row max (detached).
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).float().to(device)
        # Exclude self-comparisons from both positives and the denominator.
        self_mask = torch.eye(n, device=device)
        positive_mask = positive_mask - self_mask
        logits_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = positive_mask.sum(dim=1)
        # Mean log-likelihood over positives, for anchors that have ≥1 positive.
        valid = pos_per_anchor > 0
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / pos_per_anchor[valid]

        if mean_log_prob_pos.numel() == 0:
            # No in-batch positives (degenerate batch) → no gradient signal.
            return features.sum() * 0.0
        return -mean_log_prob_pos.mean()


class ArcFaceLoss(nn.Module):
    """ArcFace (Deng et al., 2019) — documented config alternative to SupCon.

    Additive angular margin softmax over a learnable per-identity weight matrix.
    Kept as a config switch (``cfg.contrastive.loss=arcface``); SupCon is the
    Milestone-1 default. Requires ``num_classes`` (number of train identities).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        # features assumed L2-normalised; normalise class weights too.
        cosine = F.linear(features, F.normalize(self.weight, dim=1)).clamp(-1.0, 1.0)
        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        logits = self.scale * (one_hot * target_logit + (1.0 - one_hot) * cosine)
        return F.cross_entropy(logits, labels)
