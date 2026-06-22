from __future__ import annotations

import math

import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Parameter-name prefixes that identify the two pretrained backbones. Everything
# else (encoder projections, callosum, fusion, heads) is a freshly-initialised
# module and trains at the higher learning rate.
_BACKBONE_PREFIXES = ("local_enc.backbone", "global_enc.backbone")


def _is_backbone(name: str) -> bool:
    return name.startswith(_BACKBONE_PREFIXES)


def build_param_groups(model: nn.Module, cfg: DictConfig) -> list[dict]:
    """Differential learning rates: low for pretrained backbones, high for new modules.

    phaseA_celeba_contrastive.md §4. Reads ``cfg.train.lr_backbone`` and
    ``cfg.train.lr_new``; weight decay falls back to the optimizer config.
    """
    lr_backbone = float(cfg.train.get("lr_backbone", 1e-5))
    lr_new = float(cfg.train.get("lr_new", 1e-3))
    weight_decay = float(cfg.train.optimizer.get("weight_decay", 0.05))

    backbone_params, new_params = [], []
    for name, p in model.named_parameters():
        (backbone_params if _is_backbone(name) else new_params).append(p)

    return [
        {
            "params": backbone_params,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
            "name": "backbone",
        },
        {"params": new_params, "lr": lr_new, "weight_decay": weight_decay, "name": "new"},
    ]


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Freeze/unfreeze both pretrained backbones (freeze schedule, §4)."""
    for name, p in model.named_parameters():
        if _is_backbone(name):
            p.requires_grad = requires_grad


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup → cosine decay, applied per optimizer step.

    The multiplier scales each param group's base LR, so the differential
    backbone/new LRs are preserved throughout the schedule.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
