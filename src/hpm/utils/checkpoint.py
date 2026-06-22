from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from hpm.utils.logging import config_hash, git_sha


def _rng_states() -> dict[str, Any]:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(_as_byte_tensor(states["torch"]))
    if torch.cuda.is_available() and states.get("cuda") is not None:
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in states["cuda"]])


def _as_byte_tensor(state: Any) -> torch.Tensor:
    """RNG states reload as the wrong dtype/device after torch.load; coerce to CPU ByteTensor."""
    t = state if isinstance(state, torch.Tensor) else torch.tensor(state)
    return t.cpu().to(torch.uint8)


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    cfg: DictConfig,
) -> None:
    """Write a fully-resumable checkpoint (model + optim + sched + AMP + RNG + meta).

    Note: ``callosum.lesion_gain`` is a non-persistent buffer and is intentionally
    NOT in ``state_dict``; the loader re-applies it from config (see load_checkpoint).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_states(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
    }
    # Atomic write: temp then replace, so a mid-save Colab disconnect can't corrupt it.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    cfg: DictConfig | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore training state in place.

    Re-applies ``callosum.lesion_gain`` from ``cfg`` after ``load_state_dict`` because
    it is a non-persistent buffer (see callosum_lesion_gain_convention). Returns a dict
    with ``epoch``, ``global_step``, ``best_metric`` so the caller can resume.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Non-persistent buffer: must be re-applied every load.
    if cfg is not None and hasattr(model, "callosum"):
        model.callosum.set_lesion_gain(float(cfg.callosum.lesion_gain))

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng") is not None:
        _restore_rng_states(ckpt["rng"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_metric": ckpt.get("best_metric", float("-inf")),
    }
