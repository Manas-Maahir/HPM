from __future__ import annotations

import hashlib
import logging
import subprocess

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def git_sha() -> str:
    """Current git commit SHA, or 'unknown' outside a repo."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# Backwards-compatible private alias.
_git_sha = git_sha


def config_hash(cfg: DictConfig) -> str:
    """Stable 12-char hash of the resolved config for run identity."""
    cfg_yaml = OmegaConf.to_yaml(cfg)
    return hashlib.sha256(cfg_yaml.encode()).hexdigest()[:12]


def log_run_metadata(cfg: DictConfig) -> None:
    """Log seed, config hash, and git SHA once per run before training starts."""
    log.info(
        "seed=%s  config_hash=%s  git_sha=%s",
        cfg.train.seed,
        config_hash(cfg),
        git_sha(),
    )
