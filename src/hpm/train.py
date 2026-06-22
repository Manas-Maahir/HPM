"""Phase A training entry-point (Hydra).

Usage:
    python -m hpm.train experiment=phaseA_identity_only
    python -m hpm.train experiment=phaseA_role_crossover
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from hpm.data.dataset import FaceDataset
from hpm.data.splits import make_identity_splits
from hpm.losses.weighting import LossWeighter
from hpm.models.hpm_model import HPMModel
from hpm.utils.logging import log_run_metadata
from hpm.utils.seed import seed_everything


def _build_model_cfg(cfg: DictConfig) -> DictConfig:
    """Merge model sub-config with data/train so model code sees a flat cfg.

    The model classes access cfg.d_model, cfg.local.*, etc. (model-level keys at root)
    alongside cfg.data.* and cfg.train.*. Hydra nests model under cfg.model, so we
    promote it here to match the structure expected by each module and by conftest.
    """
    # Rebuild model cfg as a non-struct config so merging in data/train (new top-level
    # keys) does not trip Hydra's struct lock on cfg.model.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    return OmegaConf.merge(
        model_cfg,
        OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True)}),
        OmegaConf.create({"train": OmegaConf.to_container(cfg.train, resolve=True)}),
    )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.train.seed)
    log_run_metadata(cfg)

    # ── Data ─────────────────────────────────────────────────────────────────
    root = Path(cfg.data.root)
    train_samples, val_samples, _ = make_identity_splits(root, cfg, cfg.train.seed)
    num_identities = len({label for _, label in train_samples})

    model_cfg = _build_model_cfg(cfg)

    train_ds = FaceDataset(train_samples, model_cfg)
    val_ds = FaceDataset(val_samples, model_cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPMModel(model_cfg).to(device)
    # lesion_gain is non-persistent: must be set after every load_state_dict
    model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)

    weighter = LossWeighter(model_cfg, num_classes=num_identities).to(device)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.train.scheduler, optimizer=optimizer)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg.train.max_epochs):
        model.train()
        train_loss = 0.0
        for x_hi, x_lo, labels in train_loader:
            x_hi = x_hi.to(device, non_blocking=True)
            x_lo = x_lo.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(x_hi, x_lo)
            # Reconstruction target: original normalised image remapped to [-1,1]
            # to match the Tanh decoder output range.
            target = x_hi * 2.0 - 1.0
            components = weighter(
                out.identity_logits,
                labels,
                out.unified_percept,
                out.local_percept,
                out.global_percept,
                target,
            )
            optimizer.zero_grad()
            components.total.backward()
            optimizer.step()
            train_loss += components.total.item()

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_hi, x_lo, labels in val_loader:
                x_hi = x_hi.to(device, non_blocking=True)
                x_lo = x_lo.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                out = model(x_hi, x_lo)
                preds = out.identity_logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        avg_train_loss = train_loss / max(len(train_loader), 1)
        print(f"epoch {epoch:03d}  train_loss={avg_train_loss:.4f}  val_acc={val_acc:.4f}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt = {
        "state_dict": model.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "seed": cfg.train.seed,
    }
    ckpt_path = Path("checkpoints") / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
