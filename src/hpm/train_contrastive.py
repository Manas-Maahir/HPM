"""Phase A supervised-contrastive trainer (CelebA, Colab-friendly).

Adds the machinery `train.py` lacks for Milestone 1: multi-view SupCon, PK
sampling, a projection head, differential LRs, AMP, an ultralytics-style progress
bar, and fully-resumable `last.pt` / `best.pt` checkpointing.

Designed to be driven from a notebook:

    from omegaconf import OmegaConf
    from hpm.train_contrastive import run
    cfg = OmegaConf.load("configs/...composed.yaml")   # or build in code
    run(cfg, resume="auto")
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from hpm.data.dataset import ContrastiveFaceDataset, FaceDataset
from hpm.data.sampler import PKSampler
from hpm.data.splits import make_splits
from hpm.eval.verification import evaluate_verification
from hpm.losses.contrastive import ArcFaceLoss, SupConLoss
from hpm.models.hpm_model import HPMModel
from hpm.train import _build_model_cfg
from hpm.utils.checkpoint import load_checkpoint, save_checkpoint
from hpm.utils.logging import log_run_metadata
from hpm.utils.optim import (
    build_param_groups,
    build_warmup_cosine_scheduler,
    set_backbone_requires_grad,
)
from hpm.utils.seed import seed_everything


def _gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / 1e9
    return 0.0


def _build_criterion(model_cfg: DictConfig, num_identities: int) -> torch.nn.Module:
    c = model_cfg.contrastive
    loss_name = str(c.get("loss", "supcon")).lower()
    if loss_name == "arcface":
        return ArcFaceLoss(
            embed_dim=c.get("proj_dim", 128),
            num_classes=num_identities,
            scale=c.get("arcface_scale", 30.0),
            margin=c.get("arcface_margin", 0.5),
        )
    return SupConLoss(temperature=c.get("temperature", 0.07))


def run(cfg: DictConfig, resume: str | bool = "auto") -> dict[str, float]:
    """Train the two-stream HPM with supervised contrastive loss.

    ``resume="auto"`` continues from ``last.pt`` if it exists (the Colab
    continuation mechanism); ``True`` forces resume; ``False`` starts fresh.
    Returns a summary dict including the best verification ROC-AUC.
    """
    seed_everything(cfg.train.seed)
    log_run_metadata(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Config: promote model sub-config to root (matches model code + tests) ──
    model_cfg = _build_model_cfg(cfg)
    OmegaConf.set_struct(model_cfg, False)

    # ── Data ───────────────────────────────────────────────────────────────────
    root = Path(cfg.data.root)
    train_samples, val_samples, _ = make_splits(root, cfg, cfg.train.seed)
    num_identities = len({label for _, label in train_samples})
    model_cfg.data.num_identities = num_identities

    sampler_cfg = model_cfg.data.sampler
    train_ds = ContrastiveFaceDataset(train_samples, model_cfg)
    val_ds = FaceDataset(val_samples, model_cfg)

    train_labels = [label for _, label in train_samples]
    pk_sampler = PKSampler(
        train_labels,
        P=sampler_cfg.P,
        K=sampler_cfg.K,
        num_batches=sampler_cfg.get("num_batches", None),
        seed=cfg.train.seed,
    )
    num_workers = int(cfg.train.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_sampler=pk_sampler, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── Model + loss ─────────────────────────────────────────────────────────────
    model = HPMModel(model_cfg).to(device)
    model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)
    if model.projection_head is None:
        raise ValueError(
            "Contrastive training requires model.contrastive.enabled=true "
            "so the projection head is built."
        )
    criterion = _build_criterion(model_cfg, num_identities).to(device)

    # ── Optimizer / scheduler / AMP ──────────────────────────────────────────────
    param_groups = build_param_groups(model, model_cfg)
    lr_new = float(cfg.train.get("lr_new", 1e-3))
    wd = float(cfg.train.optimizer.get("weight_decay", 0.05))
    if list(criterion.parameters()):  # ArcFace has a learnable weight matrix
        param_groups.append(
            {
                "params": list(criterion.parameters()),
                "lr": lr_new,
                "weight_decay": wd,
                "name": "criterion",
            }
        )
    optimizer = AdamW(param_groups)

    grad_accum = int(cfg.train.get("grad_accum", 1))
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * cfg.train.max_epochs
    warmup_steps = int(cfg.train.get("warmup_steps", steps_per_epoch))
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    use_amp = bool(cfg.train.get("amp", True)) and torch.cuda.is_available()
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    # ── Checkpoint / resume ──────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.train.checkpoint.dir)
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    save_every_steps = int(cfg.train.checkpoint.get("save_every_steps", 0))

    start_epoch, global_step, best_metric = 0, 0, float("-inf")
    should_resume = resume is True or (resume == "auto" and last_path.exists())
    if should_resume and last_path.exists():
        state = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=model_cfg,
            map_location=device,
        )
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_metric = state["best_metric"]
        print(f"resumed from {last_path} -> epoch {start_epoch}, best_roc_auc={best_metric:.4f}")

    freeze_epochs = int(cfg.train.get("freeze_epochs", 0))
    max_steps = cfg.train.get("max_steps", None)  # smoke-test cap

    def _save(path: Path, epoch: int) -> None:
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            cfg=model_cfg,
        )

    # ── Training loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if freeze_epochs > 0:
            set_backbone_requires_grad(model, epoch >= freeze_epochs)
        pk_sampler.set_epoch(epoch)
        model.train()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        running_loss = 0.0
        optimizer.zero_grad()
        print(f"\n{'Epoch':>10}{'GPU_mem':>10}{'loss':>10}{'lr':>12}")
        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            bar_format="{l_bar}{bar:10}{r_bar}",
            leave=True,
        )
        for i, (v1_hi, v1_lo, v2_hi, v2_lo, labels) in enumerate(pbar):
            v1_hi, v1_lo = v1_hi.to(device, non_blocking=True), v1_lo.to(device, non_blocking=True)
            v2_hi, v2_lo = v2_hi.to(device, non_blocking=True), v2_lo.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                out1 = model(v1_hi, v1_lo)
                out2 = model(v2_hi, v2_lo)
                embeddings = torch.cat([out1.embedding, out2.embedding], dim=0)
                two_view_labels = torch.cat([labels, labels], dim=0)
                loss = criterion(embeddings, two_view_labels) / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum

            if (i + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    _save(last_path, epoch)

            cur_lr = optimizer.param_groups[-1]["lr"]
            pbar.set_description(
                f"{epoch + 1:>4}/{cfg.train.max_epochs:<5}"
                f"{_gpu_mem_gb():>9.2f}G{running_loss / (i + 1):>10.4f}{cur_lr:>12.2e}"
            )

            if max_steps is not None and global_step >= max_steps:
                break

        # ── Validation: open-set verification (health check + best selector) ──────
        metrics = evaluate_verification(
            model,
            val_loader,
            device,
            num_pairs=int(cfg.train.checkpoint.get("verification_pairs", 10000)),
            seed=cfg.train.seed,
        )
        roc = metrics["roc_auc"]
        print(
            f"epoch {epoch:03d}  loss={running_loss / max(len(train_loader), 1):.4f}  "
            f"roc_auc={roc:.4f}  tar@far={metrics['tar_at_far']:.4f}  "
            f"acc={metrics['accuracy']:.4f}"
        )

        # Update best BEFORE saving last.pt so the resumed best_metric is correct
        # (otherwise a later, worse epoch could overwrite best.pt after a resume).
        if roc > best_metric:
            best_metric = roc
            _save(best_path, epoch)
            print(f"  * new best roc_auc={best_metric:.4f} -> {best_path}")
        _save(last_path, epoch)

        if max_steps is not None and global_step >= max_steps:
            print(f"reached max_steps={max_steps}; stopping (smoke run).")
            break

    print(f"\ntraining complete. best roc_auc={best_metric:.4f}")
    return {"best_roc_auc": best_metric, "checkpoint_dir": str(ckpt_dir)}
