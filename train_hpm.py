#!/usr/bin/env python3
"""
HPM (Hemispheric Perception Model) — Standalone Overnight Training Script
==========================================================================

Trains CNN, ViT, and full HPM models on CelebA, end-to-end, unsupervised.
Ultralytics-style: organised runs/ dir, CSV logging, best/last checkpoints.
Safe for overnight runs: per-epoch crash recovery, per-mode error isolation,
full tee log to disk.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREREQUISITES  (run ONCE before starting the overnight run)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Install dependencies
    pip install torch torchvision timm tqdm lpips
    # or from the repo root:
    pip install -e ".[dev]"

    Required:
      torch / torchvision  — deep learning framework
      timm                 — pretrained ResNet-50 + DeiT-Small backbones (MANDATORY)
      tqdm                 — progress bars
    Optional (only needed with --lambda-uni or --lambda-aux > 0):
      lpips                — perceptual reconstruction loss

STEP 2 — Get CelebA data  (~1.4 GB, choose ONE option)

  Option A  Auto-download  (needs internet; may hit Google Drive quota):
    python train_hpm.py --dataset celeba --data data/celeba --download --smoke-test

  Option B  Manual download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    Place files here:
      data/celeba/
        img_align_celeba/      ← ~202,599 aligned face JPEGs
        identity_CelebA.txt    ← "000001.jpg 2880" — one line per image
    The script resolves Kaggle double-nesting automatically.

STEP 3 — Verify setup  (~60 s, strongly recommended)
    python train_hpm.py --dataset celeba --data data/celeba --smoke-test
    All three modes (CNN / ViT / HPM) must print PASS.
    If any FAIL, fix the error before starting the overnight run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE OVERNIGHT COMMAND  (single line — start and walk away)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python train_hpm.py --dataset celeba --data data/celeba --mode all \
        --name overnight --epochs 50 --seeds 42,43,44 --workers 4

  --mode all      train CNN → ViT → HPM in sequence
  --name overnight  save everything under runs/hpm/overnight_{cnn,vit,hpm}_seed{seed}/
  --epochs 50     max epochs per model  (early-stops after --patience epochs w/o gain)
  --seeds 42,43,44  train every model on each seed → mean ± SD across seeds in the report
  --workers 4     DataLoader worker processes
  (AMP is ON by default on CUDA; add --no-amp to disable.  --patience defaults to 3.)

TO RESUME AFTER A CRASH — re-run the exact same command.
  The script reads each (model, seed) last.pt, skips completed runs, and resumes
  the interrupted one from the last finished epoch.  No flags needed.

OUTPUT FILES:
    runs/hpm/
      overnight.log                       ← full tee'd training log
      overnight_report.xlsx               ← metrics workbook (mean ± SD across seeds)
      overnight_cnn_seed42/weights/best.pt
      overnight_cnn_seed42/weights/last.pt
      overnight_cnn_seed42/results.csv
      overnight_vit_seed42/…
      overnight_hpm_seed42/…

    # Rebuild the workbook later without retraining:
    python train_hpm.py --report-only --name overnight --seeds 42,43,44

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OTHER USEFUL COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # VGGFace2 layout  (root/{n000001}/*.jpg)
    python train_hpm.py --data /path/to/vggface2 --name phaseA_run1

    # Resume a single-mode run manually
    python train_hpm.py --resume runs/hpm/myrun_hpm/weights/last.pt --mode hpm --name myrun_hpm

    # With reconstruction losses  (lpips recommended)
    python train_hpm.py --data data/celeba --dataset celeba --lambda-uni 0.1 --lambda-aux 0.05

    # Ablation: split-brain (no callosum exchange)
    python train_hpm.py --data data/celeba --dataset celeba --lesion-gain 0.0 --name split_brain

    # Ablation: no frequency split
    python train_hpm.py --data data/celeba --dataset celeba --no-freq-split --name no_freq
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    import timm
except ImportError:
    sys.exit("ERROR: timm required — pip install timm")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: tqdm required — pip install tqdm")

try:
    import lpips as _lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LocalConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    pretrained_path: str | None = None
    out_channels: int = 2048             # ResNet50 layer4 output channels


@dataclass
class GlobalConfig:
    backbone: str = "deit_small_patch16_224"
    pretrained: bool = True
    out_channels: int = 384              # DeiT-Small embed dim


@dataclass
class CallosumConfig:
    depth: int = 1
    heads: int = 4
    lesion_gain: float = 1.0            # 0.0 = split-brain; non-persistent buffer


@dataclass
class FusionConfig:
    query: str = "R"                    # "R" | "L" | "symmetric"
    ablate: bool = False                # True → concat fallback instead of attention
    heads: int = 4


@dataclass
class FreqSplitConfig:
    sigma_hi: float = 1.0              # high-pass: img − GaussianBlur(img, sigma_hi)
    sigma_lo: float = 3.0              # low-pass:  GaussianBlur(img, sigma_lo)
    enabled: bool = True               # False → both streams receive unmodified image


@dataclass
class LossWeightsConfig:
    lambda_id: float = 1.0
    lambda_uni: float = 0.0            # unified-percept reconstruction
    lambda_aux: float = 0.0            # per-stream reconstruction (local + global)


@dataclass
class DataConfig:
    name: str = "vggface2"
    root: str = "data/vggface2"
    image_size: int = 224
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    num_identities: int = 500           # filled in at runtime from actual split
    freq_split: FreqSplitConfig = field(default_factory=FreqSplitConfig)


@dataclass
class TrainConfig:
    seed: int = 42
    max_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    patience: int = 3                   # early-stopping patience (0 = disabled)
    amp: bool = True                    # automatic mixed precision (CUDA only)
    workers: int = 4
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)


@dataclass
class HPMConfig:
    mode: str = "hpm"                   # "hpm" | "cnn" | "vit"
    d_model: int = 256
    build_decoders: bool = False        # True only when any recon lambda > 0
    local: LocalConfig = field(default_factory=LocalConfig)
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    callosum: CallosumConfig = field(default_factory=CallosumConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

class LocalCNNEncoder(nn.Module):
    """High-frequency / featural stream — L-FFA analogue (ResNet50)."""

    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.local.backbone,
            pretrained=cfg.local.pretrained,
            num_classes=0,
            global_pool="",
        )
        if cfg.local.pretrained_path:
            state = torch.load(cfg.local.pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)
        self.proj = nn.Linear(cfg.local.out_channels, cfg.d_model)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (f_L [B,C,h,w], z_L [B,d_model])."""
        f_L = self.backbone.forward_features(x)          # [B, 2048, 7, 7]
        z_L = self.proj(self.pool(f_L).flatten(1))       # [B, d_model]
        return f_L, z_L


class GlobalViTEncoder(nn.Module):
    """Low-frequency / holistic stream — R-FFA analogue (DeiT-Small).

    pretrained MUST remain True — from-scratch ViTs fail on small face datasets.
    """

    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.global_.backbone,
            pretrained=cfg.global_.pretrained,
        )
        self.proj = nn.Linear(cfg.global_.out_channels, cfg.d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (T_R [B,N+1,d_model], z_R [B,d_model])."""
        tokens = self.backbone.forward_features(x)       # [B, N+1, embed_dim]
        T_R = self.proj(tokens)                          # [B, N+1, d_model]
        z_R = T_R[:, 0]                                  # CLS token
        return T_R, z_R


class CorpusCallosum(nn.Module):
    """Bidirectional cross-attention between streams.

    lesion_gain is a NON-PERSISTENT buffer — it is not serialised into checkpoints.
    After every load_state_dict call set_lesion_gain() to restore the intended value.
    """

    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.callosum.heads
        self.l_from_r = nn.MultiheadAttention(d, h, batch_first=True)
        self.r_from_l = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm_l = nn.LayerNorm(d)
        self.norm_r = nn.LayerNorm(d)
        self.register_buffer(
            "lesion_gain",
            torch.tensor(float(cfg.callosum.lesion_gain)),
            persistent=False,
        )

    def set_lesion_gain(self, value: float) -> None:
        self.lesion_gain.fill_(value)

    def forward(self, z_L: Tensor, z_R: Tensor) -> tuple[Tensor, Tensor]:
        delta_L, _ = self.l_from_r(query=z_L, key=z_R, value=z_R)
        delta_R, _ = self.r_from_l(query=z_R, key=z_L, value=z_L)
        z_L_prime = z_L + self.lesion_gain * self.norm_l(delta_L)
        z_R_prime = z_R + self.lesion_gain * self.norm_r(delta_R)
        return z_L_prime, z_R_prime


class RightDominantFusion(nn.Module):
    """Right-dominant latent fusion (query=z_R by default)."""

    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.query_side = cfg.fusion.query
        self.ablate = cfg.fusion.ablate
        if not self.ablate:
            self.attn = nn.MultiheadAttention(d, cfg.fusion.heads, batch_first=True)
            self.norm = nn.LayerNorm(d)
        else:
            self.proj = nn.Linear(2 * d, d)

    def forward(self, z_L: Tensor, z_R: Tensor) -> Tensor:
        if self.ablate:
            return self.proj(torch.cat([z_L, z_R], dim=-1))
        if self.query_side == "R":
            query, kv = z_R.unsqueeze(1), torch.stack([z_L, z_R], dim=1)
        elif self.query_side == "L":
            query, kv = z_L.unsqueeze(1), torch.stack([z_L, z_R], dim=1)
        else:  # symmetric — note: out is [B,2,d]; mean-pool to [B,d]
            query = kv = torch.stack([z_L, z_R], dim=1)
            out, _ = self.attn(query, kv, kv)
            return self.norm(out.mean(1))
        out, _ = self.attn(query, kv, kv)
        return self.norm(out.squeeze(1))


class _Reshape(nn.Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], *self.shape)


def _make_decoder(d: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d, 256 * 7 * 7),
        _Reshape(256, 7, 7),
        nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14×14
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28×28
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 56×56
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 112×112
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, 4, 2, 1),     # 224×224
        nn.Tanh(),
    )


class StreamReadoutDecoder(nn.Module):
    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        self.net = _make_decoder(cfg.d_model)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class UnifiedPerceptDecoder(nn.Module):
    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        self.net = _make_decoder(cfg.d_model)

    def forward(self, z_F: Tensor) -> Tensor:
        return self.net(z_F)


class IdentityHead(nn.Module):
    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        d, n = cfg.d_model, cfg.data.num_identities
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n))

    def forward(self, z_F: Tensor) -> Tensor:
        return self.mlp(z_F)


@dataclass
class HPMOutput:
    identity_logits: Tensor
    z_L: Tensor
    z_R: Tensor
    z_F: Tensor
    local_percept: Tensor | None = None
    global_percept: Tensor | None = None
    unified_percept: Tensor | None = None


class HPMModel(nn.Module):
    """Full Hemispheric Perception Model."""

    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        self.local_enc = LocalCNNEncoder(cfg)
        self.global_enc = GlobalViTEncoder(cfg)
        self.callosum = CorpusCallosum(cfg)
        self.fusion = RightDominantFusion(cfg)
        self.build_decoders = cfg.build_decoders
        if self.build_decoders:
            self.l_decoder = StreamReadoutDecoder(cfg)
            self.r_decoder = StreamReadoutDecoder(cfg)
            self.unified_decoder = UnifiedPerceptDecoder(cfg)
        self.identity_head = IdentityHead(cfg)

    def forward(self, x_hi: Tensor, x_lo: Tensor) -> HPMOutput:
        """x_hi = high-pass (L stream), x_lo = low-pass (R stream)."""
        f_L, z_L = self.local_enc(x_hi)                         # f_L [B,C,h,w]
        T_R, z_R = self.global_enc(x_lo)                        # T_R [B,N+1,d]

        # Project CNN spatial map → token sequence for callosum cross-attention
        z_L_seq = self.local_enc.proj(f_L.flatten(2).transpose(1, 2))  # [B, h*w, d]

        z_L_prime, z_R_prime = self.callosum(z_L_seq, T_R)      # [B, N, d] each
        z_L_pooled = z_L_prime.mean(1)                           # [B, d]
        z_R_pooled = z_R_prime[:, 0]                             # CLS  [B, d]
        z_F = self.fusion(z_L_pooled, z_R_pooled)               # [B, d]

        if self.build_decoders:
            local_percept = self.l_decoder(z_L_pooled)
            global_percept = self.r_decoder(z_R_pooled)
            unified_percept = self.unified_decoder(z_F)
        else:
            local_percept = global_percept = unified_percept = None

        return HPMOutput(
            identity_logits=self.identity_head(z_F),
            z_L=z_L_prime,
            z_R=z_R_prime,
            z_F=z_F,
            local_percept=local_percept,
            global_percept=global_percept,
            unified_percept=unified_percept,
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SingleCNNModel(nn.Module):
    """CNN-only baseline — LocalCNNEncoder + IdentityHead.

    Uses the high-pass input (x_hi) only.  Trains the same identity objective
    as the full HPM so results are directly comparable.
    """

    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        self.encoder = LocalCNNEncoder(cfg)
        self.head    = IdentityHead(cfg)

    def forward(self, x_hi: Tensor, x_lo: Tensor) -> HPMOutput:
        del x_lo  # CNN processes high-pass only
        _, z = self.encoder(x_hi)
        return HPMOutput(identity_logits=self.head(z), z_L=z, z_R=z, z_F=z)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SingleViTModel(nn.Module):
    """ViT-only baseline — GlobalViTEncoder + IdentityHead.

    Uses the low-pass input (x_lo) only.  Trains the same identity objective
    as the full HPM so results are directly comparable.
    """

    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        self.encoder = GlobalViTEncoder(cfg)
        self.head    = IdentityHead(cfg)

    def forward(self, x_hi: Tensor, x_lo: Tensor) -> HPMOutput:
        del x_hi  # ViT processes low-pass only
        _, z = self.encoder(x_lo)
        return HPMOutput(identity_logits=self.head(z), z_L=z, z_R=z, z_F=z)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: HPMConfig) -> nn.Module:
    """Instantiate the right model for the requested mode."""
    if cfg.mode == "cnn":
        return SingleCNNModel(cfg)
    if cfg.mode == "vit":
        return SingleViTModel(cfg)
    return HPMModel(cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════════════

class IdentityLoss(nn.Module):
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(logits, labels)


class ReconstructionLoss(nn.Module):
    """L1 + perceptual (LPIPS VGG) reconstruction loss."""

    def __init__(self) -> None:
        super().__init__()
        if _HAS_LPIPS:
            self._lpips = _lpips_lib.LPIPS(net="vgg")
            for p in self._lpips.parameters():
                p.requires_grad_(False)
        else:
            self._lpips = None

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        l1 = F.l1_loss(pred, target)
        if self._lpips is not None:
            perc = self._lpips(pred, target).mean()
            return l1 + 0.1 * perc
        return l1


@dataclass
class LossComponents:
    identity: Tensor
    recon_unified: Tensor
    recon_local: Tensor
    recon_global: Tensor
    total: Tensor


class LossWeighter(nn.Module):
    def __init__(self, cfg: HPMConfig) -> None:
        super().__init__()
        w = cfg.train.loss_weights
        self.lambda_id  = w.lambda_id
        self.lambda_uni = w.lambda_uni
        self.lambda_aux = w.lambda_aux
        self.id_loss    = IdentityLoss()
        self.recon_loss = ReconstructionLoss()

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        unified_pred: Tensor | None,
        local_pred: Tensor | None,
        global_pred: Tensor | None,
        target: Tensor,
    ) -> LossComponents:
        id_ = self.id_loss(logits, labels)
        zero = torch.zeros_like(id_)
        uni  = self.recon_loss(unified_pred, target) if (self.lambda_uni > 0 and unified_pred is not None) else zero
        loc  = self.recon_loss(local_pred,   target) if (self.lambda_aux > 0 and local_pred   is not None) else zero
        glob = self.recon_loss(global_pred,  target) if (self.lambda_aux > 0 and global_pred  is not None) else zero
        total = self.lambda_id * id_ + self.lambda_uni * uni + self.lambda_aux * (loc + glob)
        return LossComponents(identity=id_, recon_unified=uni, recon_local=loc, recon_global=glob, total=total)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

class FrequencySplit:
    """Splits a normalised face tensor into high-pass and low-pass views."""

    def __init__(self, cfg: HPMConfig) -> None:
        fs = cfg.data.freq_split
        self.enabled  = fs.enabled
        self.sigma_hi = fs.sigma_hi
        self.sigma_lo = fs.sigma_lo

    def _blur(self, img: Tensor, sigma: float) -> Tensor:
        ks = 2 * int(4 * sigma + 0.5) + 1
        return TF.gaussian_blur(img, kernel_size=[ks, ks], sigma=sigma)

    def __call__(self, img: Tensor) -> tuple[Tensor, Tensor]:
        if not self.enabled:
            return img, img
        low  = self._blur(img, self.sigma_lo)
        high = img - self._blur(img, self.sigma_hi)
        return high, low


class FaceDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], cfg: HPMConfig) -> None:
        self.samples    = samples
        self.image_size = cfg.data.image_size
        self.freq_split = FrequencySplit(cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        path, label = self.samples[idx]
        img   = Image.open(path).convert("RGB")
        img_t = TF.to_tensor(img)
        img_t = TF.resize(img_t, [self.image_size, self.image_size], antialias=True)
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        x_hi, x_lo = self.freq_split(img_t)
        return x_hi, x_lo, label


# ─── split builders ────────────────────────────────────────────────────────────

def _make_vggface2_splits(
    root: Path, cfg: HPMConfig, seed: int
) -> tuple[list, list, list]:
    """VGGFace2 layout: root/{n000001}/*.jpg  (identity per directory)."""
    identity_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    rng = random.Random(seed)
    shuffled = list(identity_dirs)
    rng.shuffle(shuffled)

    n      = len(shuffled)
    n_val  = max(1, int(n * cfg.data.val_fraction))
    n_test = max(1, int(n * cfg.data.test_fraction))
    id_to_int = {d.name: i for i, d in enumerate(identity_dirs)}

    def _collect(dirs: list[Path]) -> list[tuple[Path, int]]:
        out = []
        for d in dirs:
            lbl = id_to_int[d.name]
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    out.append((p, lbl))
        return out

    return (
        _collect(shuffled[: n - n_val - n_test]),
        _collect(shuffled[n - n_val - n_test : n - n_test]),
        _collect(shuffled[n - n_test :]),
    )


def _make_celeba_splits(
    root: Path, cfg: HPMConfig, seed: int
) -> tuple[list, list, list]:
    """CelebA layout: root/img_align_celeba/*.jpg + identity_CelebA.txt."""
    # Locate identity file
    identity_file = root / "identity_CelebA.txt"
    if not identity_file.exists():
        matches = list(root.rglob("identity_CelebA.txt"))
        if not matches:
            raise FileNotFoundError(f"identity_CelebA.txt not found under {root}")
        identity_file = matches[0]

    # Locate image directory (handles Kaggle double-nesting)
    image_dir = root / "img_align_celeba"
    if not image_dir.is_dir():
        matches = [p for p in root.rglob("img_align_celeba") if p.is_dir()]
        if not matches:
            raise FileNotFoundError(f"img_align_celeba/ not found under {root}")
        image_dir = matches[0]
    nested = image_dir / "img_align_celeba"
    if nested.is_dir():
        image_dir = nested

    id_to_imgs: dict[int, list[str]] = defaultdict(list)
    with open(identity_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                id_to_imgs[int(parts[1])].append(parts[0])

    identities = sorted(id_to_imgs)
    id_to_int  = {ident: i for i, ident in enumerate(identities)}   # contiguous over ALL ids

    # Closed-set classification split: partition each identity's images ~train/val/test
    # so every identity appears in train (labels 0..N-1 stay contiguous and match the
    # IdentityHead size) and val holds out images of *seen* identities -> top-1/top-5 is
    # well defined. Identity-disjoint splits would crash CrossEntropy (label >= n_classes)
    # and make accuracy undefined; that layout belongs to the contrastive trainer instead.
    rng = random.Random(seed)
    vf, tf = cfg.data.val_fraction, cfg.data.test_fraction
    train: list[tuple[Path, int]] = []
    val:   list[tuple[Path, int]] = []
    test:  list[tuple[Path, int]] = []
    for ident in identities:
        lbl  = id_to_int[ident]
        imgs = sorted(id_to_imgs[ident])
        rng.shuffle(imgs)
        n = len(imgs)
        if n == 1:
            tr, va, te = imgs, [], []
        elif n == 2:
            tr, va, te = imgs[:1], imgs[1:], []
        else:
            n_val  = max(1, int(n * vf))
            n_test = max(1, int(n * tf))
            n_tr   = n - n_val - n_test            # >=1 for n>=3 at default 0.1/0.1
            tr, va, te = imgs[:n_tr], imgs[n_tr : n_tr + n_val], imgs[n_tr + n_val :]
        train.extend((image_dir / f, lbl) for f in tr)
        val.extend((image_dir / f, lbl) for f in va)
        test.extend((image_dir / f, lbl) for f in te)

    return train, val, test


def make_splits(root: Path, cfg: HPMConfig, seed: int) -> tuple[list, list, list]:
    if cfg.data.name.lower() == "celeba":
        return _make_celeba_splits(root, cfg, seed)
    return _make_vggface2_splits(root, cfg, seed)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINER  (ultralytics-style)
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """Organises an HPM training run in the Ultralytics style.

    Output layout:
        runs/hpm/{name}/
            weights/
                best.pt   — best val/top1 so far
                last.pt   — end of most recent epoch
            results.csv   — per-epoch metrics
            args.json     — full config snapshot
    """

    def __init__(self, cfg: HPMConfig, name: str, resume: str | None = None) -> None:
        self.cfg         = cfg
        self.name        = name
        self.resume_path = resume
        self.save_dir    = Path("runs") / "hpm" / name
        self.wts_dir     = self.save_dir / "weights"
        self.wts_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path    = self.save_dir / "results.csv"
        self.best_fitness   = 0.0
        self.start_epoch    = 0
        self.no_improve     = 0
        # these are set in train()
        self.model: HPMModel
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler._LRScheduler
        self.weighter: LossWeighter
        self.device: torch.device
        self.current_epoch = 0

    # ── internal helpers ───────────────────────────────────────────────────────

    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _loaders(self) -> tuple[DataLoader, DataLoader, int]:
        root  = Path(self.cfg.data.root)
        seed  = self.cfg.train.seed
        train_s, val_s, _ = make_splits(root, self.cfg, seed)

        num_ids = len({lbl for _, lbl in train_s})
        self.cfg.data.num_identities = num_ids   # IdentityHead reads this

        w = min(self.cfg.train.workers, os.cpu_count() or 1)
        train_loader = DataLoader(
            FaceDataset(train_s, self.cfg),
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=w,
            pin_memory=True,
            persistent_workers=(w > 0),
        )
        val_loader = DataLoader(
            FaceDataset(val_s, self.cfg),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=w,
            pin_memory=True,
            persistent_workers=(w > 0),
        )
        return train_loader, val_loader, num_ids

    def _save(self, tag: str) -> None:
        torch.save(
            {
                "state_dict":   self.model.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "scheduler":    self.scheduler.state_dict(),
                "epoch":        self.current_epoch,
                "best_fitness": self.best_fitness,
                "no_improve":   self.no_improve,
                "cfg":          asdict(self.cfg),
            },
            self.wts_dir / f"{tag}.pt",
        )

    def _log_csv(self, row: dict) -> None:
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _resume(self) -> None:
        print(f"Resuming from {self.resume_path}")
        ckpt = torch.load(self.resume_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if self.cfg.mode == "hpm":
            self.model.callosum.set_lesion_gain(self.cfg.callosum.lesion_gain)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch  = ckpt["epoch"] + 1
        self.best_fitness = ckpt.get("best_fitness", 0.0)
        self.no_improve   = ckpt.get("no_improve", 0)   # keep early-stop counter across resume
        print(f"  Resumed at epoch {self.start_epoch},  best_fitness={self.best_fitness:.4f}")

    # ── main ───────────────────────────────────────────────────────────────────

    def train(self) -> None:
        random.seed(self.cfg.train.seed)
        os.environ["PYTHONHASHSEED"] = str(self.cfg.train.seed)
        torch.manual_seed(self.cfg.train.seed)
        torch.cuda.manual_seed_all(self.cfg.train.seed)

        self.device = self._device()

        # ── header ──────────────────────────────────────────────────────────────
        print()
        print("─" * 70)
        print(f"  HPM  ·  run: {self.name}")
        print(f"  device : {self.device}    amp : {self.cfg.train.amp}")
        print(f"  save   : {self.save_dir.resolve()}")
        print("─" * 70)

        # ── data ────────────────────────────────────────────────────────────────
        print("Building data splits …")
        train_loader, val_loader, num_ids = self._loaders()
        print(f"  identities : {num_ids}")
        print(f"  train      : {len(train_loader.dataset):,} images  ({len(train_loader)} batches)")
        print(f"  val        : {len(val_loader.dataset):,} images")

        # ── model ───────────────────────────────────────────────────────────────
        print("Building model …")
        self.model    = build_model(self.cfg).to(self.device)
        if self.cfg.mode == "hpm":
            self.model.callosum.set_lesion_gain(self.cfg.callosum.lesion_gain)
        self.weighter = LossWeighter(self.cfg).to(self.device)

        total_params = self.model.n_params()
        print(f"  mode       : {self.cfg.mode.upper()}")
        print(f"  parameters : {total_params:,}  ({total_params / 1e6:.1f} M)")
        if self.cfg.mode == "hpm":
            print(f"  decoders   : {'on' if self.cfg.build_decoders else 'off (M1 identity-only)'}")
            print(f"  lesion_gain: {self.cfg.callosum.lesion_gain}")
        print(f"  freq_split : {'on' if self.cfg.data.freq_split.enabled else 'off'}"
              f"  σ_hi={self.cfg.data.freq_split.sigma_hi}  σ_lo={self.cfg.data.freq_split.sigma_lo}")

        # ── optimiser & scheduler ───────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.train.max_epochs
        )
        use_amp = self.cfg.train.amp and self.device.type == "cuda"
        scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ── resume ──────────────────────────────────────────────────────────────
        if self.resume_path:
            self._resume()

        # ── persist args ────────────────────────────────────────────────────────
        (self.save_dir / "args.json").write_text(json.dumps(asdict(self.cfg), indent=2))

        # ── epoch loop ──────────────────────────────────────────────────────────
        COL = f"{'Epoch':>6}  {'GPU-mem':>8}  {'loss':>9}  {'id_loss':>8}  {'top1':>8}  {'top5':>8}  {'f1':>8}  {'lr':>9}"
        print()
        print(COL)
        print("─" * len(COL))

        for epoch in range(self.start_epoch, self.cfg.train.max_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            # ── train one epoch ─────────────────────────────────────────────────
            self.model.train()
            sum_loss = sum_id = n_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"  {epoch:3d}/{self.cfg.train.max_epochs - 1}",
                dynamic_ncols=True,   # follow the real terminal width → no bar-per-batch wrap
                mininterval=0.5,      # throttle redraws
                leave=False,
                unit="batch",
            )
            for x_hi, x_lo, labels in pbar:
                x_hi   = x_hi.to(self.device, non_blocking=True)
                x_lo   = x_lo.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    out    = self.model(x_hi, x_lo)
                    target = x_hi * 2.0 - 1.0
                    comp   = self.weighter(
                        out.identity_logits, labels,
                        out.unified_percept, out.local_percept, out.global_percept,
                        target,
                    )

                self.optimizer.zero_grad()
                scaler.scale(comp.total).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()

                sum_loss  += comp.total.item()
                sum_id    += comp.identity.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{comp.total.item():.4f}")

            self.scheduler.step()

            avg_loss = sum_loss / max(n_batches, 1)
            avg_id   = sum_id   / max(n_batches, 1)

            # ── validate ────────────────────────────────────────────────────────
            metrics = self._validate(val_loader)
            top1 = metrics["top1"]
            top5 = metrics["top5"]

            # ── console row ─────────────────────────────────────────────────────
            mem = ""
            if self.device.type == "cuda":
                mem = f"{torch.cuda.memory_reserved() / 1e9:.2f}G"
            lr  = self.scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            print(
                f"{epoch:6d}  {mem:>8}  {avg_loss:9.4f}  {avg_id:8.4f}  "
                f"{top1:8.4f}  {top5:8.4f}  {metrics['f1_macro']:8.4f}  {lr:9.2e}"
            )

            # ── CSV ─────────────────────────────────────────────────────────────
            self._log_csv({
                "epoch":                epoch,
                "train/loss":           round(avg_loss, 6),
                "train/id":             round(avg_id,   6),
                "val/top1":             round(top1,      6),
                "val/top5":             round(top5,      6),
                "val/f1_macro":         round(metrics["f1_macro"],          6),
                "val/f1_weighted":      round(metrics["f1_weighted"],       6),
                "val/precision_macro":  round(metrics["precision_macro"],   6),
                "val/recall_macro":     round(metrics["recall_macro"],      6),
                "val/balanced_acc":     round(metrics["balanced_accuracy"], 6),
                "lr":                   round(lr,        8),
                "time_s":               round(elapsed,   1),
            })

            # ── checkpoint ──────────────────────────────────────────────────────
            self._save("last")
            if top1 > self.best_fitness:
                self.best_fitness = top1
                self._save("best")
                print(f"  ↑ best  val/top1={top1:.4f}  → {self.wts_dir}/best.pt")
                self.no_improve = 0
            else:
                self.no_improve += 1

            # ── early stopping ──────────────────────────────────────────────────
            if self.cfg.train.patience > 0 and self.no_improve >= self.cfg.train.patience:
                print(f"\nEarly stop at epoch {epoch}  (no gain for {self.cfg.train.patience} epochs)")
                break

        # ── done ────────────────────────────────────────────────────────────────
        print()
        print("─" * 70)
        print(f"  Training complete  ·  results → {self.save_dir.resolve()}")
        print(f"  best val/top1 : {self.best_fitness:.4f}")
        print(f"  best.pt       : {self.wts_dir / 'best.pt'}")
        print(f"  last.pt       : {self.wts_dir / 'last.pt'}")
        print(f"  results.csv   : {self.csv_path}")
        print("─" * 70)

    # ── validation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Return top1/top5 plus macro/weighted F1, precision, recall, balanced acc.

        Accumulates top-1 predictions and labels across the whole val set, then defers
        to sklearn for the class-averaged metrics.  zero_division=0 because a held-out
        val split rarely contains every identity (macro averages over labels present).
        """
        self.model.eval()
        correct1 = correct5 = total = 0
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        for x_hi, x_lo, labels in loader:
            x_hi   = x_hi.to(self.device, non_blocking=True)
            x_lo   = x_lo.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(x_hi, x_lo).identity_logits
            k      = min(5, logits.size(1))
            _, pred_k = logits.topk(k, dim=1, largest=True, sorted=True)
            hits   = pred_k.t().eq(labels.view(1, -1).expand_as(pred_k.t()))
            correct1 += hits[:1].reshape(-1).float().sum().item()
            correct5 += hits[:k].reshape(-1).float().sum().item()
            total    += labels.size(0)
            all_pred.append(pred_k[:, 0].cpu().numpy())
            all_true.append(labels.cpu().numpy())

        denom = max(total, 1)
        top1  = correct1 / denom
        top5  = correct5 / denom

        if all_true:
            y_pred = np.concatenate(all_pred)
            y_true = np.concatenate(all_true)
            # A held-out val split legitimately predicts identities it doesn't contain;
            # silence sklearn's per-call "classes not in y_true" noise to keep logs clean.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
                f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
                recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        else:
            f1_macro = f1_weighted = precision_macro = recall_macro = balanced_accuracy = 0.0

        return {
            "top1":              top1,
            "top5":              top5,
            "f1_macro":          float(f1_macro),
            "f1_weighted":       float(f1_weighted),
            "precision_macro":   float(precision_macro),
            "recall_macro":      float(recall_macro),
            "balanced_accuracy": float(balanced_accuracy),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HPM standalone training (ultralytics-style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("paths")
    g.add_argument("--data",    default="data/vggface2", help="Dataset root directory")
    g.add_argument("--dataset", default="vggface2", choices=["vggface2", "celeba"])
    g.add_argument("--name",    default="exp",   help="Run name (saved under runs/hpm/<name>/)")
    g.add_argument("--resume",  default=None,    help="Path to last.pt to resume training")
    g.add_argument("--mode",    default="hpm",   choices=["hpm", "cnn", "vit", "all"],
                   help="hpm=full dual-stream | cnn=CNN only | vit=ViT only | all=cnn→vit→hpm")
    g.add_argument("--smoke-test", action="store_true",
                   help="Run 2-batch sanity check for all 3 modes then exit")
    g.add_argument("--report-only", action="store_true",
                   help="Rebuild <name>_report.xlsx from existing results.csv files, then exit")

    # ── training ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("training")
    g.add_argument("--epochs",   type=int,   default=50)
    g.add_argument("--batch",    type=int,   default=32)
    g.add_argument("--lr",       type=float, default=1e-4)
    g.add_argument("--wd",       type=float, default=0.05,  help="AdamW weight decay")
    g.add_argument("--seed",     type=int,   default=42,    help="Single-run seed (see --seeds for --mode all)")
    g.add_argument("--seeds",    default=None,
                   help="Comma-separated seeds for --mode all, e.g. 42,43,44 "
                        "(enables mean±SD across seeds in the Excel report; default: just --seed)")
    g.add_argument("--patience", type=int,   default=3,     help="Early-stopping patience on val/top1 (0=off)")
    g.add_argument("--workers",  type=int,   default=4,     help="DataLoader workers")
    g.add_argument("--amp",      action=argparse.BooleanOptionalAction, default=True,
                   help="Automatic mixed precision on CUDA (default on; use --no-amp to disable)")

    # ── loss weights ───────────────────────────────────────────────────────────
    g = p.add_argument_group("loss weights")
    g.add_argument("--lambda-id",  type=float, default=1.0)
    g.add_argument("--lambda-uni", type=float, default=0.0, help=">0 enables unified-percept decoder")
    g.add_argument("--lambda-aux", type=float, default=0.0, help=">0 enables per-stream decoders")

    # ── model / data ───────────────────────────────────────────────────────────
    g = p.add_argument_group("model / data")
    g.add_argument("--d-model",      type=int,   default=256)
    g.add_argument("--sigma-hi",     type=float, default=1.0,  help="High-pass Gaussian σ")
    g.add_argument("--sigma-lo",     type=float, default=3.0,  help="Low-pass Gaussian σ")
    g.add_argument("--no-freq-split",action="store_true",      help="Disable frequency split (baseline)")
    g.add_argument("--lesion-gain",  type=float, default=1.0,  help="Callosum gain (0=split-brain)")
    g.add_argument("--fusion-query", default="R", choices=["R", "L", "symmetric"])
    g.add_argument("--fusion-ablate",action="store_true",      help="Replace fusion attention with concat")
    g.add_argument("--val-fraction", type=float, default=0.1)
    g.add_argument("--test-fraction",type=float, default=0.1)
    g.add_argument("--image-size",   type=int,   default=224)
    g.add_argument("--download",     action="store_true",
                   help="Auto-download CelebA before training (--dataset celeba only)")

    return p.parse_args()


def _build_cfg(a: argparse.Namespace) -> HPMConfig:
    build_decoders = (a.mode == "hpm") and (a.lambda_uni > 0 or a.lambda_aux > 0)
    if build_decoders and not _HAS_LPIPS:
        print("Warning: lpips not installed — reconstruction loss will use L1 only.")
    return HPMConfig(
        mode           = a.mode,
        d_model        = a.d_model,
        build_decoders = build_decoders,
        local          = LocalConfig(),
        global_        = GlobalConfig(),
        callosum       = CallosumConfig(lesion_gain=a.lesion_gain),
        fusion         = FusionConfig(query=a.fusion_query, ablate=a.fusion_ablate),
        data = DataConfig(
            name          = a.dataset,
            root          = a.data,
            image_size    = a.image_size,
            val_fraction  = a.val_fraction,
            test_fraction = a.test_fraction,
            freq_split    = FreqSplitConfig(
                sigma_hi = a.sigma_hi,
                sigma_lo = a.sigma_lo,
                enabled  = not a.no_freq_split,
            ),
        ),
        train = TrainConfig(
            seed         = a.seed,
            max_epochs   = a.epochs,
            batch_size   = a.batch,
            lr           = a.lr,
            weight_decay = a.wd,
            patience     = a.patience,
            amp          = a.amp,
            workers      = a.workers,
            loss_weights = LossWeightsConfig(
                lambda_id  = a.lambda_id,
                lambda_uni = a.lambda_uni,
                lambda_aux = a.lambda_aux,
            ),
        ),
    )


_TRAIN_MODES = ["cnn", "vit", "hpm"]


# ══════════════════════════════════════════════════════════════════════════════
#  OVERNIGHT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class _TeeLogger:
    """Context manager that duplicates sys.stdout to both console and a log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def __enter__(self) -> _TeeLogger:
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_) -> None:
        sys.stdout = self._stdout
        self._file.close()


def _check_mode_state(
    wts_dir: Path, max_epochs: int
) -> tuple[str, Path | None]:
    """Inspect a mode's weights directory and return its run state.

    Returns one of:
      ("complete", None)        — last.pt epoch == max_epochs-1, skip this mode
      ("resume",  last_pt)      — partial run exists, resume from last.pt
      ("fresh",   None)         — no checkpoint found, start from scratch
    """
    last = wts_dir / "last.pt"
    if not last.exists():
        return "fresh", None
    try:
        ckpt = torch.load(last, map_location="cpu", weights_only=True)
        epoch = int(ckpt.get("epoch", -1))
    except Exception:
        return "fresh", None
    if epoch >= max_epochs - 1:
        return "complete", None
    return "resume", last


def _download_celeba(data_path: Path) -> None:
    """Download CelebA (images + identity file) via torchvision (~1.4 GB).

    torchvision downloads into {parent}/celeba/ so pass --data data/celeba
    and we derive the torchvision root as the parent directory.
    """
    try:
        from torchvision.datasets import CelebA as _TvCelebA
    except ImportError:
        sys.exit("torchvision is required for download: pip install torchvision")

    tv_root = str(data_path.parent)
    print(f"\nDownloading CelebA to {data_path.resolve()}  (~1.4 GB) …")
    print("Pulling from Google Drive via torchvision — this may take a few minutes.")
    print("If it stalls or fails (Drive quota), download manually and re-run without --download.\n")
    try:
        _TvCelebA(root=tv_root, split="all", target_type="identity", download=True)
        print(f"\nDownload complete.  Data is at {data_path.resolve()}")
    except Exception as exc:
        sys.exit(
            f"\nDownload failed: {exc}\n"
            "Download manually from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n"
            "and place img_align_celeba/ and identity_CelebA.txt inside your --data folder."
        )


def _smoke_test_all(cfg: HPMConfig, n_batches: int = 2) -> bool:
    """Run n_batches of forward+backward for every mode.  Returns True if all pass.

    Uses real data so the full pipeline (disk I/O → transforms → model → loss)
    is exercised.  Errors here will definitely crash an overnight run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data once — shared across all mode checks
    root    = Path(cfg.data.root)
    train_s, _, _ = make_splits(root, cfg, cfg.train.seed)
    cfg.data.num_identities = len({lbl for _, lbl in train_s})
    subset  = train_s[: cfg.train.batch_size * n_batches]
    loader  = DataLoader(
        FaceDataset(subset, cfg),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,   # keep it simple for the smoke test
    )

    print()
    print(f"  {'Mode':<5}  {'Status':<10}  {'Time':>6}  Details")
    print(f"  {'─'*5}  {'─'*10}  {'─'*6}  {'─'*50}")

    all_pass = True
    for mode in _TRAIN_MODES:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.mode = mode
        run_cfg.build_decoders = False   # keep smoke test light
        t0 = time.time()
        try:
            model    = build_model(run_cfg).to(device)
            if mode == "hpm":
                model.callosum.set_lesion_gain(run_cfg.callosum.lesion_gain)
            weighter  = LossWeighter(run_cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            for i, (x_hi, x_lo, labels) in enumerate(loader):
                if i >= n_batches:
                    break
                x_hi   = x_hi.to(device)
                x_lo   = x_lo.to(device)
                labels = labels.to(device)
                out    = model(x_hi, x_lo)
                target = x_hi * 2.0 - 1.0
                comp   = weighter(
                    out.identity_logits, labels,
                    out.unified_percept, out.local_percept, out.global_percept,
                    target,
                )
                optimizer.zero_grad()
                comp.total.backward()
                optimizer.step()

            n_params = model.n_params()
            elapsed  = time.time() - t0
            print(f"  {mode.upper():<5}  PASS       {elapsed:5.1f}s  {n_params/1e6:.1f}M params, {n_batches} batches OK")

        except Exception as exc:
            elapsed = time.time() - t0
            msg     = str(exc)[:60]
            print(f"  {mode.upper():<5}  FAIL       {elapsed:5.1f}s  {msg}")
            all_pass = False

    print()
    return all_pass


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    """Seeds for a multi-seed --mode all run.  Defaults to the single --seed."""
    if not args.seeds:
        return [args.seed]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    return seeds or [args.seed]


def _build_report_safe(name: str, seeds: list[int]) -> None:
    """Build the Excel report; never let a reporting hiccup abort a finished run.

    train_hpm.py is standalone, so the package module is imported lazily (and src/ is
    added to sys.path as a fallback for non-editable installs).  Missing openpyxl /
    pandas just prints guidance instead of crashing after hours of training.
    """
    try:
        try:
            from hpm.utils.results import build_report
        except ImportError:
            src = Path(__file__).resolve().parent / "src"
            if src.is_dir() and str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from hpm.utils.results import build_report

        build_report(name, _TRAIN_MODES, seeds)
    except ImportError as exc:
        print(f"  [report] skipped — {exc}. Install extras:  pip install pandas openpyxl")
    except Exception as exc:  # noqa: BLE001 — reporting must never kill training
        print(f"  [report] failed to build Excel report: {exc}")


def main() -> None:
    # Box-drawing chars in the progress tables crash on Windows' default cp1252
    # console; force UTF-8 so every run (incl. resume/overnight) prints cleanly.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    args = _parse()
    seeds = _parse_seeds(args)

    # ── report-only ───────────────────────────────────────────────────────────────
    # Rebuild the workbook from whatever results.csv files already exist, no training.
    if args.report_only:
        _build_report_safe(args.name, seeds)
        return

    # ── dataset download ────────────────────────────────────────────────────────
    if args.download:
        if args.dataset != "celeba":
            sys.exit("--download only supports --dataset celeba (VGGFace2 requires manual registration)")
        _download_celeba(Path(args.data))

    cfg  = _build_cfg(args)

    # ── smoke test ──────────────────────────────────────────────────────────────
    run_smoke = args.smoke_test or args.mode == "all"
    if run_smoke:
        print("\nSmoke-testing all 3 modes (2 batches each) …")
        ok = _smoke_test_all(cfg)
        if not ok:
            sys.exit("Smoke test FAILED — fix the errors above before the overnight run.")
        print("All modes passed smoke test.")
        if args.smoke_test and args.mode != "all":
            print("Re-run without --smoke-test to start full training.")
            return

    # ── training ────────────────────────────────────────────────────────────────
    if args.mode == "all":
        log_path = Path("runs") / "hpm" / f"{args.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _TeeLogger(log_path):
            _device_str = "cuda" if torch.cuda.is_available() else (
                "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"\n{'═' * 70}")
            print(f"  HPM OVERNIGHT RUN  ·  {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  name    : {args.name}")
            print(f"  device  : {_device_str}    amp : {args.amp}")
            print(f"  data    : {Path(args.data).resolve()}")
            print(f"  seeds   : {seeds}")
            _model_chain = ' → '.join(m.upper() for m in _TRAIN_MODES)
            print(f"  epochs  : {args.epochs} per model    models : {_model_chain}")
            print(f"  runs    : {len(seeds) * len(_TRAIN_MODES)}"
                  f"  ({len(_TRAIN_MODES)} models × {len(seeds)} seeds)")
            print(f"  log     : {log_path.resolve()}")
            print(f"{'═' * 70}")

            results: list[tuple[str, str, str]] = []  # (run, status, detail)
            total_runs = len(seeds) * len(_TRAIN_MODES)
            idx = 0

            for seed in seeds:
                for mode in _TRAIN_MODES:
                    idx += 1
                    run_name = f"{args.name}_{mode}_seed{seed}"
                    wts_dir  = Path("runs") / "hpm" / run_name / "weights"
                    state, resume_path = _check_mode_state(wts_dir, args.epochs)

                    tag = f"  [{idx}/{total_runs}]  {mode.upper()} seed={seed}"
                    print(f"\n{'═' * 70}")
                    if state == "complete":
                        print(f"{tag}  [SKIP — already complete]")
                        print(f"{'═' * 70}")
                        results.append((run_name, "SKIP", str(wts_dir / "best.pt")))
                        continue
                    elif state == "resume":
                        ck = torch.load(resume_path, map_location="cpu", weights_only=True)
                        print(f"{tag}  [RESUME from epoch {ck.get('epoch', '?')}]")
                    else:
                        print(f"{tag}  [FRESH start]")
                    print(f"{'═' * 70}")

                    run_cfg = copy.deepcopy(cfg)
                    run_cfg.mode = mode
                    run_cfg.train.seed = seed
                    run_cfg.build_decoders = (
                        mode == "hpm" and (args.lambda_uni > 0 or args.lambda_aux > 0)
                    )

                    try:
                        resume_arg = str(resume_path) if resume_path else None
                        Trainer(run_cfg, name=run_name, resume=resume_arg).train()
                        results.append((run_name, "OK", str(wts_dir / "best.pt")))
                    except Exception as exc:
                        tb = traceback.format_exc()
                        print(f"\n  ✗ {run_name} FAILED: {exc}")
                        print(tb)
                        results.append((run_name, "FAIL", str(exc)[:80]))

            # ── final summary ────────────────────────────────────────────────────
            print(f"\n{'═' * 70}")
            print(f"  OVERNIGHT SUMMARY  ·  {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'═' * 70}")
            print(f"  {'Run':<28}  {'Status':<8}  Detail")
            print(f"  {'─'*28}  {'─'*8}  {'─'*33}")
            for run_name, status, detail in results:
                print(f"  {run_name:<28}  {status:<8}  {detail}")
            print(f"{'═' * 70}")

            # ── results workbook (mean ± SD across seeds) ─────────────────────────
            print("\nBuilding results workbook …")
            _build_report_safe(args.name, seeds)
            print()
    else:
        Trainer(cfg, name=args.name, resume=args.resume).train()


if __name__ == "__main__":
    main()
