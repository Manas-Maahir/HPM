from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

from hpm.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ContrastiveAugment,
    FrequencySplit,
)


class FaceDataset(Dataset):
    """Aligned face dataset (VGGFace2 / CelebA / LFW).

    Returns (x_hi, x_lo, label) where x_hi / x_lo are the high- and low-pass
    views produced by FrequencySplit, and label is an integer identity index.
    Splits are pre-computed by hpm.data.splits to guarantee no identity leakage.
    """

    def __init__(self, samples: list[tuple[Path, int]], cfg: DictConfig) -> None:
        self.samples = samples
        self.cfg = cfg
        self.freq_split = FrequencySplit(cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (x_hi [3,H,W], x_lo [3,H,W], identity_label)."""
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_t = TF.to_tensor(img)
        img_t = TF.resize(
            img_t, [self.cfg.data.image_size, self.cfg.data.image_size], antialias=True
        )
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        x_hi, x_lo = self.freq_split(img_t)
        return x_hi, x_lo, label


class ContrastiveFaceDataset(Dataset):
    """Two-view face dataset for supervised-contrastive training.

    Returns two independently augmented views of the same image, each passed
    through FrequencySplit: ``(v1_hi, v1_lo, v2_hi, v2_lo, label)``. Positives for
    SupCon are all views sharing an identity within a PK-sampled batch.

    Use the plain ``FaceDataset`` for evaluation/verification, which needs a
    single deterministic view.
    """

    def __init__(self, samples: list[tuple[Path, int]], cfg: DictConfig) -> None:
        self.samples = samples
        self.cfg = cfg
        self.augment = ContrastiveAugment(cfg)
        self.freq_split = FrequencySplit(cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (v1_hi, v1_lo, v2_hi, v2_lo, identity_label)."""
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        v1 = self.augment(img)
        v2 = self.augment(img)
        v1_hi, v1_lo = self.freq_split(v1)
        v2_hi, v2_lo = self.freq_split(v2)
        return v1_hi, v1_lo, v2_hi, v2_lo, label
