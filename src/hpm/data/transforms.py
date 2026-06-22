from __future__ import annotations

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

# Backbones are ImageNet-pretrained — normalise with their statistics.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ContrastiveAugment:
    """Mild, holism-preserving augmentation → normalised tensor (applied per view).

    Runs BEFORE the frequency split (phaseA_celeba_contrastive.md §2). Geometric
    augmentation is deliberately gentle — a narrow RandomResizedCrop scale and no
    large rotations — so the global stream still sees intact face configuration.
    All parameters come from ``cfg.data.augmentation``; nothing is hard-coded.
    """

    def __init__(self, cfg: DictConfig) -> None:
        a = cfg.data.augmentation
        self.tf = T.Compose(
            [
                T.RandomResizedCrop(
                    cfg.data.image_size,
                    scale=(a.scale_min, a.scale_max),
                    ratio=(a.get("ratio_min", 0.9), a.get("ratio_max", 1.1)),
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=a.get("hflip_p", 0.5)),
                T.ColorJitter(
                    brightness=a.get("brightness", 0.2),
                    contrast=a.get("contrast", 0.2),
                    saturation=a.get("saturation", 0.2),
                    hue=a.get("hue", 0.0),
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, img: Image.Image) -> Tensor:
        """Return an augmented, normalised tensor [3, H, W]."""
        return self.tf(img)


class FrequencySplit:
    """Splits a face image into high-pass and low-pass views.

    High-pass = img − GaussianBlur(img, σ_hi)  → edges / local features
    Low-pass  = GaussianBlur(img, σ_lo)         → global configuration

    σ_hi and σ_lo come from cfg.data.freq_split — never hard-coded.
    Biological rationale: Sergent (1982) right=low-SF/global, left=high-SF/local.
    Treat both sigmas as primary experimental variables; run a sweep.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.enabled: bool = cfg.data.freq_split.get("enabled", True)
        self.sigma_hi: float = cfg.data.freq_split.sigma_hi
        self.sigma_lo: float = cfg.data.freq_split.sigma_lo

    def _gaussian_blur(self, img: Tensor, sigma: float) -> Tensor:
        """Apply separable Gaussian blur. img: [3,H,W] or [B,3,H,W]."""
        ks = 2 * int(4 * sigma + 0.5) + 1
        return TF.gaussian_blur(img, kernel_size=[ks, ks], sigma=sigma)

    def __call__(self, img: Tensor) -> tuple[Tensor, Tensor]:
        """Return (high_pass, low_pass) each [3,H,W].

        If enabled=False both outputs are the unmodified image (baseline controls).
        """
        if not self.enabled:
            return img, img
        low = self._gaussian_blur(img, self.sigma_lo)
        high = img - self._gaussian_blur(img, self.sigma_hi)
        return high, low
