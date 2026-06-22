from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class StreamReadoutDecoder(nn.Module):
    """Lightweight per-stream face reconstructor (diagnostic auxiliary output).

    Takes the post-callosum stream latent and upsamples to a face image.
    Purpose: (a) visualise each hemisphere's percept; (b) auxiliary loss that
    encourages both streams to encode real face content on small data.
    NOT part of the integration path — gradients flow into the encoders only.

    M1: returns zero images (stub) so shape tests pass.
    M2: replace self.net with the real transposed-conv upsampler.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.image_size: int = cfg.data.image_size
        d = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(d, 256 * 7 * 7),
            _Reshape(256, 7, 7),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 14×14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 28×28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # 56×56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),     # 112×112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),      # 224×224
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Return reconstructed face [B, 3, H, W]."""
        return self.net(z)


class UnifiedPerceptDecoder(nn.Module):
    """Decoder for the integrated unified percept from the fused latent z_F.

    The main reconstruction output; biologically the unified perceptual experience.
    Enabled at Milestone 3 (lambda_uni > 0).

    Independent weights from StreamReadoutDecoder — separate instantiation.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.image_size: int = cfg.data.image_size
        d = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(d, 256 * 7 * 7),
            _Reshape(256, 7, 7),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 14×14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 28×28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # 56×56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),     # 112×112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),      # 224×224
            nn.Tanh(),
        )

    def forward(self, z_F: Tensor) -> Tensor:
        """Return integrated face [B, 3, H, W]."""
        return self.net(z_F)


class _Reshape(nn.Module):
    """Reshape flat vector to spatial tensor; used inside Sequential decoder net."""

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], *self.shape)
