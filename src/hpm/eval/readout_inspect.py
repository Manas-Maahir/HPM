from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hpm.models.hpm_model import HPMModel


def _gradient_magnitude(img: Tensor) -> float:
    """Mean spatial gradient magnitude across the batch; proxy for image sharpness."""
    # Finite differences in x and y directions
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (dy.abs().mean() + dx.abs().mean()).item() / 2.0


class ReadoutInspector:
    """Quantitative inspection of per-stream read-out percepts.

    Local percept should exhibit low feature error / high config error.
    Global percept should exhibit high feature error / low config error.

    Metrics:
    - feature_sharpness: average spatial gradient magnitude (high = sharp local features)
    - recon_err: L1 reconstruction error vs normalised original (remapped to Tanh range)
    """

    def __call__(self, model: HPMModel, probe_loader) -> dict[str, float]:
        """Return {'local_sharpness': float, 'global_sharpness': float,
                   'local_recon_err': float, 'global_recon_err': float}."""
        model.eval()
        device = next(model.parameters()).device

        local_sharp, global_sharp = [], []
        local_err, global_err = [], []

        with torch.no_grad():
            for x_hi, x_lo, _ in probe_loader:
                x_hi = x_hi.to(device)
                x_lo = x_lo.to(device)
                out = model(x_hi, x_lo)

                # Decoders output in [-1,1] (Tanh); remap original to [-1,1] for comparison
                target = x_hi * 2.0 - 1.0

                local_sharp.append(_gradient_magnitude(out.local_percept))
                global_sharp.append(_gradient_magnitude(out.global_percept))
                local_err.append(F.l1_loss(out.local_percept, target).item())
                global_err.append(F.l1_loss(out.global_percept, target).item())

        return {
            "local_sharpness": float(sum(local_sharp) / max(len(local_sharp), 1)),
            "global_sharpness": float(sum(global_sharp) / max(len(global_sharp), 1)),
            "local_recon_err": float(sum(local_err) / max(len(local_err), 1)),
            "global_recon_err": float(sum(global_err) / max(len(global_err), 1)),
        }
