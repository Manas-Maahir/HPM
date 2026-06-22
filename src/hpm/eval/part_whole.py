from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hpm.models.hpm_model import HPMModel


def _part_accuracy(
    model: HPMModel,
    probe_loader,
    device: torch.device,
    isolate: bool,
    gallery_zL: Tensor,
    gallery_zR: Tensor,
    gallery_labels: Tensor,
) -> tuple[float, float]:
    """
    Accuracy identifying a face part's identity.
    isolate=False: whole-face context (original image)
    isolate=True:  part in isolation (top third of face, bottom zeroed)
    """
    correct_L, correct_R, total = 0, 0, 0
    with torch.no_grad():
        for x_hi, x_lo, labels in probe_loader:
            B, C, H, W = x_hi.shape
            x_hi = x_hi.to(device)
            x_lo = x_lo.to(device)
            labels = labels.to(device)

            if isolate:
                # Keep only the top third; zero out the rest
                x_hi = x_hi.clone()
                x_lo = x_lo.clone()
                x_hi[:, :, H // 3 :, :] = 0.0
                x_lo[:, :, H // 3 :, :] = 0.0

            out = model(x_hi, x_lo)
            q_zL = F.normalize(out.z_L.mean(1), dim=-1)
            q_zR = F.normalize(out.z_R[:, 0], dim=-1)

            pred_L = gallery_labels[(q_zL @ gallery_zL.T).argmax(1)]
            pred_R = gallery_labels[(q_zR @ gallery_zR.T).argmax(1)]
            correct_L += (pred_L == labels).sum().item()
            correct_R += (pred_R == labels).sum().item()
            total += B

    acc_L = correct_L / max(total, 1)
    acc_R = correct_R / max(total, 1)
    return acc_L, acc_R


class PartWholeEffect:
    """Part-recognition accuracy in whole-face context minus in isolation.

    Prediction: larger whole-face advantage in the global stream — the global
    stream benefits more from the configural context.

    context condition:   full face image
    isolation condition: top third of face only (upper features; rest zeroed)
    part_whole_effect = acc_context − acc_isolation  (higher = more holistic)
    """

    def __call__(self, model: HPMModel, probe_loader) -> dict[str, float]:
        """Return {'part_whole_global': float, 'part_whole_local': float}."""
        model.eval()
        device = next(model.parameters()).device

        # Build gallery from upright whole faces
        gal_zL, gal_zR, gal_labels = [], [], []
        with torch.no_grad():
            for x_hi, x_lo, labels in probe_loader:
                out = model(x_hi.to(device), x_lo.to(device))
                gal_zL.append(F.normalize(out.z_L.mean(1), dim=-1))
                gal_zR.append(F.normalize(out.z_R[:, 0], dim=-1))
                gal_labels.append(labels.to(device))
        gal_zL = torch.cat(gal_zL)
        gal_zR = torch.cat(gal_zR)
        gal_labels = torch.cat(gal_labels)

        acc_L_ctx, acc_R_ctx = _part_accuracy(
            model, probe_loader, device, isolate=False,
            gallery_zL=gal_zL, gallery_zR=gal_zR, gallery_labels=gal_labels,
        )
        acc_L_iso, acc_R_iso = _part_accuracy(
            model, probe_loader, device, isolate=True,
            gallery_zL=gal_zL, gallery_zR=gal_zR, gallery_labels=gal_labels,
        )

        return {
            "part_whole_global": float(acc_R_ctx - acc_R_iso),
            "part_whole_local": float(acc_L_ctx - acc_L_iso),
        }
