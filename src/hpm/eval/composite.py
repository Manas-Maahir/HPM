from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hpm.models.hpm_model import HPMModel


def _composite_accuracy(
    model: HPMModel,
    probe_loader,
    device: torch.device,
    misalign_shift: int,
    gallery_zL: Tensor,
    gallery_zR: Tensor,
    gallery_labels: Tensor,
) -> tuple[float, float]:
    """Accuracy identifying top-half identity in aligned or misaligned composites."""
    correct_L, correct_R, total = 0, 0, 0
    with torch.no_grad():
        batches = list(probe_loader)
        for i, (x_hi, x_lo, labels) in enumerate(batches):
            B, C, H, W = x_hi.shape
            x_hi = x_hi.to(device)
            x_lo = x_lo.to(device)
            labels = labels.to(device)

            # Bottom half donor: next batch (cyclic), same position within batch
            next_hi = batches[(i + 1) % len(batches)][0].to(device)
            next_lo = batches[(i + 1) % len(batches)][1].to(device)

            # Splice: keep top half of x, replace bottom half with donor
            comp_hi = x_hi.clone()
            comp_lo = x_lo.clone()
            donor_hi_bottom = next_hi[:, :, H // 2 :, :]
            donor_lo_bottom = next_lo[:, :, H // 2 :, :]
            if misalign_shift > 0:
                # Horizontal roll creates misaligned condition
                donor_hi_bottom = torch.roll(donor_hi_bottom, shifts=misalign_shift, dims=-1)
                donor_lo_bottom = torch.roll(donor_lo_bottom, shifts=misalign_shift, dims=-1)
            comp_hi[:, :, H // 2 :, :] = donor_hi_bottom
            comp_lo[:, :, H // 2 :, :] = donor_lo_bottom

            out = model(comp_hi, comp_lo)
            q_zL = F.normalize(out.z_L.mean(1), dim=-1)
            q_zR = F.normalize(out.z_R[:, 0], dim=-1)

            pred_L = gallery_labels[( q_zL @ gallery_zL.T).argmax(1)]
            pred_R = gallery_labels[(q_zR @ gallery_zR.T).argmax(1)]
            correct_L += (pred_L == labels).sum().item()
            correct_R += (pred_R == labels).sum().item()
            total += B

    acc_L = correct_L / max(total, 1)
    acc_R = correct_R / max(total, 1)
    return acc_L, acc_R


class CompositeFaceEffect:
    """Identity-match accuracy drop: top-half-A aligned vs. misaligned with bottom-half-B.

    Prediction: larger composite effect in the global stream / unified percept —
    holistic processing is disrupted more by part misalignment.

    aligned condition:   top of A + bottom of B at normal face position
    misaligned condition: same halves but bottom is shifted horizontally by W//4
    composite_effect = acc_misaligned − acc_aligned  (higher = more holistic)
    """

    def __call__(self, model: HPMModel, probe_loader) -> dict[str, float]:
        """Return {'composite_global': float, 'composite_local': float}."""
        model.eval()
        device = next(model.parameters()).device

        # Build upright gallery for k-NN matching
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

        # Infer spatial width from first batch
        first_batch = next(iter(probe_loader))
        W = first_batch[0].shape[-1]
        shift = W // 4

        acc_L_aligned, acc_R_aligned = _composite_accuracy(
            model, probe_loader, device, misalign_shift=0, gallery_zL=gal_zL,
            gallery_zR=gal_zR, gallery_labels=gal_labels,
        )
        acc_L_misaligned, acc_R_misaligned = _composite_accuracy(
            model, probe_loader, device, misalign_shift=shift, gallery_zL=gal_zL,
            gallery_zR=gal_zR, gallery_labels=gal_labels,
        )

        return {
            "composite_global": float(acc_R_misaligned - acc_R_aligned),
            "composite_local": float(acc_L_misaligned - acc_L_aligned),
        }
