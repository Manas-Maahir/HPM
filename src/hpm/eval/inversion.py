from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hpm.models.hpm_model import HPMModel


def _collect_embeddings(
    model: HPMModel,
    loader,
    device: torch.device,
    flip: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run loader through model; return (z_L_pooled, z_R_cls, labels) tensors."""
    all_zL, all_zR, all_labels = [], [], []
    with torch.no_grad():
        for x_hi, x_lo, labels in loader:
            x_hi = x_hi.to(device)
            x_lo = x_lo.to(device)
            if flip:
                x_hi = torch.flip(x_hi, dims=[-2, -1])
                x_lo = torch.flip(x_lo, dims=[-2, -1])
            out = model(x_hi, x_lo)
            all_zL.append(F.normalize(out.z_L.mean(1), dim=-1))   # pool local tokens
            all_zR.append(F.normalize(out.z_R[:, 0], dim=-1))     # CLS token
            all_labels.append(labels.to(device))
    return torch.cat(all_zL), torch.cat(all_zR), torch.cat(all_labels)


def _knn_accuracy(
    query_emb: Tensor,
    gallery_emb: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor,
) -> float:
    """1-NN accuracy: cosine similarity of query against gallery."""
    sim = query_emb @ gallery_emb.T          # [Q, G]
    pred_idx = sim.argmax(dim=1)
    return (gallery_labels[pred_idx] == query_labels).float().mean().item()


class InversionEffect:
    """Δ(accuracy upright − inverted) per stream.

    The reported signal is the *differential* between global and local streams,
    not the absolute drop in either alone. This guards against the distribution-
    shift confound: any upright-trained net degrades on inverted faces; the claim
    requires that the global stream's drop is significantly larger (Navon 1977).

    Uses 1-NN accuracy in each stream's embedding space (z_L and z_R) as the
    accuracy proxy — this isolates per-stream contribution without requiring
    separate classification heads.
    """

    def __call__(self, model: HPMModel, probe_loader) -> dict[str, float]:
        """Return {'delta_global': float, 'delta_local': float, 'differential': float}."""
        model.eval()
        device = next(model.parameters()).device

        # Gallery: upright embeddings
        gal_zL, gal_zR, gal_labels = _collect_embeddings(model, probe_loader, device, flip=False)

        # Query: inverted embeddings matched against the upright gallery
        q_zL, q_zR, q_labels = _collect_embeddings(model, probe_loader, device, flip=True)

        # Upright self-accuracy (gallery = queries, exclude diagonal)
        sim_up_L = gal_zL @ gal_zL.T
        sim_up_R = gal_zR @ gal_zR.T
        sim_up_L.fill_diagonal_(float("-inf"))
        sim_up_R.fill_diagonal_(float("-inf"))
        acc_up_L = (gal_labels[sim_up_L.argmax(1)] == gal_labels).float().mean().item()
        acc_up_R = (gal_labels[sim_up_R.argmax(1)] == gal_labels).float().mean().item()

        # Inverted accuracy: inverted query vs upright gallery
        acc_inv_L = _knn_accuracy(q_zL, gal_zL, q_labels, gal_labels)
        acc_inv_R = _knn_accuracy(q_zR, gal_zR, q_labels, gal_labels)

        delta_global = float(acc_up_R - acc_inv_R)
        delta_local = float(acc_up_L - acc_inv_L)

        return {
            "delta_global": delta_global,
            "delta_local": delta_local,
            "differential": float(delta_global - delta_local),
        }
