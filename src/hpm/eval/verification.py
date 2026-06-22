from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    which: str = "z_F",
) -> tuple[Tensor, Tensor]:
    """Run ``model`` over ``loader`` and return (L2-normalised embeddings, labels).

    ``which`` selects the embedding stream — ``z_F`` (fused, the verification
    deliverable) or ``z_L``/``z_R`` for stream-level asymmetry probing. Sequence
    latents (z_L/z_R) are mean-pooled over tokens. Uses the PRE-projection-head
    representation (SupCon convention).
    """
    model.eval()
    embs: list[Tensor] = []
    labels: list[Tensor] = []
    for x_hi, x_lo, lab in loader:
        out = model(x_hi.to(device), x_lo.to(device))
        z = getattr(out, which)
        if z.dim() > 2:
            z = z.mean(dim=1)
        embs.append(F.normalize(z, dim=1).cpu())
        labels.append(lab)
    return torch.cat(embs), torch.cat(labels)


def verification_metrics(
    embeddings: Tensor,
    labels: Tensor,
    num_pairs: int = 10000,
    far_target: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Open-set verification metrics from balanced same/different identity pairs.

    Returns ``roc_auc`` (the ``best.pt`` selector), ``tar_at_far`` (@``far_target``),
    and best-threshold verification ``accuracy``. Embeddings are assumed L2-normalised
    so the pair score is a cosine similarity (a dot product).
    """
    labels_np = labels.numpy()
    emb_np = embeddings.numpy()
    rng = np.random.default_rng(seed)

    by_id: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels_np):
        by_id[int(lab)].append(i)
    ids_with_two = [i for i, v in by_id.items() if len(v) >= 2]

    if len(ids_with_two) == 0 or len(by_id) < 2:
        return {"roc_auc": 0.5, "tar_at_far": 0.0, "accuracy": 0.5}

    n_pos = num_pairs // 2
    pos_a, pos_b = [], []
    for _ in range(n_pos):
        cid = ids_with_two[rng.integers(len(ids_with_two))]
        a, b = rng.choice(by_id[cid], size=2, replace=False)
        pos_a.append(a)
        pos_b.append(b)

    all_idx = np.arange(len(labels_np))
    neg_a, neg_b = [], []
    while len(neg_a) < n_pos:
        a, b = rng.choice(all_idx, size=2, replace=False)
        if labels_np[a] != labels_np[b]:
            neg_a.append(a)
            neg_b.append(b)

    a_idx = np.array(pos_a + neg_a)
    b_idx = np.array(pos_b + neg_b)
    y = np.concatenate([np.ones(len(pos_a)), np.zeros(len(neg_a))])
    sims = (emb_np[a_idx] * emb_np[b_idx]).sum(axis=1)

    auc = float(roc_auc_score(y, sims))
    fpr, tpr, _ = roc_curve(y, sims)
    tar = float(np.interp(far_target, fpr, tpr))
    # Balanced pairs → accuracy at a threshold is (tpr + (1 - fpr)) / 2.
    accuracy = float(np.max((tpr + (1.0 - fpr)) / 2.0))
    return {"roc_auc": auc, "tar_at_far": tar, "accuracy": accuracy}


@torch.no_grad()
def evaluate_verification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    which: str = "z_F",
    num_pairs: int = 10000,
    far_target: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Convenience wrapper: extract embeddings then compute verification metrics."""
    embeddings, labels = extract_embeddings(model, loader, device, which=which)
    return verification_metrics(
        embeddings, labels, num_pairs=num_pairs, far_target=far_target, seed=seed
    )
