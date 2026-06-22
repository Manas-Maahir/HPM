import torch
import torch.nn.functional as F

from hpm.eval.verification import verification_metrics


def test_separable_embeddings_near_perfect_auc():
    # 5 identities, each a distinct one-hot direction + tiny noise.
    g = torch.Generator().manual_seed(0)
    embs, labels = [], []
    for ident in range(5):
        centre = torch.zeros(5)
        centre[ident] = 1.0
        for _ in range(10):
            embs.append(centre + 0.01 * torch.randn(5, generator=g))
            labels.append(ident)
    embs = F.normalize(torch.stack(embs), dim=1)
    metrics = verification_metrics(embs, torch.tensor(labels), num_pairs=400, seed=0)
    assert metrics["roc_auc"] > 0.95
    assert metrics["accuracy"] > 0.9


def test_random_embeddings_chance_auc():
    g = torch.Generator().manual_seed(0)
    embs = F.normalize(torch.randn(100, 16, generator=g), dim=1)
    labels = torch.arange(100) % 10
    metrics = verification_metrics(embs, labels, num_pairs=400, seed=0)
    assert 0.4 < metrics["roc_auc"] < 0.6


def test_degenerate_single_identity_returns_chance():
    embs = F.normalize(torch.randn(10, 8), dim=1)
    labels = torch.zeros(10, dtype=torch.long)
    metrics = verification_metrics(embs, labels, num_pairs=50)
    assert metrics["roc_auc"] == 0.5
