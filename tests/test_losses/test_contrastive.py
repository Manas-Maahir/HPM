import torch
import torch.nn.functional as F

from hpm.losses.contrastive import ArcFaceLoss, SupConLoss


def _normalised(x):
    return F.normalize(x, dim=1)


def test_supcon_lower_when_classes_separated():
    loss_fn = SupConLoss(temperature=0.1)
    labels = torch.tensor([0, 0, 1, 1])

    # Well-separated: same-label embeddings nearly identical, classes far apart.
    good = _normalised(torch.tensor([[1.0, 0.01], [1.0, -0.01], [-1.0, 0.01], [-1.0, -0.01]]))
    # Mixed: labels no longer align with geometry.
    bad = _normalised(torch.tensor([[1.0, 0.01], [-1.0, 0.01], [1.0, -0.01], [-1.0, -0.01]]))

    assert loss_fn(good, labels).item() < loss_fn(bad, labels).item()


def test_supcon_no_positives_returns_zero():
    loss_fn = SupConLoss()
    feats = _normalised(torch.randn(4, 8))
    labels = torch.tensor([0, 1, 2, 3])  # every anchor is a singleton
    loss = loss_fn(feats, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_supcon_is_differentiable():
    loss_fn = SupConLoss()
    feats = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = loss_fn(F.normalize(feats, dim=1), labels)
    loss.backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()


def test_arcface_shapes_and_finite():
    loss_fn = ArcFaceLoss(embed_dim=8, num_classes=5)
    feats = _normalised(torch.randn(4, 8))
    labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(feats, labels)
    assert loss.ndim == 0 and torch.isfinite(loss)
