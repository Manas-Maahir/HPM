import torch

from hpm.losses.identity import IdentityLoss
from hpm.losses.weighting import LossWeighter


def test_identity_loss_shape():
    loss_fn = IdentityLoss(num_classes=8)
    logits = torch.randn(4, 8)
    labels = torch.randint(0, 8, (4,))
    loss = loss_fn(logits, labels)
    assert loss.shape == ()
    assert loss.item() > 0


def test_loss_weighter_identity_only(stub_cfg):
    weighter = LossWeighter(stub_cfg, num_classes=stub_cfg.data.num_identities)
    B, D, H, W = 2, stub_cfg.d_model, stub_cfg.data.image_size, stub_cfg.data.image_size
    components = weighter(
        logits=torch.randn(B, stub_cfg.data.num_identities),
        labels=torch.zeros(B, dtype=torch.long),
        unified_pred=torch.randn(B, 3, H, W),
        local_pred=torch.randn(B, 3, H, W),
        global_pred=torch.randn(B, 3, H, W),
        target=torch.randn(B, 3, H, W),
    )
    assert components.total.shape == ()
    assert components.recon_unified.item() == 0.0  # lambda_uni=0 at Milestone 1
    assert components.recon_local.item() == 0.0
