import torch

from hpm.models.heads import IdentityHead


def test_output_shape(stub_cfg):
    B, D = 2, stub_cfg.d_model
    head = IdentityHead(stub_cfg)
    logits = head(torch.randn(B, D))
    assert logits.shape == (B, stub_cfg.data.num_identities)
