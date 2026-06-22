import torch

from hpm.models.hpm_model import HPMModel, HPMOutput


def test_full_forward_shapes(stub_cfg):
    B, C, H, W = 2, 3, 224, 224
    model = HPMModel(stub_cfg)
    out: HPMOutput = model(torch.randn(B, C, H, W), torch.randn(B, C, H, W))
    assert out.identity_logits.shape == (B, stub_cfg.data.num_identities)
    assert out.local_percept.shape == (B, C, H, W)
    assert out.global_percept.shape == (B, C, H, W)
    assert out.unified_percept.shape == (B, C, H, W)
    assert out.z_F.shape == (B, stub_cfg.d_model)
