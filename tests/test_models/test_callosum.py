import pytest
import torch

from hpm.models.callosum import CorpusCallosum


def test_output_shapes(stub_cfg):
    B, N, D = 2, 197, stub_cfg.d_model
    z_L = torch.randn(B, N, D)
    z_R = torch.randn(B, N, D)
    cal = CorpusCallosum(stub_cfg)
    z_L_p, z_R_p = cal(z_L, z_R)
    assert z_L_p.shape == (B, N, D)
    assert z_R_p.shape == (B, N, D)


def test_lesion_gain_zero_blocks_exchange(stub_cfg):
    """With lesion_gain=0, outputs should equal inputs (no cross-stream information)."""
    B, N, D = 2, 197, stub_cfg.d_model
    z_L = torch.randn(B, N, D)
    z_R = torch.randn(B, N, D)
    cal = CorpusCallosum(stub_cfg)
    cal.set_lesion_gain(0.0)
    z_L_p, z_R_p = cal(z_L, z_R)
    assert torch.allclose(z_L_p, z_L, atol=1e-6)
    assert torch.allclose(z_R_p, z_R, atol=1e-6)


def test_set_lesion_gain_updates_buffer(stub_cfg):
    cal = CorpusCallosum(stub_cfg)
    cal.set_lesion_gain(0.5)
    assert cal.lesion_gain.item() == pytest.approx(0.5)


def test_lesion_gain_not_in_state_dict(stub_cfg):
    cal = CorpusCallosum(stub_cfg)
    assert "lesion_gain" not in cal.state_dict()
