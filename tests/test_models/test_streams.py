import torch

from hpm.models.streams import GlobalViTEncoder, LocalCNNEncoder


def test_local_encoder_output_shapes(stub_cfg):
    enc = LocalCNNEncoder(stub_cfg)
    x = torch.randn(2, 3, 224, 224)
    f_L, z_L = enc(x)
    assert z_L.shape == (2, stub_cfg.d_model)
    assert f_L.ndim == 4  # [B, C, h, w]


def test_global_encoder_output_shapes(stub_cfg):
    enc = GlobalViTEncoder(stub_cfg)
    x = torch.randn(2, 3, 224, 224)
    T_R, z_R = enc(x)
    assert z_R.shape == (2, stub_cfg.d_model)
    assert T_R.ndim == 3  # [B, N, d_model]
