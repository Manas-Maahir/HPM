import torch

from hpm.models.decoders import StreamReadoutDecoder, UnifiedPerceptDecoder


def test_stream_readout_output_shape(stub_cfg):
    B, D = 2, stub_cfg.d_model
    dec = StreamReadoutDecoder(stub_cfg)
    out = dec(torch.randn(B, D))
    assert out.shape == (B, 3, stub_cfg.data.image_size, stub_cfg.data.image_size)


def test_unified_decoder_output_shape(stub_cfg):
    B, D = 2, stub_cfg.d_model
    dec = UnifiedPerceptDecoder(stub_cfg)
    out = dec(torch.randn(B, D))
    assert out.shape == (B, 3, stub_cfg.data.image_size, stub_cfg.data.image_size)
