import torch
from omegaconf import OmegaConf

from hpm.models.fusion import RightDominantFusion


def _cfg_with(query="R", ablate=False):
    return OmegaConf.create({"d_model": 64, "fusion": {"query": query, "ablate": ablate, "heads": 4}})


def test_output_shape_default():
    B, D = 2, 64
    fuse = RightDominantFusion(_cfg_with())
    z_F = fuse(torch.randn(B, D), torch.randn(B, D))
    assert z_F.shape == (B, D)


def test_output_shape_ablated():
    B, D = 2, 64
    fuse = RightDominantFusion(_cfg_with(ablate=True))
    z_F = fuse(torch.randn(B, D), torch.randn(B, D))
    assert z_F.shape == (B, D)


def test_query_swap_changes_output():
    B, D = 2, 64
    z_L, z_R = torch.randn(B, D), torch.randn(B, D)
    out_R = RightDominantFusion(_cfg_with(query="R"))(z_L, z_R)
    out_L = RightDominantFusion(_cfg_with(query="L"))(z_L, z_R)
    assert not torch.allclose(out_R, out_L)
