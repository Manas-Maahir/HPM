"""Unit tests for CompositeFaceEffect on synthetic inputs.

Values on a random model are not meaningful; only the API contract is checked.
"""
import torch

from hpm.eval.composite import CompositeFaceEffect
from hpm.models.hpm_model import HPMModel


def _toy_loader(stub_cfg, n_batches: int = 3, batch_size: int = 4):
    B, C, H, W = batch_size, 3, 224, 224
    n_ids = stub_cfg.data.num_identities
    return [
        (torch.randn(B, C, H, W), torch.randn(B, C, H, W), torch.randint(0, n_ids, (B,)))
        for _ in range(n_batches)
    ]


def test_composite_face_effect_returns_expected_keys(stub_cfg):
    model = HPMModel(stub_cfg)
    result = CompositeFaceEffect()(model, _toy_loader(stub_cfg))
    assert set(result.keys()) == {"composite_global", "composite_local"}
    for v in result.values():
        assert isinstance(v, float)
