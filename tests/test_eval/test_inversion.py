"""Unit tests for InversionEffect on synthetic inputs.

Values on a random model are not meaningful; only the API contract is checked.
"""
import torch

from hpm.eval.inversion import InversionEffect
from hpm.models.hpm_model import HPMModel


def _toy_loader(stub_cfg, n_batches: int = 3, batch_size: int = 4):
    B, C, H, W = batch_size, 3, 224, 224
    n_ids = stub_cfg.data.num_identities
    return [
        (torch.randn(B, C, H, W), torch.randn(B, C, H, W), torch.randint(0, n_ids, (B,)))
        for _ in range(n_batches)
    ]


def test_inversion_effect_returns_expected_keys(stub_cfg):
    model = HPMModel(stub_cfg)
    result = InversionEffect()(model, _toy_loader(stub_cfg))
    assert set(result.keys()) == {"delta_global", "delta_local", "differential"}
    for v in result.values():
        assert isinstance(v, float)


def test_inversion_differential_is_delta_global_minus_local(stub_cfg):
    model = HPMModel(stub_cfg)
    result = InversionEffect()(model, _toy_loader(stub_cfg))
    assert abs(result["differential"] - (result["delta_global"] - result["delta_local"])) < 1e-6
