"""Unit tests for PartWholeEffect on synthetic inputs.

Values on a random model are not meaningful; only the API contract is checked.
"""
import torch

from hpm.eval.part_whole import PartWholeEffect
from hpm.models.hpm_model import HPMModel


def _toy_loader(stub_cfg, n_batches: int = 3, batch_size: int = 4):
    B, C, H, W = batch_size, 3, 224, 224
    n_ids = stub_cfg.data.num_identities
    return [
        (torch.randn(B, C, H, W), torch.randn(B, C, H, W), torch.randint(0, n_ids, (B,)))
        for _ in range(n_batches)
    ]


def test_part_whole_effect_returns_expected_keys(stub_cfg):
    model = HPMModel(stub_cfg)
    result = PartWholeEffect()(model, _toy_loader(stub_cfg))
    assert set(result.keys()) == {"part_whole_global", "part_whole_local"}
    for v in result.values():
        assert isinstance(v, float)
