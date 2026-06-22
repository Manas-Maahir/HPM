import torch
from omegaconf import OmegaConf

from hpm.data.transforms import FrequencySplit


def _cfg(enabled=True, sigma_hi=1.0, sigma_lo=3.0):
    return OmegaConf.create({"data": {"freq_split": {"sigma_hi": sigma_hi, "sigma_lo": sigma_lo, "enabled": enabled}}})


def test_output_shapes():
    img = torch.rand(3, 224, 224)
    hi, lo = FrequencySplit(_cfg())(img)
    assert hi.shape == img.shape
    assert lo.shape == img.shape


def test_disabled_returns_original():
    img = torch.rand(3, 224, 224)
    hi, lo = FrequencySplit(_cfg(enabled=False))(img)
    assert torch.allclose(hi, img)
    assert torch.allclose(lo, img)


def test_high_pass_has_less_energy_than_original():
    img = torch.rand(3, 224, 224)
    hi, _ = FrequencySplit(_cfg())(img)
    assert hi.abs().mean() < img.abs().mean()


def test_low_pass_has_less_high_freq_energy():
    img = torch.rand(3, 224, 224)
    _, lo = FrequencySplit(_cfg())(img)
    assert lo.abs().mean() <= img.abs().mean() + 1e-5
