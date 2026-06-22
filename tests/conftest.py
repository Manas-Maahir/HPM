import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def stub_cfg():
    return OmegaConf.create({
        "d_model": 64,
        "local": {"backbone": "resnet50", "pretrained": False, "pretrained_path": None, "out_channels": 2048},
        "global_": {"backbone": "deit_small_patch16_224", "pretrained": False, "out_channels": 384},
        "callosum": {"heads": 4, "lesion_gain": 1.0, "depth": 1},
        "fusion": {"query": "R", "ablate": False, "heads": 4},
        "data": {
            "freq_split": {"sigma_hi": 1.0, "sigma_lo": 3.0, "enabled": True},
            "image_size": 224,
            "num_identities": 8,
        },
        "train": {
            "seed": 0,
            "loss_weights": {"lambda_id": 1.0, "lambda_uni": 0.0, "lambda_aux": 0.0},
        },
    })


@pytest.fixture
def batch():
    B, C, H, W = 2, 3, 224, 224
    return {
        "x_hi": torch.randn(B, C, H, W),
        "x_lo": torch.randn(B, C, H, W),
        "labels": torch.zeros(B, dtype=torch.long),
    }
