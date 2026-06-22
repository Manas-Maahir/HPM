import torch
from torch.optim import AdamW

from hpm.models.hpm_model import HPMModel
from hpm.utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_restores_state(stub_cfg, batch, tmp_path):
    model = HPMModel(stub_cfg)
    opt = AdamW(model.parameters(), lr=1e-3)

    # One step so the optimizer accumulates state worth restoring.
    out = model(batch["x_hi"], batch["x_lo"])
    out.identity_logits.sum().backward()
    opt.step()

    # Set a non-default lesion_gain (a non-persistent buffer — must NOT survive save).
    model.callosum.set_lesion_gain(0.123)

    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=opt,
        scheduler=None,
        scaler=None,
        epoch=3,
        global_step=100,
        best_metric=0.7,
        cfg=stub_cfg,
    )
    # RNG was captured inside save_checkpoint; advance it now.
    x1 = torch.rand(3)

    model2 = HPMModel(stub_cfg)
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    state = load_checkpoint(path, model=model2, optimizer=opt2, cfg=stub_cfg, restore_rng=True)
    x2 = torch.rand(3)

    assert state["epoch"] == 3
    assert state["global_step"] == 100
    assert state["best_metric"] == 0.7

    # lesion_gain re-applied from cfg (1.0), not the saved 0.123 (non-persistent).
    assert float(model2.callosum.lesion_gain) == float(stub_cfg.callosum.lesion_gain)

    # Weights restored exactly.
    w1 = model.identity_head.mlp[0].weight
    w2 = model2.identity_head.mlp[0].weight
    assert torch.allclose(w1, w2)

    # RNG state restored → same draw reproduced.
    assert torch.allclose(x1, x2)
