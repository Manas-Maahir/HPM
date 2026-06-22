# HPM Handoff — Session State

This document captures the exact state of the codebase for continuation in a new
session. Read this, `CLAUDE.md`, and `implementation.md` before touching any code.

---

## What is fully implemented

Every stub body in the skeleton has been filled. The complete list:

| File | What was implemented |
|---|---|
| `src/hpm/models/heads.py` | `IdentityHead.forward` → `self.mlp(z_F)` |
| `src/hpm/models/callosum.py` | `CorpusCallosum.forward` — bidirectional cross-attn + residual; norm applied to the attention delta so `lesion_gain=0` is exact identity |
| `src/hpm/models/fusion.py` | `RightDominantFusion.forward` — R-query attn or concat ablation |
| `src/hpm/models/streams.py` | Both `LocalCNNEncoder.forward` and `GlobalViTEncoder.forward` |
| `src/hpm/models/decoders.py` | Real transposed-conv decoder (5× ConvTranspose2d → Tanh) for both decoders |
| `src/hpm/models/hpm_model.py` | `HPMModel.forward` — full assembly including CNN→token projection |
| `src/hpm/data/transforms.py` | `FrequencySplit._gaussian_blur` via `TF.gaussian_blur` |
| `src/hpm/data/splits.py` | `make_identity_splits` for VGGFace2 identity-level stratification |
| `src/hpm/data/dataset.py` | `FaceDataset.__getitem__` — PIL → tensor → normalize → freq_split |
| `src/hpm/eval/inversion.py` | `InversionEffect.__call__` — 1-NN cosine, flip augmentation in-metric |
| `src/hpm/eval/composite.py` | `CompositeFaceEffect.__call__` — splice + roll misalignment |
| `src/hpm/eval/part_whole.py` | `PartWholeEffect.__call__` — full-face vs top-third isolation |
| `src/hpm/eval/readout_inspect.py` | `ReadoutInspector.__call__` — sharpness + L1 recon error |
| `tests/test_eval/test_inversion.py` | Real toy-model tests (no more pytest.skip) |
| `tests/test_eval/test_composite.py` | Same |
| `tests/test_eval/test_part_whole.py` | Same |
| `src/hpm/train.py` | Full Hydra training loop with `_build_model_cfg()` helper |

---

## Pre-training infrastructure fixes (completed this session)

Both blocking issues and the backbone swap are now applied. The codebase is
installable and test-runnable.

| Fix | File | Detail |
|---|---|---|
| Build backend | `pyproject.toml` line 3 | Changed to `setuptools.build_meta` |
| Antialias warning | `src/hpm/data/dataset.py` line 35 | Added `antialias=True` to `TF.resize` |
| CNN backbone | `configs/model/streams.yaml` | `convnext_tiny` → `resnet50`; `out_channels` 768 → 2048; added `pretrained_path: null` |
| CNN loader API | `src/hpm/models/streams.py` | `features_only=True` → `num_classes=0, global_pool=''`; `forward` uses `forward_features` |
| Test fixture | `tests/conftest.py` | `stub_cfg` updated: backbone → `resnet50`, `out_channels` → 2048, `pretrained_path` → None |
| Control: freq_only | `configs/experiment/freq_only.yaml` | `global_.backbone` → `resnet50` (both streams now same backbone; control was otherwise invalid) |
| Control: baseline_identical | `configs/experiment/baseline_identical.yaml` | Both streams → `resnet50` (consistent with main experiment backbone) |

---

## Next step — verify and run

```bash
# 1. Install (should succeed now)
python -m pip install -e ".[dev]"

# 2. Full test run
pytest tests/ --tb=short -q

# Expected: 22 passed, 1 skipped, 0 failed
#   tests/test_models/       → all pass
#   tests/test_losses/       → all pass
#   tests/test_eval/         → all pass
#   tests/test_data/test_transforms.py → passes
#   tests/test_data/test_dataset.py    → SKIP (needs VGGFace2 on disk)
#
# NOTE: two callosum tests previously failed and are now fixed —
#   test_set_lesion_gain_updates_buffer  (test was missing `import pytest`)
#   test_lesion_gain_zero_blocks_exchange (callosum now norms the delta, not the
#   residual sum, so lesion_gain=0 is exact identity)
```

If `test_models` fails, confirm that `stub_cfg.local.out_channels` in `tests/conftest.py`
is 2048 (it should be after this session's update).

## Milestone 1 training run (after tests pass)

```bash
# Smoke test — 1 epoch
python -m hpm.train experiment=phaseA_identity_only train.max_epochs=1

# Full run
python -m hpm.train experiment=phaseA_identity_only
```

Val accuracy should improve over random baseline (~0.01% for 9131 identities).
The identity head is the only loss at M1 (`lambda_uni=0, lambda_aux=0`).

**After every `load_state_dict`:**
```python
model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)
```
`lesion_gain` is a non-persistent buffer (registered with `persistent=False`) and
resets after load. See `callosum.py`.

---

## Architecture decisions locked in

- **CNN stream:** local/featural/high-SF → ResNet50 via timm (`num_classes=0, global_pool=''` + `forward_features`); optional ArcFace weights via `pretrained_path`
- **ViT stream:** global/holistic/low-SF → DeiT-Small via timm, `forward_features`; hard constraint, do not swap
- **Callosum:** bidirectional cross-attention, `lesion_gain` non-persistent buffer
- **Fusion:** right-dominant (R is default query), swappable via `fusion.query` config
- **Decoders:** 5× ConvTranspose2d → Tanh (real implementation, not stubs)
- **Reconstruction target:** `x_hi * 2.0 - 1.0` (remap normalized to Tanh range)
- **Probe loader:** standard DataLoader yielding (x_hi, x_lo, label); each metric does its own augmentation
- **Dataset:** VGGFace2 at `data/vggface2/{identity}/{image.jpg}`; identity split at identity level (no leakage)
- **Config merging:** `_build_model_cfg()` in `train.py` promotes `cfg.model` to root so model classes see flat keys
- **ArcFace weights (optional):** InsightFace `ms1mv3_arcface_r50` → `backbone.pth`; set `pretrained_path` in `configs/model/streams.yaml`. Leave `null` for ImageNet baseline on first run.
