# HPM Implementation Guide

Step-by-step guide for filling in the skeleton. Follow the build order exactly —
each step has a verification gate before the next one starts. Architecture spec is
in `architecture.md`; this file is about *how to implement*, in what order, and
what to check at each step.

---

## Environment setup

```bash
pip install -e ".[dev]"
# Optional: ArcFace pretrained weights for local stream (see Backbone section)
pip install insightface  # only needed if using ONNX model zoo; PyTorch .pth weights are downloaded directly
```

**Pre-flight fixes (already applied to codebase):**
- `pyproject.toml` build backend fixed: `setuptools.build_meta` (was `setuptools.backends.legacy:build`, which caused `BackendUnavailable` on pip install)
- `dataset.py` `TF.resize` updated with `antialias=True` (suppresses torchvision deprecation warning)

Verify:
```bash
python -c "import timm, hydra, lpips; print('ok')"
pytest tests/ -x  # should collect but skip checkpoint-gated tests
```

---

## Build order

```
Milestone 0: eval harness (no model required)
     ↓
Milestone 1: identity-only system → asymmetry check
     ↓  (gate: asymmetry survives controls)
Milestone 2: per-stream read-out decoders
     ↓  (gate: percepts differentiated in predicted direction)
Milestone 3: unified percept decoder + loss balance
     ↓  (gate: asymmetry not degraded by recon losses)
Milestone 4: callosum lesion sweep + fusion controls
```

---

## Milestone 0 — Eval harness

### `src/hpm/eval/inversion.py`

The metric needs upright and inverted versions of the same faces. At inference,
pass both through the model and extract per-stream accuracy (or a similarity score
from the identity head embedding). The effect size is:

```
delta_global = acc_upright_global - acc_inverted_global
delta_local  = acc_upright_local  - acc_inverted_local
differential = delta_global - delta_local   ← this is the claim
```

For the probe loader: build a small dataset where each identity has both an
upright and a 180°-rotated crop. The model is run in eval mode; no gradients.

### `src/hpm/eval/composite.py`

Construct composite faces: top half of identity A + bottom half of identity B,
in two conditions: aligned (halves positioned as a normal face) and misaligned
(halves offset horizontally). The metric is:

```
composite_effect = acc_misaligned - acc_aligned   (higher = more holistic)
```

Measure this on the global stream output and local stream output separately.
Use the identity head to score top-half identity matches.

### `src/hpm/eval/part_whole.py`

Two conditions per part:
- **Context:** present the part within a whole face
- **Isolation:** present the part cropped out of the whole face

```
part_whole_effect = acc_context - acc_isolation   (higher = more holistic)
```

### Unit-testing eval metrics on synthetic data

Each metric must be callable with a random `HPMModel` and a toy probe loader
(random tensors) and return a `dict[str, float]` with the documented keys.
The *values* don't need to be meaningful — only the API contract matters at this stage.

Update `tests/test_eval/test_*.py` to actually instantiate a random model and call
the metric before adding the `pytest.skip`.

---

## Milestone 1 — Data pipeline

### `src/hpm/data/transforms.py` — `FrequencySplit._gaussian_blur`

Implement a separable Gaussian blur using `torchvision.transforms.functional.gaussian_blur`
or a manual kernel via `F.conv2d`. The kernel size should be derived from sigma
(rule of thumb: `kernel_size = 2 * int(4 * sigma + 0.5) + 1`). Keep it differentiable
so gradients can flow through data augmentation during training if needed.

```python
def _gaussian_blur(self, img: Tensor, sigma: float) -> Tensor:
    ks = 2 * int(4 * sigma + 0.5) + 1
    return TF.gaussian_blur(img, kernel_size=[ks, ks], sigma=sigma)
```

The high-pass is `img − blur(img, σ_hi)`. Clamp the result to `[0, 1]` so the
CNN doesn't receive values outside its normalisation range.

### `src/hpm/data/splits.py`

1. Enumerate all identity directories under `root`.
2. Shuffle with `random.Random(seed)` (not `random.shuffle`) so the split is
   reproducible from the seed alone.
3. Split identities (not images) into train / val / test proportions.
4. Return `(path, identity_int)` tuples; `identity_int` is the index into the
   sorted identity list — stable across runs.

### `src/hpm/data/dataset.py`

The `__getitem__` body:
```python
img = Image.open(path).convert("RGB")
img = TF.resize(img, [self.cfg.data.image_size, self.cfg.data.image_size])
img = TF.to_tensor(img)           # [0, 1]
img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
x_hi, x_lo = self.freq_split(img)
return x_hi, x_lo, label
```

Note: `FrequencySplit` is applied *after* normalisation so sigma values are in the
same units as the network's expected input range.

---

## Milestone 1 — Model modules

### Backbone choice *(implemented)*

**Local (CNN) stream — ResNet50** (`configs/model/streams.yaml`, `src/hpm/models/streams.py`)

ResNet50 via timm, optionally loaded with InsightFace ArcFace pretrained weights.
Replaced ConvNeXt-Tiny. Rationale: face-pretrained embeddings give better starting
features; ResNet50 (~25M params) is parameter-balanced with DeiT-Small (~22M).
Do NOT use iresnet100 or buffalo_l — they are ~65M params and create capacity imbalance.

**Global (ViT) stream — DeiT-Small (unchanged, hard constraint)**

DeiT-Small stays. Do NOT swap to Swin Transformer — its shifted local windows blur
the global/local architectural distinction that is load-bearing for the hypothesis.

**`configs/model/streams.yaml` (current state):**
```yaml
local:
  backbone: resnet50
  pretrained: true
  pretrained_path: null        # null = timm ImageNet; path to InsightFace .pth = ArcFace
  out_channels: 2048           # ResNet50 layer4 output channels

global_:
  backbone: deit_small_patch16_224
  pretrained: true
  out_channels: 384

image_size: 224
```

**Downloading ArcFace weights (when ready):**
PyTorch `.pth` files are at `github.com/deepinsight/insightface/tree/master/recognition/arcface_torch`.
Download `backbone.pth` for `ms1mv3_arcface_r50` and set `pretrained_path` in config.
For the ImageNet baseline (first run), leave `pretrained_path: null`.

**Control experiment configs also updated:**
- `configs/experiment/freq_only.yaml` — `global_.backbone` changed to `resnet50` (both streams now use the same backbone, preserving control validity)
- `configs/experiment/baseline_identical.yaml` — both streams changed to `resnet50` (both streams identical, consistent with main experiment backbone)

### `src/hpm/models/streams.py` *(implemented)*

**`LocalCNNEncoder.__init__`** — uses `num_classes=0, global_pool=''` (ResNet timm API):

```python
self.backbone = timm.create_model(
    cfg.local.backbone,
    pretrained=cfg.local.pretrained,
    num_classes=0,
    global_pool='',
)
if getattr(cfg.local, 'pretrained_path', None):
    state = torch.load(cfg.local.pretrained_path, map_location='cpu')
    self.backbone.load_state_dict(state, strict=False)
self.proj = nn.Linear(cfg.local.out_channels, cfg.d_model)
self.pool = nn.AdaptiveAvgPool2d(1)
```

**`LocalCNNEncoder.forward`**

```python
def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    f_L = self.backbone.forward_features(x)  # [B, 2048, 7, 7] for resnet50 at 224×224
    z_L = self.proj(self.pool(f_L).flatten(1))  # [B, d_model]
    return f_L, z_L
```

The spatial map `f_L` is kept for the callosum token projection and read-out decoder.

**`GlobalViTEncoder.forward`** — unchanged:

```python
def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    tokens = self.backbone.forward_features(x)  # [B, N+1, embed_dim] incl. CLS
    T_R = self.proj(tokens)                     # [B, N+1, d_model]
    z_R = T_R[:, 0]                             # CLS token → pooled representation
    return T_R, z_R
```

Check the exact `timm` API for `deit_small_patch16_224` — `forward_features` returns
the full token sequence including the CLS token at index 0.

### `src/hpm/models/callosum.py`

**`CorpusCallosum.forward`**

The bidirectional exchange with residual connections:

```python
def forward(self, z_L: Tensor, z_R: Tensor) -> tuple[Tensor, Tensor]:
    # z_L, z_R: [B, N, d_model] — ensure both are sequences before calling
    delta_L, _ = self.l_from_r(query=z_L, key=z_R, value=z_R)
    delta_R, _ = self.r_from_l(query=z_R, key=z_L, value=z_L)
    z_L_prime = z_L + self.lesion_gain * self.norm_l(delta_L)
    z_R_prime = z_R + self.lesion_gain * self.norm_r(delta_R)
    return z_L_prime, z_R_prime
```

**Norm the attention delta, not the residual sum.** `lesion_gain=0` must be EXACT
identity (split-brain = no exchange, stream untouched), which the lesion sweep and
the `callosum_lesion` control both depend on. If you instead write
`norm_l(z_L + gain*delta_L)`, then at gain=0 the output is `norm_l(z_L) ≠ z_L` —
`LayerNorm` at init normalises mean/variance, it is **not** identity — and
`test_lesion_gain_zero_blocks_exchange` fails. Applying the norm inside the gated
term keeps the residual exact when gain=0 and is a standard normalized-sublayer
residual at gain=1.

**Important:** the CNN encoder outputs a spatial map `f_L [B,C,h,w]`, not a token
sequence. Before passing to the callosum, flatten the spatial dims:

```python
# in HPMModel.forward:
B, C, h, w = f_L.shape
z_L_seq = f_L.flatten(2).transpose(1, 2)  # [B, h*w, C] → project → [B, h*w, d_model]
```

Or add a `to_sequence()` helper to `LocalCNNEncoder`.

### `src/hpm/models/fusion.py`

**`RightDominantFusion.forward`**

```python
def forward(self, z_L: Tensor, z_R: Tensor) -> Tensor:
    if self.ablate:
        return self.proj(torch.cat([z_L, z_R], dim=-1))

    if self.query_side == "R":
        query, kv = z_R.unsqueeze(1), torch.stack([z_L, z_R], dim=1)
    elif self.query_side == "L":
        query, kv = z_L.unsqueeze(1), torch.stack([z_L, z_R], dim=1)
    else:  # symmetric
        query = kv = torch.stack([z_L, z_R], dim=1)

    out, _ = self.attn(query, kv, kv)
    return self.norm(out.squeeze(1))
```

### `src/hpm/models/decoders.py` (Milestone 2 — implement later)

Placeholder `forward` to unblock Milestone 1 shape tests:

```python
def forward(self, z: Tensor) -> Tensor:
    # stub: return a zero image of the right shape
    B = z.shape[0]
    H = W = self.cfg.data.image_size  # store cfg in __init__
    return torch.zeros(B, 3, H, W, device=z.device)
```

Replace the stub with real transposed-conv upsampling at Milestone 2.

### `src/hpm/models/heads.py`

```python
def forward(self, z_F: Tensor) -> Tensor:
    return self.mlp(z_F)
```

### `src/hpm/models/hpm_model.py`

```python
def forward(self, x_hi: Tensor, x_lo: Tensor) -> HPMOutput:
    f_L, z_L = self.local_enc(x_hi)
    T_R, z_R = self.global_enc(x_lo)

    # Convert CNN spatial map to token sequence for callosum
    B, C, h, w = f_L.shape
    z_L_seq = f_L.flatten(2).transpose(1, 2)  # [B, h*w, d_model] after proj in encoder

    z_L_prime, z_R_prime = self.callosum(z_L_seq, T_R)

    # Pool post-callosum sequences for fusion
    z_L_pooled = z_L_prime.mean(1)   # [B, d_model]
    z_R_pooled = z_R_prime[:, 0]     # CLS token

    z_F = self.fusion(z_L_pooled, z_R_pooled)

    return HPMOutput(
        local_percept=self.l_decoder(z_L_pooled),
        global_percept=self.r_decoder(z_R_pooled),
        unified_percept=self.unified_decoder(z_F),
        identity_logits=self.identity_head(z_F),
        z_L=z_L_prime,
        z_R=z_R_prime,
        z_F=z_F,
    )
```

---

## Milestone 1 — Training loop

### `src/hpm/train.py`

Minimal loop structure:

```python
model = HPMModel(cfg)
# If resuming:
#   model.load_state_dict(torch.load(ckpt))
#   model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)

weighter = LossWeighter(cfg, num_classes=cfg.data.num_identities)
optimizer = hydra.utils.instantiate(cfg.train.optimizer, params=model.parameters())
scheduler = hydra.utils.instantiate(cfg.train.scheduler, optimizer=optimizer)

for epoch in range(cfg.train.max_epochs):
    for x_hi, x_lo, labels in train_loader:
        out = model(x_hi, x_lo)
        loss = weighter(out.identity_logits, labels, out.unified_percept,
                        out.local_percept, out.global_percept, target=x_hi).total
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
```

Use the original image (`x_hi` or the un-split face) as the reconstruction target,
not `x_lo`. The target should be the aligned, normalised input before the frequency
split.

### Checkpoint saving

Always save: `{"state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg), "seed": cfg.train.seed}`

Do NOT save `lesion_gain` (it is non-persistent). On load:

```python
ckpt = torch.load(path)
model.load_state_dict(ckpt["state_dict"])
model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)
```

---

## Milestone 2 — Read-out decoders

### `StreamReadoutDecoder` architecture

A lightweight transposed-conv upsampler from a flat latent to `[3, H, W]`:

```
Linear(d_model, 256 * 7 * 7) → Reshape(256, 7, 7)
→ ConvTranspose2d(256, 128, 4, 2, 1)  # 14×14
→ ConvTranspose2d(128, 64,  4, 2, 1)  # 28×28
→ ConvTranspose2d(64,  32,  4, 2, 1)  # 56×56
→ ConvTranspose2d(32,  16,  4, 2, 1)  # 112×112
→ ConvTranspose2d(16,  3,   4, 2, 1)  # 224×224
→ Tanh()
```

The `UnifiedPerceptDecoder` can share the same architecture — just instantiate it
separately so the two decoders have independent weights.

### Loss integration

Set `lambda_aux > 0` in `configs/train/default.yaml` (e.g., start with `0.1`).
The reconstruction target is the normalised original image remapped to `[-1, 1]`
(to match `Tanh` output): `target = img * 2 - 1`.

LPIPS expects inputs in `[-1, 1]` and imports lazily in `ReconstructionLoss.__init__`
so it only loads when `lambda_aux > 0`.

---

## Milestone 3 — Unified percept decoder

Same architecture as the stream read-out decoder. Enable with `lambda_uni > 0`.

**Loss balance sweep:** start with `lambda_id=1.0, lambda_uni=0.1, lambda_aux=0.1`.
If identity accuracy drops significantly, reduce reconstruction weights. Log every
weight combination. The `LossComponents` dataclass makes it easy to log each term
separately in your logger of choice.

---

## Milestone 4 — Lesion sweep

Run the sweep without retraining — patch `lesion_gain` at eval time:

```python
ckpt = torch.load("checkpoints/milestone1.pt")
model.load_state_dict(ckpt["state_dict"])

for gain in [0.0, 0.25, 0.5, 0.75, 1.0]:
    model.callosum.set_lesion_gain(gain)
    results[gain] = run_all_signatures(model, probe_loader)
```

Report inversion differential, composite effect, and part-whole effect as a function
of `lesion_gain`. The prediction is monotonic degradation as gain → 0.

---

## Running controls

Each control is one config override:

```bash
# Role crossover (the critical test)
python -m hpm.train experiment=role_crossover

# Split-brain
python -m hpm.train experiment=callosum_lesion

# Parameter-matched
python -m hpm.train experiment=param_matched
```

All produce a checkpoint that the same `eval/signatures.py` evaluates identically.
Never add an `if experiment == "role_crossover"` branch anywhere in the model or
eval code.

---

## Common pitfalls

| Pitfall | Fix |
|---|---|
| ViT `forward_features` API varies by timm version | Pin `timm>=0.9`; check with `timm.list_models("deit*")` |
| CNN spatial token count ≠ ViT patch token count | Callosum cross-attention handles mismatched sequence lengths naturally |
| `lesion_gain` resets to 1.0 after every `load_state_dict` | Always call `set_lesion_gain` immediately after load |
| Frequency split applied before normalisation → sigma units wrong | Apply `FrequencySplit` after `ToTensor` + `Normalize` |
| Reconstruction target is the split image, not the original | Use the pre-split normalised image as target; `target = x_hi * 2 - 1` |
| Single seed reported as a result | Always run ≥ 5 seeds; report mean ± std + p-value |
| `role_crossover` experiment forgets to swap fusion query | Config already sets `fusion.query: L`; do not add model-side logic |
| `features_only=True` with ResNet in timm → error | Use `num_classes=0, global_pool=''` + `forward_features`; `features_only` is ConvNeXt-specific |
| ArcFace `.pth` key names don't match timm ResNet | Use `strict=False` in `load_state_dict`; verify with `model.backbone.load_state_dict(state, strict=False)` |
| Swin Transformer for global stream | Do NOT use Swin — its local windows re-introduce local processing, blurring the global/local distinction |
| `conftest.py` `stub_cfg` hard-codes `out_channels: 768` | Update to `out_channels: 2048` after switching to ResNet50 |
