# architecture.md — HPM (Hemispheric Perception Model)

Technical specification of the Phase A architecture, losses, training, controls, and
data. Companion to `CLAUDE.md`. This document is the source of truth for implementation;
keep it in sync with `src/hpm/`.

---

## 1. Design summary

A monocular face image is split by spatial frequency into two biased views. A **CNN**
encodes the high-frequency view (the **local / L-FFA** stream); a pretrained **ViT**
encodes the low-frequency view (the **global / R-FFA** stream). The two latent
representations exchange information through a bidirectional **corpus-callosum**
cross-attention module with a lesion gain. Each stream feeds a **read-out decoder**
producing a diagnostic per-stream face (auxiliary supervision), and both latents feed
a **right-dominant latent fusion** module. The fused representation drives a
**unified-percept decoder** (the integrated face) and an **identity head**.

Three image outputs (local percept, global percept, unified percept) plus an identity
prediction. Integration is in latent space; per-stream images are probes, not the
fusion channel.

```
face ─┬─ high-pass ─► CNN encoder (L, local) ─┐                  ┌─► L read-out decoder ─► LOCAL percept (diag)
      │                                        ├─ callosum X-attn ┤
      └─ low-pass  ─► ViT encoder (R, global) ─┘   (lesion gain)  └─► R read-out decoder ─► GLOBAL percept (diag)
                                  │                          │
                                  └────────► latent fusion ◄─┘   (right-dominant, swappable)
                                                  │
                                     ┌────────────┴───────────┐
                                     ▼                         ▼
                          unified-percept decoder        identity head
                                     │                         │
                              UNIFIED percept            label / embedding
```

---

## 2. Inputs and frequency split

- **Phase A input:** single RGB face image, aligned & cropped, resized to `H×W`
  (default `224×224`).
- **Frequency split** (config: `data.freq_split`):
  - **Local stream input:** high-pass = `img − GaussianBlur(img, σ_hi)` (or DoG),
    emphasizing edges/features.
  - **Global stream input:** low-pass = `GaussianBlur(img, σ_lo)`, emphasizing
    configuration.
  - `σ_hi`, `σ_lo` are config parameters with a documented cycles-per-face-width
    rationale (grounded in Sergent 1982); they are NOT hard-coded. Treat the cutoff as
    a primary experimental variable and run a small sweep.
- Both streams always receive a *whole* face (the split is by frequency, not by image
  half). The half-image / visual-field split is explicitly out of scope for Phase A.

---

## 3. Modules

### 3.1 Local stream — CNN encoder (`models/streams.py`)
- Backbone: **ConvNeXt-Tiny** (default) or ResNet-50, via `timm`, **pretrained**.
- Role: featural / local mechanism. Faithful local processor (locality + hierarchical
  composition).
- Output: spatial feature map `f_L ∈ R^{C×h×w}` (pre-GAP), plus pooled token `z_L`.

### 3.2 Global stream — ViT encoder (`models/streams.py`)
- Backbone: **DeiT-Small / ViT-S/16** via `timm`, **pretrained** (mandatory — see
  constraints). Optional CNN-teacher distillation (DeiT distillation token) for extra
  small-data robustness.
- Role: holistic / global mechanism. All-to-all self-attention = second-order
  relational encoding.
- Output: patch tokens `T_R ∈ R^{N×D}` (incl. CLS), pooled token `z_R`.

> Both streams are projected to a common width `d_model` before the callosum so
> cross-attention is well-defined. Keep capacities **parameter-matched** for the
> control conditions (config `model.match_params: true`).

### 3.3 Corpus callosum — bidirectional cross-attention (`models/callosum.py`)
- Inserted at one (or more) matched depth(s); operates on token sets from each stream.
- Two cross-attention passes: `L←R` and `R←L`. Each output is scaled by a scalar
  **`lesion_gain ∈ [0,1]`** before being added back to its stream.
  - `lesion_gain = 1.0`: full interhemispheric transfer.
  - `lesion_gain = 0.0`: **split-brain** condition (no exchange).
- Biologically: exchange happens *mid-encoding*, producing access to the whole-face
  representation in each stream before specialization completes.
- Config: `model.callosum.depth`, `.heads`, `.lesion_gain`.

**Implementation — `lesion_gain` as a non-persistent buffer:**  
Register with `register_buffer("lesion_gain", torch.tensor(1.0), persistent=False)`.
This means `lesion_gain` is *not* serialised into checkpoints; the checkpoint stays
model-only and the experiment condition is a config concern, not a baked-in value.
After every `load_state_dict`, the caller **must** re-apply the config value:

```python
model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)
```

Expose `set_lesion_gain(value: float)` as a public method so eval loops can patch the
gain without re-loading the checkpoint (e.g., to run a lesion sweep on one checkpoint).  
Escape hatch: if self-contained checkpoints are ever needed, switch to `persistent=True`
and make re-application of the experiment config a hard post-`load_state_dict` rule.

### 3.4 Read-out decoders (`models/decoders.py`)
- One lightweight decoder per stream (e.g., small U-Net / transposed-conv upsampler),
  taking that stream's post-callosum representation → reconstructed face.
- Purpose: (a) **diagnostic** — visualize each hemisphere's percept; (b) **auxiliary
  loss** — deep supervision that helps both streams encode real face content on small
  data.
- **Not** part of the integration path. Gradients flow into the encoders.

### 3.5 Latent fusion — right-dominant (`models/fusion.py`)
- Cross-attention block with **query = global/R latent**, keys/values = both latents
  (R dominance), producing fused representation `z_F`.
- Implemented as a discrete, swappable module:
  - `fusion.query = R` (default, biologically right-dominant),
  - `fusion.query = L` or `symmetric` (control variants),
  - `fusion.ablate = true` (bypass → concat) for ablation.
- Config: `model.fusion.*`.

### 3.6 Unified-percept decoder + identity head (`models/decoders.py`, `models/heads.py`)
- **Unified decoder:** `z_F` (+ optional skip features) → integrated face image
  (the main reconstruction; biologically the unified percept).
- **Identity head:** MLP on pooled `z_F` → logits (cross-entropy) and/or normalized
  embedding (contrastive / ArcFace-style).
- **Wire identity head FIRST.** Add unified + read-out reconstruction only after the
  identity-only system shows the core asymmetry.

---

## 4. Losses (`losses/`)

Total loss (weights in config `train.loss_weights`):

```
L = λ_id   · L_identity
  + λ_uni  · L_recon_unified        # pixel (L1) + perceptual (LPIPS)
  + λ_aux  · (L_recon_local + L_recon_global)   # auxiliary read-out supervision
```

- Start with `λ_id = 1`, all others `0` (identity-only milestone).
- Then enable read-out aux losses, then the unified reconstruction.
- Expect to **tune the balance**: reconstruction and identity pull on `z_F`
  differently; an unweighted sum lets one dominate. Sweep weights; log them.

---

## 5. Behavioral-signature evaluation (`eval/`) — write this FIRST

Each metric takes a trained model and a probe set and returns a **per-stream effect
size with seed-wise error bars**. The reported quantity is always the **difference
between streams** (global vs. local) and vs. a featural baseline.

- **Inversion effect:** Δ(performance upright − inverted). Prediction: global stream
  Δ ≫ local stream Δ. (Guard against the distribution-shift confound: include a
  baseline and report the *differential*.)
- **Composite face effect:** identity-match accuracy drop when top-half-A is *aligned*
  vs *misaligned* with bottom-half-B. Prediction: larger composite effect in the
  global stream / unified percept.
- **Part-whole effect:** part-recognition accuracy in whole-face context minus in
  isolation. Prediction: larger whole-face advantage in the global stream.
- **Read-out inspection:** qualitative + quantitative (e.g., landmark-config error vs.
  feature-sharpness) on the per-stream percepts; local percept should show low feature
  error / high config error, global the reverse.

All metrics: unit-tested on synthetic inputs before use; run across `n_seeds` (default
≥5); reported with a statistical test (e.g., paired across seeds).

---

## 6. Control conditions (config `experiment/`)

All controls are **config variants over one code path** — never forked logic.

| Variant | Change | Tests for |
|---|---|---|
| `baseline_identical` | both streams same backbone, no bias | does generic two-stream learning alone produce asymmetry? |
| `freq_only` | frequency split, no task/role bias | is the input bias sufficient? |
| `task_only` | task/role bias, no frequency split | is the functional bias sufficient? |
| `role_crossover` | give CNN the holistic job, ViT the featural job | is asymmetry tied to mechanism (predicted) or arbitrary? |
| `callosum_lesion` | `lesion_gain = 0` | does removing transfer reproduce split-brain degradation? |
| `fusion_swap` / `fusion_ablate` | change fusion query / bypass | is right-dominant integration doing real work? |
| `param_matched` | equalize capacities | rule out parameter-count explanation |

A finding must survive `param_matched` and behave as predicted under `role_crossover`.

---

## 7. Data

- **Phase A:** a public 2D face dataset (e.g., an aligned subset of VGGFace2 / CelebA /
  LFW) for identity + reconstruction. Debug the architecture on clean data; do NOT
  debug science and a self-collected set simultaneously.
- **Phase B:** custom from-scratch **stereo** capture (left/right pairs with identity
  labels) for the depth front-end. Transfer learning matters even more here.
- Standard face alignment/crop; document the pipeline; fix splits by identity (no
  identity leakage across train/val/test).

---

## 8. Phased build order

1. **Milestone 0 — harness:** implement `eval/` signatures + tests on synthetic data.
2. **Milestone 1 — identity-only:** CNN + ViT + callosum + right-dominant fusion +
   identity head. Establish whether the global/local asymmetry emerges; run
   `freq_only`, `baseline_identical`, `param_matched`, `role_crossover` controls.
3. **Milestone 2 — read-outs:** add per-stream decoders + aux losses; inspect the two
   percepts; quantify config-vs-feature error.
4. **Milestone 3 — unified percept:** add unified decoder; tune loss balance.
5. **Milestone 4 — callosum studies:** lesion sweep; split-brain phenomenology.
6. **Phase B — depth:** stereo front-end into fusion; pose/occlusion robustness.

Do not advance a milestone until the prior one's asymmetry survives its controls.

---

## 9. Neuroscience references

- Navon, D. (1977). *Forest before the trees: The precedence of global features in
  visual perception.* Cognitive Psychology, 9, 353–383.
- Sergent, J. (1982). *The cerebral balance of power: Confrontation or cooperation?*
  J. Exp. Psychol.: Human Perception & Performance, 8, 253–272.
- Kanwisher, N., McDermott, J., & Chun, M. M. (1997). *The fusiform face area: A module
  in human extrastriate cortex specialized for face perception.* J. Neuroscience, 17,
  4302–4311.
- Right middle-fusiform holistic processing / neural composite-face effect (right-FFA
  fMRI literature).

## 10. Architecture references

- Dosovitskiy et al. (2020). *An Image is Worth 16×16 Words* (ViT). arXiv:2010.11929
- Touvron et al. (2021). *Training Data-Efficient Image Transformers & Distillation*
  (DeiT). arXiv:2012.12877
- Liu et al. (2021). *Swin Transformer.* arXiv:2103.14030 (considered; not the Phase A
  stream choice — see CLAUDE.md fidelity rationale)
- Wang et al. (2021). *Pyramid Vision Transformer (PVT).* arXiv:2102.12122 (alt. global
  backbone if reconstruction quality dominates)
- Dai et al. (2021). *CoAtNet.* arXiv:2106.04803 (explicitly excluded from individual
  streams — blends local/global)
- Liu et al. (2022). *A ConvNet for the 2020s (ConvNeXt).* arXiv:2201.03545 (local
  stream default)
