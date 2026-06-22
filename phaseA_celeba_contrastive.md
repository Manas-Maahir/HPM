# Phase A Training Plan — CelebA + Contrastive (Colab)

Scope: train the two-stream HPM (CNN local + ViT global + callosum + right-dominant
fusion) on **CelebA**, with an **identity embedding learned by supervised contrastive
loss**, on **Google Colab**. This is **Milestone 1 (identity-only)** — no decoders yet.
Companion to `architecture.md`; everything here is config-driven (`CLAUDE.md` standards).

---

## 1. Objective and success criteria

- **Engineering goal:** a stable identity embedding from the fused representation `z_F`,
  measured by open-set face verification on held-out identities.
- **Scientific goal (the real one):** does a measurable global/local asymmetry emerge,
  and does it survive `freq_only`, `param_matched`, and `role_crossover` controls?
  Verification accuracy is a *health check*, not the deliverable.
- **Done when:** verification metric is stable across ≥5 seeds AND the behavioral-
  signature harness runs on the resulting checkpoints with seed-wise error bars.

---

## 2. Data — CelebA

- **Variant:** CelebA **aligned & cropped** (`img_align_celeba`). ~202k images,
  ~10,177 identities (`identity_CelebA.txt`), 5 landmarks, 40 attributes available.
- **Splits — identity-disjoint (critical):** partition by *identity*, not by image, so
  test identities are unseen → this makes verification a true generalization test and
  prevents identity leakage. Suggested ~80/10/10 identity split. Fix the split with a
  seed and save the identity lists to disk; never regenerate on the fly.
- **Preprocessing pipeline (per image):**
  1. Use provided alignment; resize to `224×224`.
  2. Apply **frequency split** → two stream inputs:
     - local/CNN input = high-pass (`img − GaussianBlur(σ_hi)` or DoG),
     - global/ViT input = low-pass (`GaussianBlur(σ_lo)`).
     - `σ_hi`, `σ_lo` from config; run a small cutoff sweep (primary variable).
  3. Normalize per the backbones' expected mean/std.
- **Augmentation (mild — preserve holistic structure):** random resized crop with a
  *narrow* scale range, horizontal flip, light color jitter. **Avoid** aggressive crops
  or large rotations — they destroy face configuration and would undermine holistic
  learning. Augmentation is applied *before* the frequency split.
- **Landmarks/parts:** reserve CelebA 5-point landmarks (and CelebAMask-HQ part masks)
  for the local stream's part objective and the part-whole eval in later milestones.

---

## 3. Contrastive setup

- **Loss:** **Supervised Contrastive (SupCon, Khosla et al. 2020)** — uses identity
  labels so all same-identity samples in a batch are positives. More stable and
  data-efficient than triplet for ~20 images/identity. (ArcFace is the documented
  alternative if SupCon underperforms; keep it a config switch.)
- **Batch sampling — PK sampler (required):** sample **P identities × K images each**
  so positives always exist in-batch. Start P=16, K=4 → batch 64 (T4-friendly); scale
  P with bigger GPUs. Without PK sampling, contrastive signal collapses.
- **Multi-view:** two augmented views per image → two passes through the full two-stream
  model → two embeddings; positives = same identity across all views in the batch.
- **Embedding path:** `z_F` (fused) → small **MLP projection head** → **L2-normalized**
  embedding for the loss. At eval, **discard the projection head** and use the pre-head
  `z_F` embedding (SimCLR/SupCon convention). Temperature τ ≈ 0.07–0.1 (config).
- **Optional analysis embeddings:** also expose per-stream embeddings (`z_L`, `z_R`) so
  the asymmetry can be probed at the stream level, not just on the fused output. These
  are *measured*, not necessarily trained on, in Milestone 1.

---

## 4. Optimization

- **Optimizer:** AdamW, weight decay ≈ 0.05.
- **Differential learning rates:** pretrained backbones at a **low** LR (~1e-5–1e-4);
  new modules (callosum, fusion, projection head) at a **higher** LR (~1e-3).
- **Freeze schedule:** optionally freeze both backbones for the first few epochs (train
  only new modules to stabilize), then unfreeze with the low backbone LR.
- **Schedule:** linear warmup → cosine decay. Mixed precision (AMP) on.
- **Param-matching:** enforce `model.match_params` so the `param_matched` control is
  valid; log effective parameter counts per stream.

---

## 5. Colab compute plan

- **GPU:** T4 (16GB) feasible for a single run with AMP + batch 64; use **gradient
  accumulation** to reach an effective batch of 128 if contrastive benefits. Pro/Pro+
  (L4/A100) strongly recommended for the **multi-seed control matrix** (~30–50 runs).
- **Memory budget:** ViT-S (~22M) + ConvNeXt-T (~28M) + small heads fit in 16GB at 224.
  If tight: drop batch to 32 + accumulate, or freeze backbones early.
- **Checkpointing (mandatory, Colab disconnects):** save to Google Drive **every N
  steps** AND at epoch end. Each checkpoint stores model + optimizer + scheduler + AMP
  scaler + **RNG states** + epoch/step + config hash + git SHA → fully **resumable**
  after a session drop.
- **Run management:** one seed per session if needed; log to a persistent tracker
  (CSV/W&B) keyed by `(config_hash, seed, git_sha)`. Never overwrite checkpoints across
  seeds/conditions.

---

## 6. Evaluation

- **During training (health check):** open-set **verification** on held-out identities —
  cosine similarity ROC, TAR@FAR (e.g., @1e-3), and verification accuracy on balanced
  same/different identity pairs. Track every epoch.
- **On checkpoints (the deliverable):** run the `eval/` behavioral-signature harness —
  inversion, composite, part-whole — reporting **stream differences** with seed-wise
  error bars and a statistical test. (Harness is built in Milestone 0, before this.)
- **Controls:** rerun the identical eval path across `freq_only`, `baseline_identical`,
  `param_matched`, `role_crossover` config variants. A result must survive
  `param_matched` and behave as predicted under `role_crossover`.

---

## 7. Milestone-1 run order

1. Smoke test: 1 seed, few hundred steps, confirm loss decreases + verification > chance.
2. Cutoff sweep: small grid over `σ_hi`/`σ_lo`; pick a defensible operating point.
3. Full run × ≥5 seeds for the default config.
4. Control variants × ≥5 seeds each.
5. Behavioral-signature harness across all checkpoints → asymmetry result with stats.

Do not proceed to Milestone 2 (read-out decoders) until the asymmetry holds and
survives controls.

---

## 8. Risks and mitigations

- **~20 images/identity** → modest positives; PK with K=4 is fine, but monitor for
  unstable contrastive batches; consider class-balanced sampling.
- **Contrastive collapse** → L2-norm + temperature + sufficient negatives (large P);
  watch embedding variance.
- **Colab interruption** → robust resumable checkpointing (Section 5); keep epochs short
  between saves.
- **Frequency-cutoff sensitivity** → treat as a logged experimental variable; report the
  sweep, don't hide it.
- **Augmentation breaking holism** → keep geometric aug mild; verify the global stream
  still learns configuration (read-out inspection in Milestone 2).
- **Identity leakage** → enforce identity-disjoint splits; unit-test that train/test
  identity sets don't intersect.

---

## 9. References

- Khosla et al. (2020). *Supervised Contrastive Learning.* (SupCon)
- Deng et al. (2019). *ArcFace.* (alternative metric loss / face-pretraining control)
- Liu et al. (2015). *Deep Learning Face Attributes in the Wild.* (CelebA)
- Backbone + neuroscience references: see `architecture.md` §9–§10.
