# HPM Roadmap

This file tracks the phased build plan, milestone gates, and long-term trajectory.
The source of truth for architecture is `architecture.md`; the source of truth for
coding conventions is `CLAUDE.md`. This file is about *sequencing and progress*.

---

## Phase A — Monocular 2D (current)

The single guiding question: **does imposing biologically-motivated global vs. local
processing on two streams produce a measurable, reproducible asymmetry in behavioral
signatures of holistic face perception, and does it survive controls?**

---

### Milestone 0 — Evaluation Harness
*Write the measurement instruments before the model.*

**Deliverables**
- `eval/inversion.py` — `InversionEffect` callable, unit-tested on synthetic data
- `eval/composite.py` — `CompositeFaceEffect` callable, unit-tested on synthetic data
- `eval/part_whole.py` — `PartWholeEffect` callable, unit-tested on synthetic data
- `eval/readout_inspect.py` — percept quality metrics callable, unit-tested
- All eval tests pass (`pytest tests/test_eval/`)

**Gate: do NOT start Milestone 1 until**
- Every metric returns the right keys and dtypes on a random model
- The differential calculation (global − local) is tested explicitly
- n_seeds aggregation is wired up and returns a mean ± std

**Why first:** a metric written after seeing results will unconsciously fit the results.
The harness must be blind to the trained model.

---

### Milestone 1 — Identity-Only System
*Establish the core asymmetry with the simplest possible supervision.*

**Pre-training infrastructure** *(complete — codebase is now installable and test-runnable)*
- `pyproject.toml` build backend fixed (`setuptools.build_meta`)
- `dataset.py` `antialias=True` applied
- CNN backbone finalised: ResNet50 (`num_classes=0, global_pool=''` + `forward_features`) with optional ArcFace weight loading via `pretrained_path`
- Control configs `freq_only.yaml` and `baseline_identical.yaml` updated to use `resnet50` for consistency

**Deliverables** *(all stubs implemented; first training run pending)*
- `data/transforms.py` — `FrequencySplit` fully implemented and tested
- `data/splits.py` — identity-stratified splits, no leakage
- `data/dataset.py` — `FaceDataset` returning `(x_hi, x_lo, label)`
- `models/streams.py` — `LocalCNNEncoder` (ResNet50) + `GlobalViTEncoder` (DeiT-Small)
- `models/callosum.py` — `CorpusCallosum` with `set_lesion_gain()`
- `models/fusion.py` — `RightDominantFusion` (query=R default)
- `models/heads.py` — `IdentityHead` (cross-entropy)
- `models/hpm_model.py` — full forward pass; all shape tests pass
- `train.py` — trains to convergence on identity loss (`lambda_uni=0, lambda_aux=0`)
- All four behavioral signatures measured on trained checkpoint

**Controls to run before advancing**

| Config | Purpose | Expected result |
|---|---|---|
| `param_matched` | Rule out parameter count | Asymmetry persists |
| `baseline_identical` | Rule out generic two-stream | Asymmetry weaker or absent |
| `freq_only` | Isolate input bias | Asymmetry partial at best |
| `task_only` | Isolate architectural bias | Asymmetry partial at best |
| `role_crossover` | Critical mechanistic test | Asymmetry reverses or collapses |

**Gate: do NOT start Milestone 2 until**
- Global stream inversion Δ significantly > local stream Δ across ≥ 5 seeds
  (paired test, p < 0.05)
- Result survives `param_matched` and behaves as predicted under `role_crossover`
- `callosum_lesion` (`lesion_gain=0`) produces degraded behavioral signatures
- All control runs share the exact same eval path (no forked logic)

---

### Milestone 2 — Per-Stream Read-Out Decoders
*Make each hemisphere's percept visible; add auxiliary reconstruction loss.*

**Deliverables**
- `models/decoders.py` — `StreamReadoutDecoder` implemented (transposed-conv upsampler)
- Auxiliary loss enabled (`lambda_aux > 0`) in training
- Qualitative inspection: local percept shows sharp features / blurred config;
  global percept shows smooth features / intact config
- `readout_inspect.py` metrics (landmark-config RMSE, feature sharpness) report
  the expected direction for both streams

**Gate: do NOT start Milestone 3 until**
- Per-stream percepts are visually and quantitatively differentiated in the
  predicted direction (local: low feature error, high config error; global: reverse)
- Adding `lambda_aux` does not destroy the Milestone 1 behavioral asymmetry

---

### Milestone 3 — Unified Percept Decoder
*Add the integrated face output and tune the loss balance.*

**Deliverables**
- `models/decoders.py` — `UnifiedPerceptDecoder` implemented
- Unified reconstruction loss enabled (`lambda_uni > 0`)
- Loss weight sweep documented; chosen weights logged with every run
- Unified percept qualitatively resembles a coherent face (not a blend artifact)

**Gate: do NOT start Milestone 4 until**
- Identity accuracy and behavioral asymmetry from Milestone 1 are not significantly
  degraded by adding reconstruction losses
- Loss balance is stable (no single term dominating)

---

### Milestone 4 — Callosum Lesion Sweep
*Characterise the role of interhemispheric transfer.*

**Deliverables**
- Lesion gain sweep: `lesion_gain ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` on one checkpoint
  (use `model.callosum.set_lesion_gain()` — no retraining)
- Each gain level: behavioral signatures + unified percept quality reported
- Narrative: does degradation pattern resemble split-brain phenomenology?
- `fusion_swap` and `fusion_ablate` controls also run at this milestone

**Phase A completion criteria**
- Asymmetry is present, reproducible (≥ 5 seeds), and survives all mandatory controls
- All claims are stated with effect sizes, CIs, and seed distributions
- Config + seed + git SHA logged for every reported result

---

## Phase B — Stereo Depth Front-End
*Blocked until Phase A completion criteria are met.*

**Goal:** does adding stereo depth to the input improve the global stream's
robustness under pose variation and partial occlusion?

**Key additions**
- Custom stereo capture pipeline (from scratch; no off-the-shelf stereo datasets)
- `data/stereo.py` — left/right pair loader with identity labels
- Depth front-end that feeds into the existing fusion pathway
- Same behavioral-signature harness applied to stereo-trained checkpoints
- Compare holistic robustness (composite effect under pose/occlusion) vs. Phase A baseline

---

## Phase C — Perturbation Studies
*Blocked until Phase B, or optionally branched from Phase A.*

**Goal:** characterise how the fused percept degrades under targeted perturbations
of individual modules; compare to clinical phenomena (split-brain, prosopagnosia).

**Planned perturbations**
- Callosum lesion sweep (already done in M4; extend here to trained Phase B models)
- Stream-specific feature suppression (ablate spatial regions in one stream)
- Fusion perturbation: inject noise into `z_F` at inference time
- Comparison to split-brain: does `lesion_gain=0` reproduce the characteristic
  inability to integrate left-field / right-field face halves?
- Comparison to prosopagnosia: does right-stream ablation produce a pattern
  consistent with impaired holistic processing but intact featural recognition?

---

## Controls reference (Phase A)

All controls are config variants — never new code paths.

| Experiment key | What changes | What it tests |
|---|---|---|
| `phaseA_identity_only` | baseline | — |
| `baseline_identical` | same backbone, no freq split | generic two-stream |
| `freq_only` | same backbone, freq split intact | input bias alone |
| `task_only` | CNN+ViT roles, no freq split | architectural bias alone |
| `role_crossover` | CNN↔ViT roles swapped | mechanism vs. assignment |
| `callosum_lesion` | `lesion_gain=0` | interhemispheric transfer |
| `fusion_swap` | fusion query = L | right-dominance |
| `fusion_ablate` | fusion bypassed (concat) | fusion attention |
| `param_matched` | equalized capacities | parameter count |

---

## Reporting standards (all phases)

- Never report a single-seed result as a finding.
- Every claim: effect size, distribution across seeds, statistical test.
- Every run: seed + config hash + git SHA in the log.
- Controls are not optional extras — a finding without `param_matched` and
  `role_crossover` is not a finding.
