# CLAUDE.md — HPM (Hemispheric Perception Model)

Guidance for any coding agent or contributor working in this repository. Read this
file first. The full technical design lives in `architecture.md`; this file is the
orientation layer and the rules of the road.

---

## 1. What this project is

HPM is a computational neuroscience project that models **hemispheric specialization
in human face perception** with a two-stream deep network.

**Core hypothesis.** The two cerebral hemispheres process faces with *different
computational mechanisms*:

- **Left hemisphere / L-FFA → LOCAL, featural processing** (parts: eyes, nose, brows;
  high spatial frequency).
- **Right hemisphere / R-FFA → GLOBAL, holistic processing** (whole-face
  configuration and second-order spatial relations; low spatial frequency), and the
  right hemisphere is **dominant for integration** of the two.

We instantiate each mechanism with the architecture that most faithfully embodies it
(a CNN for local, a ViT for global), let them exchange information through a
"corpus-callosum" cross-attention module, and fuse them with a right-dominant module.
Each stream also reconstructs its own face (a "read-out") so we can *see* what each
hemisphere perceives, alongside a unified integrated percept.

**This is a hypothesis-testing project wearing an architecture project's clothes.**
The deliverable that matters is "a measurable global/local asymmetry emerged and
survived controls," NOT "the model is faithful" or "accuracy is high." Every
engineering decision is subordinate to making the asymmetry measurable and the
biological claim defensible.

---

## 2. Project pillars

### Research charter
Test whether imposing biologically-motivated global vs. local processing on two
streams produces *asymmetric, measurable* behavioral signatures of holistic face
perception that (a) match human lateralization and (b) cannot be explained by
parameter count, bottleneck placement, or generic two-stream learning. Success is
defined behaviorally (see Experimentation methodology), not by classification
accuracy alone.

### Architecture philosophy
- **Fidelity first, within data limits.** Each stream is a *pure* embodiment of one
  mechanism: CNN = local/featural, ViT = global/holistic. No conv+attention hybrids
  *inside* a stream — that re-blends the distinction we are trying to isolate.
- **Integrate in latent space, not pixel space.** Per-stream face images are
  diagnostic read-outs and auxiliary losses; the real combination happens on feature
  representations. Pixel-space fusion is lossy and misplaces the callosum.
- **Every biological structure is a manipulable knob.** Callosum (lesion gain),
  fusion (swap/ablate), frequency bias (perturb), stream roles (cross-over). If a
  module models a brain structure, you must be able to lesion or swap it.
- **Build the smallest claim first.** Earn complexity. Identity head before
  reconstruction; single stream sanity checks before the full system; no Phase B
  depth until Phase A asymmetry holds.

### Experimentation methodology
- **Validation harness before model.** The behavioral-signature tests (inversion,
  composite, part-whole) are written first and run on any model checkpoint.
- **Signatures are DIFFERENCES between streams, never presence in one stream.** Any
  upright-trained net fails on inverted faces from distribution shift alone; the
  signal is that the global stream's drop is *significantly larger* than the local
  stream's and larger than a featural baseline.
- **n > 1, always.** Every claim is reported across multiple random seeds with a
  distribution and a statistical test. "Consistently emerges" must be a number.
- **Controls are mandatory, not optional:** identical-capacity baseline, frequency
  bias only, task bias only, role cross-over (give the CNN the holistic job and the
  ViT the featural job), and callosum-lesion. A result that does not survive
  cross-over and parameter-matching does not support the biological claim.

### Neuroscience grounding
Each design choice traces to literature (full list in `architecture.md`):
- Frequency-split input → Sergent (1982), right=low-SF/global, left=high-SF/local.
- Global/local test paradigm → Navon (1977), global precedence.
- FFA + right dominance → Kanwisher, McDermott & Chun (1997).
- Right-FFA holistic integration + composite effect → right-fusiform fMRI work.
Do not overstate strict SF-lateralization as settled; it is task- and
exposure-dependent. State claims with that caveat.

### Coding standards
- **Language/stack:** Python 3.11+, PyTorch, `timm` for pretrained backbones,
  `hydra`/`omegaconf` for config, `pytest` for tests, `ruff` + `black` for lint/format,
  type hints required on public functions.
- **Config-driven, not hard-coded.** Every architectural knob (frequency cutoff,
  window/role assignment, callosum gain, loss weights, seed) lives in a config file,
  never as a literal in model code. Controls are *config variants*, not new code paths.
- **Reproducibility:** seed everything; log seed, config hash, and git SHA with every
  run. No result is reported without its config and seed.
- **Determinism of experiments over cleverness of code.** Prefer explicit, readable
  modules. Each brain-analog module is its own class with a documented forward signature.
- **Tests:** shape/contract tests for every module; the behavioral-signature metrics
  have unit tests on synthetic inputs before they touch a trained model.
- **No silent magic numbers** for anything neuroscience-relevant — cite the source in
  a comment or config doc.

### Long-term roadmap
- **Phase A (current):** monocular 2D faces, public dataset, identity head first then
  reconstruction read-outs, establish the asymmetry + controls.
- **Phase B:** stereo depth front-end (custom from-scratch stereo capture), depth's
  effect on holistic robustness under pose/occlusion.
- **Phase C:** perturbation studies — alter perception within/between streams, lesion
  the callosum, and characterize how the fused percept degrades; compare to split-brain
  and prosopagnosia phenomenology.

---

## 3. Repository layout (target)

```
HPM/
  CLAUDE.md              # this file
  architecture.md        # full technical spec
  configs/               # hydra configs: model, data, training, experiment variants
    model/               #   stream defs, callosum, fusion, heads
    experiment/          #   control conditions (baseline, swap, lesion, ...)
  src/hpm/
    data/                # dataset, frequency-split transforms, stereo (Phase B)
    models/
      streams.py         # CNN (local) + ViT (global) encoders
      callosum.py        # bidirectional cross-attention w/ lesion gain
      fusion.py          # right-dominant latent fusion (swappable)
      decoders.py        # per-stream read-out + unified-percept decoders
      heads.py           # identity head
      hpm_model.py       # assembles the full graph
    losses/              # identity, reconstruction, perceptual; weighting
    eval/                # behavioral signatures: inversion, composite, part-whole
    train.py
  tests/
  notebooks/             # exploratory only; nothing load-bearing lives here
```

## 4. Common commands (fill in as implemented)

```
# install
pip install -e ".[dev]"

# train Phase A baseline
python -m hpm.train experiment=phaseA_identity_only

# run a control variant
python -m hpm.train experiment=phaseA_role_crossover

# run behavioral-signature eval on a checkpoint
python -m hpm.eval.signatures checkpoint=<path>

# tests + lint
pytest && ruff check src && black --check src
```

## 5. Hard constraints (do not violate without discussion)

1. **Small dataset.** Transfer learning is mandatory; the ViT (global) stream MUST be
   initialized from pretrained/DeiT weights. From-scratch transformers will fail here.
2. **Same evaluation, every variant.** Controls differ only by config; never fork the
   eval path per condition.
3. **No pixel-space fusion as the main channel.** Integration is in latent space.
4. **No Phase B depth until Phase A asymmetry is established and survives controls.**
5. **Never report a single-seed result as a finding.**

When in doubt, optimize for *measurability of the asymmetry and defensibility of the
biological claim*, not for accuracy or architectural elegance.
