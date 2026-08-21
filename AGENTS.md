# AGENTS.md

This repository develops and audits algorithms that, for an unknown power
system observed through post-disturbance trajectories, produce a reduced-order
dynamic model that reproduces transient oscillations accurately inside the
observed excitation envelope, predicts loss of stability with a low
missed-instability (FSR) rate, exposes identified oscillation frequencies when
the data support them, and preserves an energy/Lyapunov structure so that
stability regions and margins follow without numerical transient integration.
The canonical IEEE39 dataset is the flagship evidence; synthetic systems with
known ground truth are the primary evidence for structure and stability
correctness.

Priorities are physical correctness (including passivity and energy
structure), closed-loop reintegration accuracy, leakage-free evidence, and
only then classification scores. Model selection — clustering, dynamical
order, latent/residual/delay modules — is an excitation-conditional
engineering choice made on train/validation data only; lack of uniform
support across excitations routes to per-regime selection and composition,
never to termination.

Authoritative entrypoints:

- `configs/project_manifest.json`: datasets, experiments, and workflow roles.
- `docs/architecture.md`: scientific and software boundaries.
- `docs/scientific_contract.md`: input, rollout, stability, and naming rules.
- `python -m project_harness list|audit|git-check|data-check`.

Rules:

- Never overwrite a result run. Every execution gets a new immutable run ID.
- Never use test data for fitting, normalization, calibration, or selection.
- Model inputs may contain the clearing state, declared priors, and the
  registered causal exogenous policy only.
- `Pe`, voltage, labels, event time, valid length, masks, fault metadata, and
  future summaries are never main-model features.
- Stability is derived from predicted trajectories and remains exploratory.
- Use `M2A-LYAP` only when every numerical certificate gate passes; otherwise
  use the gate-selected storage or energy-regularized name.
- Claims are bounded by certified excitation envelopes; an evaluation outside
  every certified envelope must refuse, not extrapolate. The name
  `stability-certified equivalent` additionally requires the Level 6
  certificate suite and region/margin gates (see
  `docs/structure_discovery_roadmap.md`).
- Temporary programs belong under `scratch/one_off/<run-id>/`.
