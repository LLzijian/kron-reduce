# AGENTS.md

This repository studies causal post-clear reduced dynamics on the canonical
IEEE39 dataset. Priorities are physical correctness, closed-loop reintegration,
leakage-free evidence, and only then classification scores.

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
- Temporary programs belong under `scratch/one_off/<run-id>/`.
