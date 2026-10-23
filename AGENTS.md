# AGENTS.md

This repository develops and audits algorithms that, for an unknown power
system observed through post-disturbance trajectories, produce a reduced-order
dynamic model that reproduces transient oscillations accurately inside the
observed excitation envelope, predicts loss of stability with a low
missed-instability (FSR) rate, exposes identified oscillation frequencies when
the data support them, and preserves an energy/Lyapunov structure so that
stability regions and margins follow without numerical transient integration.
Effective 2026-10-23, `kiki_ieee39_v1` is mandatory for all new real-data
training and evaluation. The old canonical IEEE39 dataset is historical
reproduction evidence. Synthetic systems with known ground truth remain the
primary evidence for structure and stability correctness.

Spatial clustering is paused. New real-data models retain all ten generator
nodes; reducing latent/dynamical order and pruning physical modules remain
research goals. Historical clustering audits and synthetic correctness tests
are preserved. Reopening real-data clustering requires a new explicit policy.

The active authority is `configs/research_policy.json`. Existing kiki test
events have prior exposure records and cannot be treated as a fresh test.
Use only source train with registered internal train/val roles for development.
Fresh confirmation requires a separately registered independent kiki dataset,
frozen choices, and the existing explicit test gate. Synthetic correctness
checks are exempt from the kiki real-data requirement.

Priorities are physical correctness (including passivity and energy
structure), closed-loop reintegration accuracy, leakage-free evidence, and
only then classification scores. Model selection — clustering, dynamical
order, latent/residual/delay modules — is an excitation-conditional
engineering choice made on train/val data only. Excitation views may
propose and audit different modules, but one supplied corpus produces one
jointly fitted model whose structure is the supported module union. Source or
regime labels never route predictions; unsupported modules are removed and
evaluations outside the certified excitation envelope refuse rather than
extrapolate.

On 2026-10-23 the user explicitly approved one pre-certification development
evaluation, `kiki_a04_development_20261023`, as registered in the active policy
and `docs/protocols/kiki_a04_development_authorization_20261023_zh.md`.
It is restricted to the frozen A04 candidates, seeds, budget, causal input
contract and internal train/val allocation inside original source train.
The purpose is to establish or reject empirical input support; this is
uncertified development evidence. Preserve every failure and every allocated
event in coverage/risk reporting. This exception permits neither deployment
predictions nor model promotion, and opens no source val/test trajectories.
Independent registered confirmation is required before releasing predictions
inside a supported range. Other evaluations retain the certified-envelope rule.

Authoritative entrypoints:

- `configs/project_manifest.json`: datasets, experiments, and workflow roles.
- `docs/architecture.md`: scientific and software boundaries.
- `docs/scientific_contract.md`: input, rollout, stability, and naming rules.
- `python -m project_harness list|audit|git-check|data-check`.

Rules:

- Agent assignment for the current research programme: core algorithm design
  and scientific acceptance/boundary decisions are delegated to Astra with
  Ultra reasoning. The coordinating agent owns integration and evidence
  verification. Bounded simple implementation and test tasks may use Luna.
  Do not invoke Claude Code or CC Orchestrator for this project.

- Never overwrite a result run. Every execution gets a new immutable run ID.
- Never use test data for fitting, normalization, calibration, or selection.
- Clearing-state task inputs contain the clearing state, declared priors and
  registered causal exogenous policy. History tasks additionally receive only
  the real observations declared by their named input contract, with a distinct
  forecast origin and comparison population.
- `Pe`, voltage, labels, event time, valid length, masks, fault metadata, and
  future summaries are never main-model features.
- Stability is derived from predicted trajectories and remains exploratory.
- Use `M2A-LYAP` only when every numerical certificate gate passes; otherwise
  use the gate-selected storage or energy-regularized name.
- Claims are bounded by certified excitation envelopes; an evaluation outside
  every certified envelope must refuse, not extrapolate, except for the
  explicitly registered, bounded A04 development evaluation above. The name
  `stability-certified equivalent` additionally requires the Level 6
  certificate suite and region/margin gates (see
  `docs/structure_discovery_roadmap.md`).
- Temporary programs belong under `scratch/one_off/<run-id>/`.
- Historical canonical experiment replay requires `--historical-reproduction`
  and unchanged registered configuration/workflow hashes. It is not new model
  selection or fresh test evidence.
- Each completed engineering/research change is committed and pushed to the
  private GitHub repository. Use `codex/` branches for subsequent work and
  review before main integration. Scientific model promotion retains its own
  acceptance and approval gates. Raw trajectories, result bodies, archives,
  credentials and local environments stay outside Git; track their manifests.
