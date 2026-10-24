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
The A04 campaign has now completed and its authorization is `consumed` in
the active policy, with a bound run, budget and result receipt. Do not reactivate
it or repeat its fits. Existing artifacts remain available for read-only audit;
any new scientific comparison needs its own registered authorization.

On 2026-10-24 the user explicitly authorized one bounded A05 paired development
evaluation, `kiki_a05_development_20261024`, as registered in the active policy
and `docs/protocols/kiki_a05_development_authorization_20261024_zh.md`. It is
restricted to experiment `paper_kiki_a05_paired`, workflow
`kiki_a05_paired_v1`, stage `paired_training`, and module
`kron_reduce.workflows.kiki_paired_training`. The comparison uses groups F/J,
seeds 324/325/326, 80 updates per fit, and at most six fits/480 new updates on
the existing frozen source train/internal train-val allocation. It explicitly
permits development scoring of all 258 train and 88 val trajectories at 1 s
and 3 s, plus all 88 val 8 s stability and numerical-refinement diagnostics.
Source val/test payloads remain closed. Labels may be read only after model,
selection, and trajectory evidence are frozen, for diagnostics. Preserve all
allocated events and failures; failed fits consume their slots and cannot be
retried. These results remain uncertified development evidence: no deployment
predictions, model promotion, or independent confirmation. The A04 authorization
remains consumed and unchanged.

On 2026-10-24 the user explicitly authorized one bounded A06 joint-refinement
development evaluation, `kiki_a06_development_20261024`, as registered in the
active policy and `docs/protocols/kiki_a06_development_authorization_20261024_zh.md`.
It is restricted to `paper_kiki_a06_refinement`, workflow
`kiki_a06_refinement_v1`, stage `joint_refinement`, and module
`kron_reduce.workflows.kiki_joint_refinement`. It starts from the three frozen
A05 J checkpoints, uses fresh Adam state, and allows exactly three fits at
seeds 324/325/326 with 240 new updates each. The update-80 checkpoint is
diagnostic; update 240 is the sole primary endpoint. The fixed exception allows
all 258 train and 88 val trajectories at 1 s and 3 s and all 88 val 8 s LOS and
numerical checks, with all 369 allocated cases retained. Source val/test payloads
remain closed; labels are available only after all training and evidence are
frozen. Failures consume their slots and cannot be retried. Results remain
uncertified development evidence with no deployment predictions or promotion.
Automatic training until a threshold passes remains disabled; A04 and A05
authorizations remain consumed and unchanged.

The A05 campaign has completed and its authorization is now `consumed`,
bound to its immutable run, six fits/480 updates, and result receipt.
Do not reactivate the authorization or repeat its fits. Existing artifacts
may be audited read-only; further scientific comparisons require their
own registered authorization. No A05 model has been promoted.

The A06 campaign has completed and its authorization is now `consumed`,
bound to its immutable run, three fits/720 updates and result receipt.
Do not reactivate it or repeat its fits. Read-only audit remains available;
a new scientific comparison requires its own registered authorization.
No A06 model has been promoted.

The A07 campaign has completed and its authorization is now `consumed`,
bound to its immutable run, six paired fits/1440 new updates and result receipt.
Do not reactivate it or repeat its fits. Read-only audit remains available;
a new scientific comparison requires its own registered authorization.
No A07 model has been promoted or selected for deployment.

Authoritative entrypoints:

- `configs/project_manifest.json`: datasets, experiments, and workflow roles.
- `docs/architecture.md`: scientific and software boundaries.
- `docs/scientific_contract.md`: input, rollout, stability, and naming rules.
- `python -m project_harness list|audit|git-check|data-check`.

Rules:

- The coordinating agent owns research runtime changes. Use an explicitly
  selected Python 3.12 environment constrained by
  `requirements/constraints-py312.txt` and invoke its Python directly.
  Do not use `uv run` in the shared checkout: it can recreate `.venv` while
  another task is using it. Prepare replacement environments under ignored
  `scratch/` and record the actual runtime with the experiment evidence.

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
  explicitly registered, bounded development evaluations above. The name
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

On 2026-10-24 the user authorized one bounded A07 tilted-objective development
comparison, `kiki_a07_development_20261024`, as registered in the active policy
and `docs/protocols/kiki_a07_development_authorization_20261024_zh.md`. It is
restricted to `paper_kiki_a07_tilted`, workflow `kiki_a07_tilted_v1`, stage
`tilted_training`, and module `kron_reduce.workflows.kiki_tilted_training`.
Groups M and T each use seeds 324/325/326 and 240 additional updates, restoring
the same-seed A06 step-240 model and exact saved Adam state. M retains the mean
event MSE objective; T uses the registered train-only risk temperatures
0.7844805082870585, 0.7768666645735861, and 0.7861097423844166. The fixed
exception allows complete train/val trajectory and val stability evaluation
with all allocated cases retained. Source val/test remain closed; labels are
diagnostic only after freeze. There is no classification-based group selection,
automatic continuation, deployment prediction, or model promotion. A04, A05,
and A06 remain consumed and unchanged.
