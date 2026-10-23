# Architecture

The repository has five non-overlapping layers:

1. Data truth: mandatory kiki real-data registry under `data/kiki_ieee39_v1/`;
   immutable source bodies resolved by `KIKI_DATA_ROOT`, historical canonical data,
   and independent synthetic truth.
2. Scientific view: data contracts, representations, splits, and causal input policy.
3. Models: equations or learned closures implementing the common dynamics API.
4. Experiments: immutable combinations of data, split, model, seed, and objectives.
5. Evidence: per-case results, audits, hashes, generated reports, and paper tables.

## Evidence and report layout

The evidence layer uses four directories with non-overlapping responsibilities:

- `results/<immutable-run-id>/` contains machine-verifiable experiment evidence,
  parameters, metrics, hashes, and gate records. A created run is never moved,
  renamed, overwritten, or edited.
- `reports/generated/` contains harness-generated cross-run summaries. These files
  are derived output, may be regenerated, and are not the numerical source of
  truth.
- `docs/reports/` contains human-curated, version-controlled research reports and
  method/result narratives. Every quantitative claim should identify its immutable
  run or evidence artifact.
- `paper/` contains manuscript and submission sources. It may cite curated reports
  for navigation, but evidence remains under `results/`.

Protocols, architecture, and scientific contracts stay under `docs/` outside
`docs/reports/`; temporary or unaccepted investigations stay under
`scratch/one_off/<run-id>/`. Files must not be placed directly in `reports/`, apart
from its directory policy README. `project_harness audit` enforces this boundary.

`project_harness` controls registry, Git state, run lifecycle, and gates. It must
not contain scientific model logic. `src/kron_reduce` contains the scientific
implementation and must not contain task numbers or result-directory constants.

The loop is a persistent, command-driven state machine, not a background service.
Each invocation validates the registered next edge, durable PASS evidence, and any
explicit approval before advancing. The project audit additionally checks workflow
identity, stage uniqueness, contiguous history, approval integrity, and the ordered
AFSD evidence chain.

The physics stack uses mass-weighted COI coordinates and one shared,
COI-preserving RK4 definition. M0 and graph-SINDy M2A identify integral
weak-form equations instead of numerically differentiated targets. The energy
candidate is an augmented port-Hamiltonian edge-memory system; its public name
is assigned only by the numerical certificate gate.

The historical data-level selector is split across three boundaries:
`selection/clusterability.py` owns the single-corpus compression decision,
`models/prunable_energy_model.py` owns the explicit block dynamics and physical
certificates, and `workflows/data_level_structure_selection.py` owns the
leakage-free 60/20/20 lifecycle and immutable artifacts. Post-freeze reporting
slices never call model fitting or selection code.

As of 2026-10-23 spatial clustering is paused for new real-data experiments:
ten generator nodes are retained. Temporal order, hidden state and module
selection remain active. `project_harness.policy` enforces the registered kiki
dataset, spatial resolution, input contract and exposed-test restriction before
creating a run. Canonical replays require an explicit mode and frozen hashes.
The runner holds a process-safe per-workflow lock, assigns collision-resistant
attempt IDs and retains exception tracebacks. Failed attempts do not advance
the scientific state or trigger automatic training retries.

GitHub CI checks metadata, policy, causality and synthetic physics without raw
private trajectories. `data-check` separately verifies frozen local metadata
and external source hashes without reading holdout trajectory arrays.

The deployment unit is one supplied corpus and one final model. Excitation
strata may be used on fit/selection data to generate candidate blocks and,
after freezing, to report where each block is observable. The candidate union
is fitted jointly and pruned by deleting whole blocks and refitting the
remainder. Dataset source, excitation regime, fault identity, and stability
label are neither model inputs nor routing keys. A lack of support narrows the
certified envelope or removes a block; it does not create a second deployment
model.

The user-approved `kiki_a04_development_20261023` evaluation provides one
bounded path to collect pre-certification development evidence on frozen
internal train/val inside original source train. Its causal empirical support
rule and 12-fit budget are registered before execution. Support auditing never
removes events from the full development comparison. The authorization binds
the experiment, input contract, allocation and protocol hashes; it is not a
generic envelope override. Deployment routing continues to refuse without a
certificate, and model promotion requires separate independent evidence.
