# Architecture

The repository has five non-overlapping layers:

1. Data truth: immutable canonical files under `data/`.
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
