# Architecture

The repository has five non-overlapping layers:

1. Data truth: immutable canonical files under `data/`.
2. Scientific view: data contracts, representations, splits, and causal input policy.
3. Models: equations or learned closures implementing the common dynamics API.
4. Experiments: immutable combinations of data, split, model, seed, and objectives.
5. Evidence: per-case results, audits, hashes, generated reports, and paper tables.

`project_harness` controls registry, Git state, run lifecycle, and gates. It must
not contain scientific model logic. `src/kron_reduce` contains the scientific
implementation and must not contain task numbers or result-directory constants.

The physics stack uses mass-weighted COI coordinates and one shared,
COI-preserving RK4 definition. M0 and graph-SINDy M2A identify integral
weak-form equations instead of numerically differentiated targets. The energy
candidate is an augmented port-Hamiltonian edge-memory system; its public name
is assigned only by the numerical certificate gate.
