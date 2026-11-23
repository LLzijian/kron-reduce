# M0-ESC v1 protocol: excitation-conditioned selection and composition

Status: registered 2026-08-20; scaffold stage; no runs yet. Workflow
definition: `configs/workflows/m0_esc_v1_excitation_conditioned_selection.json`.
This protocol implements roadmap v2 §1–§4 and supersedes the stop-rule
interpretation of the AFSD validity audits.

---

## Scientific question

Given a post-disturbance trajectory corpus from a system whose structure is
unknown, can an algorithm

1. characterize the observed excitation envelope from train data only;
2. select, per envelope regime, the simplest spatial resolution, dynamical
   order and module set that passes pre-registered accuracy gates — starting
   from the B2 weak-form swing baseline and adding latent EMF (EG), memory or
   delay, and modal port modules only when they earn their states;
3. certify, for every selection, the excitation envelope in which it is
   valid (borrowing the AFSD v3 eligibility audit and the v8 modal
   identifiability machinery as *envelope measurement*, not as stop rules);
4. preserve an energy structure by construction, so that a numerical
   certificate suite can promote the model to a *stability-certified
   equivalent* whose stability region and margin are reported without
   transient integration; and
5. compose per-regime models into a library whose union covers the observed
   excitation space, refusing — never extrapolating — outside it?

Null outcome: if no selection passes accuracy gates beyond B2, or no
certificate survives the suite, the workflow returns the named fallback
(`rollout-accurate B2 equivalent` with its certified envelope) and a data
request, not a promoted model.

## Relationship to the AFSD line

AFSD asked "which single structure does the corpus uniformly support?" and
stopped (correctly, under its own rules) at SOURCE_EXCITATION_CONFOUNDED.
ESC asks the engineering question: "which structure should be used *for the
excitation actually observed*, and where exactly does its validity end?"
Every AFSD negative result is reused as envelope evidence; nothing is
re-run or overwritten.

## Data and evidence base

1. **Synthetic benchmark registry v1 (primary)**. A versioned family of
   systems with known ground truth: modal frequencies, damping, stability
   boundaries; scripted excitation regimes (small-signal cluster-eligible,
   large-signal, mixed). Structure-discovery correctness and stability-region
   fidelity are measured here first.
2. **IEEE39 canonical (flagship)**. Unchanged rules: F1, seed 324 for
   development; frozen validation read once per frozen branch; test and F2–F5
   locked; observed `Pm(t)` rollouts are conditional upper bounds and are
   labelled as such.

Voltage, `Pe`, labels, event time and fault metadata remain excluded from
model features. Fault line may be used as a *statistical clustering unit* for
uncertainty and support, as in v6–v8.

## Selection space

Spatial: `generator_10` (identity endpoint), train-discovered partitions, and
historical fixed partitions compete per regime. Clustering is *expected* to
win in small-signal regimes (fewer identifiable features); full resolution is
the default in rich regimes (v5 evidence).

Temporal/module ladder (each rung must beat the previous on the selection
split):

- `B2`: nonnegative integral weak-form swing baseline;
- `B3`/`EG`: latent transient-EMF state (M0-3EG family);
- memory/delay families around B3 (first-order holds, edge lag);
- modal ports: passive resonant ports (v7 family) and the reciprocal
  low-rank modal port (M0-LRMRP, v8) — the DMD/DMDc machinery supplies
  candidate frequencies and an identifiability audit, never autonomous
  eigenvalue claims.

All candidates are passive-by-construction; no candidate without a supply
identity enters the accuracy gate.

## Excitation envelope and regime routing

Train-only envelope features: accelerating-power amplitude, activation
terciles (frozen, as in AFSD), block-Hankel stable rank, repeatable spectral
peaks (v8 DMD consensus), and event-count support per fault-line cluster.
Regimes are declared *before* fitting candidates. Each selected model is
paired with its envelope: the feature-space region where its support gates
passed. Deployment outside every certified envelope is a refusal with a
named data request.

## Gates (all pre-registered before controlled metrics)

Accuracy axis (selection split, then disjoint audit split):

- rollout RMSE improvement over B2 ≥ 2% at H=1000, bootstrap lower bound
  positive, sign-flip p ≤ 0.05 (as in v7b);
- zero integration failures; COI drift and RK4 convergence gates.

Stability-classification axis:

- missed-instability rate (FSR) of the trajectory-derived judgement ≤
  pre-registered bound on the eligible cohort; failed rollouts never count
  as stable.

Structure axis (synthetic ground truth only):

- identified dominant frequency error ≤ pre-registered bound (proposed 5%
  on synthetic, to be calibrated at model_unit_gate);
- MAC ≥ 0.8 against true mode shapes.

Stability-certificate axis (Level 6, synthetic first):

- supply identity, storage curvature, PSD dissipation to solver tolerance;
- certified region ⊆ numerical region of attraction (conservativeness
  required), symmetric-difference ratio ≤ bound;
- stability-margin error vs Monte-Carlo margin ≤ bound;
- integration-free FSR gated separately from the rollout classifier.

Naming ladder (strictly ordered, each name requires its axis):
`rollout-accurate equivalent` → `modal-identified equivalent` →
`stability-certified equivalent`. The v6 name `rollout-accurate dynamic
equivalent` remains the default; `M2A-LYAP` retains its own certificate gate.

## Composition and library claims

The output is a model library: entries of (model, certified envelope, earned
name, evidence run IDs). A "complete model" claim covers exactly the union of
certified envelopes and must list coverage gaps as data requests. Library
entries selected on synthetic regimes carry their envelope features with
them, so cross-dataset composition is a documented matching procedure, not an
assumption.

## Adversarial review additions (beyond the v6–v8 checklist)

- accuracy without an energy structure cannot claim any stability margin;
- envelope overreach: any evaluation outside the certified envelope voids
  the claim automatically;
- DMD/DMDc outputs remain empirical forced-response descriptors;
- regime definitions fitted after seeing candidate metrics are a FAIL;
- a library union claim without per-entry envelope evidence is a FAIL;
- synthetic ground-truth gates cannot be satisfied by IEEE39 evidence and
  vice versa.

## Stage structure (matches the harness loop shape)

`scaffold → data_verified → contracts_verified → model_unit_gate →
source_commit → hypothesis_registration → source_recommit →
controlled_experiment → residual_diagnosis → adversarial_review →
acceptance → merge_gate`

`model_unit_gate` covers: synthetic-registry recovery of B2 and each module
family, certificate-suite smoke on a known system, and refusal behaviour on
an out-of-envelope probe. `merge_gate` requires explicit approval.

## Locked claims (inherited, unchanged)

test performance; F2–F5 generalization; autonomous `Pm` prediction;
ordinary DMD as exact autonomous eigenanalysis; unique device semantics;
physical 39-bus line recovery; machine-parameter recovery; `M2A-LYAP`
naming without certificates; any stability-margin claim without the Level 6
axis passed; any library-union claim without per-entry envelope evidence;
any use of synthetic ground-truth gates to satisfy IEEE39 evidence
requirements, or the converse.
