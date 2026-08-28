# Structure discovery beyond a hand-written swing equation

Revision 4, 2026-08-28. Supersedes roadmap v3 (in git history). The main
selection unit is now the complete supplied trajectory corpus: one data-level
cluster decision is followed by one block-prunable physical model. Excitation
strata are validation views created after the model is frozen, not training
routers.

## 0. What changed since v1 of this roadmap

The AFSD line (v1–v7b, all stages completed with immutable evidence) answered
the *validity* question for the current IEEE39 F1 corpus:

- v5: no tested hard contraction suffices for the union of observed
  excitations → full ten-machine resolution is required *for this corpus*
  (`FULL_RESOLUTION_REQUIRED`).
- v6: the full-model route SS-DM is a **rollout-accurate dynamic equivalent**
  (10 s mean RMSE 0.153, zero failures); clustering is not better
  (`CLUSTER_NOT_BETTER`); the absolute modal gate FAILs.
- v7/v7b: passive resonance modules are noninferior but do not meet the 2%
  promotion gate; both REJECTED for promotion.
- v3/v4: a single unified clustering/order is **not** uniformly supported
  across excitation regimes (`SOURCE_EXCITATION_CONFOUNDED`-class stop).

Under the current framing these results motivate a direct data-level loss
test. A corpus is clustered only when discarded state information, input
heterogeneity, modal loss, synchronising closure and spectral separation all
fit inside fixed budgets. Otherwise the same algorithm retains full spatial
resolution. Mechanisms supported by different excitations are composed as
removable blocks inside the same model; they do not create a routed model
library.

## 1. Target algorithm

For an unknown system observed via post-disturbance trajectories:

```text
observed trajectories
  → one fast cluster-loss calculation on fit/selection groups
  → clustered coordinates if every loss budget passes; otherwise full resolution
  → one B2-rooted model containing supported potential, dissipation,
    node-storage, decaying-memory and oscillating-memory blocks
  → whole-block backward deletion with 1/5/10 s closed-loop noninferiority
  → one frozen energy-structure-preserving model
  → certified outputs:
      a) high-accuracy rollout inside the envelope
      b) low-FSR stability prediction from the rollout
      c) identified oscillation frequencies (when support gates pass)
      d) Lyapunov/storage certificate + stability region/margin estimate
         WITHOUT integration (the differentiating output)
  → refusal outside every certified envelope
```

## 2. Discovery hierarchy (Levels 1–5 retained, reframed)

Levels 1–5 of roadmap v1 remain in force:

- Level 1: invariant coordinates and dynamical order (block-Hankel,
  singular-value stability across folds/sources, principal angles, poles).
- Level 2: graph support from a declared candidate supergraph under
  group-sparse weak regression with symmetry/PSD constraints.
- Level 3: constitutive law per retained edge from a symmetry-compatible
  overcomplete library, weak/integral form, support-preserving.
- Level 4: energy, dissipation and ports — `x_dot = [J(x)−R(x)]∇H + G(x)u`
  with exact structural parameterizations.
- Level 5: identifiability before interpretation (persistent excitation,
  condition/profile width, selection frequency, synthetic recovery,
  held-out reintegration, invariance audits).

Reframing: the *selection unit* is the supplied corpus. Non-uniform mechanism
support is handled by composition inside the unified block model and by
whole-block validation, rather than by producing several deployment models.

## 3. Level 6 — stability certificates and regions

A candidate that claims more than rollout accuracy must carry an energy
structure by construction and pass a numerical certificate suite (reviving
and extending the M2A-LYAP gates):

- exact supply identity (dissipation inequality) to solver tolerance;
- storage Hessian positive curvature at the certified equilibrium;
- PSD dissipation and interconnection reciprocity;
- RK4 convergence and COI drift gates (existing).

On passing, the model earns the name **stability-certified equivalent** and
may output:

- a Lyapunov/storage function `V(x)`;
- a certified stability region estimate `{x : V(x) ≤ c}` with `c` from the
  controlling-UEP or largest-sublevel-set construction;
- a stability margin for a trajectory *without* integrating it
  (e.g. `c − V(x_clear)`), validated against numerical ground truth.

Metrics (new evaluation axis, pre-registered per experiment):

- stability-margin error vs Monte-Carlo numerical margin;
- certified-region fidelity vs numerical region of attraction
  (symmetric-difference ratio, and conservativeness: certified ⊆ numerical);
- missed-instability rate (FSR) of the *integration-free* judgement, gated
  separately from the rollout-based classifier.

Known hardness (recorded so gate design respects it): classical
structure-preserving models with transfer conductances admit only
path-dependent energy functions; learned Lyapunov functions with certified
regions are an open problem. M2A-LYAP (2026-07-24) passed its certificates
but failed trajectory accuracy (test ω R² 0.5998) — accuracy and certificate
are gated *jointly*, not sequentially. This axis is developed
synthetic-first (§5). The sample-based certification machinery lives in
`src/kron_reduce/evaluation/stability_region.py`.

## 4. Composition principle

Different excitations can expose different mechanisms. Candidate generation
may use that evidence, but composition occurs inside one explicit equation:
each mechanism is a storage, dissipation or memory block with reciprocal
ports. The final structure is selected jointly on the corpus and is frozen
before amplitude, information-richness or coupling-strength slices are
reported. Deployment therefore receives one model and one certified
excitation envelope, with refusal outside that envelope rather than a
training-time model router.

## 5. Evidence base

"Correctness of structure discovery" (frequency identification, stability
region) is only falsifiable against known ground truth. The evidence pyramid
is ordered:

1. **Synthetic benchmark registry (primary)**: a versioned family of
   multi-machine systems with known modal frequencies, known stability
   boundaries, and controlled excitation richness. Every new selection or
   certificate claim must recover ground truth here first.
2. **IEEE39 canonical (flagship)**: existing rules unchanged — F1/seed 324
   development, frozen validation read once per frozen branch, test and
   F2–F5 locked behind explicit gates, observed-Pm rollouts labelled as
   conditional upper bounds.
3. Cross-dataset replication tests as the registered evidence base grows.

## 6. Status ledger and next implementations

| Item | State (2026-08-28) | Under this roadmap |
|------|--------------------|--------------------|
| M0-3EG-DF | accepted single-seed anchor (F1) | retained as a fixed comparison baseline |
| SS-DM full model | rollout-accurate dynamic equivalent (v6) | retained; naming unchanged until modal/stability gates pass |
| v7/v7b resonant modules | rejected for promotion | retained as candidate-module evidence; their noninferiority evidence stays |
| v8 modal identifiability | completed: `MODAL_ENVELOPE_MAP_CREATED`; no shared modal promotion | use shared-equation poles plus excitation-local observability; do not claim unsupported frequencies |
| ESC v1 | completed historical per-envelope library | immutable historical evidence only; not the deployment architecture |
| high-order cluster validation | `VALIDATED_REFUSAL`, selected rank 10 | freeze the ten-machine spatial resolution for the current data level |
| combined ten-machine candidate | `IDENTIFICATION_FAILED`; physical and numerical gates pass, trajectory and modal gates fail | keep as a failed candidate and P1 diagnostic target; do not promote |
| M2A-LYAP | dormant since 2026-07-24 | certificate suite revived inside Level 6 |
| IEEE39 Level 6 boundary audit data | not registered | create a new checksum-locked version; never alter canonical v1 |

Recommended next implementations, in order:

1. **P1 failure diagnosis** — reproduce the combined ten-machine failure and
   separate block contribution, causal mechanical-power closure, objective
   scaling, identifiability, and modal-audit causes.
2. **P2 comparison contract** — freeze causal B2, M0-3EG-DF, M0-GMD-PM,
   SS-DM, and registered data-driven baselines under identical roles,
   horizons, integrator, and statistics.
3. **P3/P4 unified block model** — generate candidates from excitation views,
   fit the union as one model, and perform whole-block backward deletion on
   F1 train/selection/internal-audit data only.
4. **P5 Level 6 evidence** — verify algorithms on registered synthetic ground
   truth, then acquire an independent IEEE39 boundary-audit dataset. The
   Level 6 name additionally requires FSR 95% upper bound at most 5%, stable
   envelope coverage at least 50%, conservative certified regions, and the
   registered margin gates.
5. Only after P4 and P5 acceptance may F1–F5/seed 324 confirmation begin; no
   extra seed is read as a failed-confirmation rescue.
