# Structure discovery beyond a hand-written swing equation

Revision 2, 2026-08-20. Supersedes roadmap v1 (in git history). The change of
framing follows the owner directive of 2026-08-20: the deliverable is an
*algorithm for unknown systems*, and its decisive engineering value is
certified stability analysis without transient integration.

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

Under roadmap v1 these were terminal verdicts. Under the current framing they
are **envelope measurements**: each one delimits what the *observed
excitation* can support, and the algorithm must route accordingly — select
per regime, certify each envelope, compose across regimes, and state exactly
which new excitation data would enlarge the envelope. v8 (modal
identifiability) is the first envelope certification of this kind.

## 1. Target algorithm

For an unknown system observed via post-disturbance trajectories:

```text
observed trajectories
  → excitation-envelope characterization (train-only)
  → per-envelope selection of spatial resolution, dynamical order, and modules
      (B2 swing baseline; EG latent EMF states; memory/delay families;
       modal resonant / low-rank ports; clustering when the excitation is
       small-signal and feature reduction helps)
  → energy-structure-preserving fit (passive by construction)
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

Reframing: the *selection unit* at every level is the excitation regime, not
the corpus. "Not uniformly supported" routes to per-regime selection; it is
not a stop.

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

Different excitation regimes (within one dataset, or across datasets/systems)
expose different structural oscillations. The algorithm therefore maintains a
*model library*: each entry pairs a selected model with its certified
excitation envelope and its earned name. Deployment = envelope detection →
model selection → refusal if uncovered. A "complete model" claim is only ever
made over the union of certified envelopes, and the union's coverage gaps are
reported as data requests (e.g. longer near-free ringdown, repeated
persistently exciting disturbances, independent voltage/Pe channels).

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
3. Cross-dataset composition tests as the library grows.

## 6. Status ledger and next implementations

| Item | State (2026-08-20) | Under this roadmap |
|------|--------------------|--------------------|
| M0-3EG-DF | accepted single-seed anchor (F1) | retained as a library entry with its envelope |
| SS-DM full model | rollout-accurate dynamic equivalent (v6) | retained; naming unchanged until modal/stability gates pass |
| v7/v7b resonant modules | rejected for promotion | retained as library candidates; their noninferiority evidence stays |
| v8 modal identifiability | code complete, no runs | first *envelope certification*; all three terminal states feed the envelope map |
| M2A-LYAP | dormant since 2026-07-24 | certificate suite revived inside Level 6 |
| PQSC / structure forest | scratch prototypes | migrate only behind synthetic-registry evidence |

Recommended next implementations, in order:

1. **Finish v8** — envelope certification for shared modal structure on F1.
2. **Synthetic benchmark registry v1** — multi-machine families with known
   frequencies, known stability regions, and scripted excitation regimes
   (small-signal / large-signal / mixed). This unblocks every Level 6 and
   "unknown system" claim.
3. **ESC v1 workflow** (`docs/protocols/m0_esc_v1_excitation_conditioned_selection.md`)
   — the first workflow implementing §1–§4 end to end on synthetic + IEEE39
   F1.
4. Only then consider requesting a new gate for multi-fold or second-seed
   generalization; test stays locked until manuscript freeze.
