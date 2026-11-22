# Research canon

## Literature facts

- Structure-preserving power-network reduction must preserve the relationship between inertia, synchronising coupling, inputs and retained coordinates; topology-only grouping is insufficient for dynamic equivalence. Source status: background literature already cited in `docs/reports/02_afsd_structure_discovery/graph_coherent_m0_3eg_results_zh.md`.
- Port-Hamiltonian and energy-structured models can encode storage, dissipation and reciprocal ports by construction, but a structural form alone does not establish trajectory accuracy or a useful stability region. Source status: project reports and Level 6 contract in `docs/structure_discovery_roadmap.md`.
- Empirical DMD visibility depends on excitation. A shared physical equation can contain modes that are not repeated in every individual disturbance. Source: synthetic and IEEE39 evidence from M0-AFSD v8 and M0-ESC v1 immutable runs.

## Experimental facts

- `ieee39_postclear_canonical_v1` contains 315 checksum-audited events. It includes 264 stable and 51 unstable events; all unstable events belong to the dense near-critical source and cover nine fault lines.
- The focused high-order cluster validation run selected the identity partition with ten original generators and terminal state `VALIDATED_REFUSAL`.
- The latest combined ten-machine identification run ended as `IDENTIFICATION_FAILED`. It reported one model, no source routing, no test reads, no future observed-Pm use, physical and integration passes, zero false-stable certificates, and failed internal/external cohort gates.
- The same run reported negative long-horizon pooled speed R2 on both internal and external combined cohorts; it cannot support a rollout-accurate or modal-identification claim.
- The current F1 safety audit used 40 unstable cases and reported zero false-stable certificates with a Wilson 95% upper FSR bound of 0.0876. The full canonical set contains 51 unstable events, still insufficient for a 5% upper bound even before dependence by fault line is considered.
- Historical accepted evidence includes an F1/seed-324 M0-3EG-DF anchor, but it was developed in a different representation/evaluation context and is not automatically an accepted ten-machine causal model.
- P1-A04 completed its controlled F1 diagnosis with process `PASS`, scientific verdict `ACCEPT` for the diagnosis, and `KEEP_BASELINE`. The accepted diagnosis is bound to `paper_ieee39_modular_energy_p1_a04_acceptance_f1_seed324-acceptance-20260904T142540Z-8bcecd42` and merge commit `75e3fafa30fdb419cee10520b24bc7a8077d385c`.
- The P1 candidate differs jointly from B2, but all seven named module instances are individually `UNIDENTIFIABLE`; it is not a validated module-union model.
- P1 ruled out numerical integration and COI drift as the direct cause of long-horizon error. Causal mechanical-power closure improves point error, but 10 s pooled speed R2 remains negative (`-0.225783`), leaving a mixture of forcing/dynamic closure and objective-calibration mismatch unresolved.
- P1 verified the shared-pole plus excitation-local observability semantics. None of nine reportable IEEE39 poles earned observational support, so there is no identified-frequency claim.
- The existing P1 internal role has only two near-critical fault-line groups. It cannot yield a source-specific fault-line bootstrap interval; P2 must freeze a source-stratified role allocation with at least three internal lines per source before P3/P4.
- Historical M0-3EG-DF, M0-GMD-PM, SS-DM and data-driven headline values are not mutually comparable: their stored evidence differs in representation, input policy, fold/seed scope, horizon or test access. P2 therefore rebuilds all main-table numbers.

## Thermodynamic / model facts

- A valid main model receives the exact clearing state, equilibrium/declared priors and registered causal exogenous policy. It predicts future angle and speed.
- The supplied corpus produces one final equation. Excitation views can support candidate blocks and audits but do not become model inputs or routing labels.
- Energy naming is earned only after finite parameters, PSD dissipation, reciprocal ports, positive storage curvature, numerical supply identity and integration gates pass.
- `stability-certified equivalent` additionally requires the complete Level 6 region, margin, non-vacuity and FSR gates.
- Stability labels and boundary outcomes are read only after model and certificate construction are frozen.

## Supervisor / owner constraints

- Use first-principles algorithm design and adversarial review.
- Do not reach targets through shortcut metrics, leakage, trivial classifiers, forced clustering or relaxed post-result thresholds.
- Reports should lead with the conclusion and use clear language.
- Module candidates supported by different transients are combined into one model; the data are not first routed to separate deployment models.
- Every research task follows design, acquisition, execution, acceptance with merge or rejection with rollback, then reporting.

## Terminology definitions

- **模块并集模型**: one shared equation containing every independently supported block that survives joint fitting and whole-block deletion.
- **激励视图**: a frozen audit or candidate-support slice; not a model selector at deployment.
- **因果主表**: rollout beginning at clearing with `Pm_clear` held by zero-order hold or a separately registered causal Pm model.
- **条件输入敏感性**: after model freeze, replace the registered causal exogenous input by future observed `Pm(t)` without refitting. It is never a primary deployment result and is not called an upper bound unless a separate matched-input contract proves that interpretation.
- **过程通过**: implementation, evidence and audit chain are valid even if the scientific hypothesis is rejected.
- **科学接受**: the pre-registered claim gates pass and the candidate may be promoted after explicit merge approval.

## Forbidden claims

- The current combined ten-machine candidate is accepted, rollout-accurate, modal-identified or Level 6 certified.
- Source labels, amplitude bins or richness bins select different deployment models.
- Synthetic correctness proves IEEE39 performance.
- Observed future mechanical power is a causal input.
- Effective identified coefficients are true machine parameters or the 45-edge support is the physical 39-bus topology.
- Zero false-stable predictions alone establish a useful stability certificate.
- The canonical 51 unstable events establish a 5% 95%-upper FSR guarantee.

## Unresolved claims

- Which new P3 blocks have independently identifiable counterfactual effect under the P2 matched ten-machine causal contract.
- Whether a redesigned joint objective and dynamic/forcing closure can turn the still-negative long-horizon speed R2 positive.
- Which redesigned shared equation poles, if any, become observable and repeatable under each registered excitation view.
- Whether a non-vacuous IEEE39 integration-free stability region can meet Level 6 accuracy and coverage gates.
- Whether the final block structure is stable in at least four of five folds.
