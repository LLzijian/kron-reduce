# Curated research report index

This directory is the canonical home for human-curated, version-controlled
research reports. Numerical source evidence remains in immutable `results/<run-id>/`
directories; harness-generated summaries belong in `reports/generated/`.

## Report contract

Each new report should state:

1. workflow or experiment identity;
2. data scope, fold, seed, and input policy;
3. evidence status and scientific terminal state;
4. immutable run IDs or evidence paths supporting quantitative claims;
5. whether test, other folds, or canonical data were accessed;
6. limitations and claims that remain locked.

Use descriptive lowercase filenames. Add a date only when multiple frozen snapshots
of the same report must coexist. Figures and their validation scripts belong under
`docs/reports/figures/<report-id>/`.

## Current reports

| Report | Scope |
|---|---|
| [TASK3 topology and memory pair](kiki_machine_local_topology_20261115_zh.md) | Two frozen models on complete train/val/test, all 67 stable test cases classified correctly, zero test instability detections, independently recomputed |
| [Independent excitation check](machine_local_topology_transfer_20261115_zh.md) | Four frozen models across four clean synthetic systems, new stable and unstable trajectories, all matched gates passed |
| [Long-horizon coupling and memory refinement](machine_local_topology_refinement_20261115_zh.md) | Two fixed-support fits with complete synthetic train/val/test GPU scores and passed recovery gates |
| [Coupling and memory origin diagnosis](machine_local_topology_diagnosis_20261115_zh.md) | Twelve frozen shared-coefficient and origin-state comparisons on synthetic train/val |
| [Event-conditioned coupling and memory learning](machine_local_topology_learning_20261115_zh.md) | Four fixed-support fits, complete synthetic train/val/test GPU scores, memory recovery gates |
| [Event-conditioned coupling and memory S0](machine_local_topology_s0_20261115_zh.md) | Four known-coefficient synthetic cohorts, complete GPU scoring, limited rank-control recovery |
| [TASK3 machine-local momentum calibration](kiki_machine_local_memory_20261114_zh.md) | Four candidate fits, frozen A/B scoring on train 346 / val 107 / test 76, retained baseline |
| [Machine-local memory calibration](machine_local_memory_calibration_20261114_zh.md) | Fixed-support train momentum calibration, complete synthetic train/val/test, all four recovery gates passed |
| [Machine-local memory coefficient substitutions](machine_local_memory_diagnosis_20261114_zh.md) | Four controls on each complete synthetic train cohort, GPU prediction and exact frozen-model replay |
| [Joint equation and function selection](machine_local_equation_20261114_zh.md) | Three known-truth populations, full train/val/test for six equations each, CUDA inference, correct support, memory-kernel recovery gate failed |
| [Paired R20/R21 stiffness comparison](kiki_machine_local_r20_r21_20261114_zh.md) | Common start, complete train 346 / val 107 / test 76 per model, full-val checkpoint selection, trajectory and stability scores, saved-array verification |
| [Shared stiffness calibration](machine_local_stiffness_calibration_20261114_zh.md) | Complete 64-case synthetic population, one bounded fit and all three frozen controls; parameter, hidden-state and trajectory recovery |
| [Hidden-state versus dynamics diagnosis](machine_local_hidden_oracle_20261113_zh.md) | Complete 64-case synthetic population, four matched/mismatched dynamics and hidden-origin controls; recovery and saved-array verification passed |
| [Full TASK3 L10 round](kiki_machine_local_full_20261113_zh.md) | Full train 346 / val 107 / test 76, one 40-update fit, complete trajectory and stability scores, saved-result recomputation and train-only diagnosis |
| [Nonlinear history S0](machine_local_nonlinear_history_20261113_zh.md) | Six clean sine-system cases; nonlinear recovery and gradients passed; complete train/val/test trajectory and stability scores |
| [Clean differentiable learning](machine_local_clean_learning_20261113_zh.md) | One 240-step clean fit; all three models evaluated on train/val/test; matched-linear learning and saved-array checks passed |
| [Gaussian information diagnosis](machine_local_gaussian_oracle_20261112_zh.md) | Known-distribution local risk exceeds the original target; reset adds risk; numerical and saved-array audits passed, KEEP_BASELINE / STOP_T0 |
| [Joint initial-state observer](machine_local_joint_observer_20261112_zh.md) | Clean state recovery passed; 1% noise failed; rank-27 rejection and saved-array audit passed, KEEP_BASELINE / STOP_T0 |
| [Discrete-history initializer](machine_local_observer_20261112_zh.md) | Clean state recovery passed; 1% noise failed; rank-9 rejection and saved-array audit passed, KEEP_BASELINE / STOP_T0 |
| [Machine-local initializer diagnosis](machine_local_initializer_diagnosis_20261112_zh.md) | Valid saved diagnosis; route `INITIALIZER_LIMITATION_PRIORITY`; one-use authority consumed; KEEP_BASELINE |
| [Machine-local S1-base comparison](machine_local_s1_20261112_zh.md) | Twelve 240-update synthetic fits; physical/numerical/observation checks passed, all six required prediction pairs failed; REJECT / KEEP_BASELINE / STOP_T0 |
| [Machine-local S0 correctness](machine_local_s0_20261112_zh.md) | 112 frozen checks passed; 97 saved-array checks and 15 runtime receipts reviewed; zero fits or real-data reads; KEEP_BASELINE |
| [State-dependent port train comparison](kiki_state_port_train_20261111_zh.md) | Six fixed 40-update fits on 258 eligible train events; weighted trajectory gains over A06 and matched constant ports, one added rule miss per seed; 76 saved-evidence checks passed, REJECT / KEEP_BASELINE |
| [State-dependent port correctness](state_port_synthetic_20261111_zh.md) | Fixed ten-node synthetic structure, Jacobian and propagation-gradient checks; 200 saved-evidence checks passed, original summary-write failure retained, no fitting or real-data reads; KEEP_BASELINE |
| [Encoder-referenced energy-ball transfer](kiki_energy_cap_transfer_20261110_zh.md) | Completed 774 fixed-A06 train slots; improved unconstrained integral initialization but failed encoder-relative future and rule protections; REJECT, consumed authority and KEEP_BASELINE |
| [Fixed-radius confirmation](constant_c_energy_confirmation_20261109_zh.md) | Six new parameter/noise cases within the exposed matched linear family; all frozen confirmation criteria met, hidden-state failures and zero-baseline comparisons retained; consumed authority and `KEEP_BASELINE` |
| [Energy-constrained integral initializer](constant_c_energy_initializer_20261109_zh.md) | Completed 9-case, 3-radius saved-synthetic comparison: bounded initial energy and improved weak-coupling noisy futures, retained hidden-state errors and inactive mismatch controls; consumed authority and `KEEP_BASELINE` |
| [Constant-C 100 ms synthetic study](constant_c_100ms_synthetic_20261109_zh.md) | One completed 17-case study separating sampled identifiability, initialization error and visible future equivalence; zero real-data reads, consumed authority and `KEEP_BASELINE` |
| [TASK3 frozen 200 ms result receipt](kiki_task3_frozen_200ms_20261109_evidence.json) | Retrospective A03 checkpoint comparison on 105 `val` and 75 previously exposed `test` events, with a consumed authorization and `KEEP_BASELINE` |
| `kiki_a06_joint_refinement_20261024_zh.md` | Three full-model refinements, one remaining FN per seed, complete numerical gates, retained coverage and a consumed fixed budget |
| `kiki_a05_paired_training_20261024_zh.md` | Completed six-fit F/J comparison, rejection of permanent backbone freezing, retained numerical failure, zero-fit closeout repair and consumed authorization |
| `kiki_a05_recorded_error_audit_20261023_zh.md` | JSON-only decomposition of frozen A04 errors, stable/unstable tradeoffs and the proposed fixed-versus-joint backbone training diagnostic |
| `kiki_a04_development_20261023_zh.md` | Completed 12-fit A04 development comparison, trajectory and LOS findings, numerical failure, consumed authorization and retained baseline |
| `kiki_a04_execution_20261023_zh.md` | Astra-reviewed 100 ms engineering tools, synthetic observability findings, real-data prerequisite refusal and preserved failed attempt |
| `research_governance_20261023_zh.md` | Clustering pause, mandatory kiki registry, test exposure and automated engineering checks |
| `kiki_a03_stability_task3_20261022_zh.md` | Frozen A03 internal-val LOS scoring and unmatched TASK3 algorithm references |
| `graph_coherent_m0_3eg_results_zh.md` | Graph-coherent M0-3EG results |
| `m0_afsd_v3_joint_selection_validity_research_report_zh.md` | AFSD v3 selection-validity audit |
| `m0_afsd_v4_graph_compositional_discovery_research_report_zh_20260818.md` | AFSD v4 graph/module discovery key-node report |
| `m0_damping_recovery_results_zh.md` | M0 damping recovery audit |
| `m0_graph_round5_method_results_en.md` | Round 5 method/results, English |
| `m0_graph_round5_method_results_zh.md` | Round 5 method/results, Chinese |
| `m0_m2a_lyap_integral_v2_method_results.md` | Integral weak-form M0/M2A/LYAP report |
| `m0_third_order_highorder_results_zh.md` | Third-order high-order-data results |
| `m0_transient_emf_third_order_results_zh.md` | Transient-EMF third-order results |

The AFSD v4 report records the formal terminal state
`SOURCE_EXCITATION_CONFOUNDED`; it does not claim a selected F1 graph/module branch.
