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
