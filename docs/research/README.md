# Research scope and records

Check the [research policy](../../configs/research_policy.json) for current
execution authority. Registered protocols define each experiment's data access,
budget, entry point and stopping gates.

## Current research scope

The current [research plan](machine_local_structure_plan_20261112_zh.md)
prepares ten machine-local hidden states and learnable physical modules.
Its first comparison separates model order (G2/G10) from local port structure
(G10/L10), with the unchanged A06 as the performance reference. Initialization,
history propagation and future rollout use each candidate's own causal model.
The [design review](machine_local_structure_review_20261112_zh.md) covers this
preparation scope; execution requires a separately frozen protocol and budget.
The [2026-11-10 plan](research_plan_20261110_zh.md) remains the historical,
hash-bound design for the completed A1 work below.

The single [A1-S0 execution](../protocols/state_port_synthetic_v1_zh.md) saved
all numerical checks, then failed while serializing the final summary. Its
scientific authority is consumed and the failed run remains immutable. The
[separate saved-evidence review](../protocols/state_port_saved_review_20261111_zh.md)
completed 200 checks with zero mismatches and zero new model or trajectory
calls. The [result report](../reports/state_port_synthetic_20261111_zh.md) records
`PASS_FROM_SAVED_EVIDENCE` while preserving the failed process. The
[scientific result review](state_port_result_review_20261111_zh.md) accepts
these finite checks; both execution authorities are consumed. Real train
comparisons require their own design and registration; keep the baseline and
default deployment initializer.

The [A1 train comparison](../reports/kiki_state_port_train_20261111_zh.md)
completed all six 40-update fits and its separate saved arithmetic audit
(76 checks, zero mismatches). State-dependent ports improved weighted trajectory
MSE over both references, but added one train rule miss in every seed. The
[scientific review](state_port_train_result_review_20261111_zh.md) accepts the
complete evidence and confirms `REJECT / KEEP_BASELINE`. Authority
`kiki_state_port_train_v1_20261111` and its one-use saved audit are consumed.

The next preparation scope is the machine-local design's synthetic correctness
and identifiability protocol: reciprocal ports, complete causal gradients,
discrete observation sensitivity, and known-truth controls that can reject an
unsupported local structure. B0 remains an independent initializer candidate,
as originally recommended by the A1 result review; the user's 2026-11-12 request
changes the prospective priority. Potential, input and time-objective changes
remain separate. No new train run, A1 retuning, val/test access or model promotion
follows from this plan. Keep the [A1 protocol](../protocols/kiki_state_port_train_v1_zh.md),
its implementation review and all completed evidence frozen.

## Precursor evidence

The [constant-C synthetic study](../reports/constant_c_100ms_synthetic_20261109_zh.md)
completed its one registered execution: 17 cases separate discrete observation
information, initialization error and visible future equivalence. Its authority
is consumed; review the saved evidence rather than rerunning it. The fixed main
system supports accurate 100 ms recovery, while weak coupling, aliasing and
hidden-null controls delimit that result. This establishes no real-data claim.

The [energy-constrained initializer preparation](../protocols/constant_c_energy_initializer_v1_zh.md)
now defines nine saved synthetic cases and three fixed radius roles, retaining
both hidden-state and visible-future outcomes. The isolated algebra solver has
engineering tests; the [preparation review](constant_c_energy_preparation_review_20261109_zh.md)
accepts that scope. The [machine-readable plan](../../configs/proposals/constant_c_energy_initializer_preparation_v1.json)
and [saved-input manifest](../../configs/proposals/constant_c_energy_saved_sources_v1.json)
preserve the original preparation and source identities.
The separate [execution protocol](../protocols/constant_c_energy_execution_v1_zh.md)
and [exact implementation approval](constant_c_energy_execution_review_20261109_zh.md)
now bind the saved-input reader, conditional propagation and complete budgets.
The [completed comparison](../reports/constant_c_energy_initializer_20261109_zh.md)
retains all 27 records. The fixed bound controls energy inflation and improves
visible futures in the weak-coupling noisy case; hidden-state recovery still
fails its threshold. Both positive bounds remain inactive in the stiffness
mismatch case. The [scientific result review](constant_c_energy_result_review_20261109_zh.md)
accepts these bounded findings with `KEEP_BASELINE`.
Authority `constant_c_energy_initializer_v1_20261109` is consumed; phase-one
authority also remains consumed. Read-only evidence synthesis and preparation
of an independently registered follow-up remain available. No later candidate
stage or real-data comparison is authorized by this result.
The separate [fixed-radius confirmation](../reports/constant_c_energy_confirmation_20261109_zh.md)
completed its six new parameter/noise cases within the already exposed matched
linear family. All frozen future thresholds and noninferiority criteria passed;
both weak-coupling noisy cases exercised the bound. The [result review](constant_c_energy_confirmation_result_review_20261109_zh.md)
accepts this confirmation relative to integral LS with `KEEP_BASELINE`. The zero
baseline had lower visible errors in those two cases, and hidden-state recovery
still failed in all three weak-coupling cases. Authority
`constant_c_energy_confirmation_v1_20261109` is consumed. This is not external
independent validation. Further comparisons require their own design, exact
review and registration; the result does not start another execution.

The [encoder-referenced energy-ball comparison](../reports/kiki_energy_cap_transfer_20261110_zh.md)
completed all 774 fixed-A06 train slots with `REJECT / KEEP_BASELINE`:
`V=true`, the constraint was active in 666 slots, `G_I=true`, `G_E=false`,
and `D=false`. C improved the unconstrained initializer but retained roughly
eight times the encoder's future MSE and added two rule triggers per seed.
Authority `kiki_energy_cap_transfer_v1_20261109` is consumed.
This new evidence closes the tested post-hoc replacement of the encoder in
fixed A06 models under the registered history contract and energy-ball rule.
It does not test or rule out all integral initializers, joint training or
structural candidates. The empirical energy ball does not automatically
authorize differentiable training. Any follow-up needs an independent,
falsifiable mechanism, protocol and exact review.
The [result review](kiki_energy_cap_result_review_20261110_zh.md) records the
route decision. The [representative-window audit correction](../protocols/kiki_energy_cap_truth_window_audit_20261110_zh.md)
has its own saved-evidence scope and preserves the initial audit failure;
it does not reopen the scientific run.
Rejected candidates remain archived. Independent engineering extraction requires
explicit review.

See the [evidence review and constant-C 100 ms preparation](constant_c_100ms_preparation_20261109_zh.md)
for the distinction between discrete observation information, integral-estimator
error and model mismatch, and the items to freeze before a new registration.

## Records

- [Campaign history through 2026-11-09](campaign_history_through_20261109.md)
  preserves the approvals, budgets, decisions and evidence references migrated
  from the former root guide.
- [Protocols](../protocols/) contain registered experiment procedures.
- [Report index](../reports/README.md) links completed results and receipts.
