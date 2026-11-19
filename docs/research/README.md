# Research scope and records

Check the [research policy](../../configs/research_policy.json) for current
execution authority. Registered protocols define each experiment's data access,
budget, entry point and stopping gates.

## Current research scope

The [M1 first-swing protocol](../protocols/unified_first_swing_v1_zh.md)
tests the weighted rollout, one-sided speed and weak momentum objective on the
fixed T0M1 start with machine-local angle damping. The first paired fit follows
the synthetic check and scoped approval. The adaptive history initializer and
spatial coupling each have a separate paired comparison design.
The [GN prior and radius comparison](../reports/unified_first_swing_gn_decomposition_v3_20261119_zh.md)
records the complete zero-update comparison. Fixed priors with soft radii
detect all val instabilities in two configurations. The joint gate retains
the original initializer.

The [Topology–Memory Energy-Constrained ROM](topology_memory_energy_rom_zh.md)
defines the common storage, topology and memory states, candidate library,
deletion rules and phased validation for future explicit reduced models.
The first stage is the [complete train momentum and work diagnosis](../protocols/kiki_topology_memory_balance_v1_zh.md).
The [completed diagnosis](../reports/kiki_topology_memory_balance_20261116_zh.md)
isolates release from driven memory and records a positive separation residual
in 31 of the 33 witnessed unstable train events. The [completed S0–S4 comparison](../reports/topology_memory_rom_synthetic_20261116_zh.md)
recovers memory saturation in S2 and omits branch-dependent changes in coupling
strength in S1 and S3. The [fixed-coefficient history prior comparison](../protocols/topology_memory_rom_causal_v1_zh.md)
passes its [fixed-truth recovery gates](../reports/topology_memory_rom_causal_20261116_zh.md)
and exposes greater hidden-state error without the prior under noise. Full TASK3
training remains gated on structure recovery and rejection of out-of-library mismatch.
The [coupling comparison and follow-up](../reports/topology_memory_rom_coupling_strength_20261117_zh.md)
distinguish overall from branch-dependent changes with known coefficients after
improving the history solve. Joint hidden-state recovery remains below the
registered accuracy requirement when both changes occur.

## Prior topology and memory studies

The [dynamic effective coupling plan](machine_local_dynamic_coupling_plan_20261116_zh.md)
tests a shared state that varies branch stiffness through an energy gradient.
The [first protocol](../protocols/machine_local_topology_dynamic_coupling_v1_zh.md)
compares it with a static restoring term at equal fitting budgets and uses the
same 100 ms causal history contract. Phase-dependent damping remains a separate
candidate in the momentum plan.
The [synthetic comparison](../reports/machine_local_topology_dynamic_coupling_20261116_zh.md)
passes all three primary regimes. The [complete TASK3 train diagnosis](../reports/kiki_machine_local_topology_dynamic_coupling_20261116_zh.md)
finds that fixed common-mode dynamics increase unstable momentum residuals
under both damping references and all three memory controls. The registered
comparison closes at the observed-path gate.
The [zero-sum spatial coupling protocol](../protocols/machine_local_topology_spatial_coupling_v1_zh.md)
compares bounded redistribution of branch log stiffness against common-scale
change and fixed coupling on the same complete synthetic cohorts.
The [first zero-sum comparison](../reports/machine_local_topology_spatial_coupling_20261116_zh.md)
selects the generating spatial rate and fails three separation gates. The
[causal history control](../reports/machine_local_topology_spatial_coupling_causal_20261116_zh.md)
passes the separation thresholds and recovers the synthetic memory on all
train and val cases at both step sizes.
The [complete TASK3 train diagnosis](../reports/kiki_machine_local_topology_spatial_coupling_20261116_zh.md)
finds lower overall residuals and closes at the unstable-group gate. Under
spatial damping, the observed-path correction has negative alignment with the
remaining unstable residual after free memory projection.
The [paired spatial relaxation protocol](../protocols/kiki_machine_local_topology_spatial_relaxation_v1_zh.md)
learns machine-specific nonnegative relaxation rates on the complete TASK3 train
roster from a shared starting checkpoint, with frozen val and test scoring.
The [complete paired result](../reports/kiki_machine_local_topology_spatial_relaxation_20261116_zh.md)
shows lower train loss after 18 updates, with both models selecting update 0
on val. Full test yields 70 correct, including 3 of 9 unstable cases, and two
unresolved cases.

The [topology and memory plan](machine_local_topology_memory_plan_20261115_zh.md)
introduces event-conditioned edge stiffness and reciprocal machine-local memory.
The [S0 protocol](../protocols/machine_local_topology_v1_zh.md) tests four known-
coefficient combinations and the 100 ms initialization before shared-parameter
learning and function selection. The [S0 evidence](../reports/machine_local_topology_s0_20261115_zh.md)
passes the engineering checks for clean synthetic histories. The
[shared-coefficient comparison](../reports/machine_local_topology_learning_20261115_zh.md)
recovers both no-memory models and identifies recovery gaps in the two memory
models. The [origin-state diagnosis](../reports/machine_local_topology_diagnosis_20261115_zh.md)
finds that the combined model's eight-second val error is more sensitive to the
coupling correction than to the memory state. Subsequent [long-horizon refinement](../reports/machine_local_topology_refinement_20261115_zh.md)
passes the trajectory, effective-stiffness and memory-kernel recovery gates for
both fixed supports on the original clean synthetic rosters. The [independent
excitation check](../reports/machine_local_topology_transfer_20261115_zh.md) passes
all matched gates on fresh stable and unstable events. All 16 combinations
classify the designed excitations correctly. Closed-loop errors distinguish the
frozen equations on the memory truth systems. The [full TASK3 comparison](../reports/kiki_machine_local_topology_20261115_zh.md)
completed train 346, val 107 and test 76 for both frozen equations. Both models
classified 67 stable test events correctly and detected zero of nine unstable
events. The six saved forecasts were independently rescored. The objective
remains full-cohort test stability accuracy above 90%. On val, each model detected
8 of 12 unstable events, with three classified stable and one unresolved.
The [memory radius comparison](../reports/kiki_machine_local_topology_radius_20261115_zh.md)
isolated the initialization radius on the same frozen T0M1 equation and completed
all three rosters. Doubling the radius yielded 70 correct test judgments, with
three unresolved. The val guard failed on the unresolved count and the eight-second
stable trajectory error. Both the evaluation and saved-forecast audit are closed.
The [five-point witness audit](../reports/kiki_task3_train_witness_20261115_zh.md)
found a physical witness within the available three-second interval in all 35
officially unstable train events. Each radius reproduced it in six events in the
matched interval. A CUDA gradient check passed through the existing history
initializer and rollout. The [paired known-truth training check](../reports/machine_local_topology_margin_20261115_zh.md)
completed both twenty-update fits and all 24 synthetic events. The margin term
changed the val unstable margin RMSE by 0.041% relative to the trajectory-only
fit. Both fits recovered all matched triggers and classified the six val events.
The margin improvement gate failed. The subsequent [delayed-excitation check](../reports/machine_local_topology_delayed_20261115_zh.md)
reproduced six matched train misses at the shared start and measured both
loss gradients. Norm calibration changed the first Adam update substantially
while improving its projected margin descent by 0.16%. Neither twenty-update
fit recovered a matched witness. In the [seven-condition diagnosis](../reports/machine_local_topology_delayed_origin_20261115_zh.md),
damping substitution recovered every matched train and val witness at a
fixed true origin. The [causal damping fit](../reports/machine_local_topology_delayed_damping_20261115_zh.md)
recovered these witnesses from visible train trajectories and passed the
synthetic gate. The [full TASK3 damping comparison](../reports/kiki_machine_local_topology_damping_20261115_zh.md)
reduced the short-horizon train objective by 18.85%. Val instability detection
fell from 8/12 to 1/12, and the full test accuracy remained unresolved. All
three rosters and saved-prediction checks are complete. The next candidate is the
[ball-constrained history solve](machine_local_topology_ball_history_plan_20261115_zh.md)
on the original T0M1 equation. The [48-case synthetic comparison](../reports/machine_local_topology_ball_history_20261115_zh.md)
recovered the feasible clean states with zero prior. The original-prior condition
failed the trajectory and active-bound gates. The [matched no-prior protocol](../protocols/machine_local_topology_ball_history_no_prior_v1_zh.md)
compares the exact and radial updates with the same frozen zero prior. Its
[results](../reports/machine_local_topology_ball_history_no_prior_20261115_zh.md)
show lower constrained history error and higher forecast error with the exact
solve. Removing the prior within the radial method improves the matched
train and val trajectories. The [fixed-weight TASK3 comparison](../reports/kiki_machine_local_topology_no_prior_20261115_zh.md)
completed all three real cohorts and independent score recomputation. Val had
one unresolved case and test had four. Train reproduced five of 35 matched
instability windows. The [momentum-consistency diagnosis](machine_local_topology_momentum_plan_20261115_zh.md)
isolates signed force errors along observed paths. The [known-truth controls](../reports/machine_local_topology_momentum_20261115_zh.md)
recovered stiffness and damping perturbations. The [complete TASK3 train diagnosis](../reports/kiki_machine_local_topology_momentum_20261115_zh.md)
selected a positive damping factor of 0.3436767365, reducing the projected
residual SSE in stable and unstable events. The [frozen damping comparison](../reports/kiki_machine_local_topology_momentum_damping_20261115_zh.md)
reproduced 10 of 35 matched train instability windows, with five stable false
alarms. Val classified all 107 events, with 105 correct and one stable false
alarm. The eight-second val error on the 95 stable events increased from 0.218
to 0.464. The full test returned 71 correct judgments and two unresolved cases.
The val guard failed, the baseline was retained and independent rescoring
confirmed every saved judgment.
The [history-only origin transfer check](../reports/kiki_machine_local_topology_origin_transfer_20261115_zh.md)
kept the original equation fixed. Removing the history bound reduced subsequent
observed-path momentum SSE by 41.92% on unstable train events and increased it
by 33.54% on stable train events. The sampling and transfer gates failed, so
this comparison supplies no basis to relax the bound globally.
The [paired angle-damping check](../reports/machine_local_topology_angle_damping_20261115_zh.md)
recovered the known coefficient and passed trajectory and stability gates.
The [full TASK3 angle-damping comparison](../reports/kiki_machine_local_topology_angle_damping_20261115_zh.md)
used the complete train estimate at the frozen upper bound. Val classified
106/107 correctly with no stable false alarm and passed its trajectory gate.
Three train false alarms account for 99.55% of the increase in the mean stable
eight-second error. Test returned 69 correct judgments with three unresolved.
The complete rosters have been independently rescored. The [damping scale check](../reports/kiki_machine_local_topology_angle_scale_20261115_zh.md)
qualified all 346 train cases. At fixed causal origins and after free-origin
projection, increasing the scale from 1 to 1.25 rad raised residuals in both
stable and unstable train events. The change failed its registered gate.
The [synthetic spatial damping check](../reports/machine_local_topology_spatial_damping_20261115_zh.md)
recovered both reference distributions and separated spatial allocation from
the trace-matched control. The [full TASK3 spatial comparison](../reports/kiki_machine_local_topology_spatial_damping_20261115_zh.md)
retained 106/107 correct on val with zero stable false alarms and a six-term
score of 0.42674. Test returned 70 correct judgments and two unresolved, all
independently rescored. The [fixed modal probe](../reports/kiki_machine_local_topology_modal_damping_20261115_zh.md)
qualified all train cases. Adding the fixed reference-stiffness mode increased
stable residuals at both causal starts and under free-memory projection.
The [paired restoring-force check](../reports/machine_local_topology_restoring_shape_20261115_zh.md)
recovered signed coefficients on the spatial damping model while preserving
the equilibrium and linearized stiffness. The positive and negative known
systems passed the complete causal trajectory checks.
The [full TASK3 restoring-force comparison](../reports/kiki_machine_local_topology_restoring_shape_20261115_zh.md)
passed both train and val trajectory guards. Val classified 104/107 with
nine detected unstable events. Test classified 70 correctly with two
unresolved. All 529 saved predictions passed independent scoring. The
spatial damping anchor retains the stronger val stability and six-term score.
The [momentum plan](machine_local_topology_momentum_plan_20261115_zh.md#按摆动方向调节阻尼)
defines the next train mechanism check.

The [coefficient substitutions](../reports/machine_local_memory_diagnosis_20261114_zh.md)
showed compensation between the fitted visible and memory coefficients across
all three known-truth train rosters. The subsequent
[momentum calibration](../reports/machine_local_memory_calibration_20261114_zh.md)
passed all fixed-support recovery gates.
[Independent result review](machine_local_memory_calibration_result_review_20261114.json).
The [TASK3 momentum comparison](../reports/kiki_machine_local_memory_20261114_zh.md)
completed the full train, val and test rosters. All four fits stopped at a
rank-deficient nuisance block. The selected models share the same equation,
with all ten port amplitudes at the coefficient floor and degraded trajectory
and stability scores. The result is `KEEP_BASELINE`.
[Independent result review](kiki_machine_local_memory_result_review_20261114.json).

The [paired R20/R21 report](../reports/kiki_machine_local_r20_r21_20261114_zh.md)
covers both models on all 346 train, 107 val and 76 test events. Both selected
update 80 by the registered full-val rule and reached the update cap while the
val objective was still improving. On the fixed test roster, R21 had lower
eight-second trajectory error and lower stability accuracy among classifiable
cases than R20. The [scientific review](kiki_machine_local_r20_r21_result_review_20261114.json)
records `KEEP_BASELINE`. The saved-array check found zero discrepancies. Fit,
test and saved-audit authorities are consumed.

The user's full-dataset instruction now governs the first evaluation of every
real-data round. The [full TASK3 L10 plan](kiki_machine_local_full_plan_20261113_zh.md)
and [execution protocol](../protocols/kiki_machine_local_full_v1_zh.md) specify
one forty-update fit on all 346 train events, followed by all 107 val and 76 test
events. Each update covers the complete train roster through gradient accumulation.
The test stage binds the saved model and preprocessing after fitting closes.
All roles report trajectory errors and stability judgments regardless of RMSE.
The [completed full-cohort report](../reports/kiki_machine_local_full_20261113_zh.md)
now contains all 529 predictions and their stability scores. Saved-array
recomputation found zero discrepancies. Fit, test and saved-audit authorities
are consumed. Train-only diagnosis found that the observed-history residual
does not establish the cause of the future error. The next preparation separates
initialization from dynamics with synthetic true-hidden-state controls.
That [complete 64-case comparison](../reports/machine_local_hidden_oracle_20261113_zh.md)
is now finished: matched dynamics recover the hidden state, while fixed stiffness
mismatch is partly absorbed by the initializer and changes future errors in both
directions. All four groups complete train/val/test with full stability coverage;
saved-array recomputation has zero discrepancies. Its scientific and saved-audit
authorities are consumed. The [shared stiffness calibration](../reports/machine_local_stiffness_calibration_20261114_zh.md)
recovered the known scalar after 11 full train objectives. The fitted model
passed hidden-state recovery and trajectory checks on all 64 synthetic events.
The [paired design](machine_local_r20_r21_preparation_20261114_zh.md)
established a matched start and budget for the 20- and 21-parameter models.
The [joint-equation plan](machine_local_joint_equation_plan_20261114_zh.md)
extends synthetic checks to 74 original-equation coefficients and separate
harmonic-potential and modulated-damping controls. The [first protocol](../protocols/machine_local_equation_v1_zh.md)
used causal-history training at 0.5 s and 1 s with formal CUDA inference.

The [machine-local initializer diagnosis](../reports/machine_local_initializer_diagnosis_20261112_zh.md)
passed its numerical and saved-evidence checks. Its route is
`INITIALIZER_LIMITATION_PRIORITY`: the exact conditional encoder still fails
the val target under true dynamics. Its authority and saved audit are consumed.
The [discrete-history observer](../reports/machine_local_observer_20261112_zh.md)
recovered clean hidden states, but failed the 1% noise gate. Its run and saved
arithmetic audit are consumed. The [joint initial-state correction](../reports/machine_local_joint_observer_20261112_zh.md)
reduced paired train/val errors, while fresh noisy cases still failed the target.
Its saved-array audit and scientific review are complete; that authority is consumed.
The [Gaussian information diagnosis](../reports/machine_local_gaussian_oracle_20261112_zh.md)
passed technical and saved-array checks. Under its frozen generating distribution,
the local 100 ms noisy conditional risk exceeds the original target; the same-posterior
visible reset adds risk. This one-use authority and saved audit are consumed.
The user's 2026-11-13 update prioritizes quick clean predictions and complete
val/test evaluation after every frozen training round, regardless of RMSE.
The [clean learning round](../reports/machine_local_clean_learning_20261113_zh.md)
completed one 240-update fit of twenty machine-local port/time-scale parameters
through a differentiable 100 ms history solver. Truth, initial and final models
all received train/val/test scores. Numerical, gradient, truth-control, accuracy
and saved-array checks passed; the experiment and saved audit are consumed.
The [result review](machine_local_clean_learning_result_review_20261113_zh.md)
sets the next scope: verify the required clean nonlinear history interface and
its gradients, then prepare the TASK3 real-data pilot. The current linear
response solver is not an exact initializer for the sine backbone.
The [clean nonlinear history protocol](../protocols/machine_local_nonlinear_history_v1_zh.md)
completed six fixed cases and two parameter controls. Nonlinear recovery,
gradients and numerical gates passed; both controls received all train/val/test
trajectory and stability scores. The [report](../reports/machine_local_nonlinear_history_20261113_zh.md)
and [result review](machine_local_nonlinear_history_result_review_20261113_zh.md)
retain the saved-audit schema failure and its JSON-only reconciliation. No arrays
were reread for that correction. The next registered comparison uses the full
TASK3 cohorts as specified above. This no-fit S0 establishes the nonlinear
interface and gradients; real training convergence is evaluated in the new round.
The [statistical noise plan](machine_local_statistical_initializer_plan_20261112_zh.md)
is deferred. Existing noise results and consumed scopes remain unchanged.
Future real fits use TASK3's exact frozen train 346 / val 107 / test 76 cohorts;
accuracy gates govern promotion, not whether to report held-out predictions.

[Machine-local S0](../reports/machine_local_s0_20261112_zh.md) passed all 112
frozen checks. The [S1-base comparison](../reports/machine_local_s1_20261112_zh.md)
completed all twelve fits and its saved arithmetic audit. Observation, physical
and numerical checks passed; all six required prediction pairs failed. Both
one-use authorities and saved audits are consumed: REJECT / KEEP_BASELINE /
STOP_T0. The [result review](machine_local_s1_result_review_20261112_zh.md)
prioritized the now-completed initializer diagnosis; its approved review and
saved arithmetic audit are linked above.

The frozen [base research plan v2](machine_local_structure_plan_v2_20261112_zh.md)
implements ten machine-local hidden states before learning physical modules.
Its first comparison separates model order (G2/G10) from local port structure
(G10/L10), with the unchanged A06 as the performance reference. Initialization,
history propagation and future rollout use each candidate's own causal model.
The [synthetic protocol](../protocols/machine_local_synthetic_v1_zh.md) defines
S0 correctness followed by S1-base: seven observation controls and twelve fixed
fits. The user's 2026-11-12 instruction authorizes this staged implementation
and autonomous advancement through scientific gates. Each stage still requires
its new registered budget and exact implementation review; failed gates stop
downstream work. The [original design review](machine_local_structure_review_20261112_zh.md)
remains bound to its earlier preparation document.
The [full-proposal assessment](machine_local_full_proposal_assessment_20261112_zh.md)
now reviews the latest linked conversation and its complete attachment. It
supports the direction and specifies corrections to dimension-dependent encoder
regularization, deletion order, trainable parameters and execution details.
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

The current preparation scope follows the completed diagnosis route `INITIALIZER_LIMITATION_PRIORITY`.
The frozen v2 design and protocol remain the original execution contracts.
Prepare the targeted initialization correction and new synthetic confirmation
under a separately reviewed, registered scope before executing either. B0 remains an independent initializer candidate,
as originally recommended by the A1 result review; the user's 2026-11-12 request
changes the prospective priority. Potential, input and time-objective changes
remain separate. Real train work requires new learned-model synthetic confirmation and its own exact
registration. A1 retuning, source val/test access and model promotion are outside
this synthetic scope. Keep the [A1 protocol](../protocols/kiki_state_port_train_v1_zh.md),
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
