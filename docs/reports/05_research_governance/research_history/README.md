# Research scope and records

[返回主题页](../README.md)

## 复核与历史版本

<a id="doc-0a640a72f9b4"></a>

<details>
<summary>Research scope and records</summary>

<a id="doc-0a640a72f9b4-research-scope-and-records"></a>

### Research scope and records

Check the [research policy](../../../../configs/research_policy.json) for current
execution authority. Registered protocols define each experiment's data access,
budget, entry point and stopping gates.


<a id="doc-0a640a72f9b4-current-research-scope"></a>

#### Current research scope

The [M1 first-swing protocol](../../10_postfault_structure/m1_first_swing/README.md#doc-41f0fbeb5bd9)
tests the weighted rollout, one-sided speed and weak momentum objective on the
fixed T0M1 start with machine-local angle damping. The first paired fit follows
the synthetic check and scoped approval. The adaptive history initializer and
spatial coupling each have a separate paired comparison design.
The [GN prior and radius comparison](../../10_postfault_structure/m1_gn_decomposition/README.md#doc-ee04bd7e4647)
records the complete zero-update comparison. Fixed priors with soft radii
detect all val instabilities in two configurations. The joint gate retains
the original initializer.
The [V4 paired fit and frozen comparison](../../10_postfault_structure/m1_v4/README.md#doc-6c1f0e070a2c)
records the matched 18-update fits and the complete fixed test. Both fits
select update zero. The soft initializer detects 6/9 test instabilities.

[M3 统一结构发现](../../10_postfault_structure/m3/README.md#doc-3583b3e069a1)完成
54 至 74 维课程和五个物理模块的删块实验。G2 骨架的 `val` 检出为 12/12，
2 个稳定误报超过正式上限，结论为 `KEEP_BASELINE`。`test` 轨迹读取为 0。

[M4 故障后物理一致性](../../10_postfault_structure/m4_m3b/README.md#doc-82d48d3bb40e)用封存预测和观测比较
M3 与原硬 R 模型。共同匹配的主模态频率误差中位数从 0.05023 降至 0.03292 Hz。
[M3b 三组拟合](../../10_postfault_structure/m4_m3b/README.md#doc-42c15fb6425a)在正式误报上限为 2 的规则下冻结单 G2，
参数与 M3 诊断快照相同。固定 `test` 使用独立的一次性授权。
[固定 test 回执](../../10_postfault_structure/m4_m3b/README.md#doc-efc3aa20abd9)记录
单 G2 的 5/9 确认检出、零稳定误报和 2 例未决，V4 软 2R 在同一名单检出 6/9。

[M4 v2 惯量加权与储能分解](../../10_postfault_structure/m4_v2/README.md#doc-9d52b20bcffe)记录慢同调检验和两档步长的能量结果。[储能箱线图与附记](../../10_postfault_structure/m4_v2/README.md#doc-b9005f59a5ec)给出五项储能的稳定和失稳分布。

[M5 独立盲测](../../10_postfault_structure/m5_blind/README.md#doc-a2fdc902d881)显示冻结 M3b G2 在同源新仿真的 32 个失稳事件中确认检出 29 个，62 个稳定事件全部判稳。

The [Topology–Memory Energy-Constrained ROM](../../09_topology_memory/synthetic_rom/README.md#doc-4d23250443a3)
defines the common storage, topology and memory states, candidate library,
deletion rules and phased validation for future explicit reduced models.
The first stage is the [complete train momentum and work diagnosis](../../09_topology_memory/task3_momentum_work/README.md#doc-ccb939b6dee0).
The [completed diagnosis](../../09_topology_memory/task3_momentum_work/README.md#doc-7bb602349804)
isolates release from driven memory and records a positive separation residual
in 31 of the 33 witnessed unstable train events. The [completed S0–S4 comparison](../../09_topology_memory/synthetic_rom/README.md#doc-00055d87ebb1)
recovers memory saturation in S2 and omits branch-dependent changes in coupling
strength in S1 and S3. The [fixed-coefficient history prior comparison](../../09_topology_memory/causal_history/README.md#doc-3a741c25d3df)
passes its [fixed-truth recovery gates](../../09_topology_memory/causal_history/README.md#doc-c9e56591a378)
and exposes greater hidden-state error without the prior under noise. Full TASK3
training remains gated on structure recovery and rejection of out-of-library mismatch.
The [coupling comparison and follow-up](../../09_topology_memory/coupling_strength/README.md#doc-eae8ebc9bb21)
distinguish overall from branch-dependent changes with known coefficients after
improving the history solve. Joint hidden-state recovery remains below the
registered accuracy requirement when both changes occur.


<a id="doc-0a640a72f9b4-prior-topology-and-memory-studies"></a>

#### Prior topology and memory studies

The [dynamic effective coupling plan](../../08_machine_local_topology/synthetic_dynamic_coupling/README.md#doc-393bc886e31e)
tests a shared state that varies branch stiffness through an energy gradient.
The [first protocol](../../08_machine_local_topology/synthetic_dynamic_coupling/README.md#doc-931303070641)
compares it with a static restoring term at equal fitting budgets and uses the
same 100 ms causal history contract. Phase-dependent damping remains a separate
candidate in the momentum plan.
The [synthetic comparison](../../08_machine_local_topology/synthetic_dynamic_coupling/README.md#doc-c90af5eec437)
passes all three primary regimes. The [complete TASK3 train diagnosis](../../08_machine_local_topology/task3_dynamic_coupling/README.md#doc-efa81b847256)
finds that fixed common-mode dynamics increase unstable momentum residuals
under both damping references and all three memory controls. The registered
comparison closes at the observed-path gate.
The [zero-sum spatial coupling protocol](../../08_machine_local_topology/synthetic_spatial_coupling/README.md#doc-284356f266b1)
compares bounded redistribution of branch log stiffness against common-scale
change and fixed coupling on the same complete synthetic cohorts.
The [first zero-sum comparison](../../08_machine_local_topology/synthetic_spatial_coupling/README.md#doc-68f44e7dfc76)
selects the generating spatial rate and fails three separation gates. The
[causal history control](../../08_machine_local_topology/synthetic_spatial_coupling_causal/README.md#doc-1d7dcc7de0aa)
passes the separation thresholds and recovers the synthetic memory on all
train and val cases at both step sizes.
The [complete TASK3 train diagnosis](../../08_machine_local_topology/task3_spatial_coupling/README.md#doc-d5aa05bf148f)
finds lower overall residuals and closes at the unstable-group gate. Under
spatial damping, the observed-path correction has negative alignment with the
remaining unstable residual after free memory projection.
The [paired spatial relaxation protocol](../../08_machine_local_topology/task3_spatial_relaxation/README.md#doc-1b547e9cbe55)
learns machine-specific nonnegative relaxation rates on the complete TASK3 train
roster from a shared starting checkpoint, with frozen val and test scoring.
The [complete paired result](../../08_machine_local_topology/task3_spatial_relaxation/README.md#doc-0ea75e26ad8a)
shows lower train loss after 18 updates, with both models selecting update 0
on val. Full test yields 70 correct, including 3 of 9 unstable cases, and two
unresolved cases.

The [topology and memory plan](../../08_machine_local_topology/synthetic_topology/README.md#doc-415956346cb1)
introduces event-conditioned edge stiffness and reciprocal machine-local memory.
The [S0 protocol](../../08_machine_local_topology/synthetic_topology/README.md#doc-375c9059d7e6) tests four known-
coefficient combinations and the 100 ms initialization before shared-parameter
learning and function selection. The [S0 evidence](../../08_machine_local_topology/synthetic_topology/README.md#doc-649d64e1abed)
passes the engineering checks for clean synthetic histories. The
[shared-coefficient comparison](../../08_machine_local_topology/synthetic_learning/README.md#doc-07351396eae5)
recovers both no-memory models and identifies recovery gaps in the two memory
models. The [origin-state diagnosis](../../08_machine_local_topology/synthetic_diagnosis/README.md#doc-c43e5c12eeea)
finds that the combined model's eight-second val error is more sensitive to the
coupling correction than to the memory state. Subsequent [long-horizon refinement](../../08_machine_local_topology/synthetic_refinement/README.md#doc-096ac12ed14d)
passes the trajectory, effective-stiffness and memory-kernel recovery gates for
both fixed supports on the original clean synthetic rosters. The [independent
excitation check](../../08_machine_local_topology/synthetic_transfer/README.md#doc-1530b2d8d992) passes
all matched gates on fresh stable and unstable events. All 16 combinations
classify the designed excitations correctly. Closed-loop errors distinguish the
frozen equations on the memory truth systems. The [full TASK3 comparison](../../08_machine_local_topology/task3_topology_memory/README.md#doc-00b7fd2d6112)
completed train 346, val 107 and test 76 for both frozen equations. Both models
classified 67 stable test events correctly and detected zero of nine unstable
events. The six saved forecasts were independently rescored. The objective
remains full-cohort test stability accuracy above 90%. On val, each model detected
8 of 12 unstable events, with three classified stable and one unresolved.
The [memory radius comparison](../../08_machine_local_topology/task3_radius/README.md#doc-699c87b431dc)
isolated the initialization radius on the same frozen T0M1 equation and completed
all three rosters. Doubling the radius yielded 70 correct test judgments, with
three unresolved. The val guard failed on the unresolved count and the eight-second
stable trajectory error. Both the evaluation and saved-forecast audit are closed.
The [five-point witness audit](../../08_machine_local_topology/task3_witness/README.md#doc-595f500c1fc5)
found a physical witness within the available three-second interval in all 35
officially unstable train events. Each radius reproduced it in six events in the
matched interval. A CUDA gradient check passed through the existing history
initializer and rollout. The [paired known-truth training check](../../08_machine_local_topology/synthetic_margin/README.md#doc-c04b4d708da9)
completed both twenty-update fits and all 24 synthetic events. The margin term
changed the val unstable margin RMSE by 0.041% relative to the trajectory-only
fit. Both fits recovered all matched triggers and classified the six val events.
The margin improvement gate failed. The subsequent [delayed-excitation check](../../08_machine_local_topology/synthetic_delayed/README.md#doc-81598fbdd211)
reproduced six matched train misses at the shared start and measured both
loss gradients. Norm calibration changed the first Adam update substantially
while improving its projected margin descent by 0.16%. Neither twenty-update
fit recovered a matched witness. In the [seven-condition diagnosis](../../08_machine_local_topology/synthetic_delayed_origin/README.md#doc-f20b87ad9d23),
damping substitution recovered every matched train and val witness at a
fixed true origin. The [causal damping fit](../../08_machine_local_topology/synthetic_delayed_damping/README.md#doc-f783ac8a331d)
recovered these witnesses from visible train trajectories and passed the
synthetic gate. The [full TASK3 damping comparison](../../08_machine_local_topology/task3_damping/README.md#doc-3a3b55f16410)
reduced the short-horizon train objective by 18.85%. Val instability detection
fell from 8/12 to 1/12, and the full test accuracy remained unresolved. All
three rosters and saved-prediction checks are complete. The next candidate is the
[ball-constrained history solve](../../08_machine_local_topology/synthetic_ball_history/README.md#doc-dc5120c0b605)
on the original T0M1 equation. The [48-case synthetic comparison](../../08_machine_local_topology/synthetic_ball_history/README.md#doc-038fc6f10b21)
recovered the feasible clean states with zero prior. The original-prior condition
failed the trajectory and active-bound gates. The [matched no-prior protocol](../../08_machine_local_topology/synthetic_ball_history_no_prior/README.md#doc-25bef5681e95)
compares the exact and radial updates with the same frozen zero prior. Its
[results](../../08_machine_local_topology/synthetic_ball_history_no_prior/README.md#doc-6017e23c1811)
show lower constrained history error and higher forecast error with the exact
solve. Removing the prior within the radial method improves the matched
train and val trajectories. The [fixed-weight TASK3 comparison](../../08_machine_local_topology/task3_no_prior/README.md#doc-effc1602972f)
completed all three real cohorts and independent score recomputation. Val had
one unresolved case and test had four. Train reproduced five of 35 matched
instability windows. The [momentum-consistency diagnosis](../../08_machine_local_topology/synthetic_momentum/README.md#doc-f2ac6b8b3491)
isolates signed force errors along observed paths. The [known-truth controls](../../08_machine_local_topology/synthetic_momentum/README.md#doc-fc27b23f4e43)
recovered stiffness and damping perturbations. The [complete TASK3 train diagnosis](../../08_machine_local_topology/task3_momentum/README.md#doc-e34f7a9a3e3d)
selected a positive damping factor of 0.3436767365, reducing the projected
residual SSE in stable and unstable events. The [frozen damping comparison](../../08_machine_local_topology/task3_momentum_damping/README.md#doc-97c1fbb8b406)
reproduced 10 of 35 matched train instability windows, with five stable false
alarms. Val classified all 107 events, with 105 correct and one stable false
alarm. The eight-second val error on the 95 stable events increased from 0.218
to 0.464. The full test returned 71 correct judgments and two unresolved cases.
The val guard failed, the baseline was retained and independent rescoring
confirmed every saved judgment.
The [history-only origin transfer check](../../08_machine_local_topology/task3_origin_transfer/README.md#doc-eab8f261c2ca)
kept the original equation fixed. Removing the history bound reduced subsequent
observed-path momentum SSE by 41.92% on unstable train events and increased it
by 33.54% on stable train events. The sampling and transfer gates failed, so
this comparison supplies no basis to relax the bound globally.
The [paired angle-damping check](../../08_machine_local_topology/synthetic_angle_damping/README.md#doc-4a85dfb8b0af)
recovered the known coefficient and passed trajectory and stability gates.
The [full TASK3 angle-damping comparison](../../08_machine_local_topology/task3_angle_damping/README.md#doc-648afba6d2b5)
used the complete train estimate at the frozen upper bound. Val classified
106/107 correctly with no stable false alarm and passed its trajectory gate.
Three train false alarms account for 99.55% of the increase in the mean stable
eight-second error. Test returned 69 correct judgments with three unresolved.
The complete rosters have been independently rescored. The [damping scale check](../../08_machine_local_topology/task3_angle_scale/README.md#doc-71ece58bcee4)
qualified all 346 train cases. At fixed causal origins and after free-origin
projection, increasing the scale from 1 to 1.25 rad raised residuals in both
stable and unstable train events. The change failed its registered gate.
The [synthetic spatial damping check](../../08_machine_local_topology/synthetic_spatial_damping/README.md#doc-8c00922d6f72)
recovered both reference distributions and separated spatial allocation from
the trace-matched control. The [full TASK3 spatial comparison](../../08_machine_local_topology/task3_spatial_damping/README.md#doc-08e5190828ec)
retained 106/107 correct on val with zero stable false alarms and a six-term
score of 0.42674. Test returned 70 correct judgments and two unresolved, all
independently rescored. The [fixed modal probe](../../08_machine_local_topology/task3_modal_damping/README.md#doc-171beda6b4d6)
qualified all train cases. Adding the fixed reference-stiffness mode increased
stable residuals at both causal starts and under free-memory projection.
The [paired restoring-force check](../../08_machine_local_topology/synthetic_restoring_shape/README.md#doc-c576ecd78c23)
recovered signed coefficients on the spatial damping model while preserving
the equilibrium and linearized stiffness. The positive and negative known
systems passed the complete causal trajectory checks.
The [full TASK3 restoring-force comparison](../../08_machine_local_topology/task3_restoring_shape/README.md#doc-bc0a87fbb260)
passed both train and val trajectory guards. Val classified 104/107 with
nine detected unstable events. Test classified 70 correctly with two
unresolved. All 529 saved predictions passed independent scoring. The
spatial damping anchor retains the stronger val stability and six-term score.
The [momentum plan](../../08_machine_local_topology/synthetic_momentum/README.md#doc-f2ac6b8b3491-%E6%8C%89%E6%91%86%E5%8A%A8%E6%96%B9%E5%90%91%E8%B0%83%E8%8A%82%E9%98%BB%E5%B0%BC)
defines the next train mechanism check.

The [coefficient substitutions](../../08_machine_local_topology/memory_diagnosis/README.md#doc-d7906695b3c4)
showed compensation between the fitted visible and memory coefficients across
all three known-truth train rosters. The subsequent
[momentum calibration](../../08_machine_local_topology/memory_calibration/README.md#doc-4d938b5af3d2)
passed all fixed-support recovery gates.
[Independent result review](../../08_machine_local_topology/memory_calibration/_sealed/machine_local_memory_calibration_result_review_20261114.json).
The [TASK3 momentum comparison](../../08_machine_local_topology/task3_baseline_memory/README.md#doc-703c5b791810)
completed the full train, val and test rosters. All four fits stopped at a
rank-deficient nuisance block. The selected models share the same equation,
with all ten port amplitudes at the coefficient floor and degraded trajectory
and stability scores. The result is `KEEP_BASELINE`.
[Independent result review](../../08_machine_local_topology/task3_baseline_memory/_sealed/kiki_machine_local_memory_result_review_20261114.json).

The [paired R20/R21 report](../../08_machine_local_topology/task3_r20_r21/README.md#doc-f2670f1dcdb3)
covers both models on all 346 train, 107 val and 76 test events. Both selected
update 80 by the registered full-val rule and reached the update cap while the
val objective was still improving. On the fixed test roster, R21 had lower
eight-second trajectory error and lower stability accuracy among classifiable
cases than R20. The [scientific review](../../08_machine_local_topology/task3_r20_r21/_sealed/kiki_machine_local_r20_r21_result_review_20261114.json)
records `KEEP_BASELINE`. The saved-array check found zero discrepancies. Fit,
test and saved-audit authorities are consumed.

The user's full-dataset instruction now governs the first evaluation of every
real-data round. The [full TASK3 L10 plan](../../08_machine_local_topology/task3_baseline/README.md#doc-a0cab37b439d)
and [execution protocol](../../08_machine_local_topology/task3_baseline/README.md#doc-f1d0f91e9a6f) specify
one forty-update fit on all 346 train events, followed by all 107 val and 76 test
events. Each update covers the complete train roster through gradient accumulation.
The test stage binds the saved model and preprocessing after fitting closes.
All roles report trajectory errors and stability judgments regardless of RMSE.
The [completed full-cohort report](../../08_machine_local_topology/task3_baseline/README.md#doc-15de24deafe5)
now contains all 529 predictions and their stability scores. Saved-array
recomputation found zero discrepancies. Fit, test and saved-audit authorities
are consumed. Train-only diagnosis found that the observed-history residual
does not establish the cause of the future error. The next preparation separates
initialization from dynamics with synthetic true-hidden-state controls.
That [complete 64-case comparison](../../08_machine_local_topology/hidden_oracle/README.md#doc-64e9339f5396)
is now finished: matched dynamics recover the hidden state, while fixed stiffness
mismatch is partly absorbed by the initializer and changes future errors in both
directions. All four groups complete train/val/test with full stability coverage;
saved-array recomputation has zero discrepancies. Its scientific and saved-audit
authorities are consumed. The [shared stiffness calibration](../../08_machine_local_topology/stiffness_calibration/README.md#doc-196df5f42665)
recovered the known scalar after 11 full train objectives. The fitted model
passed hidden-state recovery and trajectory checks on all 64 synthetic events.
The [paired design](../../08_machine_local_topology/task3_r20_r21/README.md#doc-a8fd338fba84)
established a matched start and budget for the 20- and 21-parameter models.
The [joint-equation plan](../../08_machine_local_topology/equation/README.md#doc-052b12bfad13)
extends synthetic checks to 74 original-equation coefficients and separate
harmonic-potential and modulated-damping controls. The [first protocol](../../08_machine_local_topology/equation/README.md#doc-f8672d277241)
used causal-history training at 0.5 s and 1 s with formal CUDA inference.

The [machine-local initializer diagnosis](../../08_machine_local_topology/initializer_diagnosis/README.md#doc-d1ec1a06fe71)
passed its numerical and saved-evidence checks. Its route is
`INITIALIZER_LIMITATION_PRIORITY`: the exact conditional encoder still fails
the val target under true dynamics. Its authority and saved audit are consumed.
The [discrete-history observer](../../08_machine_local_topology/observer/README.md#doc-333ce3562b94)
recovered clean hidden states, but failed the 1% noise gate. Its run and saved
arithmetic audit are consumed. The [joint initial-state correction](../../08_machine_local_topology/joint_observer/README.md#doc-5a351c33b6b1)
reduced paired train/val errors, while fresh noisy cases still failed the target.
Its saved-array audit and scientific review are complete; that authority is consumed.
The [Gaussian information diagnosis](../../08_machine_local_topology/gaussian_oracle/README.md#doc-6bb34fcdab7d)
passed technical and saved-array checks. Under its frozen generating distribution,
the local 100 ms noisy conditional risk exceeds the original target; the same-posterior
visible reset adds risk. This one-use authority and saved audit are consumed.
The user's 2026-11-13 update prioritizes quick clean predictions and complete
val/test evaluation after every frozen training round, regardless of RMSE.
The [clean learning round](../../08_machine_local_topology/clean_learning/README.md#doc-57b3d956acd9)
completed one 240-update fit of twenty machine-local port/time-scale parameters
through a differentiable 100 ms history solver. Truth, initial and final models
all received train/val/test scores. Numerical, gradient, truth-control, accuracy
and saved-array checks passed; the experiment and saved audit are consumed.
The [result review](../../08_machine_local_topology/clean_learning/README.md#doc-2066f558784e)
sets the next scope: verify the required clean nonlinear history interface and
its gradients, then prepare the TASK3 real-data pilot. The current linear
response solver is not an exact initializer for the sine backbone.
The [clean nonlinear history protocol](../../08_machine_local_topology/nonlinear_history/README.md#doc-089395867996)
completed six fixed cases and two parameter controls. Nonlinear recovery,
gradients and numerical gates passed; both controls received all train/val/test
trajectory and stability scores. The [report](../../08_machine_local_topology/nonlinear_history/README.md#doc-c552a35ccfd1)
and [result review](../../08_machine_local_topology/nonlinear_history/README.md#doc-2a80b85f5511)
retain the saved-audit schema failure and its JSON-only reconciliation. No arrays
were reread for that correction. The next registered comparison uses the full
TASK3 cohorts as specified above. This no-fit S0 establishes the nonlinear
interface and gradients; real training convergence is evaluated in the new round.
The [statistical noise plan](../../08_machine_local_topology/statistical_initializer/README.md#doc-b45fdc06a514)
is deferred. Existing noise results and consumed scopes remain unchanged.
Future real fits use TASK3's exact frozen train 346 / val 107 / test 76 cohorts;
accuracy gates govern promotion, not whether to report held-out predictions.

[Machine-local S0](../../08_machine_local_topology/synthetic_implementation/README.md#doc-2a087778b7e9) passed all 112
frozen checks. The [S1-base comparison](../../08_machine_local_topology/synthetic_implementation/README.md#doc-c290428fd661)
completed all twelve fits and its saved arithmetic audit. Observation, physical
and numerical checks passed; all six required prediction pairs failed. Both
one-use authorities and saved audits are consumed: REJECT / KEEP_BASELINE /
STOP_T0. The [result review](../../08_machine_local_topology/synthetic_implementation/README.md#doc-33456fba3cb6)
prioritized the now-completed initializer diagnosis; its approved review and
saved arithmetic audit are linked above.

The frozen [base research plan v2](../../08_machine_local_topology/synthetic_implementation/README.md#doc-6f0ce03af776)
implements ten machine-local hidden states before learning physical modules.
Its first comparison separates model order (G2/G10) from local port structure
(G10/L10), with the unchanged A06 as the performance reference. Initialization,
history propagation and future rollout use each candidate's own causal model.
The [synthetic protocol](../../08_machine_local_topology/synthetic_implementation/README.md#doc-6888fd388cdf) defines
S0 correctness followed by S1-base: seven observation controls and twelve fixed
fits. The user's 2026-11-12 instruction authorizes this staged implementation
and autonomous advancement through scientific gates. Each stage still requires
its new registered budget and exact implementation review; failed gates stop
downstream work. The [original design review](../../08_machine_local_topology/synthetic_implementation/README.md#doc-fc90e5b9c601)
remains bound to its earlier preparation document.
The [full-proposal assessment](../../08_machine_local_topology/synthetic_implementation/README.md#doc-19bfa798076d)
now reviews the latest linked conversation and its complete attachment. It
supports the direction and specifies corrections to dimension-dependent encoder
regularization, deletion order, trainable parameters and execution details.
The [2026-11-10 plan](../cross_campaign_plan_20261110/README.md#doc-c2d77b9f9462) remains the historical,
hash-bound design for the completed A1 work below.

The single [A1-S0 execution](../../07_state_port/state_port_synthetic/README.md#doc-1716ff046240) saved
all numerical checks, then failed while serializing the final summary. Its
scientific authority is consumed and the failed run remains immutable. The
[separate saved-evidence review](../../07_state_port/state_port_synthetic/README.md#doc-e881ee12a327)
completed 200 checks with zero mismatches and zero new model or trajectory
calls. The [result report](../../07_state_port/state_port_synthetic/README.md#doc-fc70cc3f9d2d) records
`PASS_FROM_SAVED_EVIDENCE` while preserving the failed process. The
[scientific result review](../../07_state_port/state_port_synthetic/README.md#doc-21110cda5059) accepts
these finite checks; both execution authorities are consumed. Real train
comparisons require their own design and registration; keep the baseline and
default deployment initializer.

The [A1 train comparison](../../07_state_port/state_port_train/README.md#doc-62a54c0d2922)
completed all six 40-update fits and its separate saved arithmetic audit
(76 checks, zero mismatches). State-dependent ports improved weighted trajectory
MSE over both references, but added one train rule miss in every seed. The
[scientific review](../../07_state_port/state_port_train/README.md#doc-29232367d303) accepts the
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
this synthetic scope. Keep the [A1 protocol](../../07_state_port/state_port_train/README.md#doc-c1bfb16a8c3a),
its implementation review and all completed evidence frozen.


<a id="doc-0a640a72f9b4-precursor-evidence"></a>

#### Precursor evidence

The [constant-C synthetic study](../../06_constant_c_energy_initialization/constant_c_100ms_synthetic/README.md#doc-285d1a389749)
completed its one registered execution: 17 cases separate discrete observation
information, initialization error and visible future equivalence. Its authority
is consumed; review the saved evidence rather than rerunning it. The fixed main
system supports accurate 100 ms recovery, while weak coupling, aliasing and
hidden-null controls delimit that result. This establishes no real-data claim.

The [energy-constrained initializer preparation](../../06_constant_c_energy_initialization/constant_c_energy_initializer/README.md#doc-9ca7305815de)
now defines nine saved synthetic cases and three fixed radius roles, retaining
both hidden-state and visible-future outcomes. The isolated algebra solver has
engineering tests; the [preparation review](../../06_constant_c_energy_initialization/constant_c_energy_initializer/README.md#doc-874e22c7946a)
accepts that scope. The [machine-readable plan](../../../../configs/proposals/constant_c_energy_initializer_preparation_v1.json)
and [saved-input manifest](../../../../configs/proposals/constant_c_energy_saved_sources_v1.json)
preserve the original preparation and source identities.
The separate [execution protocol](../../06_constant_c_energy_initialization/constant_c_energy_initializer/README.md#doc-1390a7bda1a0)
and [exact implementation approval](../../06_constant_c_energy_initialization/constant_c_energy_initializer/README.md#doc-932696dd4820)
now bind the saved-input reader, conditional propagation and complete budgets.
The [completed comparison](../../06_constant_c_energy_initialization/constant_c_energy_initializer/README.md#doc-8935213c3884)
retains all 27 records. The fixed bound controls energy inflation and improves
visible futures in the weak-coupling noisy case; hidden-state recovery still
fails its threshold. Both positive bounds remain inactive in the stiffness
mismatch case. The [scientific result review](../../06_constant_c_energy_initialization/constant_c_energy_initializer/README.md#doc-44c815c1d4d2)
accepts these bounded findings with `KEEP_BASELINE`.
Authority `constant_c_energy_initializer_v1_20261109` is consumed; phase-one
authority also remains consumed. Read-only evidence synthesis and preparation
of an independently registered follow-up remain available. No later candidate
stage or real-data comparison is authorized by this result.
The separate [fixed-radius confirmation](../../06_constant_c_energy_initialization/constant_c_energy_confirmation/README.md#doc-31b532a5f152)
completed its six new parameter/noise cases within the already exposed matched
linear family. All frozen future thresholds and noninferiority criteria passed;
both weak-coupling noisy cases exercised the bound. The [result review](../../06_constant_c_energy_initialization/constant_c_energy_confirmation/README.md#doc-78265c298ea7)
accepts this confirmation relative to integral LS with `KEEP_BASELINE`. The zero
baseline had lower visible errors in those two cases, and hidden-state recovery
still failed in all three weak-coupling cases. Authority
`constant_c_energy_confirmation_v1_20261109` is consumed. This is not external
independent validation. Further comparisons require their own design, exact
review and registration; the result does not start another execution.

The [encoder-referenced energy-ball comparison](../../06_constant_c_energy_initialization/kiki_energy_cap_transfer/README.md#doc-4eaeefc4b469)
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
The [result review](../../06_constant_c_energy_initialization/kiki_energy_cap_transfer/README.md#doc-6c9728c858a8) records the
route decision. The [representative-window audit correction](../../06_constant_c_energy_initialization/kiki_energy_cap_transfer/README.md#doc-bab170f31860)
has its own saved-evidence scope and preserves the initial audit failure;
it does not reopen the scientific run.
Rejected candidates remain archived. Independent engineering extraction requires
explicit review.

See the [evidence review and constant-C 100 ms preparation](../../06_constant_c_energy_initialization/constant_c_100ms_synthetic/README.md#doc-d306b4c8a553)
for the distinction between discrete observation information, integral-estimator
error and model mismatch, and the items to freeze before a new registration.


<a id="doc-0a640a72f9b4-records"></a>

#### Records

- [Campaign history through 2026-11-09](#doc-4f7c46c8c54c)
  preserves the approvals, budgets, decisions and evidence references migrated
  from the former root guide.
- [Protocols](../../README.md) contain registered experiment procedures.
- [Report index](../../README.md) links completed results and receipts.

</details>

## 历史记录

<a id="doc-4f7c46c8c54c"></a>

<a id="doc-4f7c46c8c54c-research-campaign-history-through-2026-11-09"></a>

### Research campaign history through 2026-11-09

This snapshot preserves the campaign records from the root AGENTS.md at
commit ad408d4. Consult the [research index](#doc-0a640a72f9b4) for follow-up scope and
the [research policy](../../../../configs/research_policy.json) for execution authority.


<a id="doc-4f7c46c8c54c-research-campaign-records"></a>

#### Research campaign records

Use the current status in `configs/research_policy.json` for every campaign.
The records below preserve the scope of each approval and its outcome.

On 2026-10-23 the user explicitly approved one pre-certification development
evaluation, `kiki_a04_development_20261023`, as registered in the active policy
and `docs/reports/04_kiki_a03_a17_pipeline/a04_development/_sealed/kiki_a04_development_authorization_20261023_zh.md`.
It is restricted to the frozen A04 candidates, seeds, budget, causal input
contract and internal train/val allocation inside original source train.
The purpose is to establish or reject empirical input support; this is
uncertified development evidence. Preserve every failure and every allocated
event in coverage/risk reporting. This exception permits neither deployment
predictions nor model promotion, and opens no source val/test trajectories.
Independent registered confirmation is required before releasing predictions
inside a supported range. Other evaluations retain the certified-envelope rule.
The A04 campaign has now completed and its authorization is `consumed` in
the active policy, with a bound run, budget and result receipt. Do not reactivate
it or repeat its fits. Existing artifacts remain available for read-only audit;
any new scientific comparison needs its own registered authorization.

On 2026-10-24 the user explicitly authorized one bounded A05 paired development
evaluation, `kiki_a05_development_20261024`, as registered in the active policy
and `docs/reports/04_kiki_a03_a17_pipeline/a05/_sealed/kiki_a05_development_authorization_20261024_zh.md`. It is
restricted to experiment `paper_kiki_a05_paired`, workflow
`kiki_a05_paired_v1`, stage `paired_training`, and module
`kron_reduce.workflows.kiki_paired_training`. The comparison uses groups F/J,
seeds 324/325/326, 80 updates per fit, and at most six fits/480 new updates on
the existing frozen source train/internal train-val allocation. It explicitly
permits development scoring of all 258 train and 88 val trajectories at 1 s
and 3 s, plus all 88 val 8 s stability and numerical-refinement diagnostics.
Source val/test payloads remain closed. Labels may be read only after model,
selection, and trajectory evidence are frozen, for diagnostics. Preserve all
allocated events and failures; failed fits consume their slots and cannot be
retried. These results remain uncertified development evidence: no deployment
predictions, model promotion, or independent confirmation. The A04 authorization
remains consumed and unchanged.

On 2026-10-24 the user explicitly authorized one bounded A06 joint-refinement
development evaluation, `kiki_a06_development_20261024`, as registered in the
active policy and `docs/reports/04_kiki_a03_a17_pipeline/a06/_sealed/kiki_a06_development_authorization_20261024_zh.md`.
It is restricted to `paper_kiki_a06_refinement`, workflow
`kiki_a06_refinement_v1`, stage `joint_refinement`, and module
`kron_reduce.workflows.kiki_joint_refinement`. It starts from the three frozen
A05 J checkpoints, uses fresh Adam state, and allows exactly three fits at
seeds 324/325/326 with 240 new updates each. The update-80 checkpoint is
diagnostic; update 240 is the sole primary endpoint. The fixed exception allows
all 258 train and 88 val trajectories at 1 s and 3 s and all 88 val 8 s LOS and
numerical checks, with all 369 allocated cases retained. Source val/test payloads
remain closed; labels are available only after all training and evidence are
frozen. Failures consume their slots and cannot be retried. Results remain
uncertified development evidence with no deployment predictions or promotion.
Automatic training until a threshold passes remains disabled; A04 and A05
authorizations remain consumed and unchanged.

The A05 campaign has completed and its authorization is now `consumed`,
bound to its immutable run, six fits/480 updates, and result receipt.
Do not reactivate the authorization or repeat its fits. Existing artifacts
may be audited read-only; further scientific comparisons require their
own registered authorization. No A05 model has been promoted.

The A06 campaign has completed and its authorization is now `consumed`,
bound to its immutable run, three fits/720 updates and result receipt.
Do not reactivate it or repeat its fits. Read-only audit remains available;
a new scientific comparison requires its own registered authorization.
No A06 model has been promoted.

The A07 campaign has completed and its authorization is now `consumed`,
bound to its immutable run, six paired fits/1440 new updates and result receipt.
Do not reactivate it or repeat its fits. Read-only audit remains available;
a new scientific comparison requires its own registered authorization.
No A07 model has been promoted or selected for deployment.

The A08 campaign has completed and its authorization is now `consumed`,
bound to its immutable run, six paired fits/1440 new updates and result receipt.
Do not reactivate it or repeat its fits. Read-only audit remains available;
a new scientific comparison requires its own registered authorization.
No A08 model has been promoted or selected for deployment.

On 2026-10-24 the user continuation instruction authorizes the separately registered
A09 fixed-model train mechanism audit, `kiki_a09_development_20261024`, as
scientifically approved by Astra Ultra in the A08 result review. It permits
only experiment `paper_kiki_a09_train_mechanism`, workflow
`kiki_a09_train_mechanism_v1`, stage `train_mechanism`, and module
`kron_reduce.workflows.kiki_train_mechanism`. Use all three frozen A06 step240
models, all 268 allocated internal train events (258 eligible), and the exact
100 ms causal contract. No fits or optimizer updates are permitted; up to
1548 fixed-model paths, 672543 RK4 updates and 3141630 total RHS calls are
registered. Only real available targets through at most 3 s may be audited.
No labels/event times, internal/source val or test trajectories, future Pm,
Pe, voltage or hidden truth may be decoded. File-integrity byte reads are
recorded separately from decoded events. Preserve failed slots and all
uncovered events; no retry. This is uncertified train-mechanism evidence,
with no FSR claim, delivery, promotion, independent confirmation, or automatic
new training. Prior authorizations remain consumed and unchanged.

The A09 train mechanism audit has completed and its authorization is now
`consumed`, bound to its immutable run, 1548 paths, 3141630 RHS calls and
result receipt. Zero fits and updates were performed. Preserve its REJECT
and all three failed trajectory-refinement comparisons; no new training is
authorized by A09. Read-only saved-artifact audit remains available. Any
new numerical comparison requires its own registered authorization.

The separately authorized bounded A10 all-train numerical audit, `kiki_a10_development_20261024`,
is now completed and consumed in the active policy, bound to its immutable
run and result receipt. A10 reused A09 saved .005 s paths and created only .0025 s paths for the same 258 train events and three
fixed A06 models: zero fits/updates, 774 paths, 1548 rollouts, at most 896724
RK4 updates and 3812625 RHS calls. It opens no labels or internal/source val/test,
does not retry failures or further reduce the step, and cannot promote models
or authorize training. A10's full source/input/saved-replay/numerical/physical
gates all passed. Do not reactivate its authorization or repeat its paths.
Any new scientific comparison requires its own registered authorization;
A10 supports proposing one uniform time-aggregation comparison, without
automatically authorizing training. No A10 model has been promoted.

A11 completed with REJECT and KEEP_BASELINE. Its six fits/1440 updates and
all results are archived on `codex/kiki-a11-temporal-objective` at
`70c463a9286cbde2acc5f16266b19fcb6c2a1555`; PR 10 is closed without merge.
The archive remains consumed: do not repeat fits or import that branch's
rejected algorithm into main. The active policy records its archive identity.

A12 is a separately registered, one-use train input audit under
`kiki_a12_development_20261024`. Only its dedicated module CLI may execute it;
the generic project runner must reject before generic data verification.
It reads the frozen 268 train slots, uses no model/fit/prediction/force/RHS,
and freezes all unlabelled evidence before the 268 post-label metadata reads.
Internal val/source val/test and CSV/splits remain closed. Astra Ultra must
approve the exact implementation bindings before its policy becomes active.
The authorization is spent on any execution or failure, without retries.
An accepted input audit does not authorize new training, certify predictions,
verify val coverage, promote a model, or establish zero missed instability.

A12 has completed and its authorization is consumed, bound to the immutable
run and result receipt in the active policy. All 268 train inputs and source
rule/metadata checks passed; the independent saved-array replay found zero
mismatches. All ten short records already violate the joint rule by the
20 ms origin, so their support is not evidence of prediction before onset.
Do not rerun this audit or infer internal val coverage. KEEP_BASELINE applies.

A13's frozen synthetic integral-initializer study is archived with its
positive convergence cases, expected counterexamples and limited retained
evidence replay. It supports proposing a separately registered train-history
mechanism audit; it authorizes no real training or future comparison.

A14 is a separately registered one-use train-history mechanism comparison,
`kiki_a14_development_20261024`, requiring exact implementation approval by
Astra Ultra. Its dedicated CLI alone may read the ten frozen NPZ caches and
42 whitelisted members; the generic runner must reject before data checks.
It keeps fixed A06 step240 models and all 268 allocated train events, with
258 eligible histories/774 seed-event slots. Fits/updates, future paths,
labels/LOS, internal val and source val/test remain closed. The single-use
shared budget is consumed before NPZ byte checks, with no retries or resume.
Scope reservations are durable; per-call logs are flushed and range-hashed
before durable closure. Any interrupted scope is unknown/spent. Numerical
failures stay in the fixed population. V and I follow the frozen protocol;
even ACCEPT remains uncertified history evidence and KEEP_BASELINE, and
authorizes neither training nor future comparison nor model promotion.

On 2026-10-24 the user authorized one bounded A07 tilted-objective development
comparison, `kiki_a07_development_20261024`, as registered in the active policy
and `docs/reports/04_kiki_a03_a17_pipeline/a07/_sealed/kiki_a07_development_authorization_20261024_zh.md`. It is
restricted to `paper_kiki_a07_tilted`, workflow `kiki_a07_tilted_v1`, stage
`tilted_training`, and module `kron_reduce.workflows.kiki_tilted_training`.
Groups M and T each use seeds 324/325/326 and 240 additional updates, restoring
the same-seed A06 step-240 model and exact saved Adam state. M retains the mean
event MSE objective; T uses the registered train-only risk temperatures
0.7844805082870585, 0.7768666645735861, and 0.7861097423844166. The fixed
exception allows complete train/val trajectory and val stability evaluation
with all allocated cases retained. Source val/test remain closed; labels are
diagnostic only after freeze. There is no classification-based group selection,
automatic continuation, deployment prediction, or model promotion. A04, A05,
and A06 remain consumed and unchanged.

On 2026-10-24 the user authorized one bounded A08 memory-order development
comparison, `kiki_a08_development_20261024`, as registered in the active policy
and `docs/reports/04_kiki_a03_a17_pipeline/a08/_sealed/kiki_a08_development_authorization_20261024_zh.md`. It is
restricted to experiment `paper_kiki_a08_memory`, workflow
`kiki_a08_memory_v1`, stage `memory_order`, and module
`kron_reduce.workflows.kiki_memory_order`. It compares r2/r4 with seeds
324/325/326, 240 fresh-Adam updates per fit, and six fits/1440 updates total,
starting only from the bound A06 step-240 model checkpoints. It uses the frozen
source train allocation, all 258 train and 88 val trajectory evaluations, and
all 88 val 8 s stability/numerical checks, retaining all 369 allocated cases.
Source val/test remain closed; labels are diagnostic only after evidence freeze.
Failures consume their slots and cannot be retried. Results are uncertified
development evidence with no deployment predictions or model promotion.
A04 through A07 remain consumed and unchanged.

A14 has completed and its authorization is consumed, bound to the immutable
run, 774 completed slots/202788 RHS and the two independent saved-evidence
audits. V and I passed in all six fixed train-history comparisons;
Astra accepted the history mechanism and engineering main integration.
Do not rerun A14 or change the default deployment initializer. No future
trajectory, label or new FNR result was evaluated; KEEP_BASELINE applies.
Further comparisons require independent registration and Astra approval.

A15 completed with REJECT and KEEP_BASELINE: V passed, I and D failed.
Its unique 1548 future paths/5629014 RHS and all results are archived on
`codex/kiki-a15-future-transfer` at
`5ac8b571751467f677fb569df3c0a1f4c323d065`; PR 13 is closed without merge.
The authority remains consumed. Do not repeat its scientific execution or
import its rejected initializer into main. The active policy records its
archive identity; any independent engineering extraction needs its own review.

A16 is separately registered under `kiki_a16_development_20261025` for one
bounded train-history power diagnostic. Astra Ultra approved the exact
implementation and bindings on 2026-10-25; the active policy records the review.
Only its dedicated CLI may execute after the activation commit is pushed,
clean and passes CI. Keep the frozen 258 train events/774 slots, finite reads
and scalar budgets; zero models/fits/RHS/new states/future/labels/val/test.
The shared budget is spent before source reads, without retry or resume.
Results remain uncertified history evidence with KEEP_BASELINE and establish
no new missed-instability result or model promotion.

A16 has completed once and its authorization is consumed, bound to its immutable
run, 774 completed slots, 1037 core scopes and two independent saved-result
audits. The fixed history equation responds to the input correction, while
holding the old state increases both registered residual measures. No new
state, fit, future prediction or missed-instability metric was produced.
Do not rerun A16. KEEP_BASELINE applies; input-consistency controls or C(q)
training require their own registration and exact scientific approval.

A17 has completed and its authorization `kiki_a17_development_20261025` is
consumed. The result is REJECT with KEEP_BASELINE (`V=true`, `G_I=false`,
`G_E=false`); no zero missed-instability result was established. The archived
implementation is on `codex/kiki-a17-input-consistency` at
`637cdf0510a9408197e982f2dce88c30a2852af3`; PR 15 is closed without merge.
The completed run is
`paper_kiki_a17_input_consistency-input_consistency-20261025T083355586536Z-e736e260-f1ffd6d9`
from source commit `e736e26091a777f96025f70f865186eb02290d7c`, with manifest
SHA256 `708839cb627812282e0abfcc32f54c712e8afab4556b853c7253f742d88dcf2e`.
Its archived result receipt is
`docs/reports/kiki_a17_input_consistency_20261025_evidence.json` (SHA256
`40e12170da64e284f68aae55b7d7e673c425f883f6e07cdebb64e4fb8cbfcafe`); the
scientific review is
`docs/research/kiki_a17_input_consistency/a17_scientific_result_review_v1.md`
(SHA256 `cceec186606e6da19d46dbbec200749471c14231811faa86e8ae5b0f6c457c70`),
with formal review receipt SHA256
`eb2198860c0fd95c6bc7994042769d4afd3cba9f52a64c874aac02adf5077a54`.
The audit consumed 774 seed-event slots, 1548 history paths, 1548 future paths
and 5831802 RHS calls across 10356 scopes, with zero fits or optimizer updates.
Source val/test and internal val were not opened, and no labels were read.
Do not reactivate or rerun A17, or import its rejected candidate/default
initializer or code. Any independent engineering extraction requires a new
explicit review. The next work is limited to preparing the known-truth
constant-C, 100 ms information-sufficiency question; this authorizes no new
calculation.

The separate TASK3 200 ms frozen retrospective evaluation
`kiki_task3_frozen_200ms_20261109` completed with `KEEP_BASELINE`. Its one-use
authorization is consumed and bound to the immutable run and
[result receipt](../../04_kiki_a03_a17_pipeline/task3_frozen_200ms/_sealed/kiki_task3_frozen_200ms_20261109_evidence.json).
The [protocol](../../04_kiki_a03_a17_pipeline/task3_frozen_200ms/README.md#doc-d103d9170ce4) used
frozen A03 checkpoints, 105 TASK3 `val` events and 75 previously exposed
TASK3 `test` events, with zero fitting or optimizer updates. Treat the result
as a retrospective comparison. Use the saved evidence for review and keep
the consumed authorization closed.

<details>
<summary>原件与版本</summary>

- [docs/research/README.md](_sealed/README.md)
- [docs/research/campaign_history_through_20261109.md](_sealed/campaign_history_through_20261109.md)

</details>
