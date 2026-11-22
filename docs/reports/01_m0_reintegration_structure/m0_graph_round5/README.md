# Round 5 method and results: graph-structured M0

[返回主题页](../README.md)

## 运行结果与归档

- [验证归档说明](../../../../results/backup/00_early_physics_baselines/D_baseline_validations.DESCRIPTION.md)

登记文件与配置

[运行配置](../../../../configs/experiments/paper_m0_graph_round5_single_seed.toml)

## 执行协议

<a id="doc-4a5cb92063c9"></a>

<a id="doc-4a5cb92063c9-round-5-graph-potential-m0-protocol"></a>

### Round 5: graph-potential M0 protocol


<a id="doc-4a5cb92063c9-scientific-question"></a>

#### Scientific question

Round 5 tests whether a physically constrained graph potential and explicit
relative-speed dissipation improve closed-loop reintegration beyond the current
M0.  Parameter estimates are diagnostics of whether the fitted vector field is
credible; the acceptance endpoints remain reintegration accuracy, numerical
failure rate, and stability prediction from a fixed causal rollout.

This round uses seed 324 only.  The accuracy experiment uses observed
mechanical power with linear interpolation at Runge--Kutta stages and is
therefore a conditional, noncausal upper bound.  No governor model is fitted in
this round.  Test trajectories remain inaccessible until the validation gate
passes.


<a id="doc-4a5cb92063c9-fair-population-and-comparators"></a>

#### Fair population and comparators

- Fit and validation selection include only trajectories with at least 100
  recorded samples. Shorter trajectories are listed as exclusions and are not
  used for normalization, fitting, or selection.
- All three methods use the canonical exact-clearing state, the same observed
  `Pm(t)`, the same COI projections, and the same RK4 implementation with a
  maximum internal step of 0.01 s.
- `M0` is refitted rather than loaded from an earlier run.
- `M0-DD` reproduces direct numerical-derivative regression under the current
  contract. It is a diagnostic comparator, not a preferred estimator.
- `M0-GRAPH` is the Round 5 candidate.

The maximum fitting prefix is 101 samples so a 100-interval integral window is
available when the trajectory supports it.  Windows of 1, 5, 20, 40, and 100
intervals are sampled deterministically without overlap.  Each
case-by-window-length stratum contributes unit total least-squares weight, so
neither long records nor the numerous one-step windows dominate the solve.


<a id="doc-4a5cb92063c9-candidate-equation"></a>

#### Candidate equation

For incidence matrix `B`, edge angle `eta=B.T delta`, and equilibrium edge
angle `eta_eq`, the candidate is

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s[\sin\eta-\sin\eta_{eq}]
+K_c[\cos\eta-\cos\eta_{eq}]+D_eB^T\omega\}.
\]

`D_n`, `D_e`, and `K_s` are nonnegative. `K_c` may be signed, while the local
edge curvature is constrained by

\[
K_s\cos\eta_{eq}-K_c\sin\eta_{eq}\ge 0.
\]

The model is equilibrium anchored, invariant to a common angle shift, and its
internal graph force sums to zero. Parameters are found by one constrained
weak-form ridge solve for each train-only ridge value. Pruning removes an
entire edge triplet and is selected on validation data only. Learned edge
weights are effective reduced parameters and are not claimed to be the true
Kron network.


<a id="doc-4a5cb92063c9-parameter-and-modal-audit"></a>

#### Parameter and modal audit

Every fold records node and edge damping, sine and cosine coupling, local edge
curvature, active-edge count, near-zero damping counts, design rank and
condition number, quotient damping and stiffness eigenvalues, and all
linearized modal real and imaginary parts.  A candidate is ineligible if its
constrained coefficients are nonfinite, its local curvature is negative, or a
linearized quotient mode has a positive real part above numerical tolerance.

These diagnostics do not override trajectory evidence: plausible-looking
coefficients cannot rescue a model with worse closed-loop reintegration.


<a id="doc-4a5cb92063c9-validation-and-test-gates"></a>

#### Validation and test gates

The frozen validation selection uses the standardized frequency and edge-angle
trajectory loss.  The Round 5 accuracy gate additionally requires, at H=100:

1. zero rollout failures and COI drift below the existing numerical contract;
2. pooled frequency R2 above 0.8;
3. at least 20% lower mean RMSE than refitted M0;
4. improvement in at least four of five folds;
5. no degradation in worst-10% CVaR RMSE;
6. a fault-line cluster-bootstrap 95% upper confidence bound below zero;
7. `M0-GRAPH` selected in at least four folds; and
8. all fitted `M0-GRAPH` checkpoints passing their physical audit.

Only if all validation conditions pass may the full stage open test data.  The
final stability experiment then rolls the already frozen model for 5 s using
only the clearing state and `Pm_clear` held constant, independent of recorded
valid length. Stability labels are used only for final evaluation. The round is
accepted only if the accuracy gates still pass on test, classification has zero
numerical failures, and instability FSR is below 0.2.

## 报告与结果

<a id="doc-2535058946a2"></a>

<a id="doc-2535058946a2-round-5-method-and-results-graph-structured-m0"></a>

### Round 5 method and results: graph-structured M0


<a id="doc-2535058946a2-one-sentence-argument"></a>

#### One-sentence argument

On the canonical IEEE39 post-clear dataset, a locally stable graph-potential
model with explicitly learned node and edge damping did not provide reliable
closed-loop reintegration beyond a refitted M0 under observed mechanical-power
forcing; Round 5 was therefore rejected before test access and stability
classification.


<a id="doc-2535058946a2-terminology-ledger"></a>

#### Terminology ledger

| Canonical term | Definition |
|---|---|
| M0 | Equilibrium-centred swing model fitted by an integral weak form |
| M0-DD | Direct numerical-derivative comparator under the current data and RK4 contract |
| M0-GRAPH | Round 5 graph-potential model with node damping, edge damping, and sine/cosine edge forces |
| observed Pm | Recorded post-clear mechanical-power schedule, used as a conditional noncausal input |
| H=100 | The first 100 recorded samples after exact clearing, approximately 1 s |
| FINAL | Fold-specific model frozen by train/validation selection before any test access |


<a id="doc-2535058946a2-method"></a>

#### Method


<a id="doc-2535058946a2-study-design-and-evidence-boundary"></a>

##### Study design and evidence boundary

Round 5 used seed 324 and all five fault-line folds of the registered
`combined_line_5fold_v1` split. Fitting and validation selection excluded every
trajectory with fewer than 100 samples before data adaptation. This removed 31,
35, 32, 33, and 38 training events and 9, 5, 8, 5, and 8 validation events in
F1--F5, respectively. The H=100 validation population contained 177 trajectories
from 33 fault-line clusters. Test data, test labels, and stability outcomes were
not read.

The accuracy experiment supplied the recorded `Pm(t)` schedule to all methods
and linearly interpolated it at Runge--Kutta stages. It is therefore a
conditional upper bound, not a deployable causal forecast. All candidates used
the exact clearing state, the same COI projection, and the same fourth-order
Runge--Kutta integrator with an internal maximum step of 0.01 s.

SINDy identifies coefficients by sparse regression over a prescribed function
library rather than by many neural-network epochs [Brunton et al.](https://arxiv.org/abs/1509.03580).
External inputs can enter that library explicitly, as in SINDYc
[Brunton et al.](https://arxiv.org/abs/1605.06682). The present M0 fit therefore
used one train-only linear solve per ridge value. Its weak integral form avoided
placing numerical derivatives in the target, consistent with the noise-reduction
motivation of weak SINDy [Messenger and Bortz](https://epubs.siam.org/doi/abs/10.1137/20M1343166?journalCode=mmsubt).


<a id="doc-2535058946a2-comparators"></a>

##### Comparators

The refitted M0 obeyed

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s[\sin(B^T\delta)-\sin(B^T\delta_{eq})]\}.
\]

M0-DD used the same equation, current long-trajectory cohort, observed `Pm(t)`,
and RK4 rollout, but estimated acceleration with a second-order numerical
gradient and solved a bounded ridge problem. It isolated the effect of the old
direct-derivative fitting style without reintroducing old clearing offsets,
integration schemes, or selected plotting populations.


<a id="doc-2535058946a2-graph-potential-candidate"></a>

##### Graph-potential candidate

M0-GRAPH extended the reduced swing equation to

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s\Delta\sin\eta+K_c\Delta\cos\eta+D_eB^T\omega\},
\qquad \eta=B^T\delta.
\]

Here `Dn`, `De`, and `Ks` were nonnegative, `Kc` was signed, and each edge was
constrained to have nonnegative local curvature,

\[
\kappa_e=K_{s,e}\cos\eta_{eq,e}-K_{c,e}\sin\eta_{eq,e}\ge0.
\]

This structure anchors the equilibrium, preserves invariance to a common angle
shift, and balances internal force through the incidence matrix. Learned edge
weights are effective reduced coefficients; they are not interpreted as the
true Kron network.

The weak system used deterministic non-overlapping windows of 1, 5, 20, 40,
and 100 intervals within at most 101 samples. Each case-by-window-length stratum
received unit total weight. Four train-only ridge values and four validation-only
edge-pruning thresholds were evaluated. Physical inequality constraints were
encoded as a two-ray convex cone per edge. This replaced an initial SLSQP
implementation that could return a feasible-looking but unconverged solution.
The corrected implementation rejects every candidate for which the optimizer
does not report success. Constrained sparse identification follows the broader
principle that known physics and inequalities should be encoded in the
regression problem rather than checked only after fitting
[Kaptanoglu et al.](https://arxiv.org/abs/2111.08481).


<a id="doc-2535058946a2-prespecified-validation-gate"></a>

##### Prespecified validation gate

The validation gate required zero rollout failures, pooled frequency R2 above
0.8, at least 20% lower H=100 mean RMSE than refitted M0, improvement in at
least four folds, no increase in worst-10% CVaR RMSE, a fault-line cluster
bootstrap 95% upper confidence bound below zero, M0-GRAPH selection in at
least four folds, and passing graph-physics audits. Only a complete pass could
unlock test trajectories and the fixed-causal 5 s stability rollout.


<a id="doc-2535058946a2-results"></a>

#### Results


<a id="doc-2535058946a2-closed-loop-reintegration"></a>

##### Closed-loop reintegration

| Method | H=100 mean RMSE | Median RMSE | Pooled RMSE | Worst-10% CVaR | Pooled frequency R2 | Failures |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.2846 | 0.1242 | 0.4326 | 0.9839 | 0.4366 | 0% |
| M0-DD | 0.2726 | 0.1195 | 0.4112 | 0.9214 | 0.4925 | 0% |
| M0-GRAPH | 0.5135 | 0.1432 | 1.1872 | 3.1893 | -3.2248 | 0% |
| FINAL | 0.2683 | 0.1408 | 0.3996 | 0.9028 | 0.5211 | 0% |

The fold-specific selector chose M0-DD in F1, F3, and F5, and M0-GRAPH in F2
and F4. FINAL improved mean RMSE over M0 by only 5.72%, below the required 20%.
Fold improvements were +4.16%, -1.64%, +11.64%, +7.89%, and +4.07% for
F1--F5. Although four folds moved in the same direction and the clustered
paired effect was negative (difference FINAL minus M0, -0.0163; 95% bootstrap
CI -0.0286 to -0.00186), pooled frequency R2 remained 0.521, and the graph
candidate was selected in only two folds. The validation gate failed.

![Round 5 validation summary](../../../../results/figures/m0-graph-round5-solver-fix/fig_round5_validation_summary.png)


<a id="doc-2535058946a2-learned-damping-stiffness-and-modes"></a>

##### Learned damping, stiffness, and modes

All five corrected graph optimizations converged in 430--857 iterations and
passed equilibrium, force-balance, gauge-invariance, local-curvature, and
linearized-mode audits. Nevertheless, the parameter estimates were close to a
boundary solution. The number of near-zero node damping coefficients was 7,
6, 6, 4, and 6 out of 7; near-zero edge damping counts were 15, 18, 15, 14,
and 15 out of 21. Minimum quotient damping eigenvalues were approximately 0,
0, 0.031, 0.111, and 0 in F1--F5. Maximum modal real parts were all negative,
from -0.0037 to -0.1024, while minimum quotient stiffness eigenvalues remained
positive, from 9.68 to 10.08.

The normalized design condition number remained high in every fold
(7.67e3--9.28e3). More importantly, the weighted weak residual did not predict
closed-loop validation quality: F5 had the smallest weak residual (2.59) but a
mean validation RMSE of 0.741, whereas F2 had a larger weak residual (4.09) but
a mean validation RMSE of 0.149. Thus, locally admissible coefficients and a
small integrated equation residual were insufficient evidence for accurate
rollout.

![Round 5 parameter audit](../../../../results/figures/m0-graph-round5-solver-fix/fig_round5_parameter_audit.png)


<a id="doc-2535058946a2-trajectory-level-evidence"></a>

##### Trajectory-level evidence

The trajectory figure used the median H=100 difference between M0-GRAPH and
M0 within F1 and F2; it did not select the best-looking event. Even in F2,
where the graph model won the fold-level standardized selection loss, the
median event showed almost no RMSE advantage. The discrepancy was dominated by
phase and amplitude evolution after the initial transient rather than by an
incorrect clearing state.

![Round 5 trajectory examples](../../../../results/figures/m0-graph-round5-solver-fix/fig_round5_trajectory_examples.png)


<a id="doc-2535058946a2-discussion"></a>

#### Discussion

Round 5 separates physical admissibility from predictive adequacy. The graph
model was equilibrium anchored and locally stable, but these constraints only
control behavior near the operating point. They do not guarantee that a
70-parameter complete-graph representation is identifiable from 1 s of
post-clear data or that its nonlinear phase evolution is correct. The high
condition numbers, fold-dependent boundary damping, and mismatch between weak
residual and rollout error indicate that parameter redundancy is the leading
explanation. This is an inference from the diagnostics, not a demonstrated
ground-truth parameter comparison.

M0-DD produced a modest empirical improvement over M0 under the present clean
simulation data, but it did not reach the acceptance targets and does not solve
the physical-identifiability problem. Its result supports retaining a direct
derivative fit as a diagnostic initializer, not replacing the weak-form method
as the main physical estimator.

Observed `Pm(t)` was already supplied. Therefore, the current failure cannot be
attributed to the ZOH mechanical-power approximation, and introducing a Pm ODE
now would mix a new input-model error with an unresolved state-dynamics error.


<a id="doc-2535058946a2-decision-and-boundary"></a>

#### Decision and boundary

**Decision: REJECT Round 5.** Test data remained locked, so no test
reintegration metric and no stability FSR were computed. This absence is a
prespecified consequence of the failed accuracy gate, not a missing analysis.
The result does not show that graph physics is unhelpful in general; it shows
that the present 70-parameter complete-graph sine/cosine/damping library is not
reliably identifiable or predictive under the registered one-second,
single-seed protocol.


<a id="doc-2535058946a2-proposed-next-round-not-yet-executed"></a>

#### Proposed next round (not yet executed)

The next candidate should reduce graph degrees of freedom before adding any Pm
ODE or trajectory optimizer. A bounded proposal is to learn quotient damping
and stiffness in a low-dimensional modal basis, enforce positive semidefiniteness
there, and then fit only a small nonlinear correction tied to those linearized
operators. The proposal should retain M0 and M0-DD, the same long-trajectory
population, observed `Pm(t)`, and the same validation gate. It should proceed
only after explicit approval.


<a id="doc-2535058946a2-reproducibility"></a>

#### Reproducibility

- Corrected source commit: `ac3a3dc867ea96dee1ecfd91a55a5bce21745c3a`
- Corrected immutable run: `paper_m0_graph_round5_solver_fix_validation-validation_run-20260729T104104Z-ac3a3dc8`
- Unit tests: 33 passed
- Project audit: PASS
- Test access: false; test IDs read: none

<a id="doc-848aac98cfde"></a>

<a id="doc-848aac98cfde-round-5-方法与结果报告图结构-m0"></a>

### Round 5 方法与结果报告：图结构 M0


<a id="doc-848aac98cfde-核心结论"></a>

#### 核心结论

在 canonical IEEE39 故障清除后数据上，即使直接使用 observed
`Pm(t)`，并显式约束节点阻尼、边阻尼、局部刚度和线性化模态稳定，
M0-GRAPH 仍未稳定提升闭环重积分精度。因此 Round 5 在验证阶段判定为
**REJECT**；测试集未解锁，稳定性 FSR 未计算。


<a id="doc-848aac98cfde-1-方法"></a>

#### 1. 方法

本轮只使用 seed 324 和五折 `combined_line_5fold_v1`。所有有效长度小于
100 的 train/validation 轨迹均在拟合、归一化和选择前排除。H=100
验证集共 177 条轨迹，覆盖 33 个 fault-line cluster。所有模型从精确
clearing state 起步，共用 observed `Pm(t)`、RK 阶段线性插值、COI 投影
和最大内部步长 0.01 s 的 RK4。

比较对象包括：

1. `M0`：重新拟合的积分弱形式 swing 模型；
2. `M0-DD`：在当前相同数据和积分器口径下，用二阶数值导数直接拟合的
   诊断基线；
3. `M0-GRAPH`：新增节点阻尼、边阻尼及平衡点锚定的 sine/cosine
   图势能力。

M0-GRAPH 使用

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s\Delta\sin\eta+K_c\Delta\cos\eta+D_eB^T\omega\}.
\]

其中 (D_n,D_e,K_s\ge0)，(K_c) 可带符号，并约束

\[
K_s\cos\eta_{eq}-K_c\sin\eta_{eq}\ge0.
\]

训练采用 1、5、20、40、100 个间隔的非重叠积分窗口，每个
“轨迹×窗口长度”分层等权。M0 和静态 M0-GRAPH 都没有 epoch；每个
ridge 只求解一次约束方程组。首次 run 暴露出 SLSQP 未收敛候选仍可被
物理门接纳的问题。修复版使用逐边二射线凸锥参数化，并硬性拒绝
`solver_success=false` 的候选；旧 run 保留但不作为最终证据。


<a id="doc-848aac98cfde-2-重积分结果"></a>

#### 2. 重积分结果

| 模型 | H=100 mean RMSE | median RMSE | pooled RMSE | worst-10% CVaR | pooled 频率 R² | failure rate |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.2846 | 0.1242 | 0.4326 | 0.9839 | 0.4366 | 0% |
| M0-DD | 0.2726 | 0.1195 | 0.4112 | 0.9214 | 0.4925 | 0% |
| M0-GRAPH | 0.5135 | 0.1432 | 1.1872 | 3.1893 | -3.2248 | 0% |
| FINAL | 0.2683 | 0.1408 | 0.3996 | 0.9028 | 0.5211 | 0% |

按折选择结果为：F1、F3、F5 选择 M0-DD，F2、F4 选择 M0-GRAPH。
FINAL 相对 M0 的 H=100 mean RMSE 只改善 5.72%，未达到 20% 门；五折
改善分别为 +4.16%、-1.64%、+11.64%、+7.89%、+4.07%。尽管
fault-line cluster bootstrap 的 FINAL−M0 配对差为 -0.0163，95% CI
为 [-0.0286, -0.00186]，但 pooled 频率 R² 仅 0.521，且 M0-GRAPH
只在 2/5 折入选，所以整体验证门失败。

![Round 5 验证汇总](../../../../results/figures/m0-graph-round5-solver-fix/fig_round5_validation_summary.png)


<a id="doc-848aac98cfde-3-阻尼刚度与模态审计"></a>

#### 3. 阻尼、刚度与模态审计

修复后五折求解器均成功收敛，迭代次数为 430--857；平衡点锚定、内部力
平衡、共同角平移不变性、局部刚度和线性化模态门均通过。但参数落在约束
边界附近：五折近零节点阻尼数分别为 7、6、6、4、6（每折共 7 个），
近零边阻尼数为 15、18、15、14、15（每折共 21 个）。最小 quotient
阻尼特征值在 F1、F2、F5 近似为零；最大模态实部虽均为负，但 F1 只有
-0.0037，距离中性稳定很近。

设计矩阵条件数在 (7.67\times10^3) 到 (9.28\times10^3) 之间，说明
70 参数的完整图模型辨识条件较差。更重要的是，弱形式残差不能预测闭环
精度：F5 的加权弱残差最小（2.59），但验证 mean RMSE 为 0.741；F2 的
弱残差更大（4.09），验证 mean RMSE 却只有 0.149。这说明“方程残差小、
局部模态稳定”并不等于“长程相位和幅值正确”。

![Round 5 参数审计](../../../../results/figures/m0-graph-round5-solver-fix/fig_round5_parameter_audit.png)

轨迹图采用每折 M0-GRAPH−M0 H=100 RMSE 差的中位事件，不挑选最佳样本。
误差主要表现为初始瞬态之后的相位和幅值偏差，而不是 clearing state
错误。

![Round 5 典型轨迹](../../../../results/figures/m0-graph-round5-solver-fix/fig_round5_trajectory_examples.png)


<a id="doc-848aac98cfde-4-解释与边界"></a>

#### 4. 解释与边界

本轮证据表明，物理可行性与预测充分性必须分开。M0-GRAPH 的局部曲率和
线性化模态是合格的，但完整图上的 sine/cosine/边阻尼共 70 个参数存在
明显冗余；高条件数和跨折边界解支持“辨识性不足”这一解释。该解释来自
诊断关联，并非与真实 Kron 参数比较后的因果证明。

M0-DD 在当前较干净的仿真数据上有小幅经验收益，但仍未达到精度门，且
不能解决物理参数辨识问题。因此它适合作为初始化或诊断基线，不应直接
替代主弱形式模型。

本轮已经使用 observed `Pm(t)`，所以当前失败不能归因于 ZOH。此时加入
Pm ODE 会把输入模型误差与尚未解决的状态动力学误差混合，不建议继续。


<a id="doc-848aac98cfde-5-决策"></a>

#### 5. 决策

**Round 5：REJECT。** 因验证精度门失败，测试数据保持锁定；没有 test
重积分结果，也没有稳定性 FSR。这是预注册流程的正确执行，不是漏算。

建议下一轮只做一个受限改动：把 70 参数完整图降为低维 quotient 模态
阻尼/刚度模型，在低维空间保证半正定，再只增加少量与线性化算子绑定的
非线性校正。保持相同长轨迹人口、observed `Pm(t)`、M0/M0-DD 基线和
验证门；暂不增加 Pm ODE，也不做多步神经优化。该方案需下一轮明确批准
后再实施。


<a id="doc-848aac98cfde-6-可复现信息"></a>

#### 6. 可复现信息

- 修复版源码提交：`ac3a3dc867ea96dee1ecfd91a55a5bce21745c3a`
- 修复版不可变 run：`paper_m0_graph_round5_solver_fix_validation-validation_run-20260729T104104Z-ac3a3dc8`
- 测试：33 passed
- 项目审计：PASS
- 测试集读取：false；test IDs：空

<details>
<summary>原件与版本</summary>

- [docs/m0_graph_round5_protocol.md](_sealed/m0_graph_round5_protocol.md)
- [docs/reports/01_m0_reintegration_structure/m0_graph_round5_method_results_en.md](_sealed/m0_graph_round5_method_results_en.md)
- [docs/reports/01_m0_reintegration_structure/m0_graph_round5_method_results_zh.md](_sealed/m0_graph_round5_method_results_zh.md)

</details>
