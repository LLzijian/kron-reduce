# IEEE39 模块化能量等值 P1-A04 诊断报告

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/03_modular_energy_p1p2/B_p1_a04_phase.DESCRIPTION.md)

登记文件与配置

[方案配置](../../../../configs/experiments/paper_ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1.toml)

## 执行协议

<a id="doc-a9621ce55d83"></a>

<a id="doc-a9621ce55d83-p1-a03-真实-f1-受控父链预登记"></a>

### P1-A03 真实 F1 受控父链预登记

日期：2026-09-02。分支：`codex/loop-p1-a03`。协议版本：
`ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1`。

本协议把已经通过软件片段检查的五臂拟合、整块删除、模块支持和共享极点审计接入项目
harness。它只登记 P1 的 `model_unit_gate` 与 `controlled_experiment` 两个阶段，不提前给出
科学结论，也不改变 P1-A03 已冻结的轨迹、模块、模态和物理门槛。


<a id="doc-a9621ce55d83-1-第一性原理目标"></a>

#### 1. 第一性原理目标

受控实验要回答的是：一个由同一摆动骨架和模块并集构成的十机模型，是否能在完全因果的
闭环条件下，产生不能由 B2 或删除后重拟合模型解释的轨迹、储能、端口和共享极点证据。
来源和故障线路只用于等权统计与局部可观测性审计，不能进入方程、参数、闭环输入或模型
路由。若证据不足，保留负结果并维持基线；不得靠增加隐藏输入、挑选事件或降低门槛晋级。


<a id="doc-a9621ce55d83-2-冻结输入"></a>

#### 2. 冻结输入

- 数据：`ieee39_postclear_canonical_v1`，只允许 F1 的 `fit`、`selection` 和
  `internal_audit` 角色。HDF5 SHA256 为
  `3f6546411f3cdbea4de7977968a779b62228fa843e62fbc4f606972eff0e9636`；
  `dataset_index.csv` 为
  `a117bafd2067b5846e15b910f194cdccdedac248f7ede2ee0bae54699111817c`；
  `schema.json` 为
  `5e692018b7a3e9cfda68fe5846b036cf70eefddef680adc7187f9b2ebe3d3cc3`。
- 角色清单来自不可变候选运行
  `paper_ieee39_modular_energy_p1_a03_candidate_acquired_f1_seed324-candidate_acquired-20260831T130327Z-93d29d92`；
  候选 `artifact_hashes.json` SHA256 为
  `6864dfaf8805782b194d9d30fe3eda7c877cdae04a4465b3b1f85d691946daf3`，
  角色清单 SHA256 为
  `1f4d677ab53d3eeee692bc8dee694c02ddfd3718202bd85e795d8d4818e2e3cb`。
- 候选的 `frozen_model.json` SHA256 为
  `c61872fac88b483e788cd2eafe76e33a318d05218f0fe82d7f8eeaf2b499fedc`；其
  `format_version=2` 只作为已登记结构来源，不能直接作为新拟合检查点。
- 原 A03 配置 SHA256 为
  `78d9010e4a62c2fa7e947697e2077c9d3e03431399287fb9933f71f159b23526`；
  工作流定义 SHA256 为
  `a846f44eb80f6ec542150d992fb2273c98c7357c74c27652a55b627eadef58c5`。
- 固定 fold 为 F1、seed 为 324、bootstrap 为 2000 次且 seed 为 324。不得读取 test、
  F2--F5、稳定标签、`Pe`、电压、未来摘要、事件时长或有效掩码。

任何输入文件在首次读取轨迹前都必须通过类型、路径、完整文件清单和哈希检查。一次失败运行
不得修改或续写；重试使用新的 harness run ID。


<a id="doc-a9621ce55d83-3-结构投影与参数血缘"></a>

#### 3. 结构投影与参数血缘

候选旧格式只允许投影以下结构字段：`active_block_instances`、`node_tau=0.2`、
`potential_scale=0.25`、`memory_taus`、`resonances` 和 `max_internal_step=0.01`。
投影文件使用 `format_version=3`，保存源路径、源哈希及
`numerical_coefficients_inherited=false`。旧模型的阻尼、耦合、模块系数、初态或任何拟合数组
均不得读取或复制。

投影后应有一个 `NODE_HIGH_ORDER`、三个 `EDGE_MEMORY` 和三个
`RESONANT_MEMORY` 实例。`source_support` 可作为历史候选说明保留，但模型构造不得按来源选择
实例；最终仍是一个结构、一个联合参数集。五臂拟合从空系数开始，结构不在本轮重新发现。


<a id="doc-a9621ce55d83-4-模型单元门"></a>

#### 4. 模型单元门

`model_unit_gate` 在源码提交前运行，但它只检查候选合同、结构投影、泄漏防护、合成恢复、
反事实语义、模块支持、共享极点审计以及项目 harness/data/git 状态，不读取新的 IEEE39 轨迹。
它必须执行完整测试集并保存 JUnit、命令、返回码、测试数、项目 audit、data-check 和 git-check
证据。全部通过后，工作流才进入 `source_commit`；随后必须冻结一个干净提交，受控实验只运行
该提交。

模型单元门通过不代表模型科学通过。其 `scientific_verdict` 固定为空，
`promotion_decision=KEEP_BASELINE`，`level_6_claimed=false`。


<a id="doc-a9621ce55d83-5-受控父链"></a>

#### 5. 受控父链

一个 harness 结果目录包含以下不可变子目录，并按顺序执行：

1. `five_arm_fit`：读取完整 fit/selection，fit 的未来 `Pm(t)` 只作因果机械功率闭包的训练
   目标；从空参数拟合 B2、当前结构及来源平衡损失；闭包候选只用 selection 选取。
2. `five_arm_internal_evaluation`：第 3 臂检查点冻结后，internal 的 observed `Pm(t)` 只为
   第 4 臂条件上界读取；第 1、2、3、5 臂保持因果。五臂使用相同事件、视界、积分器和尺度。
3. `counterfactual_models`：每个模块实例分别置零，并从空参数重新拟合其余模块；另做全部
   模块联合置零和联合删除重拟合。不能把单模块可替代性推成联合退化。
4. `counterfactual_evaluation`：只在完整 internal audit 上评价参照、置零、删除重拟合以及
   RK4 步长检查；任一事件失败会使相应聚合不可裁定。
5. `module_support`：只用完整 fit 重建条件可辨识性，结合已封存 internal 反事实结果计算
   线路等权效应区间；selection/internal 不得重新参与参数秩或阈值选择。
6. `modal_observability`：极点只来自冻结第 3 臂共享方程；internal 轨迹 DMD 只判断各激励
   视图能否观测该极点，不生成或替换模型极点。

每个子目录都保存自己的严格 artifact manifest；父目录递归封存所有子工件及其哈希，并再次
验证父子绑定。输出只含必要的统计、访问记录和模型工件，不复制原始轨迹数组。


<a id="doc-a9621ce55d83-6-固定计算设置与裁定边界"></a>

#### 6. 固定计算设置与裁定边界

- 五臂校准使用代码登记的 `P1CalibrationSettings` 默认值，`max_nfev=16`；不得使用测试用的
  缩减迭代数。积分最大内部步长为 0.01 s，登记视界为 1、5、10 s。
- 来源—线路—事件—窗口逐级等权；模块和模态 bootstrap 均以故障线路为独立抽样单元。
- internal audit 中每个来源少于 3 条独立线路时，该来源相关模态结论固定为
  `UNDETERMINED`，不能用事件数替代线路数。
- 受控阶段只产生完整诊断证据，`scientific_verdict=null`、
  `promotion_decision=KEEP_BASELINE`、`model_unit_gate_passed=false`。残差诊断、对抗审查和
  acceptance 必须在后续独立阶段完成。
- 任何运行都不得声明 Level 6，也不得使用 `stability-certified equivalent` 名称。


<a id="doc-a9621ce55d83-7-失败回退和可写主张"></a>

#### 7. 失败、回退和可写主张

输入绑定、读取角色、拟合、积分、数值有限性、子工件验证或父封存任一失败，当前 run 终止
并保留失败位置与错误；不生成科学结论，不晋级，不覆盖旧运行。部署和论文主张继续回退到
已接受基线。允许写入论文规划的只有过程事实：协议已登记、某个门是否通过、某类证据是否
可计算；在 acceptance 前不得写“模块受支持”“振荡频率已识别”或“稳定证书成立”。

<a id="doc-e9d90d6e61d9"></a>

<a id="doc-e9d90d6e61d9-p1-a04-真实-f1-时间网格重试预登记"></a>

### P1-A04 真实 F1 时间网格重试预登记

日期：2026-09-02。分支：`codex/loop-p1-a04`。协议版本：
`ieee39_modular_energy_p1_a04_time_grid_retry_v1`。


<a id="doc-e9d90d6e61d9-1-重试原因与边界"></a>

#### 1. 重试原因与边界

P1-A03 已在提交 `78225d2c4330574609580701d3fd24d895f58f3a` 通过模型单元门，但真实
F1 父实验在首个 near-critical dense fit 事件的时间轴检查处失败，拟合未开始。失败 run 为
`paper_ieee39_modular_energy_p1_a03_controlled_experiment_f1_seed324-controlled_experiment-20260901T173705Z-78225d2c`。
该 run 保留且 A03 源码冻结，不回写、不续跑。

只读诊断表明，F1 三个开发角色共 178 个事件全部严格递增并覆盖 10 s。109 个事件与
`0:0.01:10` 完全一致；69 个事件在清除后的首个正时间点存在 0.001--0.009 s 相位偏移，
之后相邻点严格为 0.01 s。最大包围索引为 1001，最大源间隔为
`0.010000000000002451 s`。因此 A04 只修复数据表示的时间基准，不改变模型、角色、门槛、
模块并集、损失、视界或统计单位。


<a id="doc-e9d90d6e61d9-2-冻结输入和复用证据"></a>

#### 2. 冻结输入和复用证据

- 候选运行、角色清单、canonical HDF5/index/schema、结构投影、五臂设置和全部科学门槛与
  A03 受控父链协议完全相同。
- 复用 A03 已通过的模型门仅作历史输入；A04 修改读取器后必须重新执行完整模型单元门。
- A04 的 `retry_registered` 先从角色清单取得完整 F1 `fit/selection/internal_audit` ID，只读取
  对应 HDF5 `postclear/t` 前 1002 点并生成不可变网格审计。不得读取轨迹、`Pm`、`Pe`、
  电压、标签、事件时长、有效掩码、test 或 F2--F5。
- seed 仍为 324，bootstrap 仍为 2000 次；一个语料仍只拟合一个无来源路由的模型。


<a id="doc-e9d90d6e61d9-3-唯一允许的时间统一算法"></a>

#### 3. 唯一允许的时间统一算法

目标网格固定为 1001 点 `t_k = 0.01 k, k=0,...,1000`。每个事件在读取任何状态数组前必须
满足：

1. 源时间首点严格为 0，时间有限且严格递增；
2. 前 1002 个源点中存在首个 `t >= 10` 的包围点；禁止外推；
3. 从首点到包围点的最大间隔不超过 `0.0100000001 s`；
4. 首个正时间点之后的间隔相对 0.01 s 的绝对误差不超过 `1e-12`；
5. 若源网格已与目标网格在 `1e-12` 内一致，逐元素复制，不调用插值；否则对每个发电机通道
   使用一次线性插值；不得平滑、滤波、填零、删事件或改变目标网格。

`delta`、`omega` 和仅作闭包目标/条件上界的 observed `Pm(t)` 使用同一已审计网格变换。
`Pm_clear` 始终取源数据 `t=0` 的原始值，不从插值结果回推。因果主模型仍只接收
`Pm_clear` ZOH；observed `Pm(t)` 不进入主模型。

每次变换记录事件、角色、来源、源点数、首个正时间、包围索引、最大间隔、是否插值和目标
网格哈希。不能只记录成功事件；任一事件失败则整个角色读取失败。


<a id="doc-e9d90d6e61d9-4-对抗检查"></a>

#### 4. 对抗检查

单元与合成测试必须主动攻击：非单调时间、重复时间、缺少 10 s 包围点、大于上限的间隔、
首点非零、非有限时间、通道长度不一致、插值外推、状态与 observed `Pm` 使用不同索引、
`Pm_clear` 被插值改写，以及通过 valid length、标签或来源改变模型路径。

精确在线网格输入必须逐位保持；线性函数必须恢复到舍入误差；COI 投影、功率平衡、初态相等、
1001 点完整视界和无未来 `Pm` 主输入合同必须继续通过。


<a id="doc-e9d90d6e61d9-5-loop-与裁定"></a>

#### 5. Loop 与裁定

A04 工作流为：

`retry_registered → model_unit_gate → source_commit → controlled_experiment → residual_diagnosis → adversarial_review → acceptance → merge_gate/rollback_report → report`。

模型单元门、受控父链、残差诊断和科学门槛沿用 A03。任何过程失败均保留新 run 并保持基线；
受控阶段仍只输出 `scientific_verdict=null` 的诊断摘要，不能在 acceptance 前晋级。A04 不声明
Level 6，不使用稳定认证名称。

## 报告与结果

<a id="doc-f98c5cb5bd43"></a>

<a id="doc-f98c5cb5bd43-ieee39-模块化能量等值-p1-a04-诊断报告"></a>

### IEEE39 模块化能量等值 P1-A04 诊断报告


<a id="doc-f98c5cb5bd43-结论"></a>

#### 结论

- 科学结论：`ACCEPT`
- 模型晋级：`KEEP_BASELINE`
- 边界：Accepted P1 diagnosis on F1 fit/selection/internal-audit evidence only. The candidate model is not promoted; there is no test/F2-F5 performance, identified-frequency, stability-certificate, or Level 6 claim.


<a id="doc-f98c5cb5bd43-三个问题"></a>

#### 三个问题


<a id="doc-f98c5cb5bd43-q1_modules_change_prediction_or_degenerate_to_b2"></a>

##### Q1_modules_change_prediction_or_degenerate_to_B2

The jointly deleted model is statistically worse in at least one registered cell, so the whole candidate does not collapse to B2.  However, no named module has independent predictive support; the observed module-status counts are {'UNIDENTIFIABLE': 7}. The candidate therefore cannot be promoted as a validated module-union model.

边界：Joint-B2 refutation is not individual-module support. Nonzero coefficients are not independent contributions; zero coefficients and failed profile gates remain negative evidence.


<a id="doc-f98c5cb5bd43-q2_why_long_horizon_speed_error_is_large"></a>

##### Q2_why_long_horizon_speed_error_is_large

Numerical integration and COI preservation are not the cause. A causal mechanical-power closure improves the registered point error, but 10 s pooled speed R2 remains -0.225783. The weak local objective improves only 0.673%, all three closed-loop calibrations are rejected after solver failure, and the overall fault-line interval is unavailable. The evidence identifies an unresolved mixture of forcing/dynamic closure and objective-calibration mismatch, not one proven cause.

边界：Point improvements do not replace a missing overall confidence interval. The observed-Pm arm is noncausal and parameter-confounded, and the causal closure has not achieved positive 10 s speed R2.


<a id="doc-f98c5cb5bd43-q3_modal_audit_shared_poles_and_local_observability"></a>

##### Q3_modal_audit_shared_poles_and_local_observability

The implementation follows the v8 semantics: poles come from the one frozen shared equation and trajectory DMD only measures excitation-local observability. On IEEE39, none of the 9 reportable poles is observed; view statuses are {'UNDETERMINED': 9, 'UNSUPPORTED': 9}. No identified-frequency claim is available.

边界：Internal equation poles are not identified IEEE39 oscillation frequencies unless an eligible excitation view supplies the preregistered support.


<a id="doc-f98c5cb5bd43-下一步"></a>

#### 下一步

进入 P2 公平基线与评价合同冻结；当前候选不作为已接受部署模型。

<details>
<summary>原件与版本</summary>

- [docs/protocols/ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1_zh.md](_sealed/ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1_zh.md)
- [docs/protocols/ieee39_modular_energy_p1_a04_time_grid_retry_v1_zh.md](_sealed/ieee39_modular_energy_p1_a04_time_grid_retry_v1_zh.md)
- [docs/reports/03_modular_energy_p1p2/paper_ieee39_modular_energy_p1_a04_report_f1_seed324-report-20260904T150027Z-45123100_p1_a04_zh.md](_sealed/paper_ieee39_modular_energy_p1_a04_report_f1_seed324-report-20260904T150027Z-45123100_p1_a04_zh.md)

</details>
