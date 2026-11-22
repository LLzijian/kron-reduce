# P1-A03 真实 F1 受控父链预登记

日期：2026-09-02。分支：`codex/loop-p1-a03`。协议版本：
`ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1`。

本协议把已经通过软件片段检查的五臂拟合、整块删除、模块支持和共享极点审计接入项目
harness。它只登记 P1 的 `model_unit_gate` 与 `controlled_experiment` 两个阶段，不提前给出
科学结论，也不改变 P1-A03 已冻结的轨迹、模块、模态和物理门槛。

## 1. 第一性原理目标

受控实验要回答的是：一个由同一摆动骨架和模块并集构成的十机模型，是否能在完全因果的
闭环条件下，产生不能由 B2 或删除后重拟合模型解释的轨迹、储能、端口和共享极点证据。
来源和故障线路只用于等权统计与局部可观测性审计，不能进入方程、参数、闭环输入或模型
路由。若证据不足，保留负结果并维持基线；不得靠增加隐藏输入、挑选事件或降低门槛晋级。

## 2. 冻结输入

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

## 3. 结构投影与参数血缘

候选旧格式只允许投影以下结构字段：`active_block_instances`、`node_tau=0.2`、
`potential_scale=0.25`、`memory_taus`、`resonances` 和 `max_internal_step=0.01`。
投影文件使用 `format_version=3`，保存源路径、源哈希及
`numerical_coefficients_inherited=false`。旧模型的阻尼、耦合、模块系数、初态或任何拟合数组
均不得读取或复制。

投影后应有一个 `NODE_HIGH_ORDER`、三个 `EDGE_MEMORY` 和三个
`RESONANT_MEMORY` 实例。`source_support` 可作为历史候选说明保留，但模型构造不得按来源选择
实例；最终仍是一个结构、一个联合参数集。五臂拟合从空系数开始，结构不在本轮重新发现。

## 4. 模型单元门

`model_unit_gate` 在源码提交前运行，但它只检查候选合同、结构投影、泄漏防护、合成恢复、
反事实语义、模块支持、共享极点审计以及项目 harness/data/git 状态，不读取新的 IEEE39 轨迹。
它必须执行完整测试集并保存 JUnit、命令、返回码、测试数、项目 audit、data-check 和 git-check
证据。全部通过后，工作流才进入 `source_commit`；随后必须冻结一个干净提交，受控实验只运行
该提交。

模型单元门通过不代表模型科学通过。其 `scientific_verdict` 固定为空，
`promotion_decision=KEEP_BASELINE`，`level_6_claimed=false`。

## 5. 受控父链

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

## 6. 固定计算设置与裁定边界

- 五臂校准使用代码登记的 `P1CalibrationSettings` 默认值，`max_nfev=16`；不得使用测试用的
  缩减迭代数。积分最大内部步长为 0.01 s，登记视界为 1、5、10 s。
- 来源—线路—事件—窗口逐级等权；模块和模态 bootstrap 均以故障线路为独立抽样单元。
- internal audit 中每个来源少于 3 条独立线路时，该来源相关模态结论固定为
  `UNDETERMINED`，不能用事件数替代线路数。
- 受控阶段只产生完整诊断证据，`scientific_verdict=null`、
  `promotion_decision=KEEP_BASELINE`、`model_unit_gate_passed=false`。残差诊断、对抗审查和
  acceptance 必须在后续独立阶段完成。
- 任何运行都不得声明 Level 6，也不得使用 `stability-certified equivalent` 名称。

## 7. 失败、回退和可写主张

输入绑定、读取角色、拟合、积分、数值有限性、子工件验证或父封存任一失败，当前 run 终止
并保留失败位置与错误；不生成科学结论，不晋级，不覆盖旧运行。部署和论文主张继续回退到
已接受基线。允许写入论文规划的只有过程事实：协议已登记、某个门是否通过、某类证据是否
可计算；在 acceptance 前不得写“模块受支持”“振荡频率已识别”或“稳定证书成立”。

