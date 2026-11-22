# IEEE39 模块化能量 P1 A01 冻结协议

版本：`ieee39_modular_energy_p1_a01_v1`。本协议只注册 P1 当前实现诊断的执行边界和门槛，不产生新的科学结论。

## 数据、模型与访问边界

- 分支固定为 `codex/loop-p1-a01`；数据集为 `ieee39_postclear_canonical_v1`；开发折为 F1，seed 为 324。
- 参照 run 固定为 `paper_combined_ten_machine_identification_f1_seed324-identification_run-20260825T015128Z-d4d50de4`。
- 一个语料只拟合一个模型（`model_count=1`）；禁止 source routing。
- test 以及 F2--F5 全程锁定，不得用于拟合、归一化、校准、选择或门槛生成。
- 主表机械功率为 `Pm_clear` 的 ZOH；observed `Pm(t)` 仅作为条件上界，不得成为主模型输入。
- bootstrap 固定 2000 draws、seed 324；任何模态视图少于 3 条独立故障线路即为 `UNDETERMINED`。

## 五个受控臂

1. `b2_zoh_original_loss`：B2 参照模型、`Pm_clear` ZOH 与原始损失。
2. `current_zoh_original_loss`：当前候选、`Pm_clear` ZOH 与原始损失。
3. `current_zoh_balanced_loss`：当前候选、`Pm_clear` ZOH 与来源平衡损失。
4. `current_observed_pm_balanced_loss_conditional_upper_bound`：当前候选、observed `Pm(t)` 与来源平衡损失的条件上界；该臂不能晋级，也不能成为主表或部署模型。
5. `current_causal_pm_closure_balanced_loss`：当前候选、因果机械功率闭包与来源平衡损失。

各臂在相同 F1 数据角色、视界、积分器和统计口径下运行，保存角度/速度分开指标、逐线路配对差与最差 10% 轨迹。

## 双重模块审计与门槛

模块审计必须同时给出：(1) 不重拟合参数的模块置零反事实，直接测量动力效应；(2) 整块删除后重新拟合全部剩余参数，测量该模块能否被替代。两类审计均按逐事件、逐视界保存预测变化，并记录参数范数、储能贡献和端口功率。阈值冻结如下：

- `tau_y >= max(1e-3, 10 * trajectory_numerical_error)`；`tau_phys >= max(1e-4, 100 * supply_identity_error)`，其中 `supply_identity_error` 是供能恒等式误差；删除退化不超过 2%。
- 可辨识性：sigma ratio 不低于 `1e-6`、condition 不高于 `1e6`、profile 容差不超过 2%。
- 速度改善至少 5%，非目标退化不超过 2%；因果闭包至少恢复 observed 条件上界差距的 50%。
- 平衡残差不超过 `1e-7`、可见度至少 `1e-3`、有限窗能量至少 1%。
- COI 漂移不超过 `1e-10`、供能恒等式误差不超过 `1e-8`、RK4 收敛比为 8--32。

## 模态规则

频率主张来自共享方程极点；激励视图只审计局部可观性。可对照模态需满足频率相对误差不超过 5%、阻尼绝对误差不超过 0.05、MAC 不低于 0.8，且至少一个预登记视图的 bootstrap 支持不低于 0.8。少于三条独立线路时结论必须是 `UNDETERMINED`。

## 工件清单与路由

必须保存 `module_counterfactual_metrics.csv`、`module_energy_port_audit.csv`、`pm_closure_factorial.csv`、`loss_scale_gradient_audit.csv`、`shared_pole_local_observability.csv`、验收 JSON、工件哈希和简明报告。

工作流严格为：`scaffold -> plan_registered -> candidate_acquired -> model_unit_gate -> source_commit -> controlled_experiment -> residual_diagnosis -> adversarial_review -> acceptance`。`ACCEPT` 只能到需人工批准的 `merge_gate`，随后到 `report`；`RETRY` 或 `REJECT` 到 `rollback_report`，随后到 `report`。`report` 是唯一终态。P1 的 `ACCEPT` 也只是接受诊断，不晋级模型，`promotion_decision` 必须为 `KEEP_BASELINE`。验收必须记录状态、过程状态、科学裁定、晋级决定、回退目标、主张边界和可用于论文的主张；回退报告还必须记录负证据 run 与下一假设；最终报告必须记录终止分支、非空验收 run、非空工件哈希和报告路径。来自 `merge_gate` 的报告必须记录 `merge_commit`；来自 `rollback_report` 的报告必须记录 `rollback_target` 和 `negative_evidence_run`。
