# 08 · 逐机隐状态与拓扑修正

S0 的实现核对通过，S1 的六组必检合成预测配对未过门。后续通过合成真值和 TASK3 对照拆查初值、耦合、阻尼和恢复力误差。各报告分别记录局部门槛和模型选择，研究汇总串起了保留基线的依据。

## 实验与研究记录

| 轮次与材料 | 同轮资料 |
|---|---|
| [synthetic_implementation · 逐机隐状态模型 S0 正确性验证](synthetic_implementation/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [clean_learning · Clean learning 单轮结果（2026-11-13）](clean_learning/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [nonlinear_history · 非线性历史初始化 S0 结果](nonlinear_history/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [gaussian_oracle · Gaussian information oracle 诊断结果（2026-11-12）](gaussian_oracle/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [initializer_diagnosis · 逐机隐状态初始化诊断](initializer_diagnosis/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [joint_observer · Joint observer 确认结果（2026-11-12）](joint_observer/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [observer · 逐机模型离散历史初始化验证](observer/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [hidden_oracle · 逐机隐状态：初始化与动力学误差的完整合成对照](hidden_oracle/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [equation · 逐机方程与函数项恢复：合成结果](equation/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [memory_calibration · 合成记忆核系数校准](memory_calibration/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [memory_diagnosis · 合成记忆核替换诊断](memory_diagnosis/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [stiffness_calibration · 共享刚度参数的完整合成验证](stiffness_calibration/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_angle_damping · 功角相关阻尼的合成配对验证](synthetic_angle_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_ball_history · 固定拓扑下的能量球历史求解](synthetic_ball_history/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_ball_history_no_prior · 零先验条件下的历史求解几何](synthetic_ball_history_no_prior/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_delayed · 延迟失步下的余量梯度标定与合成对照](synthetic_delayed/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_delayed_damping · 单参数因果阻尼拟合结果](synthetic_delayed_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_delayed_origin · 延迟失步的起报隐态与动力学参数对照](synthetic_delayed_origin/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_diagnosis · 冻结共享系数下的耦合与记忆起报点诊断](synthetic_diagnosis/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_learning · 逐机耦合与记忆模型的共享系数学习](synthetic_learning/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_margin · 五点失步余量的合成训练结果](synthetic_margin/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_momentum · 逐机记忆动量诊断的合成验证](synthetic_momentum/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_refinement · 耦合与记忆模型的长时域精化](synthetic_refinement/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_restoring_shape · 合成恢复力形状识别与闭环预测](synthetic_restoring_shape/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_topology · 逐机耦合与记忆隐变量 S0 合成结果](synthetic_topology/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_spatial_damping · 逐机功角阻尼的合成配对验证](synthetic_spatial_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_transfer · 四种耦合与记忆方程在新合成激励下的表现](synthetic_transfer/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_dynamic_coupling · 动态耦合合成验证](synthetic_dynamic_coupling/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_spatial_coupling · 零和空间耦合合成验证](synthetic_spatial_coupling/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [synthetic_spatial_coupling_causal · 零和耦合通过因果历史恢复对照](synthetic_spatial_coupling_causal/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_baseline · KIKI TASK3 全名单 L10 学习结果](task3_baseline/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [task3_baseline_memory · TASK3 逐机记忆动量校准结果](task3_baseline_memory/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_r20_r21 · R20/R21 完整 TASK3 配对结果](task3_r20_r21/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_topology_memory · TASK3 逐机记忆与事件耦合修正对照](task3_topology_memory/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_witness · TASK3 train 五点失步证据核查](task3_witness/README.md) | 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_angle_damping · TASK3 功角阻尼完整评估](task3_angle_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_angle_scale · TASK3 train 功角尺度的固定初态对照](task3_angle_scale/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_damping · TASK3 逐机记忆模型的阻尼校准](task3_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_modal_damping · TASK3 train 固定模态阻尼对照](task3_modal_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_momentum · TASK3 train 逐机记忆动量诊断](task3_momentum/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_momentum_damping · TASK3 动量辨识阻尼的完整评估](task3_momentum_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_no_prior · TASK3 零先验径向初始化](task3_no_prior/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_origin_transfer · TASK3 train 隐藏初态的跨时段动量检验](task3_origin_transfer/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_radius · TASK3 冻结逐机记忆半径对照](task3_radius/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_restoring_shape · TASK3 恢复力形状完整评估](task3_restoring_shape/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_spatial_damping · TASK3 逐机功角阻尼完整评估](task3_spatial_damping/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_dynamic_coupling · TASK3 完整 train 共同动态耦合诊断](task3_dynamic_coupling/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_spatial_coupling · TASK3 train 零和空间耦合路径诊断](task3_spatial_coupling/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [task3_spatial_relaxation · 逐机零和耦合的完整 TASK3 对照](task3_spatial_relaxation/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [statistical_initializer · 逐机模型的 train-only 统计初始化计划](statistical_initializer/README.md) | 研究方案 |
| [synthetic_closeout · 逐机耦合与记忆 S0 结果补录](synthetic_closeout/README.md) | 执行协议 · 历史记录 · 回执 |

## 任务组归档

[任务组总说明](../../../results/SUMMARY_08_machine_local_topology.md)

<details>
<summary>按小任务找归档（16 个 ZIP）</summary>

| 小任务与包内起读文件 | 文件 |
|---|---|
| [S0 · 逐机隐状态模型实现正确性验证](../../../results/backup/08_machine_local_topology/01_s0_implementation.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/01_s0_implementation.zip) |
| [S0 实现检查：归档证据指南](../../../results/backup/08_machine_local_topology/01_s0_implementation_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/01_s0_implementation_saved_checks.zip) |
| [S1-base · 逐机模型合成学习对照](../../../results/backup/08_machine_local_topology/02_s1_synthetic_learning.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/02_s1_synthetic_learning.zip) |
| [S1 合成学习检查：归档证据指南](../../../results/backup/08_machine_local_topology/02_s1_synthetic_learning_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/02_s1_synthetic_learning_saved_checks.zip) |
| [KIKI · 逐机基线记忆模型（全量 / 记忆 / r20-r21）](../../../results/backup/08_machine_local_topology/03_kiki_baseline_memory.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/03_kiki_baseline_memory.zip) |
| [KIKI 基线记忆检查：归档证据指南](../../../results/backup/08_machine_local_topology/03_kiki_baseline_memory_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/03_kiki_baseline_memory_saved_checks.zip) |
| [拓扑诊断与学习（diagnosis / learning / refinement / margin）](../../../results/backup/08_machine_local_topology/04_topology_diagnosis_learning.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/04_topology_diagnosis_learning.zip) |
| [拓扑诊断与学习检查：归档证据指南](../../../results/backup/08_machine_local_topology/04_topology_diagnosis_learning_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/04_topology_diagnosis_learning_saved_checks.zip) |
| [阻尼类拓扑修正（damping / angle / spatial / modal / angle-scale）](../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/05_damping_corrections.zip) |
| [阻尼修正检查：归档证据指南](../../../results/backup/08_machine_local_topology/05_damping_corrections_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/05_damping_corrections_saved_checks.zip) |
| [动量与耦合修正（momentum / dynamic_coupling / spatial_coupling）](../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/06_momentum_coupling.zip) |
| [动量耦合检查：归档证据指南](../../../results/backup/08_machine_local_topology/06_momentum_coupling_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/06_momentum_coupling_saved_checks.zip) |
| [恢复力 / 延迟 / 球历史 / 半径 / 空间松弛修正](../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.zip) |
| [恢复力与延迟历史检查：归档证据指南](../../../results/backup/08_machine_local_topology/07_restoring_delayed_history_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/07_restoring_delayed_history_saved_checks.zip) |
| [观察器、oracle 与参数校准诊断](../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.zip) |
| [观察器与校准诊断检查：归档证据指南](../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis_saved_checks.DESCRIPTION.md) | [ZIP](../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis_saved_checks.zip) |

</details>

[返回成果索引](../README.md)
