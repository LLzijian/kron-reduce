# 04 · KIKI A03–A17 方法改进

早期资料验证了历史输入和初值恢复的作用，随后逐轮比较短历史、目标函数、记忆阶、记录窗口、积分初始化和机械功率输入。正式结果和回执保留每轮的选择、终点及数据范围。200 ms 项是对冻结检查点的回顾性评分。

## 早期历史输入试验

- [隐状态初始化的合成验证与后续算法任务](causal_latent_initialization_20261019_zh.md)
- [隐状态学习与 IEEE39 F1 历史输入试验](history_learning_f1_20261019_zh.md)
- [P2-A03：历史隐状态与非线性骨架的受控试验](nonlinear_history_a03_20261021_zh.md)

## A03–A17 与冻结比较

- [Kiki 数据上的 A03 五方案横向对照](kiki_a03_comparison_20261022_zh.md)
- [当前模型的稳定性预测与 TASK3 算法对照](kiki_a03_stability_task3_20261022_zh.md)。 Frozen A03 internal-val LOS scoring and unmatched TASK3 algorithm references
- [A04：100 ms历史模型的冻结开发比较](kiki_a04_development_20261023_zh.md) · [核验回执](kiki_a04_development_20261023_evidence.json)。 Completed 12-fit A04 development comparison, trajectory and LOS findings, numerical failure, consumed authorization and retained baseline
- [A04工程与合成机制集成结果](kiki_a04_execution_20261023_zh.md) · [核验回执](kiki_a04_execution_20261023_evidence.json)。 Astra-reviewed 100 ms engineering tools, synthetic observability findings, real-data prerequisite refusal and preserved failed attempt
- [下一轮算法设计依据：保留记忆收益，检验物理骨架的共同调整](kiki_a05_recorded_error_audit_20261023_zh.md) · [核验回执](kiki_a05_recorded_error_audit_20261023_evidence.json)。 JSON-only decomposition of frozen A04 errors, stable/unstable tradeoffs and the proposed fixed-versus-joint backbone training diagnostic
- [A05：共同训练优于永久固定物理骨架](kiki_a05_paired_training_20261024_zh.md) · [核验回执](kiki_a05_paired_training_20261024_evidence.json)。 Completed six-fit F/J comparison, rejection of permanent backbone freezing, retained numerical failure, zero-fit closeout repair and consumed authorization
- [A06：共同微调将失稳漏判降至每次1例](kiki_a06_joint_refinement_20261024_zh.md) · [核验回执](kiki_a06_joint_refinement_20261024_evidence.json)。 Three full-model refinements, one remaining FN per seed, complete numerical gates, retained coverage and a consumed fixed budget
- [A07：固定目标对照未达到零漏判，拒绝本次终点](kiki_a07_tilted_objective_20261024_zh.md) · [核验回执](kiki_a07_tilted_objective_20261024_evidence.json)
- [A08：四维被动记忆改善轨迹，失稳漏判保持 1/14](kiki_a08_memory_order_20261024_zh.md) · [核验回执](kiki_a08_memory_order_20261024_evidence.json)
- [A09：train 的五点越界形状审计](kiki_a09_train_mechanism_20261024_zh.md) · [核验回执](kiki_a09_train_mechanism_20261024_evidence.json)
- [A10：全体 train 的积分精化审计](kiki_a10_train_refinement_20261024_zh.md) · [核验回执](kiki_a10_train_refinement_20261024_evidence.json)
- [268 条 train 通过记录 20 ms 输入合同审计](kiki_a12_record20_20261024_zh.md) · [核验回执](kiki_a12_record20_20261024_evidence.json)
- [固定模型积分初态估计：合成正确性证据](kiki_a13_integral_synthetic_20261024_zh.md) · [核验回执](kiki_a13_integral_synthetic_20261024_evidence.json)
- [A14：固定模型的积分初态将历史重建误差降低约一半](kiki_a14_integral_history_20261024_zh.md) · [核验回执](kiki_a14_integral_history_20261024_evidence.json)
- [A16：历史机械功率近似与初态方程](kiki_a16_pm_impulse_20261025_zh.md) · [核验回执](kiki_a16_pm_impulse_20261025_evidence.json)

其他回执与附件

- [kiki_a03_stability_20261022_verification.json](kiki_a03_stability_20261022_verification.json)
- [TASK3 frozen 200 ms result receipt](kiki_task3_frozen_200ms_20261109_evidence.json)。 Retrospective A03 checkpoint comparison on 105 `val` and 75 previously exposed `test` events, with a consumed authorization and `KEEP_BASELINE`

## 任务组归档

[任务组总说明和小包入口](../../../results/SUMMARY_04_kiki_a03_a17_pipeline.md)

[返回成果索引](../README.md)
