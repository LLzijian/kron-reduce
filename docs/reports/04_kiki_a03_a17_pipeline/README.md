# 04 · KIKI A03–A17 方法改进

早期资料验证了历史输入和初值恢复的作用，随后逐轮比较短历史、目标函数、记忆阶、记录窗口、积分初始化和机械功率输入。正式结果和回执保留每轮的选择、终点及数据范围。200 ms 项是对冻结检查点的回顾性评分。

## 早期历史输入试验

- [隐状态初始化的合成验证与后续算法任务](causal_latent_initialization_20261019_zh.md) · [方案](../../research/causal_latent_initialization_design_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/causal_latent_synth.DESCRIPTION.md)
- [隐状态学习与 IEEE39 F1 历史输入试验](history_learning_f1_20261019_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/ieee39_p2.DESCRIPTION.md)。方案与输入、训练和评分口径见报告正文。
- [P2-A03：历史隐状态与非线性骨架的受控试验](nonlinear_history_a03_20261021_zh.md) · [协议](../../protocols/nonlinear_history_a03_v1_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/ieee39_p2.DESCRIPTION.md)

## A03–A17 与冻结比较

- [Kiki 数据上的 A03 五方案横向对照](kiki_a03_comparison_20261022_zh.md) · [协议](../../protocols/kiki_a03_rerun_20261022_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_p2_a03_rerun.DESCRIPTION.md)
- [当前模型的稳定性预测与 TASK3 算法对照](kiki_a03_stability_task3_20261022_zh.md) · [协议](../../protocols/kiki_a03_stability_20261022_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a03.DESCRIPTION.md)。 Frozen A03 internal-val LOS scoring and unmatched TASK3 algorithm references
- [A04：100 ms历史模型的冻结开发比较](kiki_a04_development_20261023_zh.md) · [核验回执](kiki_a04_development_20261023_evidence.json) · [方案](../../protocols/kiki_a04_short_history_design_20261023_zh.md) · [授权协议](../../protocols/kiki_a04_development_authorization_20261023_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a04.DESCRIPTION.md)。 Completed 12-fit A04 development comparison, trajectory and LOS findings, numerical failure, consumed authorization and retained baseline
- [A04工程与合成机制集成结果](kiki_a04_execution_20261023_zh.md) · [核验回执](kiki_a04_execution_20261023_evidence.json) · [执行前置条件](../../protocols/kiki_a04_envelope_prerequisite_20261023_zh.md)。 Astra-reviewed 100 ms engineering tools, synthetic observability findings, real-data prerequisite refusal and preserved failed attempt
- [下一轮算法设计依据：保留记忆收益，检验物理骨架的共同调整](kiki_a05_recorded_error_audit_20261023_zh.md) · [核验回执](kiki_a05_recorded_error_audit_20261023_evidence.json) · [方案](../../research/kiki_a05_astra_core_design_20261023_zh.md) · [审计协议](../../protocols/kiki_a04_recorded_error_audit_20261023_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/A04_short_history_diag.DESCRIPTION.md)。 JSON-only decomposition of frozen A04 errors, stable/unstable tradeoffs and the proposed fixed-versus-joint backbone training diagnostic
- [A05：共同训练优于永久固定物理骨架](kiki_a05_paired_training_20261024_zh.md) · [核验回执](kiki_a05_paired_training_20261024_evidence.json) · [方案](../../research/kiki_a05_astra_core_design_20261023_zh.md) · [授权协议](../../protocols/kiki_a05_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a05.DESCRIPTION.md)。 Completed six-fit F/J comparison, rejection of permanent backbone freezing, retained numerical failure, zero-fit closeout repair and consumed authorization
- [A06：共同微调将失稳漏判降至每次1例](kiki_a06_joint_refinement_20261024_zh.md) · [核验回执](kiki_a06_joint_refinement_20261024_evidence.json) · [方案](../../research/kiki_a06_astra_design_20261024_zh.md) · [授权协议](../../protocols/kiki_a06_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a06.DESCRIPTION.md)。 Three full-model refinements, one remaining FN per seed, complete numerical gates, retained coverage and a consumed fixed budget
- [A07：固定目标对照未达到零漏判，拒绝本次终点](kiki_a07_tilted_objective_20261024_zh.md) · [核验回执](kiki_a07_tilted_objective_20261024_evidence.json) · [方案](../../research/kiki_a07_astra_design_20261024_zh.md) · [授权协议](../../protocols/kiki_a07_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a07.DESCRIPTION.md)
- [A08：四维被动记忆改善轨迹，失稳漏判保持 1/14](kiki_a08_memory_order_20261024_zh.md) · [核验回执](kiki_a08_memory_order_20261024_evidence.json) · [方案](../../research/kiki_a08_astra_design_20261024_zh.md) · [授权协议](../../protocols/kiki_a08_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a08.DESCRIPTION.md)
- [A09：train 的五点越界形状审计](kiki_a09_train_mechanism_20261024_zh.md) · [核验回执](kiki_a09_train_mechanism_20261024_evidence.json) · [方案](../../research/kiki_a09_astra_design_20261024_zh.md) · [授权协议](../../protocols/kiki_a09_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a09.DESCRIPTION.md)
- [A10：全体 train 的积分精化审计](kiki_a10_train_refinement_20261024_zh.md) · [核验回执](kiki_a10_train_refinement_20261024_evidence.json) · [方案](../../research/kiki_a10_astra_design_20261024_zh.md) · [授权协议](../../protocols/kiki_a10_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a10.DESCRIPTION.md)
- [268 条 train 通过记录 20 ms 输入合同审计](kiki_a12_record20_20261024_zh.md) · [核验回执](kiki_a12_record20_20261024_evidence.json) · [方案登记](../../../configs/proposals/kiki_a12_record20_reads.json) · [授权协议](../../protocols/kiki_a12_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a12.DESCRIPTION.md)
- [固定模型积分初态估计：合成正确性证据](kiki_a13_integral_synthetic_20261024_zh.md) · [核验回执](kiki_a13_integral_synthetic_20261024_evidence.json) · [方案与固定协议](../../research/kiki_a13_integral_synthetic/integral_initializer_fixed_protocol_zh.md) · [本机保存摘要](../../../scratch/one_off/kiki-initializer-synthetic-20261024/run-20261024T181442748/summary.json)
- [A14：固定模型的积分初态将历史重建误差降低约一半](kiki_a14_integral_history_20261024_zh.md) · [核验回执](kiki_a14_integral_history_20261024_evidence.json) · [方案登记](../../../configs/proposals/kiki_a14_integral_history_reads.json) · [授权协议](../../protocols/kiki_a14_development_authorization_20261024_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a14.DESCRIPTION.md)
- [A16：历史机械功率近似与初态方程](kiki_a16_pm_impulse_20261025_zh.md) · [核验回执](kiki_a16_pm_impulse_20261025_evidence.json) · [方案](../../research/kiki_a16_pm_impulse/design_zh.md) · [授权协议](../../protocols/kiki_a16_development_authorization_20261025_zh.md) · [归档说明](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a16.DESCRIPTION.md)

其他回执与附件

- [kiki_a03_stability_20261022_verification.json](kiki_a03_stability_20261022_verification.json)
- [TASK3 frozen 200 ms result receipt](kiki_task3_frozen_200ms_20261109_evidence.json) · [方案登记](../../../configs/proposals/kiki_task3_frozen_200ms.json) · [协议](../../protocols/kiki_task3_frozen_200ms_20261109_zh.md) · [val 评分结果](../../../results/kiki_task3_frozen_200ms_20261109-20261109T024831303717Z-aa86374c/val-sine-no-history-loss.json)。 Retrospective A03 checkpoint comparison on 105 `val` and 75 previously exposed `test` events, with a consumed authorization and `KEEP_BASELINE`

## 任务组归档

[任务组总说明](../../../results/SUMMARY_04_kiki_a03_a17_pipeline.md)

<details>
<summary>按小任务找归档（19 个 ZIP）</summary>

| 小任务与包内起读文件 | 文件 |
|---|---|
| [A04 短历史 / 记录错误诊断](../../../results/backup/04_kiki_a03_a17_pipeline/A04_short_history_diag.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/A04_short_history_diag.zip) |
| [因果潜在 合成审计](../../../results/backup/04_kiki_a03_a17_pipeline/causal_latent_synth.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/causal_latent_synth.zip) |
| [IEEE39 P2 · 非线性/历史机制](../../../results/backup/04_kiki_a03_a17_pipeline/ieee39_p2.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/ieee39_p2.zip) |
| [KIKI A03 · 稳定性评估](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a03.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a03.zip) |
| [KIKI A04 · 短历史](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a04.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a04.zip) |
| [KIKI A05 · paired + closeout](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a05.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a05.zip) |
| [KIKI A06 · 联合精化](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a06.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a06.zip) |
| [KIKI A07 · tilted 目标](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a07.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a07.zip) |
| [KIKI A08 · 记忆阶数](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a08.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a08.zip) |
| [KIKI A09 · 训练机制](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a09.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a09.zip) |
| [KIKI A10 · 训练精化](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a10.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a10.zip) |
| [KIKI A11 · 时序目标](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a11.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a11.zip) |
| [KIKI A12 · 记录20输入审计](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a12.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a12.zip) |
| [KIKI A14 · 积分历史](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a14.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a14.zip) |
| [KIKI A15 · 未来迁移](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a15.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a15.zip) |
| [KIKI A16 · Pm 冲量](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a16.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a16.zip) |
| [KIKI A17 · 输入一致性](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a17.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a17.zip) |
| [KIKI P2-a03 · 重跑](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_p2_a03_rerun.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_p2_a03_rerun.zip) |
| [非线性历史 合成](../../../results/backup/04_kiki_a03_a17_pipeline/nonlinear_history_synth.DESCRIPTION.md) | [ZIP](../../../results/backup/04_kiki_a03_a17_pipeline/nonlinear_history_synth.zip) |

</details>

[返回成果索引](../README.md)




