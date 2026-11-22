# 04 · KIKI A03–A17 方法改进

早期资料验证了历史输入和初值恢复的作用，随后逐轮比较短历史、目标函数、记忆阶、记录窗口、积分初始化和机械功率输入。正式结果和回执保留每轮的选择、终点及数据范围。200 ms 项是对冻结检查点的回顾性评分。

## 实验与研究记录

| 轮次与材料 | 同轮资料 |
|---|---|
| [causal_latent · 隐状态初始化的合成验证与后续算法任务](causal_latent/README.md) | 研究方案 · 报告与结果 · 运行或归档 |
| [history_learning_f1 · 隐状态学习与 IEEE39 F1 历史输入试验](history_learning_f1/README.md) | 报告与结果 · 运行或归档 |
| [a03 · P2-A03：历史隐状态与非线性骨架的受控试验](a03/README.md) | 研究方案 · 执行协议 · 报告与结果 · 运行或归档 |
| [a03_kiki_rerun · Kiki 数据上的 A03 五方案横向对照](a03_kiki_rerun/README.md) | 执行协议 · 报告与结果 · 运行或归档 |
| [a03_stability · 当前模型的稳定性预测与 TASK3 算法对照](a03_stability/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [a04_development · A04：100 ms历史模型的冻结开发比较](a04_development/README.md) | 研究方案 · 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [a04_execution · A04工程与合成机制集成结果](a04_execution/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 |
| [a05 · A05：共同训练优于永久固定物理骨架](a05/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 附记 · 回执 · 运行或归档 |
| [a05_recorded_error_audit · 下一轮算法设计依据：保留记忆收益，检验物理骨架的共同调整](a05_recorded_error_audit/README.md) | 执行协议 · 报告与结果 · 历史记录 · 回执 · 运行或归档 |
| [a06 · A06：共同微调将失稳漏判降至每次1例](a06/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [a07 · A07：固定目标对照未达到零漏判，拒绝本次终点](a07/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [a08 · A08：四维被动记忆改善轨迹，失稳漏判保持 1/14](a08/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [a09 · A09：train 的五点越界形状审计](a09/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [a10 · A10：全体 train 的积分精化审计](a10/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [a12 · 268 条 train 通过记录 20 ms 输入合同审计](a12/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [a13 · 固定模型积分初态估计：合成正确性证据](a13/README.md) | 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 附记 · 回执 |
| [a14 · A14：固定模型的积分初态将历史重建误差降低约一半](a14/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 附记 · 回执 · 运行或归档 |
| [a16 · A16：历史机械功率近似与初态方程](a16/README.md) | 研究方案 · 执行协议 · 报告与结果 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |
| [task3_frozen_200ms · TASK3 200 ms 冻结参数评估](task3_frozen_200ms/README.md) | 执行协议 · 复核与历史版本 · 历史记录 · 回执 · 运行或归档 |

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
