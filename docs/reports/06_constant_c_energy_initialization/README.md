# 06 · 短历史初始化与能量约束

常数 C 合成试验区分隐藏初值恢复和可见未来预测。固定能量上界改善了部分弱耦合算例的可见预测。固定 A06 模型的 KIKI train 比较保留原编码器，具体结果见迁移报告与收尾核验。

## 历次报告

- [Constant-C 100 ms synthetic study](constant_c_100ms_synthetic_20261109_zh.md) · [核验回执](constant_c_100ms_synthetic_20261109_evidence.json) · [方案登记](../../../configs/proposals/constant_c_100ms_synthetic_v1.json) · [协议](../../protocols/constant_c_100ms_synthetic_v1_zh.md) · [运行结果](../../../results/constant_c_100ms_synthetic_v1-synthetic_audit-20261109T072255936025Z-fcbcd287-40178a64/summary.json) · [归档说明](../../../results/backup/06_constant_c_energy_initialization/01_constant_c_100ms_synthetic.DESCRIPTION.md)。 One completed 17-case study separating sampled identifiability, initialization error and visible future equivalence; zero real-data reads, consumed authority and `KEEP_BASELINE`
- [Fixed-radius confirmation](constant_c_energy_confirmation_20261109_zh.md) · [核验回执](constant_c_energy_confirmation_20261109_evidence.json) · [方案登记](../../../configs/proposals/constant_c_energy_confirmation_v1.json) · [协议](../../protocols/constant_c_energy_confirmation_v1_zh.md) · [运行结果](../../../results/constant_c_energy_confirmation_v1-synthetic_confirmation-20261109T130353444021Z-1edf0ebd-832d4649/summary.json) · [归档说明](../../../results/backup/06_constant_c_energy_initialization/03_constant_c_energy_confirmation.DESCRIPTION.md)。 Six new parameter/noise cases within the exposed matched linear family; all frozen confirmation criteria met, hidden-state failures and zero-baseline comparisons retained; consumed authority and `KEEP_BASELINE`
- [Energy-constrained integral initializer](constant_c_energy_initializer_20261109_zh.md) · [核验回执](constant_c_energy_initializer_20261109_evidence.json) · [方案准备登记](../../../configs/proposals/constant_c_energy_initializer_preparation_v1.json) · [协议](../../protocols/constant_c_energy_initializer_v1_zh.md) · [运行结果](../../../results/constant_c_energy_initializer_v1-synthetic_comparison-20261109T101705739984Z-4d06ae48-ba48733d/summary.json) · [归档说明](../../../results/backup/06_constant_c_energy_initialization/02_constant_c_energy_initializer.DESCRIPTION.md)。 Completed 9-case, 3-radius saved-synthetic comparison: bounded initial energy and improved weak-coupling noisy futures, retained hidden-state errors and inactive mismatch controls; consumed authority and `KEEP_BASELINE`
- [Encoder-referenced energy-ball transfer](kiki_energy_cap_transfer_20261110_zh.md) · [核验回执](kiki_energy_cap_transfer_20261110_evidence.json) · [方案登记](../../../configs/proposals/kiki_energy_cap_transfer_v1.json) · [协议](../../protocols/kiki_energy_cap_transfer_v1_zh.md) · [运行结果](../../../results/kiki_energy_cap_transfer_v1-train_transfer-20261109T150903551189Z-89135853-cf169575/summary.json) · [归档说明](../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.DESCRIPTION.md)。 Completed 774 fixed-A06 train slots; improved unconstrained integral initialization but failed encoder-relative future and rule protections; REJECT, consumed authority and KEEP_BASELINE

## 任务组归档

[任务组总说明](../../../results/SUMMARY_06_constant_c_energy_initialization.md)

<details>
<summary>按小任务找归档（4 个 ZIP）</summary>

| 小任务与包内起读文件 | 文件 |
|---|---|
| [01 常数 C、100 ms 已知真值研究](../../../results/backup/06_constant_c_energy_initialization/01_constant_c_100ms_synthetic.DESCRIPTION.md) | [ZIP](../../../results/backup/06_constant_c_energy_initialization/01_constant_c_100ms_synthetic.zip) |
| [02 能量约束积分初始化器](../../../results/backup/06_constant_c_energy_initialization/02_constant_c_energy_initializer.DESCRIPTION.md) | [ZIP](../../../results/backup/06_constant_c_energy_initialization/02_constant_c_energy_initializer.zip) |
| [03 固定能量上界确认](../../../results/backup/06_constant_c_energy_initialization/03_constant_c_energy_confirmation.DESCRIPTION.md) | [ZIP](../../../results/backup/06_constant_c_energy_initialization/03_constant_c_energy_confirmation.zip) |
| [04 kiki train 能量球迁移](../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.DESCRIPTION.md) | [ZIP](../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.zip) |

</details>

[返回成果索引](../README.md)
