# 短历史初始化与能量约束

常数 C 合成试验区分隐藏初值恢复和可见未来预测。固定能量上界改善了部分弱耦合算例的可见预测。固定 A06 模型的 KIKI train 比较保留原编码器，具体结果见迁移报告与收尾核验。

## 历次报告

- [Constant-C 100 ms synthetic study](../constant_c_100ms_synthetic_20261109_zh.md) · [核验回执](../constant_c_100ms_synthetic_20261109_evidence.json)。 One completed 17-case study separating sampled identifiability, initialization error and visible future equivalence; zero real-data reads, consumed authority and `KEEP_BASELINE`
- [Fixed-radius confirmation](../constant_c_energy_confirmation_20261109_zh.md) · [核验回执](../constant_c_energy_confirmation_20261109_evidence.json)。 Six new parameter/noise cases within the exposed matched linear family; all frozen confirmation criteria met, hidden-state failures and zero-baseline comparisons retained; consumed authority and `KEEP_BASELINE`
- [Energy-constrained integral initializer](../constant_c_energy_initializer_20261109_zh.md) · [核验回执](../constant_c_energy_initializer_20261109_evidence.json)。 Completed 9-case, 3-radius saved-synthetic comparison: bounded initial energy and improved weak-coupling noisy futures, retained hidden-state errors and inactive mismatch controls; consumed authority and `KEEP_BASELINE`
- [Encoder-referenced energy-ball transfer](../kiki_energy_cap_transfer_20261110_zh.md) · [核验回执](../kiki_energy_cap_transfer_20261110_evidence.json)。 Completed 774 fixed-A06 train slots; improved unconstrained integral initialization but failed encoder-relative future and rule protections; REJECT, consumed authority and KEEP_BASELINE

## 任务组归档

[任务组总说明和小包入口](../../../results/SUMMARY_07_constant_c_energy_initialization.md)

[返回成果索引](../README.md)
