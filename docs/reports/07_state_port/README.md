# 07 · 状态端口结构与 KIKI train 对照

合成保存证据核验通过，并保留原运行的汇总写入失败记录。KIKI train 的配对比较有轨迹收益，三个 seed 各多漏一例规则触发，候选判定为 REJECT 并保留基线。

## 历次报告

- [State-dependent port correctness](state_port_synthetic_20261111_zh.md) · [核验回执](state_port_synthetic_20261111_evidence.json) · [方案登记](../../../configs/proposals/state_port_synthetic_v1.json) · [协议](../../protocols/state_port_synthetic_v1_zh.md) · [原运行结果（含失败状态）](../../../results/state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0/summary.json) · [保存审核归档说明](../../../results/backup/07_state_port/01_state_port_synthetic_saved_audit.DESCRIPTION.md)。 Fixed ten-node synthetic structure, Jacobian and propagation-gradient checks; 200 saved-evidence checks passed, original summary-write failure retained, no fitting or real-data reads; KEEP_BASELINE
- [State-dependent port train comparison](kiki_state_port_train_20261111_zh.md) · [核验回执](kiki_state_port_train_20261111_evidence.json) · [方案登记](../../../configs/proposals/kiki_state_port_train_v1.json) · [协议](../../protocols/kiki_state_port_train_v1_zh.md) · [运行结果](../../../results/kiki_state_port_train_v1-train_mechanism-20261111T012130667376Z-5fb30e45-5ac1c107/summary.json) · [保存审核归档说明](../../../results/backup/07_state_port/02_kiki_state_port_train_saved_audit.DESCRIPTION.md)。 Six fixed 40-update fits on 258 eligible train events; weighted trajectory gains over A06 and matched constant ports, one added rule miss per seed; 76 saved-evidence checks passed, REJECT / KEEP_BASELINE

## 任务组归档

[任务组总说明](../../../results/SUMMARY_07_state_port.md)

<details>
<summary>按小任务找归档（2 个 ZIP）</summary>

| 小任务与包内起读文件 | 文件 |
|---|---|
| [01 状态端口合成保存审核](../../../results/backup/07_state_port/01_state_port_synthetic_saved_audit.DESCRIPTION.md) | [ZIP](../../../results/backup/07_state_port/01_state_port_synthetic_saved_audit.zip) |
| [02 kiki 状态端口 train 保存审核](../../../results/backup/07_state_port/02_kiki_state_port_train_saved_audit.DESCRIPTION.md) | [ZIP](../../../results/backup/07_state_port/02_kiki_state_port_train_saved_audit.zip) |

</details>

[返回成果索引](../README.md)
