# 状态端口结构与 KIKI train 对照

合成保存证据核验通过，并保留原运行的汇总写入失败记录。KIKI train 的配对比较有轨迹收益，三个 seed 各多漏一例规则触发，候选判定为 REJECT 并保留基线。

## 历次报告

- [State-dependent port correctness](../state_port_synthetic_20261111_zh.md) · [核验回执](../state_port_synthetic_20261111_evidence.json)。 Fixed ten-node synthetic structure, Jacobian and propagation-gradient checks; 200 saved-evidence checks passed, original summary-write failure retained, no fitting or real-data reads; KEEP_BASELINE
- [State-dependent port train comparison](../kiki_state_port_train_20261111_zh.md) · [核验回执](../kiki_state_port_train_20261111_evidence.json)。 Six fixed 40-update fits on 258 eligible train events; weighted trajectory gains over A06 and matched constant ports, one added rule miss per seed; 76 saved-evidence checks passed, REJECT / KEEP_BASELINE

## 任务组归档

[任务组总说明和小包入口](../../../results/SUMMARY_08_state_port.md)

[返回成果索引](../README.md)
