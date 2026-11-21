# 故障后统一结构与独立盲测

M3b 冻结了单 G2 模型。M4 检查了故障后主模态和起报储能。M5 对同源新仿真的 94 个封存事件独立评分，冻结模型检出 29/32 例失稳，62 例稳定事件均判稳。固定 test 对照和独立盲测各自的样本范围见对应报告。

## M1 首摆与初值

- [TASK3 M1 first-swing loss ablation](unified_first_swing_fit_20261119_zh.md) · [核验回执](../unified_first_swing_fit_20261119_evidence.json)。 Complete train and val, 18 updates per group, both selecting the shared starting checkpoint
- [TASK3 M1 gradient and initializer probes](unified_first_swing_probes_v2_20261119_zh.md) · [核验回执](../unified_first_swing_probes_v2_20261119_evidence.json)。 Full train and val diagnostics with zero updates, adaptive GN detected 12/12 val instabilities and exceeded the stable trajectory limit
- [TASK3 M1 GN prior and radius comparison](../unified_first_swing_gn_decomposition_v3_20261119_zh.md) · [核验回执](../unified_first_swing_gn_decomposition_v3_20261119_evidence.json)。 Complete train and val, two fixed-prior soft-radius settings detected 12/12 val instabilities, no setting passed the joint gate
- [TASK3 M1 V4 fit and frozen comparison](unified_first_swing_v4_20261120_zh.md) · [拟合回执](../unified_first_swing_v4_fit_20261119_evidence.json) · [固定 test 回执](../unified_first_swing_v4_test_20261119_evidence.json)。 Both fits selected update zero. The soft initializer detected 6/9 fixed test instabilities with 0 stable false alarms.

其他回执与附件

- [unified_first_swing_synthetic_20261119_evidence.json](unified_first_swing_synthetic_20261119_evidence.json)

## M3 与 M3b 结构发现

- [TASK3 M3 统一结构发现](../unified_discovery_m3_v1_20261120_zh.md) · [核验回执](../unified_discovery_m3_v1_20261120_evidence.json)。 完整 `train/val` 双步长评分。G2 骨架检出 `val` 12/12 例失稳，2 个稳定误报触发 `KEEP_BASELINE`。
- [TASK3 M3b 三组配对拟合](../unified_discovery_m3b_v1_20261121_zh.md) · [核验回执](../unified_discovery_m3b_v1_20261121_evidence.json)。 同一 54 维起点的三组双步长评分，按两例误报的上限接受并冻结单 G2。
- [TASK3 M3b 固定 test 对照](../unified_discovery_m3b_test_v1_20261121_zh.md) · [核验回执](../unified_discovery_m3b_test_v1_20261121_evidence.json)。 冻结单 G2 在已曝光名单检出 5/9，稳定误报 0，未决 2。与两份 V4 冻结评分并列。

## M4 故障后物理核对

- [TASK3 M4 故障后物理一致性](../postfault_m4_v1_20261121_zh.md) · [核验回执](../postfault_m4_v1_20261121_evidence.json)。 完整 `train/val` 归档分析，给出主模态、45 条有效边、失步割集与起报储能。
- [M4 v2　惯量加权的机组分群与起报储能](../postfault_m4_v2_20261121_zh.md) · [核验回执](../postfault_m4_v2_20261121_evidence.json) · [首次失败回执](../postfault_m4_v2_failed_20261121_evidence.json)

## M5 封存盲测

- [M5　IEEE 39 节点独立盲测](../blind_confirmation_m5_v1_20261121_zh.md) · [核验回执](../blind_confirmation_m5_v1_20261121_evidence.json)
- [盲测数据使用说明](../../../data/kiki_ieee39_blind_v2/README.md) · [数据字典](../../../data/kiki_ieee39_blind_v2/schema.json)

其他回执与附件

- [blind_dataset_build_m5_v1_20261121_evidence.json](../blind_dataset_build_m5_v1_20261121_evidence.json)

## TASK3 文献方法对照与未决复评

- 并列 `val`、固定 `test` 和 `blind_v2` 的轨迹评分与失稳判定。[三组名单对照](../task3_three_cohort_comparison_20261121_zh.md) · [核验回执](../task3_three_cohort_comparison_20261121_evidence.json)
- 对 311 个原 8 秒未决的模型与事件组合延长预测并记录判定变化。[15 秒复评报告](../task3_unsettled_15s_v1_20261122_zh.md) · [核验回执](../task3_unsettled_15s_v1_20261122_evidence.json)

## 图件与复算

**M4 物理核对图**

[origin_storage.png](../figures/postfault_m4_v1/origin_storage.png) · [origin_storage.svg](../figures/postfault_m4_v1/origin_storage.svg) · [top10_topology.png](../figures/postfault_m4_v1/top10_topology.png) · [top10_topology.svg](../figures/postfault_m4_v1/top10_topology.svg)

**M4 v2 慢同调与储能分布**

[energy_decomposition_auroc.png](../figures/postfault_m4_v2/energy_decomposition_auroc.png) · [energy_decomposition_auroc.svg](../figures/postfault_m4_v2/energy_decomposition_auroc.svg) · [energy_decomposition_distribution.png](../figures/postfault_m4_v2/energy_decomposition_distribution.png) · [energy_decomposition_distribution.svg](../figures/postfault_m4_v2/energy_decomposition_distribution.svg) · [energy_decomposition_distribution_zh.md](../figures/postfault_m4_v2/energy_decomposition_distribution_zh.md) · [slow_coherency_comparison.png](../figures/postfault_m4_v2/slow_coherency_comparison.png) · [slow_coherency_comparison.svg](../figures/postfault_m4_v2/slow_coherency_comparison.svg)

**M3 结构发现课程图**

[curriculum_20261120.png](../figures/unified_discovery_m3_v1/curriculum_20261120.png) · [curriculum_20261120.svg](../figures/unified_discovery_m3_v1/curriculum_20261120.svg)

[返回成果索引](../README.md)
