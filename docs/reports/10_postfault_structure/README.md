# 10 · 故障后统一结构与独立盲测

M3b 冻结了单 G2 模型。M4 检查了故障后主模态和起报储能。M5 对同源新仿真的 94 个封存事件独立评分，冻结模型检出 29/32 例失稳，62 例稳定事件均判稳。固定 test 对照和独立盲测各自的样本范围见对应报告。

## M1 首摆与初值

- [TASK3 M1 first-swing loss ablation](unified_first_swing_fit_20261119_zh.md) · [方案](../../../configs/experiments/unified_first_swing_v1.toml) · [协议](../../../docs/protocols/unified_first_swing_v1_zh.md) · [运行结果](../../../results/unified_first_swing_v1-fit-20261119T051218221066Z-fdfecba6-4c03db51/) · [核验回执](unified_first_swing_fit_20261119_evidence.json)。 Complete train and val, 18 updates per group, both selecting the shared starting checkpoint
- [TASK3 M1 gradient and initializer probes](unified_first_swing_probes_v2_20261119_zh.md) · [方案](../../../configs/experiments/unified_first_swing_probes_v2.toml) · [协议](../../../docs/protocols/unified_first_swing_probes_v2_zh.md) · [运行结果](../../../results/unified_first_swing_probes_v2-probe-20261119T064313252198Z-822c9fed-c0907457/) · [核验回执](unified_first_swing_probes_v2_20261119_evidence.json)。 Full train and val diagnostics with zero updates, adaptive GN detected 12/12 val instabilities and exceeded the stable trajectory limit
- [TASK3 M1 GN prior and radius comparison](unified_first_swing_gn_decomposition_v3_20261119_zh.md) · [方案](../../../configs/experiments/unified_first_swing_gn_decomposition_v3.toml) · [协议](../../../docs/protocols/unified_first_swing_gn_decomposition_v3_zh.md) · [运行结果](../../../results/unified_first_swing_gn_decomposition_v3-probe-20261119T074514989765Z-5637f04a-73c62955/) · [核验回执](unified_first_swing_gn_decomposition_v3_20261119_evidence.json)。 Complete train and val, two fixed-prior soft-radius settings detected 12/12 val instabilities, no setting passed the joint gate
- [TASK3 M1 V4 fit and frozen comparison](unified_first_swing_v4_20261120_zh.md) · [方案](../../../configs/experiments/unified_first_swing_v4_fit.toml) · [协议](../../../docs/protocols/unified_first_swing_v4_fit_zh.md) · [拟合结果](../../../results/unified_first_swing_v4_fit-fit-20261119T085346160446Z-4256b8a8-2aac3232/) · [test 结果](../../../results/unified_first_swing_v4_test-test-20261120T062857089605Z-e44f1587-a79403a1/) · [拟合回执](unified_first_swing_v4_fit_20261119_evidence.json) · [固定 test 回执](unified_first_swing_v4_test_20261119_evidence.json)。 Both fits selected update zero. The soft initializer detected 6/9 fixed test instabilities with 0 stable false alarms.

其他回执与附件

- [unified_first_swing_synthetic_20261119_evidence.json](unified_first_swing_synthetic_20261119_evidence.json) · [运行结果](../../../results/unified_first_swing_fixture_v1-synthetic-20261119T045408141342Z-4f7c040a-7298d370)

## M3 与 M3b 结构发现

- [TASK3 M3 统一结构发现](unified_discovery_m3_v1_20261120_zh.md) · [方案](../../../configs/experiments/unified_discovery_m3_v1.toml) · [协议](../../../docs/protocols/unified_discovery_m3_v1_zh.md) · [运行结果](../../../results/unified_discovery_m3_v1-discovery-20261120T080201366232Z-dc27983e-6970d859/) · [核验回执](unified_discovery_m3_v1_20261120_evidence.json)。 完整 `train/val` 双步长评分。G2 骨架检出 `val` 12/12 例失稳，2 个稳定误报触发 `KEEP_BASELINE`。
- [TASK3 M3b 三组配对拟合](unified_discovery_m3b_v1_20261121_zh.md) · [执行配置](../../../configs/experiments/postfault_m4_m3b_v1.toml) · [协议](../../../docs/protocols/postfault_m4_m3b_v1_zh.md) · [运行结果](../../../results/unified_discovery_m3b_v1-fit-20261121T034533646318Z-ceaa6a72-4795a249/) · [核验回执](unified_discovery_m3b_v1_20261121_evidence.json)。 同一 54 维起点的三组双步长评分，按两例误报的上限接受并冻结单 G2。
- [TASK3 M3b 固定 test 对照](unified_discovery_m3b_test_v1_20261121_zh.md) · [工作流登记](../../../configs/workflows/unified_discovery_m3b_test_v1.json) · [运行结果](../../../results/unified_discovery_m3b_test_v1-test-20261121T044042004702Z-7984cefd-3f6fcbfd/) · [核验回执](unified_discovery_m3b_test_v1_20261121_evidence.json)。 冻结单 G2 在已曝光名单检出 5/9，稳定误报 0，未决 2。与两份 V4 冻结评分并列。

## M4 故障后物理核对

- [TASK3 M4 故障后物理一致性](postfault_m4_v1_20261121_zh.md) · [核验回执](postfault_m4_v1_20261121_evidence.json) · [运行结果](../../../results/postfault_m4_v1-m4-20261121T034423510207Z-ceaa6a72-6fb69813/summary.json)。 完整 `train/val` 归档分析，给出主模态、45 条有效边、失步割集与起报储能。
- [M4 v2　惯量加权的机组分群与起报储能](postfault_m4_v2_20261121_zh.md) · [方案](../../../configs/experiments/postfault_m4_v2.toml) · [协议](../../../docs/protocols/postfault_m4_v2_zh.md) · [分析运行结果](../../../results/postfault_m4_v2-analysis-20261121T060746860331Z-c955592a-e210fd93/) · [核验回执](postfault_m4_v2_20261121_evidence.json) · [首次失败运行结果](../../../results/postfault_m4_v2-analysis-20261121T060227625654Z-4906e508-e1311f97/) · [首次失败回执](postfault_m4_v2_failed_20261121_evidence.json)

## M5 封存盲测

- [M5　IEEE 39 节点独立盲测](blind_confirmation_m5_v1_20261121_zh.md) · [方案](../../../configs/experiments/blind_confirmation_m5_v1.toml) · [协议](../../../docs/protocols/blind_confirmation_m5_v1_zh.md) · [运行结果](../../../results/blind_confirmation_m5_v1-blind-20261121T064636317312Z-26fe26a3-b73dd81d/) · [核验回执](blind_confirmation_m5_v1_20261121_evidence.json)
- [盲测数据使用说明](../../../data/kiki_ieee39_blind_v2/README.md) · [数据字典](../../../data/kiki_ieee39_blind_v2/schema.json)

其他回执与附件

- [blind_dataset_build_m5_v1_20261121_evidence.json](blind_dataset_build_m5_v1_20261121_evidence.json) · [数据集构建运行结果](../../../results/blind_dataset_build_m5_v1-build_receipt-20261121T064503861579Z-8e63d1d3-c86e61e2/)

## TASK3 文献方法对照与未决复评

- 并列 `val`、固定 `test` 和 `blind_v2` 的轨迹评分与失稳判定。[三组名单对照](task3_three_cohort_comparison_20261121_zh.md) · [核验回执](task3_three_cohort_comparison_20261121_evidence.json)
- 对 311 个原 8 秒未决的模型与事件组合延长预测并记录判定变化。[15 秒复评方案](../../../configs/proposals/task3_unsettled_15s_v1.json) · [协议](../../protocols/task3_unsettled_15s_v1_zh.md) · [15 秒复评报告](task3_unsettled_15s_v1_20261122_zh.md) · [运行结果](../../../results/task3_unsettled_15s_v1-reevaluate-20261121T174426106655Z-8f0b6862-09b10a2e/) · [核验回执](task3_unsettled_15s_v1_20261122_evidence.json) · [首次中断运行结果](../../../results/task3_unsettled_15s_v1-reevaluate-20261121T173831296967Z-4a0056fc-12340b3f/) · [首次复评中断回执](task3_unsettled_15s_v1_20261122_interruption_evidence.json)

## 图件与复算

**M4 物理核对图**

[origin_storage.png](figures/postfault_m4_v1/origin_storage.png) · [origin_storage.svg](figures/postfault_m4_v1/origin_storage.svg) · [top10_topology.png](figures/postfault_m4_v1/top10_topology.png) · [top10_topology.svg](figures/postfault_m4_v1/top10_topology.svg)

**M4 v2 慢同调与储能分布**

[energy_decomposition_auroc.png](figures/postfault_m4_v2/energy_decomposition_auroc.png) · [energy_decomposition_auroc.svg](figures/postfault_m4_v2/energy_decomposition_auroc.svg) · [energy_decomposition_distribution.png](figures/postfault_m4_v2/energy_decomposition_distribution.png) · [energy_decomposition_distribution.svg](figures/postfault_m4_v2/energy_decomposition_distribution.svg) · [energy_decomposition_distribution_zh.md](figures/postfault_m4_v2/energy_decomposition_distribution_zh.md) · [slow_coherency_comparison.png](figures/postfault_m4_v2/slow_coherency_comparison.png) · [slow_coherency_comparison.svg](figures/postfault_m4_v2/slow_coherency_comparison.svg)

**M3 结构发现课程图**

[curriculum_20261120.png](figures/unified_discovery_m3_v1/curriculum_20261120.png) · [curriculum_20261120.svg](figures/unified_discovery_m3_v1/curriculum_20261120.svg)

[返回成果索引](../README.md)
