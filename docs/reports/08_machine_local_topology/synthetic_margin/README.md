# 五点失步余量的合成训练结果

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/04_topology_diagnosis_learning.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_margin_v2.json)

## 执行协议

<a id="doc-d5e3a4358386"></a>

<a id="doc-d5e3a4358386-五点失步余量的同起点合成训练对照"></a>

### 五点失步余量的同起点合成训练对照

同一套 T0M1 真值产生 24 个固定事件，`train=12`、`val=6`、`test=6`，各划分高速与低速激励各半。沿用独立激励生成器，种子固定为 2026111502，高速偏移幅值从 10.5 至 12 rad/s 等距取值。两档独立 NumPy 真值使用 1.25 和 0.625 ms 步长。起报设在第 11 个观测点。完整真值用于 8 秒评价，train 的稳定事件监督至 3 秒，失稳事件监督至首次五点失步确认点且至多 3 秒。train 和 val 要求两类齐全、历史未失步及双步长真值一致。每例保存数值资格。保留预设事件和生成失败记录。

两个组从同一份 22 参数初值出发。刚度与阻尼共享尺度为 0.90、1.10，十机端口幅值均为 0.55，十个记忆时间常数在 0.08 至 0.55 秒之间按几何级数排布。模型用 100 ms 历史、恒值机械功率和五次有界 Gauss–Newton 初始化，隐态半径为 1。历史尺度取完整 train 的均方根，目标尺度对每例真实可用的未来逐通道先取均方再在 train 等权平均并开方。两组共用尺度。

两组采用完全相同的 1 秒与 3 秒等权事件轨迹损失。候选将每例五点失步最大有符号余量与真值对应余量的平方差加入目标，权重为 1。两条路径都拼接 11 点真实历史和各自的未来，余量窗至少包含一个未来样本，且匹配同一事件的监督长度。长度仅用于计算损失。两组各执行 20 次完整 train 更新，Adam 学习率为 0.001、梯度范数裁剪为 1，使用 5 ms CUDA float64 RK4。固定训练长度用于测量同计算量下目标函数的差异。数值失败保留最后有效权重并记录失败。

先保存共同初值与两个终点的权重，再用 2.5 和 1.25 ms GPU 双步长评价三个模型的完整 train、val、test。报告各划分轨迹误差、稳定性准确率、混淆计数、未决及覆盖。另报匹配实测时段的余量 RMSE 和触发数。数值资格沿用合成双步长约定。能量恒等式核对 train 和 val 的真值起报状态，误差至多 \(10^{-10}\)。

用完整 val 评价是否进入真实数据准备。候选的失稳匹配余量 RMSE 至多为共同初值和轨迹对照中较小值的 95%，匹配时段的失稳触发数至少与轨迹对照持平。候选的 3 秒和 8 秒池化归一化 RMSE 均至多为对照的 105%，分别检查整个 val 和其真实稳定事件。候选须全体可判定、稳定误报为零，判对数至少持平。两组均完成全量更新并通过能量恒等式。test 在模型冻结后用于报告。部署决定为 `KEEP_BASELINE`。

本轮单次执行，两次合成拟合。预算包含 77,760 次 NumPy RHS、至多 1,330 次积分传播、3,211,203 次 Torch RHS、196,000 次切线 RHS、3,840 次 QR、SVD、三角求解和对应乘法各一项，以及至多 43 次梯度、40 次优化更新。正式评分的批次数值失败可按既有代码逐例隔离一次，该预算已计入。真实数据读取为零。[TASK3 train 的五点证据](../task3_witness/README.md#doc-595f500c1fc5)

## 报告与结果

<a id="doc-c04b4d708da9"></a>

<a id="doc-c04b4d708da9-五点失步余量的合成训练结果"></a>

### 五点失步余量的合成训练结果

加入五点失步余量后，val 失稳余量 RMSE 从 0.000180045 降至 0.000179971，降幅为 0.041%，未达到预登记的 5%。保持 `KEEP_BASELINE`。
[执行协议](#doc-d5e3a4358386)

两组在同一套 T0M1 真值上从同一初值各完成 20 次完整 train 更新。24 个固定事件覆盖 train 12、val 6、test 6。候选加入预测与真值的五点失步余量平方差。


<a id="doc-c04b4d708da9-匹配监督时段的失稳预测"></a>

#### 匹配监督时段的失稳预测

每例使用 100 ms 历史和真实可用的监督时段，最长 3 秒。表中余量 RMSE 无量纲。

| 模型 | 数据划分 | 失稳余量 RMSE | 匹配时段触发数 |
|---|---|---:|---:|
| 共同初值 | train | 0.000259754 | 6/6 |
| 共同初值 | val | 0.000249401 | 3/3 |
| 共同初值 | test | 0.000124232 | 3/3 |
| 轨迹目标 | train | 0.000187306 | 6/6 |
| 轨迹目标 | val | 0.000180045 | 3/3 |
| 轨迹目标 | test | 8.95446e-05 | 3/3 |
| 轨迹与余量目标 | train | 0.000187175 | 6/6 |
| 轨迹与余量目标 | val | 0.000179971 | 3/3 |
| 轨迹与余量目标 | test | 8.94625e-05 | 3/3 |


<a id="doc-c04b4d708da9-完整轨迹与稳定性"></a>

#### 完整轨迹与稳定性

| 模型 | 数据划分 | 1 秒 NRMSE | 3 秒 NRMSE | 8 秒 NRMSE | TP/TN/FP/FN | 未决／未覆盖／数值失败 | 完整准确率 | 可判定准确率 |
|---|---|---:|---:|---:|---|---|---:|---:|
| 共同初值 | train | 0.0397332 | 0.17126 | 0.439807 | 6/6/0/0 | 0/0/0 | 100.00% | 100.00% |
| 共同初值 | val | 0.0421969 | 0.18059 | 0.442521 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |
| 共同初值 | test | 0.0442285 | 0.192842 | 0.485776 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |
| 轨迹目标 | train | 0.0299484 | 0.134667 | 0.407107 | 6/6/0/0 | 0/0/0 | 100.00% | 100.00% |
| 轨迹目标 | val | 0.0319106 | 0.142648 | 0.386901 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |
| 轨迹目标 | test | 0.033583 | 0.153543 | 0.413835 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |
| 轨迹与余量目标 | train | 0.0299566 | 0.134785 | 0.407692 | 6/6/0/0 | 0/0/0 | 100.00% | 100.00% |
| 轨迹与余量目标 | val | 0.0319176 | 0.142768 | 0.387386 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |
| 轨迹与余量目标 | test | 0.0335922 | 0.153669 | 0.413991 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |

轨迹表采用等事件权重的池化归一化 RMSE，GPU 双步长为 2.5 ms 和 1.25 ms。表内各模型和划分的稳定性覆盖率、失稳召回率均为 100%，漏判率、稳定误报率均为 0%。

val 的匹配时段失稳检出、3 秒和 8 秒轨迹误差、稳定性覆盖、正确数和误报均通过对应条件。共同初值已复现 train 6/6、val 3/3 个匹配时段的失步。合成 train 与 val 的 9 个真实失稳在起报后 70–90 ms 首次确认。真实 TASK3 train 的首次确认中位数为 340 ms，35 个失稳事件中有 28 个晚于 90 ms。下一项合成准备增加延迟失步和临界激励，并拆分轨迹与余量梯度贡献。[TASK3 train 证据](../task3_witness/README.md#doc-595f500c1fc5) · [后续研究方案](../synthetic_topology/README.md#doc-415956346cb1)

运行 `machine_local_topology_margin_v2-margin_v2-20261115T072851312779Z-b523e31a-05f7989b`。实际耗时 185.3 秒。

首轮完成两组拟合后，在 CUDA 权重导出时停止。修正版先将权重复制到 CPU 再保存，核验通过后以新的授权复现。两轮的真值、预处理和两组训练日志哈希一致。[首轮失败回执](_sealed/machine_local_topology_margin_v1_serialization_failure_20261115.json)

[证据回执](_sealed/machine_local_topology_margin_20261115_evidence.json) · [结果审核](_sealed/machine_local_topology_margin_result_review_20261115.json)

## 回执与附件

- [machine_local_topology_margin_20261115_evidence.json](_sealed/machine_local_topology_margin_20261115_evidence.json)
- [machine_local_topology_margin_result_review_20261115.json](_sealed/machine_local_topology_margin_result_review_20261115.json)
- [machine_local_topology_margin_review_20261115.json](_sealed/machine_local_topology_margin_review_20261115.json)
- [machine_local_topology_margin_v1_serialization_failure_20261115.json](_sealed/machine_local_topology_margin_v1_serialization_failure_20261115.json)
- [machine_local_topology_margin_v2_review_20261115.json](_sealed/machine_local_topology_margin_v2_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_margin_v1_zh.md](_sealed/machine_local_topology_margin_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/machine_local_topology_margin_v1_zh.md](_sealed/original/machine_local_topology_margin_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_margin_20261115_zh.md](_sealed/machine_local_topology_margin_20261115_zh.md)

</details>
