# 零和耦合通过因果历史恢复对照

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_spatial_coupling_causal_v1.json)

## 执行协议

<a id="doc-9d2bf0d08608"></a>

<a id="doc-9d2bf0d08608-零和耦合的因果历史恢复对照"></a>

### 零和耦合的因果历史恢复对照

[首次验证](../synthetic_spatial_coupling/README.md#doc-68f44e7dfc76)中，零和模型选中了真值速率，原记忆半径外的六个 train 样本贡献了其 92.37% 的双时域平方误差。本轮检验干净历史足以恢复初态时，原模型能否达到预设的机制区分门限。

沿用[首次协议](../synthetic_spatial_coupling/README.md#doc-284356f266b1)的三种真值、完整 train 24、val 12、test 12、双步长、功率、方程、候选速率网格、6 次选择、30 个 train 候选评分与九组全量预测。模型从 11 帧可见历史估计十维记忆初态，耦合初态为零。起报时替换可见状态，保留传播所得的记忆与耦合。采用径向初始化，半径取冻结真值最大记忆范数 1.5 的两倍，即 3，二次先验权重取零。真值记忆只用于生成和评分，两项改动共同检验初态恢复对结果的影响。

保留首次协议的数值和科学阈值，包括最优与次优损失比 0.95、双时域损失比 0.8、val 稳定组 8 s 轨迹误差比 0.9。新半径覆盖原全部病例，因此轨迹、耦合和稳定门覆盖对应完整 train、val 群体。增加双步长记忆恢复门。对每种真值的匹配模型，逐例检查全部 36 个 train 和 val 样本在历史第一帧及起报帧的十维记忆，两处误差均用 \(\max(0.5,\|z_h^*\|_2)\) 归一化，每项至多为 \(10^{-3}\)。test 只报告。

本次单独登记，真实数据读取为零。各门通过后，另行登记完整 TASK3 train 路径诊断。真实诊断使用该分支预定的真实初始化设置，并独立比较固定因果起报状态和自由记忆投影。

## 报告与结果

<a id="doc-1d7dcc7de0aa"></a>

<a id="doc-1d7dcc7de0aa-零和耦合通过因果历史恢复对照"></a>

### 零和耦合通过因果历史恢复对照

三种真值各覆盖完整 train 24、val 12、test 12。共同尺度真值选中共同速率 1，支路重分配真值选中空间速率 1，固定刚度真值使两者都选中零。所有门禁通过。[冻结协议](#doc-9d2bf0d08608)

| 真值 | 匹配模型 8 s 归一化 RMSE，val | 匹配模型 8 s 归一化 RMSE，test | 固定刚度对照，val／test |
|---|---:|---:|---:|
| 固定刚度 | 5.58×10⁻¹¹ | 6.28×10⁻¹¹ | 5.58×10⁻¹¹／6.28×10⁻¹¹ |
| 共同尺度 | 5.65×10⁻¹¹ | 7.01×10⁻¹¹ | 0.01886／0.01951 |
| 支路重分配 | 5.61×10⁻¹¹ | 6.22×10⁻¹¹ | 0.01425／0.01456 |

归一化 RMSE 为事件等权均方误差的平方根。每种真值下，两族赢家和固定刚度对照的 val、test 稳定性准确率与覆盖率均为 100%，各划分 TP/TN/FP/FN＝3/9/0/0。匹配模型在全部 train、val 样本上通过了双步长的历史第一帧和起报帧记忆恢复门。空间真值下，匹配模型的 train 双时域 MSE 为 1.60×10⁻²¹，次优候选为 3.18×10⁻⁸。[结果回执](_sealed/machine_local_topology_spatial_coupling_causal_20261116_evidence.json)

同一批真值在[首次对照](../synthetic_spatial_coupling/README.md#doc-68f44e7dfc76)中使用半径 1 和二次先验，空间模型未通过三项科学门。本轮将径向半径设为 3、先验权重设为零，两项合并改动消除了干净合成历史上的主要初始化误差。每个初态均从可见历史求得，真值只用于评分。结论适用于本协议的干净历史、方程族与速率网格。真实 train 路径诊断采用独立冻结设置。

## 回执与附件

- [machine_local_topology_spatial_coupling_causal_20261116_evidence.json](_sealed/machine_local_topology_spatial_coupling_causal_20261116_evidence.json)
- [machine_local_topology_spatial_coupling_causal_result_review_20261116.json](_sealed/machine_local_topology_spatial_coupling_causal_result_review_20261116.json)
- [machine_local_topology_spatial_coupling_causal_v1_review_20261116.json](_sealed/machine_local_topology_spatial_coupling_causal_v1_review_20261116.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_spatial_coupling_causal_v1_zh.md](_sealed/machine_local_topology_spatial_coupling_causal_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/machine_local_topology_spatial_coupling_causal_v1_zh.md](_sealed/original/machine_local_topology_spatial_coupling_causal_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_spatial_coupling_causal_20261116_zh.md](_sealed/machine_local_topology_spatial_coupling_causal_20261116_zh.md)

</details>
