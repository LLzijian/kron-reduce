# TASK3 train 功角尺度的固定初态对照

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/kiki_machine_local_topology_angle_scale_v1.json)

## 执行协议

<a id="doc-0c9531893d0a"></a>

<a id="doc-0c9531893d0a-task3-train-功角尺度的固定初态对照"></a>

### TASK3 train 功角尺度的固定初态对照

以冻结功角阻尼模型的 \(\rho=2/3\) 为锚，比较相对功角尺度 1 rad 与预设 1.25 rad。共享刚度、可见线性阻尼和逐机记忆系数固定。完整 train 346 使用原尺度、原半径与二次先验，由 11 帧历史分别用 2.5 ms、1.25 ms 积分估计初态。两个尺度共用每档估得的记忆初态。

沿每例的完整实测速度传播记忆。从起报点开始至最多 3 s 的实测区间评分，以质量和区间数归一化。对同一未来算子消去自由记忆初态，作为第三种对照。每条路径的两个尺度均使用四点 Gauss 力积分，并做八点核查。三个对照均要求满秩、有效覆盖和求积差至多 \(10^{-10}\)。各自的稳定组残差平方和至少下降 5%，全 train 和失稳组的残差平方和增幅分别至多 1%。

一次 train 诊断，零共享参数拟合，GPU 计算历史初态。实测未来只参与力平衡评分，模型闭环未来传播次数为零。打开 static 与 development 缓存各一次，载入检查点两次，保留失败记录。精确的两档历史求解和异常隔离预算见方案 JSON。无 val/test 轨迹读取。

## 报告与结果

<a id="doc-71ece58bcee4"></a>

<a id="doc-71ece58bcee4-task3-train-功角尺度的固定初态对照"></a>

### TASK3 train 功角尺度的固定初态对照

把功角尺度从 1 rad 增至 1.25 rad 后，完整 train 的后续动量残差增大。稳定组和失稳组在两档因果初态及自由初态投影下方向一致。346 例全部通过数值资格，增大尺度的候选未通过预登记残差门。

固定 \(\rho=2/3\)。用原尺度和 100 ms 可见历史估计起点记忆，每档步长估得的初态同时用于两个尺度。沿同一实测速度传播记忆，计算起报后最多 3 s 的动量残差平方和。表中数值为 1.25 rad 的残差相对 1 rad 的变化。[协议](#doc-0c9531893d0a)

| train 分组 | 样本数 | 2.5 ms 因果初态 | 1.25 ms 因果初态 | 自由初态投影 |
|---|---:|---:|---:|---:|
| 全部 | 346 | +6.71% | +6.71% | +3.14% |
| 稳定 | 311 | +1.20% | +1.20% | +1.32% |
| 失稳 | 35 | +10.21% | +10.21% | +7.25% |

稳定组要求下降至少 5%，全 train 与失稳组要求增幅至多 1%。三项对照均未达到。结论适用于这两个尺度在当前方程上的实测路径一致性。运行只读取 train，闭环未来求解次数为零，保留原功角尺度。

运行 `kiki_machine_local_topology_angle_scale_v1-train_momentum-20261115T115504524767Z-1d3ecb45-fde33d42`。保存结果复核包括完整名单、双步长历史求解和算子预算。[运行回执](_sealed/kiki_machine_local_topology_angle_scale_20261115_evidence.json) · [独立复核](_sealed/kiki_machine_local_topology_angle_scale_result_review_20261115.json)

## 回执与附件

- [kiki_machine_local_topology_angle_scale_20261115_evidence.json](_sealed/kiki_machine_local_topology_angle_scale_20261115_evidence.json)
- [kiki_machine_local_topology_angle_scale_result_review_20261115.json](_sealed/kiki_machine_local_topology_angle_scale_result_review_20261115.json)
- [kiki_machine_local_topology_angle_scale_v1_review_20261115.json](_sealed/kiki_machine_local_topology_angle_scale_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_angle_scale_v1_zh.md](_sealed/kiki_machine_local_topology_angle_scale_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_angle_scale_20261115_zh.md](_sealed/kiki_machine_local_topology_angle_scale_20261115_zh.md)

</details>
