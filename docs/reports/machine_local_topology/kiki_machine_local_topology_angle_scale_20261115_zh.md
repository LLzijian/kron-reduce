# TASK3 train 功角尺度的固定初态对照

把功角尺度从 1 rad 增至 1.25 rad 后，完整 train 的后续动量残差增大。稳定组和失稳组在两档因果初态及自由初态投影下方向一致。346 例全部通过数值资格，增大尺度的候选未通过预登记残差门。

固定 \(\rho=2/3\)。用原尺度和 100 ms 可见历史估计起点记忆，每档步长估得的初态同时用于两个尺度。沿同一实测速度传播记忆，计算起报后最多 3 s 的动量残差平方和。表中数值为 1.25 rad 的残差相对 1 rad 的变化。[协议](../../protocols/kiki_machine_local_topology_angle_scale_v1_zh.md)

| train 分组 | 样本数 | 2.5 ms 因果初态 | 1.25 ms 因果初态 | 自由初态投影 |
|---|---:|---:|---:|---:|
| 全部 | 346 | +6.71% | +6.71% | +3.14% |
| 稳定 | 311 | +1.20% | +1.20% | +1.32% |
| 失稳 | 35 | +10.21% | +10.21% | +7.25% |

稳定组要求下降至少 5%，全 train 与失稳组要求增幅至多 1%。三项对照均未达到。结论适用于这两个尺度在当前方程上的实测路径一致性。运行只读取 train，闭环未来求解次数为零，保留原功角尺度。

运行 `kiki_machine_local_topology_angle_scale_v1-train_momentum-20261115T115504524767Z-1d3ecb45-fde33d42`。保存结果复核包括完整名单、双步长历史求解和算子预算。[运行回执](../kiki_machine_local_topology_angle_scale_20261115_evidence.json) · [独立复核](../../research/kiki_machine_local_topology_angle_scale_result_review_20261115.json)
