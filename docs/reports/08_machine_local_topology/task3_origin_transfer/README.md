# TASK3 train 隐藏初态的跨时段动量检验

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/kiki_machine_local_topology_origin_transfer_v1.json)

## 执行协议

<a id="doc-595d6b4d1734"></a>

<a id="doc-595d6b4d1734-task3-train-隐藏初态的跨时段动量检验"></a>

### TASK3 train 隐藏初态的跨时段动量检验

降低可见阻尼后，匹配失步窗口从 6/35 增至 10/35，val 的稳定轨迹误差也明显上升。原方程的 100 ms 动量诊断中，346 例均为满秩，299 例的约束解达到原半径。检验较大的隐态能否从短历史中稳健求出，并解释之后的实测运动，再决定是否研究新的耗散函数。[闭环对照](../task3_momentum_damping/README.md#doc-97c1fbb8b406) · [动量诊断](../task3_momentum/README.md#doc-e34f7a9a3e3d)

固定原 T0M1 第 12 步的完整方程，沿用原 ZOH 功率输入。对完整 train 346 的 11 帧历史，分别求 10 ms 无约束和原半径约束解。以每隔一帧保留始末点的 20 ms 观测计算另一个无约束解。比较两档历史起点隐态的差，分母为原半径和 10 ms 无约束解范数中的较大者。

两个 10 ms 初态在评分前冻结。逐机记忆按实测速度的分段线性输入传播，从起报点开始，直到起报后三秒或可用记录结束，统计质量和时长归一化的动量残差平方和。两种估计共用实测路径、记忆响应和区间权重。未来速度只用于离线检查，隐态估计只读取 100 ms 历史。

检查门要求 346 例的两档历史系统均为秩 10 且结果有效，抽样差异逐例至多 5%。无约束解的起报后残差平方和须在完整 train、311 个稳定样本及 35 个失稳样本上分别比约束解低至少 20%。保留全部病例和失败记录。过门后单独审查因果积分初始化及能量上界。部署保持 `KEEP_BASELINE`。

单次 train 诊断，零共享参数拟合。打开 static 与 development 缓存各一次，载入原模型检查点两次。最多 692 次 SVD、346 次球约束求解、112450 个动量区间。无 val/test 数据读取和闭环预测。

## 报告与结果

<a id="doc-eab8f261c2ca"></a>

<a id="doc-eab8f261c2ca-task3-train-隐藏初态的跨时段动量检验"></a>

### TASK3 train 隐藏初态的跨时段动量检验

固定原 T0M1 方程后，放开历史起点的记忆半径降低了失稳样本的后续动量残差，同时增大了稳定样本的残差。完整 train 的 346 例均得到有效解。两档抽样一致性和残差改善均未通过预登记门槛，保留原能量上界。[冻结协议](#doc-595d6b4d1734)

用 100 ms 实测历史求 10 ms 无约束解、原半径约束解和 20 ms 抽样复核解。随后沿相同实测速度传播记忆，从起报点到最多 3 s 的可用实测段评分。表中比值为无约束解的质量和时长归一化残差平方和除以原半径约束解的平方和。抽样合格指两个无约束解的相对差不超过 5%。

| train 范围 | 10 ms 约束解触顶 | 抽样合格 | 最大抽样差 | 后续残差平方和比值 | 残差变化 |
|---|---:|---:|---:|---:|---:|
| 全部 346 例 | 299/346 | 331/346 | 7.18% | 0.83057 | 降低 16.94% |
| 稳定 311 例 | 264/311 | 298/311 | 7.18% | 1.33541 | 增加 33.54% |
| 失稳 35 例 | 35/35 | 33/35 | 7.04% | 0.58081 | 降低 41.92% |

原 T0M1 的弱动量系统在 10 ms 和 20 ms 下均为秩 10。放开半径后，仅失稳组达到 20% 残差改善门槛。该检验使用起报后的实测速度评价动力学一致性，未做闭环预测。数据日志验证只读取完整 train，未读取 val 或 test 的轨迹。

运行 `kiki_machine_local_topology_origin_transfer_v1-train_momentum-20261115T110223793521Z-11e9ffa4-23dd8516`。静态和开发缓存各打开一次，使用 692 次 SVD、346 次球约束求解和 103531 个动量区间。保存结果已复核，部署保持 `KEEP_BASELINE`。[运行回执](_sealed/kiki_machine_local_topology_origin_transfer_20261115_evidence.json) · [独立审核](_sealed/kiki_machine_local_topology_origin_transfer_result_review_20261115.json)

## 回执与附件

- [kiki_machine_local_topology_origin_transfer_20261115_evidence.json](_sealed/kiki_machine_local_topology_origin_transfer_20261115_evidence.json)
- [kiki_machine_local_topology_origin_transfer_result_review_20261115.json](_sealed/kiki_machine_local_topology_origin_transfer_result_review_20261115.json)
- [kiki_machine_local_topology_origin_transfer_v1_review_20261115.json](_sealed/kiki_machine_local_topology_origin_transfer_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_origin_transfer_v1_zh.md](_sealed/kiki_machine_local_topology_origin_transfer_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/kiki_machine_local_topology_origin_transfer_v1_zh.md](_sealed/original/kiki_machine_local_topology_origin_transfer_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_origin_transfer_20261115_zh.md](_sealed/kiki_machine_local_topology_origin_transfer_20261115_zh.md)

</details>
