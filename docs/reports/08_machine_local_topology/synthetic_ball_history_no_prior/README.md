# 零先验条件下的历史求解几何

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_ball_history_no_prior_v1.json)

## 执行协议

<a id="doc-25bef5681e95"></a>

<a id="doc-25bef5681e95-固定能量球下零先验初始化对照"></a>

### 固定能量球下零先验初始化对照

[第一轮合成对照](../synthetic_ball_history/README.md#doc-038fc6f10b21)中，零先验的精确能量球方法通过干净可行真值恢复。保留原先验时，train 和 val 的 27 个可行病例中有 4 例超过 8 s 轨迹误差门槛，9 个球外真值病例中有 4 例的约束最优解落在球内。本轮补齐零先验的径向裁剪对照，检验同一先验和半径下的求解几何。

沿用第一轮完整合成 train 24、val 12、test 12 的冻结轨迹、输入、两档真值资格、尺度与角色。半径为 1，五次历史更新，历史和预测都用原固定拓扑逐机方程。候选采用已核验的精确零先验预测。新增运行原径向裁剪加零先验，在 CUDA float64 上以 2.5 ms 和 1.25 ms 完成全部 48 例及完整 8 s 预测。配对依据为来源哈希和事件 ID。保留两组的完整轨迹与稳定性统计。test 只报告。

进入真实对照的门槛使用 train、val。此前零先验干净真值恢复门须通过。本轮每个角色的两组预测须全部具备数值资格。精确组在该角色须全部正确且无未决。球外真值按实际历史目标检验，每个角色精确组各例须触及半径，目标相对裁剪至多上升 $10^{-10}$，至少一例下降 $10^{-10}$。3 s 和 8 s 整体归一化 RMSE 至多为裁剪组的 1.05 倍。球外病例的真值恢复误差用于刻画半径限制。

一次执行，零拟合，零真实数据读取。只解码一份冻结合成真值 NPZ。预算含两档步长的整批运算和失败后的逐例隔离一次。传播 686 次，Torch 右端 2022720 次，切线右端 117600 次，QR、SVD、三角求解和矩阵乘各 960 次。通过后再单独登记真实 TASK3 对照，保持原始 T0M1 权重、半径和历史尺度，采用冻结的零先验精确求解。

## 报告与结果

<a id="doc-6017e23c1811"></a>

<a id="doc-6017e23c1811-零先验条件下的历史求解几何"></a>

### 零先验条件下的历史求解几何

在相同能量球和零先验下，精确求解降低了全部 12 个球外真值病例的历史目标。两组均正确判定全部 48 例。精确解在 train 和 val 的 3 s、8 s 轨迹误差超过径向裁剪的 1.05 倍门槛，候选未进入真实 TASK3 对照。球外病例的最优历史拟合与预测误差出现分歧。

| 方法 | 划分 | 3 s 合并归一化 RMSE | 8 s 合并归一化 RMSE | TP/TN/FP/FN | 完整准确率 |
|---|---|---:|---:|---|---:|
| 径向裁剪，零先验 | train | 0.002022 | 0.001519 | 6/18/0/0 | 24/24 |
| 精确球约束，零先验 | train | 0.002381 | 0.001801 | 6/18/0/0 | 24/24 |
| 径向裁剪，零先验 | val | 0.001749 | 0.001322 | 3/9/0/0 | 12/12 |
| 精确球约束，零先验 | val | 0.002174 | 0.001636 | 3/9/0/0 | 12/12 |
| 径向裁剪，零先验 | test | 0.002150 | 0.001605 | 3/9/0/0 | 12/12 |
| 精确球约束，零先验 | test | 0.002577 | 0.001943 | 3/9/0/0 | 12/12 |

两组均无未决、未覆盖或数值失败。独立核对了完整名单、操作预算、来源哈希、逐病例和汇总计数。[执行回执](_sealed/machine_local_topology_ball_history_no_prior_20261115_evidence.json) · [结果审查](_sealed/machine_local_topology_ball_history_no_prior_result_review_20261115.json)

保留径向裁剪时，去除原二次先验使 train 和 val 的 8 s 合并归一化 RMSE 分别下降 18.7% 和 24.7%。两个划分的失稳匹配窗口仍分别为 6/6、3/3。球内 9 个非零真值病例的起点隐态恢复误差除以真值范数后最大为 $4.4\times10^{-9}$。这组前后配对支持为原径向方法的零先验候选单独登记真实对照。[原先验合成结果](../synthetic_ball_history/README.md#doc-038fc6f10b21)

## 回执与附件

- [machine_local_topology_ball_history_no_prior_20261115_evidence.json](_sealed/machine_local_topology_ball_history_no_prior_20261115_evidence.json)
- [machine_local_topology_ball_history_no_prior_result_review_20261115.json](_sealed/machine_local_topology_ball_history_no_prior_result_review_20261115.json)
- [machine_local_topology_ball_history_no_prior_v1_review_20261115.json](_sealed/machine_local_topology_ball_history_no_prior_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_ball_history_no_prior_v1_zh.md](_sealed/machine_local_topology_ball_history_no_prior_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/machine_local_topology_ball_history_no_prior_v1_zh.md](_sealed/original/machine_local_topology_ball_history_no_prior_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_ball_history_no_prior_20261115_zh.md](_sealed/machine_local_topology_ball_history_no_prior_20261115_zh.md)

</details>
