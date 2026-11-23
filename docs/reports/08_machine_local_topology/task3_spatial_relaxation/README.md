# 逐机零和耦合的完整 TASK3 对照

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[拟合方案](../../../../configs/proposals/kiki_machine_local_topology_spatial_relaxation_fit_v1.json) · [test 方案](../../../../configs/proposals/kiki_machine_local_topology_spatial_relaxation_test_v1.json)

## 执行协议

<a id="doc-1b547e9cbe55"></a>

<a id="doc-1b547e9cbe55-task3-逐机空间耦合速率配对训练"></a>

### TASK3 逐机空间耦合速率配对训练

检验逐机非负速率能否在保持能量收支和非负耗散的同时改变支路重分配的方向。此前的固定速率候选已完成[合成真值辨识](../synthetic_spatial_coupling_causal/README.md#doc-1d7dcc7de0aa)和[完整 train 路径诊断](../task3_spatial_coupling/README.md#doc-d5aa05bf148f)。真实路径的失稳组门失败。本轮按用户授权独立登记，实际拟合并报告两个冻结模型的完整预测。

两组均从已冻结 T0M1 第 12 步的权重起步，使用同一逐机功角阻尼 \(\rho=2/3\)、空间混合权重 1。每组学习相同的 22 个共享系数，即整体刚度与阻尼尺度、十机端口强度及十个时间常数。刚度和阻尼尺度相对基础骨架限定在 \([0.5,2]\)。固定刚度对照组使用逐机记忆。空间候选组增加九维动态坐标，历史第一帧取零，并在完整历史上传播。十机速率 \(\gamma_i\in[0,2]\,\mathrm{s}^{-1}\) 均从零开始，因此两组的初始可见预测一致。

对 Helmert 零和基 \(Q\)，设 \(G=Q^\top\operatorname{diag}(\gamma_i)Q/\kappa\)，其中 \(\kappa\) 固定为训练起点的有效边刚度之和。空间坐标满足 \(\dot\xi=-G\nabla_\xi H\)，储能变化增加耗散项 \(-\nabla_\xi H^\top G\nabla_\xi H\)。十机同速率时恢复原共同速率的空间动力学。边对数倍率继续零和，倍率界继续为 \([0.5,2]\)。两组从同一 100 ms 可见历史估计十维记忆初态，使用同一组 train 尺度和半径、五次径向 Gauss–Newton 与 \(10^{-6}\) 二次先验。起报重置可见功角和速度，保留传播后的记忆与空间耦合状态。

每次参数更新累积完整 train 346 的梯度，批次至多 128。训练用 5 ms RK4，损失为事件等权的 1 s 和 3 s 归一化 MSE 平均。两组共享参数的 Adam 学习率为 0.006，空间组速率学习率为 0.04。两类参数分别将梯度范数裁剪至 1，每次更新后将速率投影到签核范围。

每组最多更新 36 次，初始及每隔 6 次更新用完整 val 107 做双步长 8 s 评分。无未覆盖或数值失败时，优先选最小有限 TASK3 六项轨迹分数，同分取最早步骤。其余检查点按无有效判定数、未判对数和步数排序。至少完成 18 次更新后，若两组连续三次完整 val 检查均无有效进步，则联合停止。有效进步包括进入更优的检查点类别、无效分数下覆盖或错判计数改善，以及有效分数相对下降至少 0.1%。数值训练失败时保留各组已评检查点与失败记录。

冻结两组检查点和输入尺度后，完整评分 train、val，再由独立 test 阶段读取完整 76 例。两阶段冻结推理均用 CUDA float64、RK4 2.5 ms 与 1.25 ms、最长 8 s、批量至多 32。轨迹门限不决定是否运行 test。各划分报告归一化轨迹误差、TP/TN/FP/FN、失稳召回、漏判、稳定误报、可判准确率和覆盖率。完整名单准确率有未决时记为空。test 只供冻结基准比较。

训练前用构造数据核对等速率退化、零起点梯度、历史和未来方向导数、带输入能量收支和实际训练批次的 GPU 反传。注册文件固定算子预算、数据名单、代码与所有启动资产的哈希。

## 报告与结果

<a id="doc-0ea75e26ad8a"></a>

<a id="doc-0ea75e26ad8a-逐机零和耦合的完整-task3-对照"></a>

### 逐机零和耦合的完整 TASK3 对照

固定耦合组和逐机速率组各完成 18 次全量 train 更新。两组都从冻结 T0M1 第 12 步及同一个逐机功角阻尼模型起步。空间组的十个速率初值为零，两组共用 100 ms 初始化规则与相同的 22 个可训练系数。[训练协议](#doc-1b547e9cbe55)


<a id="doc-0ea75e26ad8a-训练与检查点"></a>

#### 训练与检查点

| 更新步 | 固定耦合 train MSE | 逐机速率 train MSE | 固定耦合 val 六项分数 | 逐机速率 val 六项分数 |
|---:|---:|---:|---:|---:|
| 0 | 0.830735 | 0.830735 | 0.42674 | 0.42674 |
| 6 | 0.611364 | 0.600079 | 0.42804 | 0.42816 |
| 12 | 0.512062 | 0.486601 | 0.43128 | 0.43149 |
| 18 | 0.421758 | 0.398971 | 0.43622 | 0.43653 |

train MSE 是 1 s 和 3 s 归一化轨迹平方误差按事件等权平均。首行取第一次更新前的共同损失，随后各行取执行该次更新时计算的损失。val 分数取该步更新完成后完整 107 例的冻结 TASK3 六项轨迹指标。逐机速率组最后一次 train 损失比固定组低 5.4%。第 18 步后，速率向量第 6、第 10 项分别为 0.581、0.469 \(\mathrm{s}^{-1}\)，其余为零。三次完整 val 检查分数均高于起点，两组联合早停并选中第 0 步。冻结模型中十个空间速率均为零。本轮学习降低了 train 损失，未改善 val 轨迹分数。[拟合与冻结回执](_sealed/kiki_machine_local_topology_spatial_relaxation_fit_20261116_evidence.json)


<a id="doc-0ea75e26ad8a-完整冻结评分"></a>

#### 完整冻结评分

两组的冻结评分一致。归一化 RMSE 为每个时域上可评分事件的平均值，括号内给出该项样本数。表中六项分数为 TASK3 固定口径。

| 划分 | 1 s | 2 s | 5 s | 8 s | 六项分数 |
|---|---:|---:|---:|---:|---:|
| train 346 | 0.30763（315） | 0.31716（311） | 0.45207（311） | 0.64004（311） | 0.55176 |
| val 107 | 0.29920（96） | 0.26251（95） | 0.22625（95） | 0.22020（95） | 0.42674 |
| test 76 | 0.29395（67） | 0.24672（67） | 0.20933（67） | 0.18779（67） | 0.41890 |

稳定性来自最长 8 s 预测轨迹。两档积分在已分类事件的稳定性判断一致。所有样本均完成双步长计算，无未覆盖或数值失败。完整准确率在有未决时记为 `null`。失稳召回和漏判率按可判定失稳计算，未决列出实数。

| 划分 | TP/TN/FP/FN | 未决 | 完整准确率 | 可判准确率 | 判定覆盖率 | 失稳召回／漏判率 | 稳定误报率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train 346 | 27/308/3/4 | 4 | `null` | 97.95% | 98.84% | 87.10%／12.90% | 0.96% |
| val 107 | 11/95/0/1 | 0 | 99.07% | 99.07% | 100% | 91.67%／8.33% | 0% |
| test 76 | 3/67/0/4 | 2 | `null` | 94.59% | 97.37% | 42.86%／57.14% | 0% |

test 共判对 70 例。9 个真实失稳样本中，3 个正确检出，4 个判为稳定，2 个未决。两个模型均按冻结检查点完成 test。第 0 步的结果刻画共同起点的预测表现，空间组训练中学出的非零速率未进入 test。[测试回执](_sealed/kiki_machine_local_topology_spatial_relaxation_test_20261116_evidence.json)

训练与 train、val 复评共用时 30.62 分钟，独立 test 用时 72.39 秒。完整名单、操作预算、早停与冻结文件经独立核验。逐例轨迹、损失与分组分数均保存在回执绑定的不可变目录。

## 回执与附件

- [kiki_machine_local_topology_spatial_relaxation_fit_20261116_evidence.json](_sealed/kiki_machine_local_topology_spatial_relaxation_fit_20261116_evidence.json)
- [kiki_machine_local_topology_spatial_relaxation_test_20261116_evidence.json](_sealed/kiki_machine_local_topology_spatial_relaxation_test_20261116_evidence.json)
- [kiki_machine_local_topology_spatial_relaxation_fit_result_review_20261116.json](_sealed/kiki_machine_local_topology_spatial_relaxation_fit_result_review_20261116.json)
- [kiki_machine_local_topology_spatial_relaxation_fit_review_20261116.json](_sealed/kiki_machine_local_topology_spatial_relaxation_fit_review_20261116.json)
- [kiki_machine_local_topology_spatial_relaxation_interface_20261116.json](_sealed/kiki_machine_local_topology_spatial_relaxation_interface_20261116.json)
- [kiki_machine_local_topology_spatial_relaxation_test_result_review_20261116.json](_sealed/kiki_machine_local_topology_spatial_relaxation_test_result_review_20261116.json)
- [kiki_machine_local_topology_spatial_relaxation_test_review_20261116.json](_sealed/kiki_machine_local_topology_spatial_relaxation_test_review_20261116.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_spatial_relaxation_v1_zh.md](_sealed/kiki_machine_local_topology_spatial_relaxation_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/kiki_machine_local_topology_spatial_relaxation_v1_zh.md](_sealed/original/kiki_machine_local_topology_spatial_relaxation_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_spatial_relaxation_20261116_zh.md](_sealed/kiki_machine_local_topology_spatial_relaxation_20261116_zh.md)

</details>
