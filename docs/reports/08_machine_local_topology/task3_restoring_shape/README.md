# TASK3 恢复力形状完整评估

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[拟合方案](../../../../configs/proposals/kiki_machine_local_topology_restoring_shape_fit_v1.json) · [test 方案](../../../../configs/proposals/kiki_machine_local_topology_restoring_shape_test_v1.json)

## 执行协议

<a id="doc-3f46ffb1ebec"></a>

<a id="doc-3f46ffb1ebec-task3-恢复力形状的完整-train-拟合"></a>

### TASK3 恢复力形状的完整 train 拟合

以逐机功角阻尼为比较锚点，使用原 T0M1 第 12 步的支路刚度、阻尼、逐机记忆、100 ms 历史合同和零阶保持功率。阻尼取 \(\rho=2/3\)、空间系数 \(\lambda=1\)、尺度 1 rad。恢复力修正沿用已通过的合成方案。[已知真值协议](../synthetic_restoring_shape/README.md#doc-029c3a1cfb71)

仅打开完整 TASK3 train 的两个来源缓存，保留 346 例的起报前历史和起报后最长 3 s 的实际可用观测。使用同一动量权重、隐初态列空间投影和空间阻尼基准，独立求解形状系数 \(\alpha\in[-1/4,1/4]\) 和下界为 \(10^{-6}\) 的正全局刚度乘数。全局刚度用于弱动量对照。隐初态投影使用观测路径，用于估计共享系数。闭环预测另行使用 100 ms 历史估计事件隐初态。

工程门要求完整名单、346 个秩 10 算子、四点与八点求积差至多 \(10^{-10}\) 以及两个有限可辨识拟合。科学门要求形状预测的弱残差平方和相对零改动下降至少 5%，稳定组和失稳组各有正收益，并在三个统计范围均至多等于对应全局刚度对照。工程合格且系数非零时，冻结形状候选并登记完整 TASK3 的 GPU 预测，完成 train 346、val 107、test 76 和独立评分复核。弱残差科学门单独报告。

单次执行，最多载入两个模型文件，计算一次阻尼平方根、346 次 SVD，拟合两次。实测与复核区间预算各 107,260。预测、val 与 test 数据读取次数均为零。

<a id="doc-7fac27d8aff5"></a>

<a id="doc-7fac27d8aff5-task3-恢复力形状完整比较"></a>

### TASK3 恢复力形状完整比较

将完整 train 弱形式拟合得到的形状系数加入逐机功角阻尼模型。训练和形状拟合在评分前关闭。使用原 T0M1 检查点、支路参数、逐机记忆以及同一阻尼设置。初始化使用候选方程和 11 帧 100 ms 历史，因果输入为起报输入功率的零阶保持。[train 拟合协议](#doc-3f46ffb1ebec)

按冻结的 TASK3 名单评估 train 346、val 107、test 76。历史和未来传播均使用 CUDA float64，步长 2.5 ms 和 1.25 ms。报告各时间窗的轨迹归一化 RMSE、失稳样本实际可用区间的误差、五点失步判断和覆盖率。对保存预测独立复算。沿用 T0M1 对照和预登记的 train 与 val 门，报告同一逐机阻尼锚点的保存结果。

完整评估在一个不可重复的已登记执行中完成。RMSE 和 train、val 门的结果均保留。test 为已曝光的固定比较名单，评分用于报告。完整 test 目标要求 76 例全判定且至少 69 例正确。

## 报告与结果

<a id="doc-bc0a87fbb260"></a>

<a id="doc-bc0a87fbb260-task3-恢复力形状完整评估"></a>

### TASK3 恢复力形状完整评估

恢复力修正把 train 的稳定误报从 3 例降到 2 例，val 判对 104/107。test 判对 70 例，74 例可判定，2 例未决，完整准确率为 `null`。依照 val 的失稳检出与六项轨迹分数，下一轮以逐机功角阻尼为研究锚点。


<a id="doc-bc0a87fbb260-冻结模型与训练证据"></a>

#### 冻结模型与训练证据

在逐机功角阻尼上加入保持平衡点与线性刚度的支路恢复力项。完整 train 的形状系数为 −0.03621840651061638。工程检查通过。相对零修正，形状候选的弱动量残差平方和在 train 346 例下降 0.87%，稳定 311 例下降 1.40%，失稳 35 例上升 0.28%。失稳组的全局刚度对照下降 0.18%，弱动量科学门未通过。[拟合回执](_sealed/kiki_machine_local_topology_restoring_shape_fit_20261115_evidence.json)

评分使用自己的候选方程完成 100 ms 历史初始化与未来预测，冻结功率零阶保持、原记忆半径和先验。CUDA float64 使用 2.5 ms 和 1.25 ms 两档步长。train 346、val 107、test 76 均完成预测，独立复算的 529 例得分一致。[执行协议](#doc-7fac27d8aff5) · [执行回执](_sealed/kiki_machine_local_topology_restoring_shape_20261115_evidence.json) · [独立复算](_sealed/kiki_machine_local_topology_restoring_shape_saved_audit_20261115_evidence.json)


<a id="doc-bc0a87fbb260-轨迹误差"></a>

#### 轨迹误差

各列采用 TASK3 的事件平均归一化 RMSE。括号内是该时间窗参与评分的事件数，失稳列使用各失稳样本的实际可用时段。

| 划分 | 模型 | 1 s | 2 s | 5 s | 8 s | 失稳样本 |
|---|---|---:|---:|---:|---:|---:|
| train | 逐机阻尼 | 0.30763（315） | 0.31716（311） | 0.45207（311） | 0.64004（311） | 0.81614（35） |
| train | 恢复力修正 | 0.29043（315） | 0.28264（311） | 0.35417（311） | 0.47098（311） | 0.86669（35） |
| val | 逐机阻尼 | 0.29920（96） | 0.26251（95） | 0.22625（95） | 0.22020（95） | 0.76979（12） |
| val | 恢复力修正 | 0.28995（96） | 0.25288（95） | 0.22200（95） | 0.21807（95） | 0.82213（12） |
| test | 逐机阻尼 | 0.29395（67） | 0.24672（67） | 0.20933（67） | 0.18779（67） | 1.01852（9） |
| test | 恢复力修正 | 0.29074（67） | 0.24459（67） | 0.20923（67） | 0.18811（67） | 1.05813（9） |

val 的六项轨迹分数从 0.42674 升至 0.43126。恢复力修正的 train 与 val 均通过预登记闭环门，train 两档步长各复现 7/35 个匹配失步窗口。[逐机阻尼对照](../task3_spatial_damping/README.md#doc-08e5190828ec)


<a id="doc-bc0a87fbb260-稳定性"></a>

#### 稳定性

五点失步判据以失稳为阳性。可判定准确率、失稳召回率、漏判率和稳定误报率以可判定样本为分母。存在未决时完整准确率为 `null`。两组预测均无未覆盖或数值失败。

| 划分 | 模型 | 已判定/名单 | TP/TN/FP/FN | 未决 | 覆盖率 | 完整准确率 | 可判定准确率 | 失稳召回率 | 漏判率 | 稳定误报率 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| train | 逐机阻尼 | 342/346 | 27/308/3/4 | 4 | 98.84% | `null` | 97.95% | 87.10% | 12.90% | 0.96% |
| train | 恢复力修正 | 342/346 | 27/309/2/4 | 4 | 98.84% | `null` | 98.25% | 87.10% | 12.90% | 0.64% |
| val | 逐机阻尼 | 107/107 | 11/95/0/1 | 0 | 100.00% | 99.07% | 99.07% | 91.67% | 8.33% | 0.00% |
| val | 恢复力修正 | 107/107 | 9/95/0/3 | 0 | 100.00% | 97.20% | 97.20% | 75.00% | 25.00% | 0.00% |
| test | 逐机阻尼 | 74/76 | 3/67/0/4 | 2 | 97.37% | `null` | 94.59% | 42.86% | 57.14% | 0.00% |
| test | 恢复力修正 | 74/76 | 3/67/0/4 | 2 | 97.37% | `null` | 94.59% | 42.86% | 57.14% | 0.00% |

恢复力修正在 test 的 67 个真实稳定样本全部判对。9 个真实失稳样本中，3 个判为失稳、4 个判为稳定、2 个未决。完整 test 准确率目标仍需闭合。部署决定为 `KEEP_BASELINE`。

## 回执与附件

- [kiki_machine_local_topology_restoring_shape_20261115_evidence.json](_sealed/kiki_machine_local_topology_restoring_shape_20261115_evidence.json)
- [kiki_machine_local_topology_restoring_shape_fit_20261115_evidence.json](_sealed/kiki_machine_local_topology_restoring_shape_fit_20261115_evidence.json)
- [kiki_machine_local_topology_restoring_shape_saved_audit_20261115_evidence.json](_sealed/kiki_machine_local_topology_restoring_shape_saved_audit_20261115_evidence.json)
- [kiki_machine_local_topology_restoring_shape_fit_result_review_20261115.json](_sealed/kiki_machine_local_topology_restoring_shape_fit_result_review_20261115.json)
- [kiki_machine_local_topology_restoring_shape_fit_v1_review_20261115.json](_sealed/kiki_machine_local_topology_restoring_shape_fit_v1_review_20261115.json)
- [kiki_machine_local_topology_restoring_shape_saved_audit_review_20261115.json](_sealed/kiki_machine_local_topology_restoring_shape_saved_audit_review_20261115.json)
- [kiki_machine_local_topology_restoring_shape_test_result_review_20261115.json](_sealed/kiki_machine_local_topology_restoring_shape_test_result_review_20261115.json)
- [kiki_machine_local_topology_restoring_shape_test_v1_review_20261115.json](_sealed/kiki_machine_local_topology_restoring_shape_test_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_restoring_shape_fit_v1_zh.md](_sealed/kiki_machine_local_topology_restoring_shape_fit_v1_zh.md)
- [docs/protocols/kiki_machine_local_topology_restoring_shape_v1_zh.md](_sealed/kiki_machine_local_topology_restoring_shape_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_restoring_shape_20261115_zh.md](_sealed/kiki_machine_local_topology_restoring_shape_20261115_zh.md)

</details>
