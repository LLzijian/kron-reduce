# 延迟失步下的余量梯度标定与合成对照

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_delayed_v1.json)

## 执行协议

<a id="doc-320cc6364df2"></a>

<a id="doc-320cc6364df2-延迟失步合成对照与余量梯度标定"></a>

### 延迟失步合成对照与余量梯度标定

沿用独立 T0M1 真值方程和 100 ms 历史。12 个预先冻结的双机组合各生成 1 个稳定事件和 1 个延迟失稳事件，完整 train、val、test 分别为 12、6、6。机对由种子 2026111503 在 45 个无序机对中不放回抽取。对每个机对，令 \(d=1_F-\sum_{j\in F}M_j/\sum_jM_j\)，功角和转速均以该 COI 方向给定。平衡角、机械输入和隐藏初态为零。

稳定事件在历史起点取 \(q=0.12d\)、\(v=-0.12d\)。失稳事件取 \(q=-2.5d\)，速度差依次试算 13.5、14.5、15.5、16.5、17.5 rad/s。先完成全部 72 条候选的 1.25 和 0.625 ms NumPy RK4 真值，再按各机对的固定速度顺序取第一个两档均确认的失稳事件。两档确认索引须一致，首次确认在起报后 0.20 至 0.80 秒，首次确认窗的归一化余量在 \([10^{-4},0.05]\) 内。稳定事件须在完整 8 秒两档均判稳定。所有试算的资格与确认时刻存档。任一配额无合格真值即停止，并保留失败名单。所选路径还须通过正式训练尺度下的双步长差核验。

共同初值有意将共享刚度设为真值的 1.15 倍、阻尼设为 1.35 倍，用于检验过强恢复力和耗散造成的失步漏判。十机端口幅值均为 0.55，时间常数按 0.08 至 0.55 秒的几何序列排布。两组的历史隐态估计都采用五次 Gauss–Newton，记忆半径 1、先验权重 0，使用同一份 train 归一化。失稳监督到首次五点确认，稳定监督到 3 秒。共同初值必须在完整 train 的六个失稳中至少漏掉两个匹配窗口，否则只评价共同初值并保存诊断。

仅用完整 train 在共同初值计算 1/3 秒轨迹目标及匹配失步余量目标的梯度。记录梯度分量、范数和夹角。两项范数均须有限且大于 \(10^{-12}\)。候选权重取轨迹梯度与余量梯度二范数之比。按范数裁剪 1 和 Adam 的首步公式试算更新方向，联合目标与轨迹目标之差须达到轨迹目标首步方向范数的 5%，随后冻结权重。梯度门失败时只评价共同初值。

通过初始资格后，两组各做 20 次完整 train 更新。对照用轨迹目标，候选加标定后的余量项。CUDA float64、Adam 学习率 0.001、裁剪 1、训练步长 5 ms。共同初值和终点权重先复制到 CPU 保存，再对完整 train、val、test 用 2.5/1.25 ms GPU 双步长评分。沿用上一轮 val 的 5% 余量改善、零稳定误报、完整判定和轨迹误差门。两组均须在 train 六个失稳事件上具备完整数值资格，候选还须比同起点轨迹组多复现至少一个失稳窗口。test 用于报告。模型部署决定为 `KEEP_BASELINE`。

一次执行。至多两次拟合、40 次参数更新、45 次梯度、77,760 次批量 NumPy RHS、1,337 次传播、3,214,083 次 Torch RHS、196,400 次切线 RHS、3,900 次 QR、SVD、三角求解和对应乘法各一类。正式评分批次失败后至多逐例隔离一次，计入上限。真实数据读取为零。[上一轮证据](../synthetic_margin/README.md#doc-c04b4d708da9)

## 报告与结果

<a id="doc-81598fbdd211"></a>

<a id="doc-81598fbdd211-延迟失步下的余量梯度标定与合成对照"></a>

### 延迟失步下的余量梯度标定与合成对照

[执行协议](#doc-320cc6364df2)

两组各完成 20 次完整 train 更新。加入标定余量后，合成 val 的匹配失稳余量 RMSE 从 0.0311416 变为 0.0311432，完整 train 和 val 仍未恢复任何匹配失步窗口。验收条件未满足，保留现有基线。

完整合成 train 的共同初值漏掉 6/6 个匹配失步窗口。轨迹和余量梯度的范数分别为 0.0134812 和 0.0069634，夹角余弦为 0.997673。范数比为 1.93601，首步 Adam 更新的相对差为 42.59%。


<a id="doc-81598fbdd211-完整的轨迹与稳定性评价"></a>

#### 完整的轨迹与稳定性评价

合成 train、val、test 分别为 12、6、6。8 秒 NRMSE 为等事件权重的池化归一化 RMSE，GPU 双步长为 2.5/1.25 ms。

| 模型 | 数据划分 | 失稳余量 RMSE | 匹配失步 | 3 秒 NRMSE | 8 秒 NRMSE | TP/TN/FP/FN | 未决／未覆盖／数值失败 | 完整准确率 | 可判定准确率 |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| 共同初值 | train | 0.0371679 | 0/6 | 0.557358 | 1.18234 | 3/6/0/0 | 3/0/0 | `null` | 100.00% |
| 共同初值 | val | 0.0344209 | 0/3 | 0.909116 | 2.08682 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |
| 共同初值 | test | 0.0481814 | 0/3 | 1.02543 | 1.6154 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |
| 轨迹目标 | train | 0.0330076 | 0/6 | 0.506048 | 1.0687 | 4/6/0/0 | 2/0/0 | `null` | 100.00% |
| 轨迹目标 | val | 0.0311416 | 0/3 | 0.832798 | 1.85045 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |
| 轨迹目标 | test | 0.0433598 | 0/3 | 0.944521 | 1.48821 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |
| 轨迹与标定余量目标 | train | 0.0330052 | 0/6 | 0.505608 | 1.06593 | 4/6/0/0 | 2/0/0 | `null` | 100.00% |
| 轨迹与标定余量目标 | val | 0.0311432 | 0/3 | 0.832292 | 1.84365 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |
| 轨迹与标定余量目标 | test | 0.0433667 | 0/3 | 0.944476 | 1.48761 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |

val 失稳余量未通过，val 匹配失步检出数不降通过，val 轨迹误差通过，val 稳定性覆盖与正确数未通过，train 失稳窗口恢复未通过。

两组在完整合成 val 和 test 均有 1 个失稳样本未决。各自覆盖率为 83.33%，可判定失稳召回率为 100%，按全失稳分母确认检出 2/3。稳定误报率为 0%。

两个损失的梯度几乎同向。首步 Adam 更新的相对差为 42.59%，但把更新投影到余量下降方向后，预计首步下降量仅多 0.157%。两组训练到第 20 步时轨迹目标仍在下降。用 2.5 ms 轨迹诊断时，两个终点及共同初值在合成 train 和 val 的 9 个失稳事件上都触及历史起点的记忆储能上限 0.5，这些真值的历史起点隐态均为零。下一步分别替换起报隐态与动力学系数。[后续方案](../synthetic_topology/README.md#doc-415956346cb1)

运行 `machine_local_topology_delayed_v1-delayed-20261115T075847142292Z-aa3aa014-c7d6d0af`，用时 196.5 秒。
[证据回执](_sealed/machine_local_topology_delayed_20261115_evidence.json) · [结果核对](_sealed/machine_local_topology_delayed_result_review_20261115.json)

## 回执与附件

- [machine_local_topology_delayed_20261115_evidence.json](_sealed/machine_local_topology_delayed_20261115_evidence.json)
- [machine_local_topology_delayed_result_review_20261115.json](_sealed/machine_local_topology_delayed_result_review_20261115.json)
- [machine_local_topology_delayed_review_20261115.json](_sealed/machine_local_topology_delayed_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_delayed_v1_zh.md](_sealed/machine_local_topology_delayed_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/machine_local_topology_delayed_v1_zh.md](_sealed/original/machine_local_topology_delayed_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_delayed_20261115_zh.md](_sealed/machine_local_topology_delayed_20261115_zh.md)

</details>
