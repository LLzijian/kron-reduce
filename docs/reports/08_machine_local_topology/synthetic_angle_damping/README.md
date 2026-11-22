# 功角相关阻尼的合成配对验证

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_angle_damping_v1.json)

## 执行协议

<a id="doc-1fc52ce43e80"></a>

<a id="doc-1fc52ce43e80-功角相关阻尼的合成配对验证"></a>

### 功角相关阻尼的合成配对验证

同一组已封存的 48 例初始条件生成常值阻尼和功角相关阻尼的真值。沿用 train 24、val 12、test 12。零强度直接使用已有两档步长轨迹。正强度 \(\rho=0.5\) 用独立 NumPy 真值算子重算，步长为 1.25 ms 和 0.625 ms。刚度、逐机记忆、初始条件及 ZOH 功率固定。[方程与导数](../synthetic_momentum/README.md#doc-f2ac6b8b3491-%E4%B8%8B%E4%B8%80%E9%A1%B9%E5%90%88%E6%88%90%E5%AF%B9%E7%85%A7)

每种真值分别用完整 train 在实测路径上拟合正的常值倍率和单一功角系数 \(0\leq\rho\leq2/3\)。两个方向共用一个投影，消去同一逐机记忆起点的自由响应。采用分段线性观测和四点 Gauss 积分，八点独立核查，拟合最多使用起报后 3 s。闭环输入只用 100 ms 历史，保留原径向初始化和合成记忆半径 1、先验 \(10^{-6}\)。支撑范围外的 12 例也参与拟合与完整评分，单列半径内的轨迹门。系数和尺度先冻结，再评分 val 和 test。两种预测均采用两档 CUDA float64 步长 2.5 ms 和 1.25 ms，预测 8 s。

工程门在 train 和 val 检查数值资格、观测动量秩 10、功率残差及求积差不超过 \(10^{-10}\)。单元测试核验零系数时右端、切线和因果预测逐元素一致。两种真值的功角系数误差和零强度真值的常值倍率误差均至多 0.001。半径可行的 train 与 val 样本须恢复真值判定，触发时间相差至多 20 ms，8 s 归一化 RMSE 至多 0.01。正强度真值中，train、val 的动量残差平方和须至多为常值阻尼候选的 5%，常值候选的残差平方和须高于真值系数参考残差的 100 倍。半径可行的 val 稳定组 8 s 归一化 RMSE 至多为常值阻尼候选的 90%。train、val 的匹配失步触发数不得下降，稳定误报不得增加。完整合成 test 的轨迹与稳定性结果只作报告。

单次合成运行，真实数据读取为零。精确操作上限以配对方案 JSON 为准，含 1 次已封存 truth NPZ 解码、4 次单参数闭式拟合以及全部 48 例在两种真值与两种候选下的双步长评分。通过合成门后，真实训练和冻结比较另行登记。

## 报告与结果

<a id="doc-4a85dfb8b0af"></a>

<a id="doc-4a85dfb8b0af-功角相关阻尼的合成配对验证"></a>

### 功角相关阻尼的合成配对验证

正强度配对从完整合成 train 恢复系数 0.50000293，真值为 0.5。半径内的 6 个 val 稳定样本中，8 s 归一化 RMSE 从常值阻尼候选的 0.0136847 降至 0.000655232。工程门和科学门均通过。


<a id="doc-4a85dfb8b0af-设置与证据"></a>

#### 设置与证据

本次配对实验使用已封存的 48 个解析合成初值，按 train 24、val 12、test 12 划分。相同的刚度、记忆系数、初值与 ZOH 功率下，比较常值阻尼和功角相关阻尼。

\[
D(q)=\left(1-\rho\frac{s}{1+s}\right)D_0,
\qquad s=\operatorname{mean}((B^Tq)^2),\quad 0\leq\rho\leq 2/3.
\]

两种候选都在 train 实测路径上拟合一个标量；随后冻结参数，再评分 val 和 test。闭环预测仍采用原来的 100 ms 观测历史、径向隐状态初始化、记忆半径 1、先验权重 $10^{-6}$，并以两档积分步长预测 8 s。test 只用于最终报告。

运行 `machine_local_topology_angle_damping_v1-angle_damping-20261115T112513267447Z-ff9676a4-7b106d35`。独立审核从保存的 JSON 复核了四组完整名单、拟合、弱残差、混淆矩阵和门槛。[冻结协议](#doc-1fc52ce43e80) · [回执](_sealed/machine_local_topology_angle_damping_20261115_evidence.json)


<a id="doc-4a85dfb8b0af-系数恢复与路径残差"></a>

#### 系数恢复与路径残差

| 合成真值 | 功角系数估计 | 常值候选的相对阻尼系数 | train 弱残差比（功角/常值） | val 弱残差比（功角/常值） |
|---|---:|---:|---:|---:|
| ρ=0 | 0.00000636 | -0.00000341 | — | — |
| ρ=0.5 | 0.50000293 | -0.457424（总倍率 0.542576） | 0.000000006998 | 0.000000007801 |

零强度的功角估计与零真值一致，常值项通过零效应门。正强度下功角系数恢复误差为 $2.93\times10^{-6}$；train、val 的常值候选残差分别超过真值离散参考残差约 (1.367\times10^8) 倍和 (1.221\times10^8) 倍。两种真值的能量恒等式误差均在 (4\times10^{-16}) 以下。


<a id="doc-4a85dfb8b0af-因果预测"></a>

#### 因果预测

半径内的 train 与 val 共 27 例。正强度真值下，功角候选在这些事件上的最大 8 s 归一化 RMSE 为 0.00137774，低于预设 0.01 门限。val 稳定组的 8 s 误差比为 0.04788，通过 0.90 门限。

正强度功角候选的全名单汇总 NRMSE 采用事件等权。

| 划分 | 1 s | 2 s | 5 s | 8 s |
|---|---:|---:|---:|---:|
| train（24） | 0.002115 | 0.002143 | 0.001939 | 0.001583 |
| val（12） | 0.001984 | 0.002012 | 0.001818 | 0.001483 |
| test（12） | 0.002190 | 0.002194 | 0.001975 | 0.001612 |

在 val 全名单的 8 s 汇总上，正强度功角候选 NRMSE 为 0.001483，常值候选为 0.078113（比例 0.01899）。test 对应为 0.001612 与 0.089881（比例 0.01794），只作报告。


<a id="doc-4a85dfb8b0af-稳定性结果"></a>

#### 稳定性结果

稳定性标签由合成真值与预测轨迹按相同规则得到，失稳为阳性。两种真值、两种候选在每个划分都覆盖全部样本，无未解析、数值失败或未覆盖样本；因此全名单准确率均为 100%。

| 划分 | 样本 | TP | TN | FP | FN | 覆盖率 | 准确率 | 失稳召回率 | 稳定误报率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 24 | 6 | 18 | 0 | 0 | 100% | 100% | 100% | 0% |
| val | 12 | 3 | 9 | 0 | 0 | 100% | 100% | 100% | 0% |
| test | 12 | 3 | 9 | 0 | 0 | 100% | 100% | 100% | 0% |

两种真值与两种候选的混淆计数相同。正强度对照在匹配时段内复现了 train 的 6/6 个和 val 的 3/3 个失步窗口。本轮真实数据读取为零。

## 回执与附件

- [machine_local_topology_angle_damping_20261115_evidence.json](_sealed/machine_local_topology_angle_damping_20261115_evidence.json)
- [machine_local_topology_angle_damping_result_review_20261115.json](_sealed/machine_local_topology_angle_damping_result_review_20261115.json)
- [machine_local_topology_angle_damping_v1_review_20261115.json](_sealed/machine_local_topology_angle_damping_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_angle_damping_v1_zh.md](_sealed/machine_local_topology_angle_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_angle_damping_20261115_zh.md](_sealed/machine_local_topology_angle_damping_20261115_zh.md)

</details>
