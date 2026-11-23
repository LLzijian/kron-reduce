# 四种耦合与记忆方程在新合成激励下的表现

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_transfer_v1.json)

## 执行协议

<a id="doc-1c95f352f620"></a>

<a id="doc-1c95f352f620-耦合与记忆方程的独立合成激励验证"></a>

### 耦合与记忆方程的独立合成激励验证

检验已训练的四个方程能否解释新事件，并用冻结失步规则评价起报后的稳定性。T0M0、T1M0 来自 S1，T0M1、T1M1 来自长时域精化。四个方程分别评价四种同协议真值，共 16 组对照。每种真值使用相同 24 个新初值和恒值机械输入，固定 `train=12`、`val=6`、`test=6`。每个划分各含一半低速和高速激励。真值参数、激励种子、ID 和角色在执行前冻结。

初始角在 ±0.14 rad 内抽样后转为 COI 坐标。初始速度基值在 ±0.12 rad/s 内抽样。高速组另外给随机排列的五机和另五机施加符号相反的速度偏移，偏移幅值按事件顺序从 10.5 到 12 rad/s 等距取值，最后转为 COI 坐标。高速激励以 100 ms 观测之后触发失步为设计目标。标签由真值轨迹按 TASK3 规则产生。保留全部预设事件，类别不足或起报前已失步纳入覆盖门槛。

用独立 NumPy 模型以 1.25/0.625 ms RK4 生成真值。方程、五次有界 Gauss–Newton 和可见末点重置沿用既有实现。每个真值系统用完整 train 一次计算历史加权尺度和轨迹评分尺度，供同系统的四个候选共用。新历史尺度会影响加权初始化，属于事先声明的 train 归一化调整。候选只接收该事件的 11 点历史和恒值平衡投影机械功率，使用 GPU 上 2.5/1.25 ms 双步长推理 8 秒。

四个方程与各自匹配真值的 train、val 需有完整历史资格并同时含稳定和失稳事件。每个匹配组合在 val 的稳定性准确率须超过 90%，完整 train、val 每例的 3 秒与 8 秒合并归一化 RMSE 分别至多 0.01 和 0.03，有效边刚度最大相对误差至多 5%。真值双步长差至多 1e-6，预测双步长差至多 1e-5，分类和触发时刻按既有双步长规则核对。匹配模型的预测起点核对能量恒等式，归一化误差至多 1e-10。

每种真值与候选的完整 train、val、test 均记录轨迹误差、稳定性准确率、混淆矩阵、覆盖、未决和失败，同时按低速、高速组报告 8 秒归一化 RMSE、功角 RMSE（rad）和转速 RMSE（rad/s）。真值数值核验覆盖三个划分，未通过的事件计为数值失败和未覆盖。test 用于报告。四个匹配组合承担验证门槛，另外 12 组交叉结果用于解释方程对不同激励的响应。本轮零拟合、零真实数据读取，保持 `KEEP_BASELINE`。

单次执行。四组真值计 311040 次批量 NumPy RHS。每个组合每档步长执行一次完整批次，批次数值失败后至多逐例隔离一次。计入隔离后，预算上限为 4600 次传播、16272004 次 Torch RHS、720000 次切线 RHS、5760 次 QR、SVD、三角求解和对应乘法各一项，以及四次能量微分。历史求解的线性代数按事件计数，RK4 RHS 按批量调用计数。

## 报告与结果

<a id="doc-1530b2d8d992"></a>

<a id="doc-1530b2d8d992-四种耦合与记忆方程在新合成激励下的表现"></a>

### 四种耦合与记忆方程在新合成激励下的表现

四个冻结方程均通过独立合成验证。在包含耦合修正和逐机记忆的真值系统上，联合模型的 test 8 秒合并归一化 RMSE 为 0.002134，固定耦合的逐机记忆模型为 0.006425，仅含耦合修正的模型为 0.1701。四个方程各自沿用前两轮选中的系数，初始化历史权重与评分尺度按协议从新合成 train 一次确定。[执行协议](#doc-1c95f352f620)

每种真值含 24 个预定事件，train 12、val 6、test 6。四种真值共用激励初值，每个划分各有一半低速和高速激励。16 组对照都完成 8 秒 GPU 双步长预测，最早真值失步发生在起报后 70 ms。各组 train 的稳定性准确率为 12/12，val 和 test 均为 6/6。val、test 的 TP/TN/FP/FN 分别为 3/3/0/0，覆盖率和失稳召回率均为 100%，漏判率、误报率、未决、未覆盖和数值失败均为零。


<a id="doc-1530b2d8d992-闭环轨迹"></a>

#### 闭环轨迹

表中为各真值内使用相同 train 尺度计算的 8 秒合并归一化 RMSE。T1 表示启用节点耦合修正，M1 表示启用逐机记忆。所有数值来自全名单。

| 真值 | 冻结模型 | train | val | test |
|---|---|---:|---:|---:|
| T0M0 | **T0M0** | 2.373e-7 | 2.021e-7 | 2.696e-7 |
| T0M0 | T1M0 | 4.4587e-07 | 1.721e-7 | 1.851e-7 |
| T0M0 | T0M1 | 0.13452 | 0.13491 | 0.14690 |
| T0M0 | T1M1 | 0.12877 | 0.14195 | 0.14850 |
| T1M0 | T0M0 | 0.0061258 | 0.005953 | 0.004217 |
| T1M0 | **T1M0** | 4.966e-7 | 2.772e-7 | 6.020e-7 |
| T1M0 | T0M1 | 0.13485 | 0.13499 | 0.14679 |
| T1M0 | T1M1 | 0.12907 | 0.14161 | 0.14837 |
| T0M1 | T0M0 | 0.15053 | 0.15344 | 0.16672 |
| T0M1 | T1M0 | 0.15146 | 0.15270 | 0.16969 |
| T0M1 | **T0M1** | 0.0005521 | 0.0006421 | 0.0008164 |
| T0M1 | T1M1 | 0.0007398 | 0.003373 | 0.002765 |
| T1M1 | T0M0 | 0.14977 | 0.15397 | 0.16716 |
| T1M1 | T1M0 | 0.15076 | 0.15323 | 0.17014 |
| T1M1 | T0M1 | 0.0058161 | 0.013022 | 0.006425 |
| T1M1 | **T1M1** | 0.0006689 | 0.003853 | 0.002134 |

匹配的 T1M1 模型在 test 的 3 秒合并归一化 RMSE 为 0.000568。该模型的低速 3 例在 8 秒内功角与转速合并 RMSE 分别为 (3.62\times10^{-5}) rad 和 (1.48\times10^{-5}) rad/s，高速 3 例分别为 0.01859 rad 和 0.008708 rad/s。各组按激励划分的物理量误差、1/2/3/5/8 秒归一化误差以及逐事件结果见证据收据和原始运行。全部结构都能区分本轮低速稳定与高速失稳，结构之间的差异体现在闭环轨迹。


<a id="doc-1530b2d8d992-证据与后续范围"></a>

#### 证据与后续范围

运行 `machine_local_topology_transfer_v1-transfer-20261115T044343858959Z-cd2990d9-99b9d085`，耗时 89.25 秒，RTX 4060 Ti，Torch 2.13.0+cu130。逐事件数值资格、全部划分的评分汇总、工件哈希与算子预算已复核。本轮零拟合、零真实数据读取，试验结论为 `KEEP_BASELINE`。真实 TASK3 目标按完整 76 例评分，要求全部可判定且至少判对 69 例。

[证据收据](_sealed/machine_local_topology_transfer_20261115_evidence.json) · [结果复核](_sealed/machine_local_topology_transfer_20261115_result_review.json)

## 回执与附件

- [machine_local_topology_transfer_20261115_evidence.json](_sealed/machine_local_topology_transfer_20261115_evidence.json)
- [machine_local_topology_transfer_20261115_result_review.json](_sealed/machine_local_topology_transfer_20261115_result_review.json)
- [machine_local_topology_transfer_review_20261115.json](_sealed/machine_local_topology_transfer_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_transfer_v1_zh.md](_sealed/machine_local_topology_transfer_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_transfer_20261115_zh.md](_sealed/machine_local_topology_transfer_20261115_zh.md)

</details>
