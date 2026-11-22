# 动态耦合合成验证

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_dynamic_coupling_v1.json)

## 研究方案

<a id="doc-393bc886e31e"></a>

<a id="doc-393bc886e31e-动态有效耦合与逐机记忆"></a>

### 动态有效耦合与逐机记忆

检验短历史因果预测是否需要随时间演化的有效边刚度。最新完整 TASK3 逐机阻尼在 val 判对 106/107，test 判对 70 例并有 2 例未决。恢复力形状修正的 val 失稳检出为 9/12，逐机阻尼为 11/12。更早的静态耦合 T1M1 在真实数据上由 val 选中第 0 步。动态耦合对失步时序和稳定预测的作用由配对实验衡量。[最新对话](chatgpt-conversation://6af25e1f-db88-83e9-a3ae-d02442950a25) · [逐机阻尼结果](../task3_spatial_damping/README.md#doc-08e5190828ec) · [静态耦合比较](../task3_topology_memory/README.md#doc-00b7fd2d6112)


<a id="doc-393bc886e31e-方程"></a>

#### 方程

第一项候选增加一个共同耦合状态 \(c\)，用 \(k_e(c)=k_e^0\exp \ell(c)\) 调节同一套边。取 \(\ell(c)=2b\tanh(c/b)\)，因此各边倍率在 \([e^{-2b},e^{2b}]\) 内。设 \(U(q,c)=\sum_e k_e(c)\phi_e(q)\)，其中 \(\phi_e\) 为原模型的移位势能，\(H= \tfrac12 v^\top Mv+U+\tfrac12z^\top z+\tfrac12\kappa c^2\)。沿同一可见恢复力与互易记忆方程加入

\[
\dot c=-\mu[\kappa c+\ell'(c)U(q,c)],\qquad
\dot H=v^\top p-v^\top D(q)v-z^\top\Lambda z-\mu(\partial_cH)^2 .
\]

首轮固定 \(b,\kappa\)，估计跨事件共享的 \(\mu\ge0\)。从 \(c=0\) 开始且 \(\mu=0\) 时，原固定耦合模型精确退化。倍率界限制动态刚度，轨迹的有限性、步长一致性和稳定性另由实验检验。\(c\) 表示约化方程的有效耦合。逐机记忆由速度驱动并通过加性端口力反馈。

九维零和坐标作为独立候选。对完全图，其对数边倍率之和为零，检验空间重分配。共同尺度与空间重分配分别建立证据后才考虑组合。按摆动方向变化的阻尼属于另一项耗散对照。


<a id="doc-393bc886e31e-因果与验证顺序"></a>

#### 因果与验证顺序

1. 从历史第一帧固定 \(c=0\)。用完整 100 ms 可见历史和当前候选方程估计逐机记忆初态，联合传播两个状态。起报时替换可见功角和转速，保留传播得到的 \(c,z\)。合成核对非零平衡角下的功率收支、解析切线、耦合倍率和起报连续性。
2. 在同一组初始条件下验证固定耦合、动态耦合和仅有静态恢复力变化三个真值。静态形状修正共用观测合同、主干、参数数量和训练预算。用 train 拟合、val 判断机制，报告全量合成 test。额外记录非零耦合前史对零初态先验的影响。
3. 合成通过后，另行登记完整 TASK3 train 的实测路径动量诊断，并保留固定因果初态与自由记忆投影的区别。实际预测仍由 100 ms 历史独立起报。对原 T0M1、逐机阻尼、动态耦合、两者联合做成对比较。各组按固定名单完成 GPU 双步长 train 346、val 107、test 76 评估和独立复算。

轨迹归一化 RMSE、匹配失步窗口、混淆矩阵与覆盖率共同评价预测。对动态结构另外记录刚度倍率的时间路径和新增松弛耗散。test 用于报告，结构与超参数依据 train 和预声明的 val 规则确定。


<a id="doc-393bc886e31e-零和空间重分配"></a>

#### 零和空间重分配

将受限九维坐标 \(u(\xi)=\xi/\sqrt{1+\|\xi\|^2/b_s^2}\) 映射为支路对数倍率 \(Su\)。完全图上的支路对数倍率之和恒为零。各边倍率的几何平均为 1，算术总和随状态变化。共同模型的各边倍率相等。两者使用同一边倍率界 \([0.5,2]\)、候选速率和因果历史。零和模型的储能梯度包含径向映射的雅可比，观测历史切线同时包含它的二阶项。[配对合成协议](../synthetic_spatial_coupling/README.md#doc-284356f266b1)

通过几何区分、已知真值恢复和全量合成门后，沿完整 TASK3 train 实测路径比较两个阻尼锚点上的零和修正。模型的记忆起报固定为各锚点的因果估计，同时计算自由记忆投影。路径门通过后冻结候选并评估 TASK3 的 train 346、val 107、test 76，全量报告轨迹误差、稳定性与覆盖率。

合成因果对照通过。完整 TASK3 train 在两个阻尼模型下均通过数值、全体和稳定组门，失稳组门失败。逐机功角阻尼下的自由记忆投影表明，保存的零和冲量方向与剩余失稳残差负相关。后续结构对照应检验能够改变空间力方向的耦合动力学，并用相同的固定因果记忆和自由投影核对。新候选走独立登记。[合成结果](../synthetic_spatial_coupling_causal/README.md#doc-1d7dcc7de0aa) · [真实 train 结果](../task3_spatial_coupling/README.md#doc-d5aa05bf148f)


<a id="doc-393bc886e31e-共同尺度的实验结论"></a>

#### 共同尺度的实验结论

固定、动态、静态恢复力三组合成真值均选中匹配的网格候选，完整预测通过门槛。真实 train 诊断中，两档因果记忆下的共同耦合使常值阻尼和逐机阻尼的失稳组残差平方和分别上升 6.95% 和 7.28%。自由记忆投影下分别上升 16.88% 和 17.26%。当前共同尺度停在路径诊断。后续独立检查零和空间分配，先用已知真值确认共同尺度与支路重分配能被区分。[合成对照](#doc-c90af5eec437) · [真实 train 诊断](../task3_dynamic_coupling/README.md#doc-efa81b847256)

## 执行协议

<a id="doc-931303070641"></a>

<a id="doc-931303070641-共同耦合动态的已知真值对照"></a>

### 共同耦合动态的已知真值对照

沿用封存的合成名单与初始条件，完整 train 24、val 12、test 12。固定逐机记忆、基础支路刚度、功率、时钟及逐机阻尼 \(\rho=2/3,\lambda=1\)。共同耦合用 \(\ell(c)=2b\tanh(c/b)\)，其中 \(b=\log(2)/2\)、\(\kappa=\sum_e k_e^0=5.4\)。以 \(\gamma=\mu\kappa\) 表示恢复速率。真值用独立 NumPy RK4 的 1.25 ms 和 0.625 ms 步长生成。[方程设计](#doc-393bc886e31e)

| 主真值 | \(\gamma\)（每秒） | 静态形状系数 \(\alpha\) | 历史第一帧 \(c\) |
|---|---:|---:|---:|
| 固定 | 0 | 0 | 0 |
| 动态耦合 | 1 | 0 | 0 |
| 静态恢复力 | 0 | 0.125 | 0 |

在每种真值的完整 train 上，分别选择动态系数 \(\gamma\in\{0,0.25,0.5,1,2\}\) 与静态系数 \(\alpha\in\{-0.25,-0.125,0,0.125,0.25\}\)。每族五个候选采用 CUDA float64、2.5 ms 步长及事件等权的 1 s 与 3 s 归一化 MSE 平均值。仅 train 可用于候选选择和尺度。损失精确并列时先取绝对值更小的系数，再取数值更小者。候选出现数值失败时保留该记录，整族均失败时评分零系数对照并记录选择失败。两个候选族各自选定后完整评分三个划分，采用 2.5 ms、1.25 ms 两档步长和 8 s 预测。

所有候选使用相同的 100 ms 因果历史估计器、预设记忆半径 1 和二次先验 \(10^{-6}\)。动态候选只估计原十维逐机记忆初态，耦合初态固定为零。起报时替换可见功角及速度，保留历史传播得到的记忆和耦合状态。另设第四组真值 \(c(-100\,\mathrm{ms})=0.05\)，用主动态组冻结的模型和尺度完成完整 48 例预测，单列零初态先验的效果。


<a id="doc-931303070641-通过标准"></a>

#### 通过标准

工程门检查 30 次 train 候选评分有限，主真值 train 和 val 双步长归一化差至多 \(10^{-6}\)，两族预测相应差至多 \(10^{-5}\)，质心和带输入功率恒等式误差至多 \(10^{-10}\)。动态模型两档步长均要求历史第一帧状态为零、历史末端耦合精确传入起报点。单元校验覆盖非零参考角、解析切线、刚度倍率与零系数退化。第四组单独报告数值资格。

固定真值两族均须选择零。动态真值须选择 \(\gamma=1\)，静态真值须选择 \(\alpha=0.125\)。两个非零机制下，匹配族的 train 最小损失至多为次优网格点的 95%。匹配族在完整 train 和 val 的双时域 MSE 各自至多为另一族的 80%，差值均须超过对应真值双步长 MSE 的 10 倍。

各主真值的 27 个半径内 train 和 val 事件须恢复标签与触发时刻，8 s 归一化 RMSE 至多 0.01。动态真值的耦合状态最大绝对误差至多 \(0.01b\)。两个非零机制下，匹配族的半径内 val 稳定样本 8 s 归一化 RMSE 至多为另一族的 90%。完整 train 和 val 的匹配失步数不得下降，误报不得增加。test 和非零耦合初态组用于报告。

单次登记覆盖 4 组双步长真值、6 次有限网格选择、30 个完整 train 候选批次及 7 组完整双步长预测。所有候选保留误差与覆盖，真实数据读取为零。主门通过后另行登记真实 train 路径诊断。主门失败保留终态和所有结果。

## 报告与结果

<a id="doc-c90af5eec437"></a>

<a id="doc-c90af5eec437-动态耦合合成验证"></a>

### 动态耦合合成验证

本轮合成验证比较动态有效耦合与静态恢复力形状修正。工程门和科学门均通过。48 个已知真值样本按 train 24、val 12、test 12 评估，三个主真值组均选中对应机制。


<a id="doc-c90af5eec437-主真值结果"></a>

#### 主真值结果

表中 8 s 归一化 RMSE 按 train、val、test 顺序列出，由各事件的归一化平方误差等权平均后开方。动态候选的估计量为速率，静态候选的估计量为恢复力形状系数。

| 真值机制 | train 选中速率 / 形状系数 | 动态候选 NRMSE | 静态候选 NRMSE |
|---|---|---|---|
| 零效应 | 0 / 0 | 0.001446 / 0.001352 / 0.001472 | 0.001446 / 0.001352 / 0.001472 |
| 动态耦合 | 1 / 0 | 0.001531 / 0.001421 / 0.001519 | 0.012981 / 0.022458 / 0.022839 |
| 静态恢复形状 | 0 / 0.125 | 0.009164 / 0.017835 / 0.014316 | 0.001472 / 0.001371 / 0.001485 |

动态真值中，train 选择速率 1。静态真值中，train 选择形状系数 0.125，动态候选回到零。固定真值的两个候选均选择零，预测结果一致。

三个主真值下，选中候选的稳定性准确率和覆盖率在 train、val、test 均为 100%。train 的 TP/TN/FP/FN 为 6/18/0/0，val 和 test 均为 3/9/0/0。标签按合成真值与预测轨迹分别判定，正类为失稳。


<a id="doc-c90af5eec437-非零耦合初态"></a>

#### 非零耦合初态

该组在历史第一帧的真值耦合状态为 0.05，模型在同一帧取零并传播至起报点。模型沿用主动态组的冻结速率 1 与尺度。本组用于报告零初态先验的影响。全量 48 例完成评分，8 s 归一化 RMSE 在 train、val、test 分别为 0.002606、0.002873、0.002394。三个划分的稳定性准确率和覆盖率均为 100%。


<a id="doc-c90af5eec437-评估范围"></a>

#### 评估范围

动态速率在冻结网格 `[0, 0.25, 0.5, 1, 2]` 中选择，静态形状系数在冻结网格 `[-0.25, -0.125, 0, 0.125, 0.25]` 中选择。动态耦合的限幅和恢复强度固定为 0.34657 和 5.4。train 用于选择，val 检查机制，test 用于报告。结果支持这组合成真值与有限网格下的机制区分。


<a id="doc-c90af5eec437-证据"></a>

#### 证据

结果回执见 [machine_local_topology_dynamic_coupling_20261116_evidence.json](_sealed/machine_local_topology_dynamic_coupling_20261116_evidence.json)。独立复核确认名单、训练网格选择、完整评分组、稳定性和能量检查均通过，且未读取真实数据，复核记录见 [machine_local_topology_dynamic_coupling_result_review_20261116.json](_sealed/machine_local_topology_dynamic_coupling_result_review_20261116.json)。

## 回执与附件

- [machine_local_topology_dynamic_coupling_20261116_evidence.json](_sealed/machine_local_topology_dynamic_coupling_20261116_evidence.json)
- [machine_local_topology_dynamic_coupling_result_review_20261116.json](_sealed/machine_local_topology_dynamic_coupling_result_review_20261116.json)
- [machine_local_topology_dynamic_coupling_v1_review_20261116.json](_sealed/machine_local_topology_dynamic_coupling_v1_review_20261116.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_dynamic_coupling_v1_zh.md](_sealed/machine_local_topology_dynamic_coupling_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_dynamic_coupling_20261116_zh.md](_sealed/machine_local_topology_dynamic_coupling_20261116_zh.md)
- [docs/research/machine_local_dynamic_coupling_plan_20261116_zh.md](_sealed/machine_local_dynamic_coupling_plan_20261116_zh.md)

</details>
