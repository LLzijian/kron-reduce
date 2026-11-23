# 零和空间耦合合成验证

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_spatial_coupling_v1.json)

## 执行协议

<a id="doc-284356f266b1"></a>

<a id="doc-284356f266b1-零和空间耦合与共同尺度的配对合成验证"></a>

### 零和空间耦合与共同尺度的配对合成验证

固定原逐机记忆、基础支路刚度与逐机功角阻尼 \(\rho=2/3,\lambda=1\)。沿用合成名单以及原初始可见状态、记忆和功率。每种真值含 train 24、val 12、test 12。独立 NumPy 以 1.25 ms 和 0.625 ms 生成真值，采样间隔为 10 ms。[耦合动力学方案](../synthetic_dynamic_coupling/README.md#doc-393bc886e31e)

共同尺度使用 \(\ell_e=2b_c\tanh(c/b_c)\)，其中 \(b_c=\log 2/2\)。零和空间模型取 \(S=|B|^\top Q\)，\(Q^\top Q=I\)、\(Q^\top\mathbf1=0\)。令 \(u=\xi/\sqrt{1+\|\xi\|^2/b_s^2}\)、\(b_s=\log2/\max_e\|S_e\|\)，取 \(\ell=Su\)。两种模型以 \(e^{\ell_e}\) 乘原支路刚度，倍率均在 \([0.5,2]\) 内。零和模型保持各边倍率的几何平均为 1。两者均以储能梯度推进耦合坐标，\(\kappa=\sum k_e^0=5.4\)、\(\mu=\gamma/\kappa\)。

三组真值分别为固定边刚度、共同动态 \(\gamma=1\,\mathrm{s}^{-1}\)、零和动态 \(\gamma=1\,\mathrm{s}^{-1}\)。每组用完整 train 和同一训练尺度分别在 \(\{0,0.25,0.5,1,2\}\) 中选择两族速率。损失为事件等权的 1 s、3 s 归一化 MSE 平均值，CUDA float64 步长为 2.5 ms。每族五次候选评价，损失精确并列取较低速率。失败记录原样保留，整族选择失败时评分零速率对照。两族赢家和固定刚度模型均进行 8 s 全量三划分评分，CUDA 步长为 2.5 ms 和 1.25 ms。

所有候选使用 100 ms 因果历史，记忆半径 1、二次先验 \(10^{-6}\)。耦合坐标在历史第一帧设为零，只估计十维记忆初态。起报替换可见状态，保留传播所得的记忆和耦合状态。数值门要求 train 候选均有效，train、val 的真值和预测双步长归一化差分别至多 \(10^{-6}\) 和 \(10^{-5}\)，质心和带输入能量恒等式误差至多 \(10^{-10}\)。动态候选两档步长均要求历史第一帧耦合为零并精确传递起报耦合。共同模型在边间的对数倍率标准差和零和模型的对数倍率和须在 \(10^{-12}\) 内符合各自恒等式。倍率界使用 \(10^{-12}\) 舍入容差。

固定真值须使两族都选择零。非零真值须使匹配族选择 1，匹配族的 train 最优损失至多为次优网格点的 95%。与另一动态模型及固定模型相比，匹配族完整 train 与 val 的双时域损失均至多为对照的 80%，差值均须超过独立真值双步长 MSE 的十倍。各匹配模型在原 27 个半径内 train、val 事件上须恢复标签与触发时刻，允许触发差至多两帧，8 s 归一化 RMSE 至多 0.01。非零真值还要求这 27 例的耦合状态最大欧氏误差至多 \(0.01b\)、边对数倍率最大误差至多 0.01。两个非零真值下，匹配族在半径内 val 稳定样本上的 8 s 误差至多为每个对照的 90%，完整 train 与 val 的匹配失步数不降且误报不增。

两组非零真值在起报后三秒的 train、val 边对数倍率均方根至少为 \(10^{-3}\)。空间真值去除每个时刻沿基础恢复加速度方向的分量后，剩余新增加速度平方和至少占新增总平方和的 5%。这项指标直接衡量共同倍率能否形成同一力方向。test 只报告。

单次登记含三组双步长真值、六次网格选择、30 个完整 train 候选批次及九组完整双步长预测。真实数据读取为零。各门通过后另行登记完整 TASK3 train 路径诊断。

## 报告与结果

<a id="doc-68f44e7dfc76"></a>

<a id="doc-68f44e7dfc76-零和空间耦合合成验证"></a>

### 零和空间耦合合成验证

在三种独立已知真值上，比较固定边刚度、共同尺度变化和零和支路重分配。每种真值覆盖完整 train 24、val 12、test 12。模型使用 100 ms 因果历史估计逐机记忆，以 CUDA 双步长预测 8 s。归一化 RMSE 由各事件的归一化平方误差等权平均后开方。[冻结协议](#doc-284356f266b1)

| 真值 | 共同尺度胜出速率 | 零和空间胜出速率 | 8 s 归一化 RMSE，val（共同／空间／固定） | 8 s 归一化 RMSE，test（共同／空间／固定） |
|---|---:|---:|---|---|
| 固定刚度 | 0 | 0 | 0.001352／0.001352／0.001352 | 0.001472／0.001472／0.001472 |
| 共同尺度 | 1 | 2 | 0.001421／0.014248／0.022458 | 0.001519／0.013313／0.022839 |
| 支路重分配 | 0 | 1 | 0.014634／0.001355／0.014634 | 0.014988／0.001473／0.014988 |

九组预测在 val 和 test 的准确率及覆盖率均为 100%，每组均为 TP/TN/FP/FN＝3/9/0/0。空间真值的恢复加速度变化中，垂直于原恢复力方向的分量占新增平方和的 10.82%（train）和 9.66%（val）。该分量超出了共同尺度变化能产生的方向。

工程门通过，空间真值下三项科学门失败。匹配空间模型的 train 最优／次优候选损失比为 0.9922，门限为 0.95。匹配／固定模型的完整 train 双时域 MSE 比为 0.9155，门限为 0.8。在原半径内 val 稳定样本上，匹配／固定模型的 8 s 归一化 RMSE 比为 1.0222，门限为 0.9。匹配空间模型 92.37% 的 train 双时域平方误差来自六个记忆初态在球外的样本。独立对照将检验这部分历史恢复误差是否掩盖了模型区分能力。[独立结果审查](_sealed/machine_local_topology_spatial_coupling_result_review_20261116.json)

本轮授权已用尽，真实数据读取为零。完整 run ID、哈希、运行环境和评分见[结果回执](_sealed/machine_local_topology_spatial_coupling_20261116_evidence.json)。

## 回执与附件

- [machine_local_topology_spatial_coupling_20261116_evidence.json](_sealed/machine_local_topology_spatial_coupling_20261116_evidence.json)
- [machine_local_topology_spatial_coupling_result_review_20261116.json](_sealed/machine_local_topology_spatial_coupling_result_review_20261116.json)
- [machine_local_topology_spatial_coupling_v1_review_20261116.json](_sealed/machine_local_topology_spatial_coupling_v1_review_20261116.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_spatial_coupling_v1_zh.md](_sealed/machine_local_topology_spatial_coupling_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_spatial_coupling_20261116_zh.md](_sealed/machine_local_topology_spatial_coupling_20261116_zh.md)

</details>
