# 逐机功角阻尼的合成配对验证

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_spatial_damping_v1.json)

## 执行协议

<a id="doc-05f14349c042"></a>

<a id="doc-05f14349c042-逐机功角阻尼的合成配对验证"></a>

### 逐机功角阻尼的合成配对验证

固定阻尼强度 \(\rho=2/3\) 和尺度 1 rad，分别以全局分配 \(\lambda=0\) 和逐机分配 \(\lambda=1\) 为真值。两个真值从同一组封存的 48 例初始条件重算，train 24、val 12、test 12。刚度、逐机记忆、注入功率和时钟固定。独立 NumPy 真值使用 1.25 ms、0.625 ms 两档步长。[方程设计](../synthetic_momentum/README.md#doc-f2ac6b8b3491-%E4%B8%8B%E4%B8%80%E9%A1%B9%E7%A9%BA%E9%97%B4%E5%AF%B9%E7%85%A7)

完整 train 通过弱动量方程拟合单一 \(\lambda\)。用同一投影消去逐例的自由记忆初态，四点 Gauss 积分由八点核查。除原全局阻尼外，另比较每个状态下阻尼矩阵迹与逐机模型相等的标量阻尼。后者只参与动量残差诊断。训练后冻结系数与尺度，以半径 1、二次先验 \(10^{-6}\) 和 100 ms 历史初始化；全局对照和拟合候选均用 CUDA float64 的 2.5 ms、1.25 ms 步长，完整预测 8 s。

工程门包括零混合精确退化、耗散与切线核查、train 和 val 的双步长资格、秩 10，功率和求积误差至多 \(10^{-10}\)。两个真值下原始与截断后的系数误差均至多 0.001。逐机真值的 train 和 val 上，拟合残差须至多为两种全局对照各自的 5%，且两对照的残差各自须高于真值参考残差的 100 倍。半径内 train、val 恢复失稳标签与触发时刻，8 s 归一化 RMSE 至多 0.01。逐机真值的半径内 val 稳定组 8 s 误差至多为原全局对照的 90%，train 和 val 的匹配失步数不得下降，误报不得增加。所有病例保留在完整报告中，test 只报告。

单次合成配对，真实数据读取为零。预算包括 1 次封存真值数组解码、2 次标量拟合、3 次 NumPy 阻尼平方根构造和双步长完整评估。模型右端闭包内的 Torch 平方根构造随历史与预测传播调用计入预算说明。通过后，真实拟合与完整 TASK3 预测分开登记。

## 报告与结果

<a id="doc-8c00922d6f72"></a>

<a id="doc-8c00922d6f72-逐机功角阻尼的合成配对验证"></a>

### 逐机功角阻尼的合成配对验证

完整合成 train 恢复了两种空间分配。全局真值 \(\lambda=0\) 的拟合值为 0.00015961。逐机真值 \(\lambda=1\) 的原始估计为 1.00014392，按预登记范围截为 1。工程门与科学门均通过。

固定强度 \(\rho=2/3\) 和尺度 1 rad，从同一组 48 例初态生成两组真值。两组各使用 train 24、val 12、test 12。用完整 train 的实测路径拟合，冻结后做 100 ms 因果初始化和 8 s 预测。逐机分配保持原储能和耗散上下界。[协议](#doc-05f14349c042)

逐机真值的动量残差在三种对照下如下。迹匹配的标量对照在每个状态使用与逐机模型相同的阻尼矩阵迹。

| 划分 | 样本 | 原全局阻尼 SSE | 迹匹配 SSE | 逐机阻尼 SSE |
|---|---:|---:|---:|---:|
| train | 24 | 0.1449 | 0.2673 | 3.396e-8 |
| val | 12 | 0.08098 | 0.1529 | 2.044e-8 |
| test | 12 | 0.1429 | 0.2654 | 3.357e-8 |

逐机真值下，合并 NRMSE 是逐例归一化 MSE 的均值开方。完整 val 的 8 s 误差从全局模型的 0.040919 降至逐机模型的 0.001352。半径内的 6 个 val 稳定样本中，8 s 误差降至全局模型的 88.10%。

| 划分 | 模型 | 1 s | 2 s | 5 s | 8 s |
|---|---|---:|---:|---:|---:|
| train | 全局 | 0.010308 | 0.012292 | 0.016866 | 0.021566 |
| train | 逐机 | 0.001930 | 0.001959 | 0.001772 | 0.001446 |
| val | 全局 | 0.016600 | 0.020265 | 0.029342 | 0.040919 |
| val | 逐机 | 0.001806 | 0.001836 | 0.001657 | 0.001352 |
| test | 全局 | 0.019830 | 0.024322 | 0.035927 | 0.050677 |
| test | 逐机 | 0.001991 | 0.002000 | 0.001802 | 0.001472 |

两种真值、两种候选的稳定性计数相同。train 的 TP/TN/FP/FN 为 6/18/0/0，val 与 test 均为 3/9/0/0，完整准确率和覆盖率均为 100%。两个逐机真值候选都复现了 train 6/6 与 val 3/3 个匹配失步窗口。

运行 `machine_local_topology_spatial_damping_v1-spatial_damping-20261115T121431844839Z-ba8eebb1-2d083c3f`。独立复核完成四组全名单、系数、残差和门槛。真实数据读取为零。[回执](_sealed/machine_local_topology_spatial_damping_20261115_evidence.json) · [独立复核](_sealed/machine_local_topology_spatial_damping_result_review_20261115.json)

## 回执与附件

- [machine_local_topology_spatial_damping_20261115_evidence.json](_sealed/machine_local_topology_spatial_damping_20261115_evidence.json)
- [machine_local_topology_spatial_damping_result_review_20261115.json](_sealed/machine_local_topology_spatial_damping_result_review_20261115.json)
- [machine_local_topology_spatial_damping_v1_review_20261115.json](_sealed/machine_local_topology_spatial_damping_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_spatial_damping_v1_zh.md](_sealed/machine_local_topology_spatial_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_spatial_damping_20261115_zh.md](_sealed/machine_local_topology_spatial_damping_20261115_zh.md)

</details>
