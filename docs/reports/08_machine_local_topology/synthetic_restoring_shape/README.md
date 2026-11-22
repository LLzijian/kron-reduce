# 合成恢复力形状识别与闭环预测

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_restoring_shape_v1.json)

## 执行协议

<a id="doc-029c3a1cfb71"></a>

<a id="doc-029c3a1cfb71-保持近平衡刚度的恢复力合成对照"></a>

### 保持近平衡刚度的恢复力合成对照

支路恢复力使用研究计划中的切线保持项。其系数为 \(\alpha\in[-1/4,1/4]\)，势能由修正后的恢复力积分得到。使用非零平衡角单元测试核对平衡点、解析切线和功率恒等式。零系数恢复原方程与因果预测。[方程与候选范围](../synthetic_momentum/README.md#doc-f2ac6b8b3491-%E4%BF%9D%E6%8C%81%E8%BF%91%E5%B9%B3%E8%A1%A1%E5%88%9A%E5%BA%A6%E7%9A%84%E6%81%A2%E5%A4%8D%E5%8A%9B%E5%AF%B9%E7%85%A7)

同一组 48 例分别按 \(\alpha=-1/8,0,1/8\) 重算真值，保留 train 24、val 12、test 12。同一逐机功角阻尼 \(\rho=2/3\)、\(\lambda=1\) 和记忆用于三组方程。真值采用 NumPy RK4，步长为 1.25 ms 和 0.625 ms。每组只用 train 分别拟合形状和全局刚度缩放。全局刚度乘数 \(1+\beta\) 的下界为 \(10^{-6}\)。两次标量拟合使用同一实测路径、动量权重和逐例隐初态投影。求积使用分段线性功角的解析正弦积分，阻尼积分用四点 Gauss 求积并与八点比较。

固定形状参数和 train 尺度后，零修正及拟合形状分别用自身方程完成 100 ms 历史初始化与八秒闭环预测。半径为 1，二次先验为 \(10^{-6}\)，预测采用 CUDA float64，步长为 2.5 ms 和 1.25 ms。两组均评估完整三个划分。全局刚度缩放用于弱动量对照。

原始及截断形状系数距真值均须至多 0.001。零真值下原始及截断刚度乘数距 1 均须至多 0.001。两个非零真值下，train、val 的形状拟合残差须至多为零修正及全局刚度拟合各自的 5%，两对照均须超过真值系数残差的 100 倍。完整 train 与 val 校核双步长资格、秩 10、功率与求积误差至多 \(10^{-10}\)。半径内 train、val 须恢复真值失步标签和触发时刻，八秒归一化 RMSE 至多 0.01。两种非零真值下，半径内 val 稳定组的八秒归一化 RMSE 须至多为零修正组的 90%，全量 train、val 的匹配失步数不得下降，误报不得增加。test 只报告。

单次执行，零真实数据读取，六次标量拟合。使用登记的算子预算为三组双步长真值和六组双步长因果预测存证。Torch 阻尼平方根构造随历史和预测闭包核算。通过后独立登记真实 train 拟合。

## 报告与结果

<a id="doc-c576ecd78c23"></a>

<a id="doc-c576ecd78c23-合成恢复力形状识别与闭环预测"></a>

### 合成恢复力形状识别与闭环预测

同一批已知真值事件的三组对照同时通过工程和科学门。正、负形状系数的恢复误差均低于 \(4.4\times10^{-6}\)，形状候选在两种非零真值下改善了八秒闭环轨迹。

每种真值均使用完整 train 24、val 12、test 12。八秒归一化 RMSE 由各事件的归一化平方误差等权平均后开方。

| 真值 α | 划分 | 基线 NRMSE | 形状候选 NRMSE |
|---:|---|---:|---:|
| −0.125 | train | 0.009379 | 0.001424 |
| −0.125 | val | 0.020078 | 0.001335 |
| −0.125 | test | 0.016280 | 0.001459 |
| 0 | train | 0.001446 | 0.001446 |
| 0 | val | 0.001352 | 0.001352 |
| 0 | test | 0.001472 | 0.001472 |
| +0.125 | train | 0.009164 | 0.001472 |
| +0.125 | val | 0.017835 | 0.001371 |
| +0.125 | test | 0.014316 | 0.001485 |

在真值 α=−0.125 和 +0.125 下，train 拟合分别恢复 −0.1249957 和 +0.1250043。半径内 val 稳定组的 8 秒 NRMSE 比值（候选/基线）分别为 0.5761 和 0.5851，对应降低 42.4% 和 41.5%。零形状真值下拟合 α 为 0.00000429，候选与基线的 8 秒误差几乎相同。

六组预测的稳定性准确率均为 100%。每个真值的 train 均为 TP/TN/FP/FN = 6/18/0/0，val 和 test 均为 3/9/0/0。各组均无未决、未覆盖或数值失败。test 用于报告。

[执行回执](_sealed/machine_local_topology_restoring_shape_20261115_evidence.json) · [独立复核](_sealed/machine_local_topology_restoring_shape_result_review_20261115.json)

## 回执与附件

- [machine_local_topology_restoring_shape_20261115_evidence.json](_sealed/machine_local_topology_restoring_shape_20261115_evidence.json)
- [machine_local_topology_restoring_shape_result_review_20261115.json](_sealed/machine_local_topology_restoring_shape_result_review_20261115.json)
- [machine_local_topology_restoring_shape_v1_review_20261115.json](_sealed/machine_local_topology_restoring_shape_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_restoring_shape_v1_zh.md](_sealed/machine_local_topology_restoring_shape_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_restoring_shape_20261115_zh.md](_sealed/machine_local_topology_restoring_shape_20261115_zh.md)

</details>
