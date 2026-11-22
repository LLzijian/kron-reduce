# TASK3 train 零和空间耦合路径诊断

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/kiki_machine_local_topology_spatial_coupling_v1.json)

## 执行协议

<a id="doc-dfa9af650e98"></a>

<a id="doc-dfa9af650e98-task3-完整-train-零和空间耦合路径诊断"></a>

### TASK3 完整 train 零和空间耦合路径诊断

检验零和支路重分配的修正力是否解释真实失稳样本的动量残差。原 T0M1 第 12 步及同参数的逐机功角阻尼分别作为对照。固定合成通过的恢复速率 \(\gamma=1\,\mathrm{s}^{-1}\)、边倍率界 \([0.5,2]\)，取 \(b_s=\log2/\sqrt{1.6}\)、\(\kappa=\sum k_e^0\)、\(\mu=\gamma/\kappa\)。[配对合成协议](../synthetic_spatial_coupling_causal/README.md#doc-9d2bf0d08608)

完整 train 包含 346 例，稳定和失稳分别为 311 和 35 例。每例从历史第一帧的零耦合状态出发，用分段线性实测功角连续推进九维 \(\xi\)。以相同 RK4 阶段累计相对固定刚度的质心投影力冲量，采用 5 ms 和 2.5 ms 两档步长。两个阻尼对照共用这一条 \(\xi\) 路径。逐例保存起报坐标、倍率范围、边对数倍率和与残差。

两个阻尼模型分别以 CUDA float64 的 2.5 ms、1.25 ms 步长从 100 ms 观测估计记忆起报状态。在每个对照下固定该起报状态，配对比较原方程与耦合修正。自由记忆初态投影提供第三组实测路径诊断。评分只使用起报后最多 3 s 实际可用区间，按质量、步长和区间数平方根归一。保存耦合冲量的平方范数，区分修正方向与原残差的对齐程度和修正幅度，不据此选择新速率。

工程门要求 346 例齐全、每例秩 10。阻尼四点与八点求积差至多 \(10^{-10}\)，耦合坐标最大欧氏差和归一化冲量差各至多 \(10^{-6}\)。两档积分的边对数倍率和至多偏离零 \(10^{-12}\)，倍率界使用同一舍入容差。两阻尼对照的两档因果起报和自由投影均须满足下表。

| train 范围 | 修正后残差平方和与对照之比上限 |
|---|---:|
| 全部 346 例 | 1.01 |
| 稳定 311 例 | 1.01 |
| 失稳 35 例 | 0.99 |

单次执行只载入两项模型资产并打开两个授权缓存。候选参数冻结，拟合次数为零。真实访问限于完整 train，闭环积分限于因果历史估计。门通过后再登记完整 TASK3 闭环预测；各划分分别报告轨迹误差、稳定性和覆盖率。

## 报告与结果

<a id="doc-d5aa05bf148f"></a>

<a id="doc-d5aa05bf148f-task3-train-零和空间耦合路径诊断"></a>

### TASK3 train 零和空间耦合路径诊断

沿完整 TASK3 train 的实测轨迹检查固定速率 1 的零和支路修正。全部 346 例通过数值门。全体和稳定组的动量残差平方和均降低，失稳组未通过门禁。[冻结协议](#doc-dfa9af650e98)

| TASK3 train 范围 | 常值阻尼，因果记忆 | 常值阻尼，自由投影 | 逐机功角阻尼，因果记忆 | 逐机功角阻尼，自由投影 |
|---|---:|---:|---:|---:|
| 全部 346 例 | −1.74% | −1.37% | −1.76% | −1.31% |
| 稳定 311 例 | −2.85% | −2.02% | −3.19% | −2.54% |
| 失稳 35 例 | −1.35% | −0.52% | −0.76% | +1.59% |

表中为修正相对各自固定刚度对照的残差平方和变化，使用起报后最多 3 s 的实际观测。两档因果记忆步长 2.5 ms 与 1.25 ms 的变化率在表中精度下一致。失稳组门禁要求六项都至少下降 1%，实测只有常值阻尼的两档因果记忆达标。

记原残差为 \(r\)，零和修正在相同观测路径上的带符号冲量为 \(d\)。将已保存的冲量乘上非负倍数 \(a\) 时，残差平方和变化为 \(a^2\|d\|^2-2a\langle r,d\rangle\)。在逐机功角阻尼的 35 个失稳样本中，因果记忆下沿这条方向最多只能改善 0.87%。自由记忆投影后，\(\langle r,d\rangle=-12.37\)，任何正倍数都会增加残差。这是已保存冲量方向的诊断。改变耦合速率会改变时间路径，仍需独立验证。[独立审查](_sealed/kiki_machine_local_topology_spatial_coupling_result_review_20261116.json)

两档路径积分的最大归一化冲量差为 7.82×10⁻⁹，最大坐标差为 1.04×10⁻¹⁰。边对数倍率和的最大绝对值为 2.50×10⁻¹⁶，全部倍率在 0.85515 至 1.37874 内。缓存访问仅覆盖两项登记资产和完整 train。由于失稳组门禁未通过，本轮停在路径诊断，未产生该候选的真实 val、test 轨迹误差和稳定性准确率。[结果回执](_sealed/kiki_machine_local_topology_spatial_coupling_20261116_evidence.json)

## 回执与附件

- [kiki_machine_local_topology_spatial_coupling_20261116_evidence.json](_sealed/kiki_machine_local_topology_spatial_coupling_20261116_evidence.json)
- [kiki_machine_local_topology_spatial_coupling_result_review_20261116.json](_sealed/kiki_machine_local_topology_spatial_coupling_result_review_20261116.json)
- [kiki_machine_local_topology_spatial_coupling_v1_review_20261116.json](_sealed/kiki_machine_local_topology_spatial_coupling_v1_review_20261116.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_spatial_coupling_v1_zh.md](_sealed/kiki_machine_local_topology_spatial_coupling_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_spatial_coupling_20261116_zh.md](_sealed/kiki_machine_local_topology_spatial_coupling_20261116_zh.md)

</details>
