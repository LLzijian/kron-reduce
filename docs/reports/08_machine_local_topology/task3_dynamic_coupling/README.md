# TASK3 完整 train 共同动态耦合诊断

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/kiki_machine_local_topology_dynamic_coupling_v1.json)

## 执行协议

<a id="doc-4e4a40a93520"></a>

<a id="doc-4e4a40a93520-task3-完整-train-动态耦合路径诊断"></a>

### TASK3 完整 train 动态耦合路径诊断

检验共同耦合的修正力是否解释真实失稳样本的动量残差。使用原 T0M1 第 12 步模型及同参数的逐机功角阻尼模型作为两项对照。固定合成通过的恢复速率 \(\gamma=1\,\mathrm{s}^{-1}\)、倍率界 \(b=\log(2)/2\)，以实际冻结模型的基础边刚度之和作为 \(\kappa\)，令 \(\mu=\gamma/\kappa\)。[合成协议](../synthetic_dynamic_coupling/README.md#doc-931303070641)

仅使用完整 train 346，含 311 例稳定和 35 例失稳。每例将 \(c\) 在历史第一帧设为零，在分段线性实测功角上连续传播。用相同 RK4 阶段计算相对固定刚度的投影力冲量，步长取 5 ms 和 2.5 ms。持有同一 \(c\) 轨迹分别检查两个阻尼对照。保留起报 \(c\)、各边共同倍率范围和逐例残差。

两种阻尼对照各自从 100 ms 观察估计记忆起报状态，使用 CUDA float64 的 2.5 ms、1.25 ms 两档步长。固定同一个因果记忆起报状态比较该阻尼下的原方程和动力学修正。另设自由记忆投影作为实测路径诊断。动力学、初态、尺度和判据的参数均冻结。

评价只使用起报后最多 3 s 的实际可用动量区间。弱方程按质量、步长和区间数平方根归一。工程门要求 346 例齐全、每例秩 10、阻尼四点与八点求积差至多 \(10^{-10}\)，耦合坐标和归一化冲量的双步长差各至多 \(10^{-6}\)。两个阻尼对照下的两档因果起报与自由记忆投影都须通过以下残差平方和门槛。

| train 范围 | 修正后与原方程之比上限 |
|---|---:|
| 全部 346 例 | 1.01 |
| 稳定 311 例 | 1.01 |
| 失稳 35 例 | 0.99 |

单次执行仅加载两项模型资产，打开两个授权缓存。两套阻尼各在两档步长完成历史起报。观测路径推进 5 ms、2.5 ms 两档并记录算子次数。val 和 test 轨迹读取、模型拟合及起报后的闭环预测数均为零。通过后再独立登记完整 TASK3 四种模型的闭环比较。实测路径残差与因果闭环精度分别报告。

## 报告与结果

<a id="doc-efa81b847256"></a>

<a id="doc-efa81b847256-task3-完整-train-共同动态耦合诊断"></a>

### TASK3 完整 train 共同动态耦合诊断

固定共同动态耦合在两个阻尼对照中均降低了稳定组动量残差，同时提高了失稳组残差。完整 train 346 例通过工程门，全部样本与失稳样本的科学门未通过。按执行协议，本轮停在路径诊断。


<a id="doc-efa81b847256-比较条件"></a>

#### 比较条件

使用原 T0M1 和同参数的逐机功角阻尼。耦合恢复速率为 \(1\,\mathrm{s}^{-1}\)，共用实测功角驱动的 \(c\) 路径。各阻尼模型用自己的 100 ms 历史方程估计因果记忆起报，再配对比较加修正前后的动量残差。自由记忆投影使用同一列空间。保留 98,341 个实测区间，起报后参与评分的区间为 94,881 个。[执行协议](#doc-4e4a40a93520) · [合成结果](../synthetic_dynamic_coupling/README.md#doc-c90af5eec437)


<a id="doc-efa81b847256-起报后最多-3-s-的弱动量残差"></a>

#### 起报后最多 3 s 的弱动量残差

表中为加共同动态耦合后相对各自阻尼模型的残差平方和变化。区间按质量、步长和每例实际区间数归一。负号表示残差下降。

| train 分组 | 阻尼对照 | 2.5 ms 因果记忆 | 1.25 ms 因果记忆 | 自由记忆投影 |
|---|---|---:|---:|---:|
| 全部 346 例 | 常值阻尼 | +3.95% | +3.95% | +7.20% |
| 全部 346 例 | 逐机阻尼 | +1.98% | +1.98% | +3.49% |
| 稳定 311 例 | 常值阻尼 | −4.51% | −4.51% | −0.35% |
| 稳定 311 例 | 逐机阻尼 | −5.65% | −5.65% | −2.32% |
| 失稳 35 例 | 常值阻尼 | +6.95% | +6.95% | +16.88% |
| 失稳 35 例 | 逐机阻尼 | +7.28% | +7.28% | +17.26% |

稳定组的残差下降集中在一部分事件。两档因果记忆条件下，常值阻尼有 76/311 例改善，逐机阻尼有 80/311 例改善。失稳组对应为 3/35 和 4/35。自由记忆投影后，失稳组总残差上升的方向一致。结论适用于本轮冻结的共同尺度、恢复速率与两个原模型。

346 例弱动量算子均为秩 10。阻尼求积最大差为 \(3.91\times10^{-14}\)，耦合坐标和归一化力冲量的双步长最大差分别为 \(1.91\times10^{-9}\) 和 \(3.14\times10^{-8}\)。[执行回执](_sealed/kiki_machine_local_topology_dynamic_coupling_20261116_evidence.json) · [独立复核](_sealed/kiki_machine_local_topology_dynamic_coupling_result_review_20261116.json)

## 回执与附件

- [kiki_machine_local_topology_dynamic_coupling_20261116_evidence.json](_sealed/kiki_machine_local_topology_dynamic_coupling_20261116_evidence.json)
- [kiki_machine_local_topology_dynamic_coupling_result_review_20261116.json](_sealed/kiki_machine_local_topology_dynamic_coupling_result_review_20261116.json)
- [kiki_machine_local_topology_dynamic_coupling_v1_review_20261116.json](_sealed/kiki_machine_local_topology_dynamic_coupling_v1_review_20261116.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_dynamic_coupling_v1_zh.md](_sealed/kiki_machine_local_topology_dynamic_coupling_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_dynamic_coupling_20261116_zh.md](_sealed/kiki_machine_local_topology_dynamic_coupling_20261116_zh.md)

</details>
