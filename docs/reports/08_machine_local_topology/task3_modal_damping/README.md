# TASK3 train 固定模态阻尼对照

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/kiki_machine_local_topology_modal_damping_v1.json)

## 执行协议

<a id="doc-66253afa19fe"></a>

<a id="doc-66253afa19fe-task3-train-参考刚度模态的固定阻尼对照"></a>

### TASK3 train 参考刚度模态的固定阻尼对照

以逐机阻尼模型为参照。全局与逐机方案的 val 稳定性计数相同，逐机方案的预登记六项分数为 0.42674，全局方案为 0.42819。固定逐机混合系数 1、阻尼强度 \(2/3\)、尺度 1 rad、原记忆半径和先验。[完整评估](../task3_spatial_damping/README.md#doc-08e5190828ec)

从冻结参考刚度 \(K=B\operatorname{diag}(k)B^\top\) 构造 \(M^{-1/2}KM^{-1/2}\)。令 \(w\) 为最大特征值对应向量，取 \(b=M^{1/2}w/\lVert M^{1/2}w\rVert_2\)。要求九个正参考模态、最大特征值相对间隙至少 \(10^{-8}\)，且 \(|\mathbf 1^\top b|\le10^{-10}\)。比较附加阻尼 \(\Delta D=\eta bb^\top\)，其中 \(\eta=\tfrac12 b^\top D_0 b>0\)。这一项增加耗散 \(\eta(b^\top v)^2\)，与刚度、记忆及储能方程共存。

在完整 train 346 上，由原逐机模型的 100 ms 历史各解一次 2.5 ms 和 1.25 ms 初态。每档起点隐态在有无附加阻尼的两种力平衡中共同使用。沿相同的实测速度传播记忆，仅评分起报后最多 3 s 的实测区间。附加力对分段线性速度的积分用梯形公式精确计算，原逐机阻尼的四点求积做八点复核。第三个对照消去自由记忆初态。保存每例残差、交叉项和方向平方和。

三个对照均要求 346 例完整有效、秩 10、求积差至多 \(10^{-10}\)。稳定 311 例的残差平方和须下降至少 1%，全 train 和失稳 35 例的增幅分别至多 1%。这项诊断执行一次，构造阻尼平方根和参考刚度谱各一次，完整预算见冻结方案。系数拟合、未来闭环预测和 val/test 轨迹读取次数均为零。通过后另行登记该单一模态的合成验证与完整预测。

## 报告与结果

<a id="doc-171beda6b4d6"></a>

<a id="doc-171beda6b4d6-task3-train-固定模态阻尼对照"></a>

### TASK3 train 固定模态阻尼对照

完整 train 的 346 例均通过数值检查。参考刚度的最大模态相对间隙为 0.19676，预设的秩一附加阻尼系数为 0.37140。附加阻尼提高了稳定组的动量残差，未通过预登记门槛。

固定逐机阻尼模型，使用相同的 100 ms 因果初态和实测速度。在定下的单一模态上增加耗散，计算起报后至多 3 s 的弱动量残差。表中是附加阻尼相对于逐机参照的残差平方和变化。[协议](#doc-66253afa19fe)

| train 分组 | 样本数 | 2.5 ms 因果初态 | 1.25 ms 因果初态 | 自由初态投影 |
|---|---:|---:|---:|---:|
| 全部 | 346 | +0.0084% | +0.0084% | +1.03% |
| 稳定 | 311 | +1.00% | +1.00% | +1.36% |
| 失稳 | 35 | −0.67% | −0.67% | +0.24% |

三个对照的稳定组均未满足残差下降至少 1% 的要求。自由初态的全 train 残差增幅也超过 1%。保留原逐机方程。本轮读取范围为 train，拟合和闭环未来传播次数均为零。

运行 `kiki_machine_local_topology_modal_damping_v1-train_momentum-20261115T124237995489Z-dfc29c8a-a2df2391`。独立复核覆盖完整名单、模式资格、残差比及算子预算。[回执](_sealed/kiki_machine_local_topology_modal_damping_20261115_evidence.json) · [独立复核](_sealed/kiki_machine_local_topology_modal_damping_result_review_20261115.json)

## 回执与附件

- [kiki_machine_local_topology_modal_damping_20261115_evidence.json](_sealed/kiki_machine_local_topology_modal_damping_20261115_evidence.json)
- [kiki_machine_local_topology_modal_damping_result_review_20261115.json](_sealed/kiki_machine_local_topology_modal_damping_result_review_20261115.json)
- [kiki_machine_local_topology_modal_damping_v1_review_20261115.json](_sealed/kiki_machine_local_topology_modal_damping_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_modal_damping_v1_zh.md](_sealed/kiki_machine_local_topology_modal_damping_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/kiki_machine_local_topology_modal_damping_v1_zh.md](_sealed/original/kiki_machine_local_topology_modal_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_modal_damping_20261115_zh.md](_sealed/kiki_machine_local_topology_modal_damping_20261115_zh.md)

</details>
