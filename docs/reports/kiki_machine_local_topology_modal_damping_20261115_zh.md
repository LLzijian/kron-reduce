# TASK3 train 固定模态阻尼对照

完整 train 的 346 例均通过数值检查。参考刚度的最大模态相对间隙为 0.19676，预设的秩一附加阻尼系数为 0.37140。附加阻尼提高了稳定组的动量残差，未通过预登记门槛。

固定逐机阻尼模型，使用相同的 100 ms 因果初态和实测速度。在定下的单一模态上增加耗散，计算起报后至多 3 s 的弱动量残差。表中是附加阻尼相对于逐机参照的残差平方和变化。[协议](../protocols/kiki_machine_local_topology_modal_damping_v1_zh.md)

| train 分组 | 样本数 | 2.5 ms 因果初态 | 1.25 ms 因果初态 | 自由初态投影 |
|---|---:|---:|---:|---:|
| 全部 | 346 | +0.0084% | +0.0084% | +1.03% |
| 稳定 | 311 | +1.00% | +1.00% | +1.36% |
| 失稳 | 35 | −0.67% | −0.67% | +0.24% |

三个对照的稳定组均未满足残差下降至少 1% 的要求。自由初态的全 train 残差增幅也超过 1%。保留原逐机方程。本轮读取范围为 train，拟合和闭环未来传播次数均为零。

运行 `kiki_machine_local_topology_modal_damping_v1-train_momentum-20261115T124237995489Z-dfc29c8a-a2df2391`。独立复核覆盖完整名单、模式资格、残差比及算子预算。[回执](kiki_machine_local_topology_modal_damping_20261115_evidence.json) · [独立复核](../research/kiki_machine_local_topology_modal_damping_result_review_20261115.json)
