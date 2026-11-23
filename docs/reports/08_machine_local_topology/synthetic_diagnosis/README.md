# 冻结共享系数下的耦合与记忆起报点诊断

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/04_topology_diagnosis_learning.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_diagnosis_v1.json)

## 执行协议

<a id="doc-a05926e988e1"></a>

<a id="doc-a05926e988e1-逐机耦合与记忆模型起报点交叉诊断"></a>

### 逐机耦合与记忆模型起报点交叉诊断

对 S1 的 T1M1 联合模型，交叉替换共享系数、起报点耦合修正和记忆状态，分别取选中估计值与合成真值，共八组。对只有记忆的 T0M1 交叉替换共享系数和记忆起报点，共四组。两组系统均覆盖此前冻结的合成 train 6、val 3。每例的角度、转速和零阶保持机械功率保持相同。真值按独立 NumPy 方程，以 0.625 ms RK4 生成。

在每个系统上，先用选中系数及 100 ms 观测，以 1.25 ms 步长执行原五次有界初始化。所有替换共用同一份估计隐态，交叉时直接换起报点对应的状态块。采用固定归一化储能坐标来交换记忆状态。共享系数替换会同时改变可见动力学和记忆端口、衰减率，保留由此产生的交互作用。

从冻结起报点预测 8 秒，用 CUDA float64 和 2.5/1.25 ms 两档步长。报告逐事件及分 split 的 1、2、3、5、8 秒归一化 RMSE、资格覆盖和 TASK3 规则派生的稳定性判断。用 8 秒 NMSE 比较逐块替换与联合替换，保留替换顺序依赖。全部资格通过为工程完成。已有 S0 记录提供真系数的因果初始化对照，本轮全真值格使用真系数与真起报点。

一次执行，零拟合、零真实数据、零 test 轨迹、零历史数组解码。只读两个 S1 选中方程 JSON，绑定已归档的原运行清单与证据收据。两组初始化和十二组双步长预测允许批次失败后逐例隔离一次。上限为 103680 次 NumPy RHS、380 次 Torch 传播、4646400 次 Torch RHS、32000 次切线 RHS，QR、SVD、三角求解和相关乘法各 180 次。部署决策为 KEEP_BASELINE。

## 报告与结果

<a id="doc-c43e5c12eeea"></a>

<a id="doc-c43e5c12eeea-冻结共享系数下的耦合与记忆起报点诊断"></a>

### 冻结共享系数下的耦合与记忆起报点诊断

在完整合成 train 6、val 3 上，分别替换共享系数、起报点耦合修正和记忆状态。每格使用同一真实可见末点、机械功率和同一份因果估计。表内为 8 秒合并归一化 RMSE。

| 真值系统 | 共享系数 | 耦合修正 | 记忆起报点 | train | val |
|---|---|---|---|---:|---:|
| T0M1 | 拟合 | 固定 | 估计 | 0.001869 | 0.002509 |
| T0M1 | 拟合 | 固定 | 真值 | 0.001930 | 0.002291 |
| T0M1 | 真值 | 固定 | 估计 | 0.000664 | 0.000969 |
| T0M1 | 真值 | 固定 | 真值 | 2.70e-12 | 3.30e-12 |
| T1M1 | 拟合 | 估计 | 估计 | 0.014624 | 0.029479 |
| T1M1 | 拟合 | 估计 | 真值 | 0.015329 | 0.029427 |
| T1M1 | 拟合 | 真值 | 估计 | 0.013205 | 0.012290 |
| T1M1 | 拟合 | 真值 | 真值 | 0.013725 | 0.012800 |
| T1M1 | 真值 | 估计 | 估计 | 0.009730 | 0.027561 |
| T1M1 | 真值 | 估计 | 真值 | 0.009119 | 0.027010 |
| T1M1 | 真值 | 真值 | 估计 | 0.004336 | 0.003257 |
| T1M1 | 真值 | 真值 | 真值 | 2.71e-12 | 3.30e-12 |

在 T1M1 的 val 上，单独修正耦合隐态有最大的长时误差降幅。保留真实耦合修正后，余下的误差随共享系数和记忆状态组合而变。在 T0M1 的 val 上，换真共享系数的误差降幅大于单换真记忆初态。各次替换会改变方程与初态之间的配合，联合收益随替换组合而变。

十二组预测均通过双步长和完整覆盖检查。真值均为稳定样本，所有组的 train、val 准确率为 6/6、3/3，未决与数值失败为零。失稳召回率为 `null`。本轮诊断范围为合成 train 和 val。

原估计耦合误差进入联合模型的长时预测，单独把真值记忆塞回起报点的收益很小。后续使用 1 秒和 3 秒等权目标继续学习共享方程，并使用相同目标选择完整 val 检查点。[长时域精化协议](../synthetic_refinement/README.md#doc-e6a3caf6a434)

[诊断协议](#doc-a05926e988e1) · [证据收据](_sealed/machine_local_topology_diagnosis_20261115_evidence.json)

## 回执与附件

- [machine_local_topology_diagnosis_20261115_evidence.json](_sealed/machine_local_topology_diagnosis_20261115_evidence.json)
- [machine_local_topology_diagnosis_review_20261115.json](_sealed/machine_local_topology_diagnosis_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_diagnosis_v1_zh.md](_sealed/machine_local_topology_diagnosis_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_diagnosis_20261115_zh.md](_sealed/machine_local_topology_diagnosis_20261115_zh.md)

</details>
