# 耦合与记忆模型的长时域精化

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/04_topology_diagnosis_learning.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_refinement_v1.json)

## 执行协议

<a id="doc-e6a3caf6a434"></a>

<a id="doc-e6a3caf6a434-耦合与记忆模型长时域共同精化"></a>

### 耦合与记忆模型长时域共同精化

起报点交叉诊断发现，T1M1 的 val 8 秒误差随耦合修正换回真值明显下降。共享系数直接换回真值而保留原估计隐态时，误差仍接近原值。采用更长的闭环目标继续调整共享系数，让模型在较长区间同时预测持续的耦合力和衰减的记忆力。

重新生成相同的两个合成系统 T0M1 和 T1M1，保留完整 train 6、val 3、test 3 的已登记 ID。初值绑定 S1 的选中方程，T0M1 已完成 240 步且选中第 240 步，T1M1 已完成 170 步且选中第 110 步。保持两类节点、记忆状态的 100 ms 因果估计、RK4 同阶段切线和五次有界 Gauss–Newton。开放共享刚度、阻尼尺度，以及各机端口和衰减率。Adam 新建状态，学习率 0.003，范数裁剪为 1。两组各一次拟合，最多新增 180 步。

训练和完整 val 均使用 1 秒与 3 秒归一化 MSE 各一半，5 ms RK4 用同一次 3 秒自由积分获取两项。采用 S1 的 train 尺度与全 train 梯度方向核验，update 0 纳入选择，之后每 10 步检查完整 val，额外检查最后有效状态。完成至少 80 步后，连续 5 个有效 val 检查未超过预设容差则停止。容差沿用相对 0.1% 与绝对 (10^{-12}) 中较大者。选择最早严格最小的检查点，保留实际新增和选中步数。

初始与选中模型都冻结后，完成该系统的完整 train、val、test 双步长 GPU 评分，即使拟合或科学门失败。评价和接受门沿用 S1。报告 1、2、3、5、8 秒轨迹误差、稳定性准确率和覆盖、逐机参数与记忆核误差、有效边刚度及瞬时能量残差。test 用于报告。该方案同时改变训练窗和优化起点及步长，以同一冻结起点检验整套精化的净改进。

单次执行，零真实数据、零既有轨迹加载、零数组检查点解码。两次拟合、360 次新增更新，38 次完整 val 检查，含方向核验和能量诊断共 364 次自动微分。NumPy RHS 上限 155520。Torch 上限包括 3752 次传播、3390722 次 RHS、297600 次历史切线 RHS 和各 13350 次 QR、SVD、三角求解与相关乘法。预测预算含失败后的逐例隔离。部署决策为 KEEP_BASELINE。

## 报告与结果

<a id="doc-096ac12ed14d"></a>

<a id="doc-096ac12ed14d-耦合与记忆模型的长时域精化"></a>

### 耦合与记忆模型的长时域精化

[交叉诊断](../synthetic_diagnosis/README.md#doc-c43e5c12eeea)显示，联合模型在合成 val 上的误差对耦合修正起报值更敏感。本轮从冻结的 S1 系数出发，用 1 秒和 3 秒闭环损失共同调整共享系数，保持 100 ms 观测、因果初始化及方程结构。仅含逐机记忆的 T0M1 和同时含耦合修正与逐机记忆的 T1M1 均通过本轮门槛。T1M1 的 val 最大有效边刚度相对误差从 34.68% 降至 0.158%。

| 模型 | 新增／选中步 | 选中点 val 归一化 MSE | 记忆核相对误差（初始 → 选中） | 端口／时间常数最大相对误差 | val 有效边刚度最大相对误差 |
|---|---:|---:|---:|---:|---:|
| T0M1 | 180／180 | 2.3325e-08 | 6.901% → 3.882% | 8.038%／0.4706% | 0.03727% |
| T1M1 | 180／160 | 1.9737e-09 | 16.81% → 4.311% | 9.538%／0.1101% | 0.1576% |

两组均达到新增 180 步的上限，以完整 val 的 1 秒和 3 秒归一化 MSE 等权平均选择检查点。精化前的检查点为 S1 中 T0M1 的第 240 步和 T1M1 的第 110 步。

| 模型 | 数据划分 | 3 秒合并归一化 RMSE（初始 → 选中） | 8 秒合并归一化 RMSE（初始 → 选中） | 选中模型稳定性准确率 | 可判定／名单 | TP/TN/FP/FN | 未决／未覆盖／数值失败 |
|---|---|---:|---:|---:|---:|---|---|
| T0M1 | train | 0.0013854 → 0.00014038 | 0.0018687 → 0.0001743 | 100% | 6/6 | 0/6/0/0 | 0/0/0 |
| T0M1 | val | 0.0021433 → 0.00018994 | 0.0025089 → 0.00021372 | 100% | 3/3 | 0/3/0/0 | 0/0/0 |
| T0M1 | test | 0.0012687 → 0.00010614 | 0.0014288 → 0.0001281 | 100% | 3/3 | 0/3/0/0 | 0/0/0 |
| T1M1 | train | 0.010576 → 3.8273e-05 | 0.014624 → 7.7103e-05 | 100% | 6/6 | 0/6/0/0 | 0/0/0 |
| T1M1 | val | 0.024018 → 6.0994e-05 | 0.029479 → 0.00013322 | 100% | 3/3 | 0/3/0/0 | 0/0/0 |
| T1M1 | test | 0.0089464 → 7.3288e-05 | 0.011893 → 0.00011967 | 100% | 3/3 | 0/3/0/0 | 0/0/0 |

归一化尺度由各组 train 确定。两个真值系统沿用同一无噪声合成名单，train 6、val 3、test 3，均为稳定样本。各划分覆盖率与可判定准确率均为 100%，稳定误报率为零，失稳召回率和漏判率为 `null`。正式推理使用 GPU 双步长检验。test 用于报告。

运行 `machine_local_topology_refinement_v1-refinement-20261115T035654984080Z-5ef99a7e-97131108`，源提交 `5ef99a7e6b4d82b7ab7920f2515e43577ca4e182`。存档复核覆盖逐事件标量、检查点选择、工件哈希及算子预算。本轮从已有解重启 Adam，并调整学习率与损失时域，观测到的改进来自这套方案的共同作用。真实 TASK3 尚未启动，本轮部署结论为 `KEEP_BASELINE`。

[执行协议](#doc-e6a3caf6a434) · [证据收据](_sealed/machine_local_topology_refinement_20261115_evidence.json) · [结果复核](_sealed/machine_local_topology_refinement_20261115_result_review.json)

## 回执与附件

- [machine_local_topology_refinement_20261115_evidence.json](_sealed/machine_local_topology_refinement_20261115_evidence.json)
- [machine_local_topology_refinement_20261115_result_review.json](_sealed/machine_local_topology_refinement_20261115_result_review.json)
- [machine_local_topology_refinement_review_20261115.json](_sealed/machine_local_topology_refinement_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_refinement_v1_zh.md](_sealed/machine_local_topology_refinement_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_refinement_20261115_zh.md](_sealed/machine_local_topology_refinement_20261115_zh.md)

</details>
