# 方程物理意义与七节点聚合代码审查

## 审查范围

- canonical 数据适配、COI 投影与七节点常量；
- 弱积分 M0、M0-3E 与冻结 M0-GMD 的拟合和闭环重积分；
- 历史 higher-order 七节点实现；
- F1/seed 324 的数据谱系、选择标记及 test 隔离。

## 关键发现

### P0：历史与当前七节点分区不是同一对象

历史 `legacy/source/3.10/support/common.py` 使用
`[[0,7],[1,2],[3],[4],[5,6],[8],[9]]`，当前
`src/kron_reduce/physics/constants.py` 使用
`[[0,2],[1],[3],[4],[5],[6,7],[8,9]]`。因此历史 higher-order 的正面结果不能自动迁移到当前 canonical 七节点表示。v2 必须以 matched inter-area 指标同时比较 current、legacy 与 train-only discovered 三种分区。

### P0：M0-3E 名称不保证三阶状态被使用

`TransientEmfM0Model.fit` 从包含 `alpha=0` 的候选集中按验证误差选择。当 `alpha=0` 时，三阶电势状态不影响力方程，模型实际退化为 anchor M0。报告必须同时给出 `selected_alpha` 和 `selected_tau`；只有 `alpha>0` 才能把结果作为 higher-order 机制证据。

### P1：canonical schema 一致不等于物理参数一致

当前 combined 数据由 `ieee39_highorder_dataset` 与
`ieee39_near_critical_unstable_v1_dense` 两个 source 组成。schema 固定了通道、COI 口径和共享的 `M_prior`，但没有为每条事件保存原仿真器的阻尼、励磁、调速器及网络参数哈希。源特定有效 M0 参数差异只能证明一个共同闭合对两个响应域的兼容性不足，不能直接证明真实参数改变。

### P1：冻结 GMD 的结构证书不是物理参数证书

GMD 通过 incidence/零总力基保证内部力平衡，通过角差特征保证共同角平移不变，并限制正时间常数。这些性质证明模型结构与摆动方程商空间兼容，但不能唯一识别真实机器阻尼或 Kron 拓扑。必须另用真实路径残差方向、解释能量、残差功率和闭环模态保持做经验验证。

### P1：observed $P_m(t)$ 只构成条件上界

当前冻结方程和本轮重积分读取完整 observed $P_m(t)$。它适合隔离电气与残差闭合误差，但不是因果在线预测。任何工程部署主张都必须在独立工作流中用只读取 clearing state 的 $P_m$ ODE 复核全部门控。

### P2：七节点的“低通”需要可证伪定义

质量正交投影严格保证能量分解，却不自动保证频率低通。只有验证集上 cluster-consensus 分量保留主要质量能量、补空间中位频率更高且高频能量占比更高，才允许称为经验时间尺度分离。严格 graph low-pass 还需要图移位算子和频率响应衰减比，本项目当前没有该证书。

## 处置

- 新增三分区同协议审计与 train-only 分区发现；
- 新增 high-order/dense/combined 三套弱形式拟合与有效参数一致性表；
- higher-order 门加入 `alpha>0`；
- 报告自动拒绝机器参数恢复、物理拓扑恢复、严格图低通和聚类算法新颖性四类过强主张；
- test 数据继续封闭，标签不参与拟合、归一化、候选或阈值选择。
