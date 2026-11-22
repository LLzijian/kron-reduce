# 方程物理意义与七节点聚合审计协议

[返回主题页](../README.md)

## 执行协议

<a id="doc-7d627b5590d2"></a>

<a id="doc-7d627b5590d2-方程物理意义与七节点聚合审计协议"></a>

### 方程物理意义与七节点聚合审计协议


<a id="doc-7d627b5590d2-研究问题"></a>

#### 研究问题

本审计只使用 `combined_line_5fold_v1/F1` 的训练集和验证集以及 seed 324，测试集保持封闭。审计回答两个问题：

1. 冻结的 M0-GMD 方程是否保留了可观测的机电振荡特征，其残差闭合是否具有可解释的空间、时间和功率结构？
2. 固定七节点划分在当前 canonical 数据集上是否仍得到数据支持；它能否被严格描述为质量正交的相干投影，并在经验上表现出低通效应和更好的辨识条件？


<a id="doc-7d627b5590d2-第一性原理分解"></a>

#### 第一性原理分解

令 $R ∈ ℝ^{7×10}$ 为惯量加权聚合矩阵，$A ∈ ℝ^{10×7}$ 为组内复制矩阵，
并定义 $P=AR$。每个聚合坐标是组内惯量加权平均，且 $RA=I_7$。
因此 $P^2=P$ 且 $P^\top M=MP$：七节点化是十机状态空间上的
$M$-正交 cluster-consensus 投影。由此得到精确能量分解

\[
\|x\|_M^2=\|Px\|_M^2+\|(I-P)x\|_M^2.
\]

该代数事实本身不等于低通滤波。只有当验证轨迹显示补空间 ((I-P)x) 更集中于高时间频率、固定分组优于随机同型分组，并且七节点模型保留跨区预测性能时，才允许使用“经验慢相干低通前端”的表述。由于没有预先指定图拉普拉斯及其频率响应，本研究不宣称严格的图信号处理低通滤波器。

M0-GMD 的残差闭合满足零总内力、共同角平移不变和正时间常数，但这些结构证书仍不足以证明其系数是真实机器参数。审计分别测量：闭环轨迹误差；频谱重叠与主频；DMD 有效模态的频率、阻尼比和模态保证准则（MAC）；真实力残差的方向拟合与残差功率；静态、记忆、线性与非线性项消融。


<a id="doc-7d627b5590d2-冻结选择与数据隔离"></a>

#### 冻结选择与数据隔离

- M0-GMD 使用既有不可变 checkpoint，不重新选择其特征、秩、时间常数或系数。
- observed \(P_m(t)\) 仅用于隔离电气/闭合方程的条件重积分能力，不形成因果部署声明。
- 新的十节点与七节点 M0/M0-3E 比较只在训练集拟合并在验证集选 ridge 或三阶候选。
- 结果可按数据源与稳定标签做冻结后的描述性分层，但标签不进入任何拟合、归一化、候选选择或阈值选择。
- `test_ids_read` 必须为空；任何测试集读取均使整个审计无效。


<a id="doc-7d627b5590d2-v2-分区谱系修正与数据源一致性"></a>

#### v2 分区谱系修正与数据源一致性

历史 higher-order 实验使用的七节点分区为
`[[0,7],[1,2],[3],[4],[5,6],[8],[9]]`，而当前 canonical 适配器使用
`[[0,2],[1],[3],[4],[5],[6,7],[8,9]]`。二者不得再以同一“七节点模型”合并报告。
v2 在同一 F1/seed 324 协议下比较 current、legacy 以及 train-only discovered 三种分区。

train-only 分区以三对互不重叠机组为约束，最小化按 high-order/dense 数据源平衡后的角度增量与速度轨迹标准化 RMS 距离。分区一经训练集确定便冻结；能量保留、随机分区分位数、时间尺度分离、条件数和闭环误差仅在其后评估。该程序是可复现的分区发现规则，不宣称 slow-coherency 聚类算法本身具有新颖性。

数据源一致性通过三套弱形式拟合比较：combined、high-order only 与 dense only。比较源特定验证误差以及独立拟合的有效阻尼/耦合向量。差异只允许解释为“同一有效闭合对两个响应域的兼容性证据”；在没有原始仿真配置及参数哈希时，不得断言真实物理参数发生变化。


<a id="doc-7d627b5590d2-工程指标"></a>

#### 工程指标

工程指标参照电力系统模型验证与模态监测实践：轨迹形状、峰值和峰值时刻、均方误差、频谱、振荡频率、阻尼和模态形状。门槛仅用于本研究的主张校准，不冒充 NERC/WECC 的通用合格线：频谱重叠目标 0.80，主频误差不超过 0.10 Hz，MAC 目标 0.80，动态质量能量保留目标 95%，固定分组在随机同型分组中的目标分位数为 95%。阻尼估计对窗口与非线性敏感，作为连续量报告而不设单一硬门。


<a id="doc-7d627b5590d2-接受逻辑"></a>

#### 接受逻辑

计算和过程完整性与科学结论分开判定。只要数据隔离、投影恒等式、有限数值、可复现产物、测试、项目审计和对抗性审查全部通过，本审计可被过程接受，即使某个科学假设被否定。每项科学主张单独标记为 `SUPPORTED`、`CONDITIONAL` 或 `REJECTED`，不以总体平均掩盖数据源差异。


<a id="doc-7d627b5590d2-文献依据"></a>

#### 文献依据

- Hauer, J. F. *IEEE Trans. Power Syst.* **6**, 1062–1068 (1991). DOI: 10.1109/59.119247.
- Trudnowski, D. J. et al. *IEEE Trans. Power Syst.* **23**, 680–690 (2008). DOI: 10.1109/TPWRS.2008.919415.
- Hatziargyriou, N. et al. *IEEE Trans. Power Syst.* **36**, 3271–3281 (2021). DOI: 10.1109/TPWRS.2020.3041774.
- Chow, J. H. et al. *IEEE Trans. Power Syst.* **10**, 680–685 (1995). DOI: 10.1109/59.387903.
- Tyuryukanov, I. et al. *IEEE Trans. Power Syst.* **36**, 1482–1492 (2021). DOI: 10.1109/TPWRS.2020.3009628.
- Koochi, M. H. R. *IET Gener. Transm. Distrib.* **13** (2019). DOI: 10.1049/iet-gtd.2018.7028.
- Ramakrishna, R., Wai, H.-T. & Scaglione, A. arXiv:2008.01305 (2020).
- Zhang, L. & Schaeffer, H. *Multiscale Model. Simul.* **17**, 948–972 (2019). DOI: 10.1137/18M1189828.
- Russo, B. & Laiu, M. P. *SIAM J. Appl. Dyn. Syst.* (2023). DOI: 10.1137/22M1526782.
- NERC MOD-027-1 and MOD-033-1; WECC Generating Facility Data, Testing and Model Validation Requirements (2020).

## 复核与历史版本

<a id="doc-b9103ba73880"></a>

<details>
<summary>方程物理意义与七节点聚合代码审查</summary>

<a id="doc-b9103ba73880-方程物理意义与七节点聚合代码审查"></a>

### 方程物理意义与七节点聚合代码审查


<a id="doc-b9103ba73880-审查范围"></a>

#### 审查范围

- canonical 数据适配、COI 投影与七节点常量；
- 弱积分 M0、M0-3E 与冻结 M0-GMD 的拟合和闭环重积分；
- 历史 higher-order 七节点实现；
- F1/seed 324 的数据谱系、选择标记及 test 隔离。


<a id="doc-b9103ba73880-关键发现"></a>

#### 关键发现


<a id="doc-b9103ba73880-p0历史与当前七节点分区不是同一对象"></a>

##### P0：历史与当前七节点分区不是同一对象

历史 `legacy/source/3.10/support/common.py` 使用
`[[0,7],[1,2],[3],[4],[5,6],[8],[9]]`，当前
`src/kron_reduce/physics/constants.py` 使用
`[[0,2],[1],[3],[4],[5],[6,7],[8,9]]`。因此历史 higher-order 的正面结果不能自动迁移到当前 canonical 七节点表示。v2 必须以 matched inter-area 指标同时比较 current、legacy 与 train-only discovered 三种分区。


<a id="doc-b9103ba73880-p0m0-3e-名称不保证三阶状态被使用"></a>

##### P0：M0-3E 名称不保证三阶状态被使用

`TransientEmfM0Model.fit` 从包含 `alpha=0` 的候选集中按验证误差选择。当 `alpha=0` 时，三阶电势状态不影响力方程，模型实际退化为 anchor M0。报告必须同时给出 `selected_alpha` 和 `selected_tau`；只有 `alpha>0` 才能把结果作为 higher-order 机制证据。


<a id="doc-b9103ba73880-p1canonical-schema-一致不等于物理参数一致"></a>

##### P1：canonical schema 一致不等于物理参数一致

当前 combined 数据由 `ieee39_highorder_dataset` 与
`ieee39_near_critical_unstable_v1_dense` 两个 source 组成。schema 固定了通道、COI 口径和共享的 `M_prior`，但没有为每条事件保存原仿真器的阻尼、励磁、调速器及网络参数哈希。源特定有效 M0 参数差异只能证明一个共同闭合对两个响应域的兼容性不足，不能直接证明真实参数改变。


<a id="doc-b9103ba73880-p1冻结-gmd-的结构证书不是物理参数证书"></a>

##### P1：冻结 GMD 的结构证书不是物理参数证书

GMD 通过 incidence/零总力基保证内部力平衡，通过角差特征保证共同角平移不变，并限制正时间常数。这些性质证明模型结构与摆动方程商空间兼容，但不能唯一识别真实机器阻尼或 Kron 拓扑。必须另用真实路径残差方向、解释能量、残差功率和闭环模态保持做经验验证。


<a id="doc-b9103ba73880-p1observed-p_mt-只构成条件上界"></a>

##### P1：observed $P_m(t)$ 只构成条件上界

当前冻结方程和本轮重积分读取完整 observed $P_m(t)$。它适合隔离电气与残差闭合误差，但不是因果在线预测。任何工程部署主张都必须在独立工作流中用只读取 clearing state 的 $P_m$ ODE 复核全部门控。


<a id="doc-b9103ba73880-p2七节点的低通需要可证伪定义"></a>

##### P2：七节点的“低通”需要可证伪定义

质量正交投影严格保证能量分解，却不自动保证频率低通。只有验证集上 cluster-consensus 分量保留主要质量能量、补空间中位频率更高且高频能量占比更高，才允许称为经验时间尺度分离。严格 graph low-pass 还需要图移位算子和频率响应衰减比，本项目当前没有该证书。


<a id="doc-b9103ba73880-处置"></a>

#### 处置

- 新增三分区同协议审计与 train-only 分区发现；
- 新增 high-order/dense/combined 三套弱形式拟合与有效参数一致性表；
- higher-order 门加入 `alpha>0`；
- 报告自动拒绝机器参数恢复、物理拓扑恢复、严格图低通和聚类算法新颖性四类过强主张；
- test 数据继续封闭，标签不参与拟合、归一化、候选或阈值选择。

</details>

<details>
<summary>原件与版本</summary>

- [docs/research/physical_cluster_audit_protocol.md](_sealed/physical_cluster_audit_protocol.md)
- [docs/research/physical_cluster_code_review.md](_sealed/physical_cluster_code_review.md)

</details>
