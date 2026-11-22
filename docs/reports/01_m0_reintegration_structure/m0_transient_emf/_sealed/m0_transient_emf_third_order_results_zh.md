# 应力门控暂态电势三阶 M0：方法、单种子验证结果与接受结论

## 摘要

本轮在 F1、seed 324 上构建并验证了节点暂态电势三阶模型 M0-3EG。模型不再把第三状态解释为抽象的边记忆，而是为每个聚合节点引入归一化暂态电势 (q_i)，并用同一图结构同时构造有功功率与 Kirchhoff 一致的电流项。为避免三阶自由度破坏小扰动精度，模型增加了只依赖当前功角的平滑大扰动激活；阈值以下严格退化为二阶 M0，阈值以上才允许暂态电势降低有效输电能力。

在 observed (P_m(t)) 条件重积分下，M0-3EG 的 H=100 平均综合 RMSE 为 0.19936，相对旧混合来源 M0 的 0.31817 改善 37.34%。同源 highorder 验证误差与二阶物理锚点完全一致，dense 压力轨迹平均 RMSE 从 0.73355 降至 0.45479，改善 38.00%。模型冻结后，在统一 5 s、(P_{m,\mathrm{clear}}) 零阶保持的因果分类审计中，9 个不稳定样本全部识别且无误报，FSR=0、FUR=0。所有预设门均通过，因此本轮结论为 **ACCEPT（仅限 F1/seed324 验证证据）**，尚不能表述为最终 test 或多折显著性结论。

## 1. 为什么二阶模型在两类轨迹上表现不同

二阶 swing 方程把节点电压幅值视为常量，边功率只随相角差变化。这一假设在 highorder 小扰动轨迹上已经足够，因此用同源长轨迹恢复的二阶 M0 可达到 H=100 平均 RMSE 0.02057。dense 轨迹包含更强的功角应力，同一二阶锚点的平均 RMSE 升至 0.73355，说明固定传输能力假设在大扰动区失效。

三阶同步机/单轴模型的基本物理启示是：有功边功率应按 (E'_iE'_j\sin(\delta_i-\delta_j)) 缩放，暂态电势与功角动态可能共同决定失稳，而非只增加一个任意统计残差。相关理论与模型背景见 Sharafutdinov 等关于三阶模型角度—电压耦合的研究 [1]、Thümler 等关于 Kirchhoff 一致性和动态传输能力的讨论 [2]，以及三阶同步机非线性模型有效性研究 [3]。图结构与能量一致性的建模原则也与端口哈密顿电网表示相符 [4]。

本数据审计已经排除了“highorder 与 dense 使用了不同平衡点或明显不同基础网络文件”这一简单解释，但两类轨迹反演出的有效 (D,K) 差异较大。结合 dense 的大扰动特征，更合理的解释是：低阶常系数模型把未观测的电压、励磁和网络代数动态折算进了有效参数，导致跨运行区间失配。因此本轮不继续增加无约束多项式，而是显式恢复暂态传输能力这一缺失物理状态。

## 2. 模型方程

### 2.1 二阶物理锚点

首先只用同源 highorder 训练轨迹，通过积分弱形式恢复 (D_i\ge0) 与 (K_e\ge0)：

\[
m_i\dot\omega_i=\Delta P_{m,i}-D_i\omega_i
-\left[B\left(K\odot\Delta\sin\eta\right)\right]_i,
\qquad \Delta P_m=P_m-P_{m,\mathrm{eq}}.
\]

得到的阻尼为

\[
D=[0.89780,\,0.15300,\,0.32739,\,0.24208,\,3.93\times10^{-14},\,0.15473,\,1.19\times10^{-16}].
\]

其中 5 个节点得到正阻尼、2 个节点仍贴近非负边界。因此这一步改善了“所有阻尼整体塌缩”的问题，但不能把聚合节点 (D_i) 宣称为真实机组阻尼；它仍是 COI 商空间和聚合图上的有效可辨识量。更重要的是，在后续三阶选择中 (D,K) 逐位冻结，禁止通过再次压低阻尼换取轨迹误差。

### 2.2 节点暂态电势三阶动力学

M0-3EG 使用

\[
\dot\delta=\omega,
\]

\[
m\odot\dot\omega=\Delta P_m-D\odot\omega
-B\left[K\odot\left(q_uq_v\sin\eta-\sin\eta_{\mathrm{eq}}\right)\right],
\]

\[
\tau\dot q=\mathbf 1-q-
a(\sigma)\frac{\alpha}{k_{\mathrm{ref}}}
\left[i_d(q,\eta)-i_d(\mathbf1,\eta_{\mathrm{eq}})\right].
\]

图电流项为

\[
i_{d,i}(q,\eta)=\sum_{j}K_{ij}\left(q_i-q_j\cos\eta_{ij}\right),
\]

它与摆动方程使用同一个 incidence 图和同一组 (K_e)，避免为第三状态引入与功率图无关的自由连接。

### 2.3 大扰动激活与严格回退

当前状态的无量纲边角应力定义为

\[
\sigma(\delta)=\max_e\left|\sin\eta_e-\sin\eta_{e,\mathrm{eq}}\right|.
\]

激活 (a(\sigma)) 在阈值以下为 0，在阈值以上经过宽度 (w) 的 smoothstep 平滑过渡到 1。由此得到两个关键性质：

1. 当 (\sigma\) 低于阈值时，(q\equiv\mathbf1)，M0-3EG 与二阶锚点逐步完全一致；
2. 大扰动时 (q_iq_j) 可降低有效边功率，表达暂态传输能力下降。

训练集清除状态的 (sigma) 分位数用于生成候选阈值，不使用标签、验证集或 test。候选阈值为

\[
[0,\,0.03621,\,0.12059,\,0.36892,\,0.53907],
\]

平滑宽度为 0.12572。最终选择

\[
\alpha=1.0,\qquad \tau=0.5\ \mathrm{s},\qquad
\sigma_{\mathrm{th}}=0.53907.
\]

清除时 (q(0)) 由三阶方程右端为零的准稳态线性系统求得，只读取清除角度、平衡角度、已冻结图参数和声明先验，不读取真实电压、(P_e)、未来状态、标签或事件长度。

## 3. 训练与选择协议

- 折与种子：F1、seed 324，仅一个 seed。
- 物理锚点：highorder 同源训练轨迹，积分弱形式一次求解 (D,K)。
- 三阶候选：
  - (\alpha\in\{0,0.3,1,3\})；
  - (\tau\in\{0.5,1,2\}\) s；
  - 阈值来自 train-only 清除应力分位数；
  - (\alpha=0) 是严格二阶回退。
- 支持样本：训练 204 条（valid length 至少 9）；长视界验证 34 条（至少 100 点）。
- 选择目标：对每个数据来源分别用训练尺度标准化，再对来源等权平均；同时要求每个来源的标准化验证损失均不得劣于二阶锚点。
- 条件重积分：observed (P_m(t))，线性 RK 阶段插值。
- 分类审计：模型选择完成并写入 selection marker 后，读取 43 条验证样本标签；统一使用 500 点、0.01 s、(P_{m,\mathrm{clear}}) 常值输入。
- test：未读取。

这里没有神经网络 epoch。计算量来自候选三阶 ODE 的闭环重积分。模型选择后未再调整 (D,K,\alpha,\tau) 或阈值。

## 4. 重积分结果

### 4.1 H=100 总体结果

| 方法 | Mean RMSE | Median RMSE | P90 RMSE | pooled ω (R^2) | Failure rate |
|---|---:|---:|---:|---:|---:|
| M0-combined-long | 0.31817 | 0.05825 | 1.07935 | 0.0124 | 0 |
| M0-source-anchor | 0.31415 | 0.03674 | 1.11169 | -0.0266 | 0 |
| M0-3E（无门控） | 0.20465 | 0.04027 | 0.64868 | 0.6226 | 0 |
| **M0-3EG（应力门控）** | **0.19936** | **0.03674** | **0.57157** | **0.6651** | **0** |

M0-3EG 相对旧 M0 的 mean RMSE 改善 37.34%，相对自身二阶锚点改善 36.54%，相对无门控 M0-3E 继续改善 2.58%。P90 同样显著下降，说明收益不是只由少数容易样本造成。

### 4.2 分来源结果

| 来源 | M0-source-anchor | M0-3E 无门控 | M0-3EG | M0-3EG 相对锚点 |
|---|---:|---:|---:|---:|
| highorder，n=20 | 0.02057 | 0.02286 | **0.02057** | **0.00%（严格保持）** |
| dense，n=14 | 0.73355 | 0.46435 | **0.45479** | **改善 38.00%** |

第一轮无门控 M0-3E 虽然总体改善 35.68%，但使 highorder 平均误差退化 11.14%，因此自动判定为 REJECT。第二轮应力门控使 highorder 回到与物理锚点完全相同，同时进一步改善 dense，解决了这一 Pareto 冲突。

### 4.3 数值审计

- 最大 COI 漂移：(4.44\times10^{-16})。
- highorder 代表轨迹：(dt/dt2=1.46\times10^{-8})，(dt2/dt4=1.89\times10^{-9})，收敛比 7.73。
- dense 代表轨迹：(dt/dt2=3.60\times10^{-8})，(dt2/dt4=5.24\times10^{-9})，收敛比 6.87。
- 平衡点三阶残差：0。
- 所有长视界和分类 rollout failure rate：0。

## 5. 固定因果分类结果

| 方法 | TP | FN | FP | TN | FSR | FUR | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0-combined-long | 2 | 7 | 0 | 34 | 0.7778 | 0 | 0.8372 | 0.6111 |
| M0-source-anchor | 2 | 7 | 0 | 34 | 0.7778 | 0 | 0.8372 | 0.6111 |
| M0-3E 无门控 | 9 | 0 | 1 | 33 | 0 | 0.0294 | 0.9767 | 0.9853 |
| **M0-3EG** | **9** | **0** | **0** | **34** | **0** | **0** | **1.0000** | **1.0000** |

分类标签未参与方程拟合、阈值生成或候选选择。该结果支持“暂态传输能力是 dense 不稳定轨迹的重要缺失状态”这一机制解释，但由于仍是同一验证折上的单 seed 审计，不能据此声称总体分类已经达到 100%。

## 6. 图像证据

最终图采用各来源内 H=100 M0-3EG 误差最接近该来源中位数的验证样本，而非挑选最佳案例。highorder 代表为 `scenario_002`，dense 代表为 `ncunstable_dense_0081`。图中黑线为实际数据，灰线为旧混合来源 M0，蓝线为同源二阶锚点，橙线为 M0-3EG。highorder 中蓝线与橙线严格重合；dense 中三阶模型修正了二阶模型的振荡相位和幅值。

图及可追溯源数据见[暂态电势图件归档说明](../../../results/backup/01_m0_reintegration_structure/02_transient_emf_reintegration.DESCRIPTION.md)。使用包内 `results/reports/m0-3eg-transient-emf-f1-seed324-20260730-v3/`。

- `fig_m0_3eg_reintegration_examples.{svg,pdf,png}`
- `figure_source_data.csv`
- `figure_metadata.json`

## 7. 接受结论与限制

### 接受

M0-3EG 通过本轮全部门：非平凡三阶状态、总体误差改善、highorder 不退化、dense 改善、零 rollout failure、COI 守恒、(D,K) 严格冻结、步长收敛、物理审计通过、FSR<0.2 且优于 M0。因此接受该结构作为下一阶段 M0 主候选。

### 不接受的表述

当前证据不足以支持以下说法：

1. (q_i) 是被数据直接测得的真实机组暂态电势；它只是满足三阶物理结构的聚合隐状态。
2. 当前 (D_i,K_e) 是真实机组阻尼或真实 Kron 网络参数。
3. 分类在未知 test、其他折或其他 seed 上必然为 100%。
4. observed (P_m(t)) 结果是完全因果的机械功率预测。精度结果仍属于条件重积分上界；只有固定 (P_{m,\mathrm{clear}}) 分类审计是因果 ZOH 输入。

## 8. 下一步

1. 保持用户要求的单 seed 策略，不立即启动五折全量计算。
2. 先将 M0-3EG 作为冻结基座，检查 M2A/LYAP 残差是否仍能在不破坏应力门控和能量证书的条件下提供至少 2% 增益。
3. 在 observed (P_m(t)) 的效果已经达标后，再按既定顺序增加只依赖 (P_{m,\mathrm{clear}}) 与预测速度的一阶 governor ODE，评估能否接近条件重积分精度。
4. 在最终 test 前固定阈值策略、候选表、selection marker 与 checkpoint；多折/多 seed 只在用户批准后运行。

## 参考文献

1. Sharafutdinov K, et al. Rotor-angle versus voltage instability in the third-order model for synchronous generators. *Chaos* 28, 033117 (2018). <https://arxiv.org/abs/1706.06396>
2. Thümler L, et al. Absence of pure voltage instabilities in the third order model of power grid dynamics (2021). <https://arxiv.org/abs/2111.15590>
3. Validity testing of third-order nonlinear models for synchronous generators. *Electric Power Systems Research* (2009). <https://www.sciencedirect.com/science/article/abs/pii/S0378779608003246>
4. A port-Hamiltonian approach to power network modeling and analysis. *European Journal of Control* (2013). <https://www.sciencedirect.com/science/article/pii/S094735801300157X>

## 证据位置

- 接受结果：`results/diagnostics/m0-3eg-stress-gated-f1-seed324-20260730/acceptance.json`
- 候选表：`results/diagnostics/m0-3eg-stress-gated-f1-seed324-20260730/candidate_selection.csv`
- 逐样本指标：`results/diagnostics/m0-3eg-stress-gated-f1-seed324-20260730/per_case_metrics.csv`
- 分类明细：`results/diagnostics/m0-3eg-stress-gated-f1-seed324-20260730/classification_rows.csv`
- selection marker：`results/diagnostics/m0-3eg-stress-gated-f1-seed324-20260730/selection_marker.json`
- checkpoint：`results/diagnostics/m0-3eg-stress-gated-f1-seed324-20260730/checkpoints/M0-3EG.npz`
