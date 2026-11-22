# A1-S0：状态依赖端口合成正确性协议

日期：2026-11-10  
状态：待独立批准。允许范围仅限本协议列出的 A1-S0；批准收据须绑定实现、预算和本协议。workflow 必须设置 synthetic_correctness 显式审批门，未取得对应批准收据不得执行。

## 目的和范围

以一组固定已知参数检查常值端口和状态依赖端口的能量结构、COI、数值实现及完整梯度。零真实载荷、零拟合、零优化更新、零初始化求解。通过仅表示固定实现通过本协议的结构与梯度检查，可另行设计 A1 train；不代表真实数据收益，也不授权其他范围。

## 固定参数

节点 \(i=0,\ldots,9\)，边按公共 complete_incidence(10) 顺序 \(e=0,\ldots,44\)，角度用弧度。令
\[
P=I-\frac{m\mathbf1^\top}{\mathbf1^\top m},\quad Q=P^\top.
\]
使用解析 Helmert 力基 \(N\)，满足 \(\mathbf1^\top N=0,\ N^\top N=I\)，不得用非均匀质量角度 COI 基替代。第 \(j\) 列（\(j=0,\ldots,8\)）前 \(j+1\) 项为 \(1/\sqrt{(j+1)(j+2)}\)，第 \(j+2\) 项为 \(-(j+1)/\sqrt{(j+1)(j+2)}\)，其余为零。

| 参数 | 固定值 |
|---|---|
| 质量 | \(m_i=1+0.1i\) |
| 正弦边刚度 | \(k_e=0.6+0.01e\) |
| 阻尼 | \(D=0.15NN^\top\) |
| 隐藏矩阵 | \(J=\begin{bmatrix}0&-0.7\\0.7&0\end{bmatrix},\ R=\operatorname{diag}(0.12,0.18)\) |
| 基线端口 | \(C_0=N[:,0:2]\operatorname{diag}(0.35,0.25)\) |
| 静态平衡先验 | \(q_{\rm eq}=0.1Q[\cos(i+1)]\) |
| 常输入 | \(p=0.03P[\sin(i+1)]\) |
| 非零参数 | \(\Theta^\star_{j,:}=0.25[\cos(j+1),\sin(2(j+1))]\) |

优化坐标为 \(\Theta\in\mathbb R^{9\times2}\)，物理增量 \(\Delta=s_C\tanh(\Theta)\)，\(s_C=\lVert C_0\rVert_F/\sqrt{18}\)。定义
\[
C_{\rm const}=C_0+N\Delta,\quad
C_{\rm state}(q)=C_0+P\operatorname{diag}(\sin(Qq))N\Delta.
\]
PyTorch 外层保留 \(s_C\tanh(\Theta)\) 链式梯度。互连两侧分别用 \(-PC(q)z\)、\(C(q)^\top Qv\)；\(H\) 不依赖 \(C\)，不加 \(\dot C\) 项，状态 Jacobian 和反传须包含 \(C'(q)\)。\(\Theta=0\) 时回到 \(C_0\)。

固定状态：令 \(a_i=\sin((i+1)\pi/5), b_i=\cos((i+1)\pi/5)\)。

- \(x_1=(0.6Qa,0.08Qb,[0.4,-0.3])\)
- \(x_2=(0.9Qb,-0.05Qa,[-0.25,0.45])\)
- \(x_{\rm zero}=(0,0,0)\)
- 仅单侧互易负例：\(x_{\rm neg}=(Qe_0,Qe_0,[1,0])\)，\(\Theta_{\rm neg}[0,0]=0.5\)，其余为零。

五个条件是 baseline \(C_0\)、const0、state0、const*、state*。前三者能量与向量场一致；后两者轨迹可不同，轨迹差异不是预测收益。

## 检查与阈值

**长路径：** 输出时钟 \(0:0.01:3\) s，最大步长 \(h=0.005,0.0025\) s。NumPy baseline 在 \(x_1,x_2\) 两档步长共 4 条；const0/state0 在 \(x_1,h=0.0025\) 共 2 条；const*/state* 在 \(x_1,x_2\) 两档步长共 8 条。PyTorch 在 \(x_1,h=0.0025\) 对 const*/state* 各 1 条。所有路径使用指定完整初态，不读外部目标、不重置状态；NumPy 逐存储状态检查有限值、质量 COI 和物理恒等式，PyTorch 检查有限值、质量 COI 及与 NumPy 的一致性。

**点态、Jacobian 和短传播梯度：** 在 \(x_{\rm zero},x_1,x_2\) 检查五条件能量恒等式；在 \(x_1,x_2\) 做非零端口下 NumPy/PyTorch RHS 一致性，共 4 对，并查 \(\Theta=0\) 基线一致性。在 \(x_1,x_2\) 检查非零端口 \(q\to q+0.7\mathbf1\) 平移不变性。\(x_1,\Theta^\star\) 检查两种端口完整 \(22\times40\) state+parameter Jacobian；\(x_1,\Theta=0\) 检查完整 \(22\times18\) 参数 Jacobian。中心差分 \(\epsilon=10^{-5},10^{-6}\) 两档均须通过。传播梯度在 \(x_1,\Theta^\star\)、两种端口、时钟 \(0:0.01:0.1\) s、\(h=0.005\) 下检查。\(L_{\rm check}\) 是 11 个存储点、20 个可见分量平方的均值，仅为梯度检查量，不是训练目标；比较完整 18 参数梯度与两档中心差分。NumPy 短路径 144 条，PyTorch 两条。

**隔离负例：** 单侧修改互易耦合须违反能量抵消；state 端口计算 \(C\) 时对 \(q\) detach 的实现须被非零 \(\Theta\) 完整 state-Jacobian 检查拒绝。错误实现只存在于隔离检查函数。

| 检查 | 阈值 |
|---|---|
| 零 \(\Delta\) NumPy 基线路径 | maxabs \(\le10^{-12}\) |
| NumPy/PyTorch 对应长路径 | maxabs \(\le10^{-10}\) |
| 质量 COI | \(|m^\top q|,|m^\top v|\le10^{-10}\) |
| 交换功率/能量恒等式 | 误差除以 max(1,各功率项绝对值之和) \(\le10^{-11}\) |
| 两档步长 | 全部 22 状态逐点 maxabs \(\le10^{-6}\) |
| 完整 Jacobian/传播梯度 | maxabs差 \(\le10^{-8}+2\cdot10^{-6}\times\) maxabs(FD)，两档 \(\epsilon\) 均满足 |
| 负例检出 | 违反量大于对应容差 10 倍；否则该负例未有效受检，整轮不通过 |

两档步长是预定数值检查，不是连续 ODE 误差严格证书。全部正确实现门通过且两个负例有效检出方可接受；失败即停止并保留证据，不放宽阈值或换参数重试。

## 预算和执行边界

成功完成的固定预算：NumPy RHS 69,017 次；PyTorch RHS 9,769 次；5 次 Jacobian、110 次 VJP、2 次标量反传；162 条 rollout、3 次 probe（setup、algebra、Jacobian）。另计 614 次 model construction 和 158 次 public rollout checker，共 772 次 PassiveLatentClosure validation，每次 3 次 eigvalsh，即 2,316 次参数校验；这不是可观测性或谱研究。机器计划与独立批准收据绑定这些上限，科学执行后不得更改。

保存全部正负 RHS、144 条短路径正负 loss、2 条 torch 名义路径及梯度；失败时保存 last_rhs。各 scope 失败即停止、保留证据，不续跑。按项记录 attempted/returned、计数和失败状态；反传及重计算不得记零。SVD、expm、LS 均为零。

完成后允许一次独立保存证据核验：只解码本次运行的 `inputs.npz`、`algebra.npz`、`jacobians.npz`、`paths.npz` 各一次及其 JSON 索引，核对有限差分、loss、误差与门槛的算术和调用记录；可检查本次完整 artifact manifest 的文件哈希。不得重新调用模型、RHS、积分器、求导或其他科学算子，核验结果另外保存。运行本身的失败不得由核验覆盖。

只执行本协议固定确定性参数，不读真实或已保存轨迹数组、标签、val/test，不进行拟合、优化、初始化求解或 A1-S0 外附加计算。需独立批准收据及 workflow 的 synthetic_correctness 显式审批门。接受结论限于：“固定 A1 实现通过本范围的结构与梯度检查，可另行设计下一次比较。”A1 train、数据访问、模型升级和部署均需另行登记批准。

理论依据：[用户提供的研究讨论](chatgpt-conversation://6af25e1f-db88-83e9-a3ae-d02442950a25)、[研究方案](../research/research_plan_20261110_zh.md)、[A16 闭环理论](../research/kiki_a16_pm_impulse/astra_next_closed_loop_theory_zh.md)、[能量屏障补充推导](../research/kiki_a16_pm_impulse/astra_cq_energy_barrier_addendum_zh.md)。

