# 状态依赖端口：结构与完整梯度检查

[返回主题页](../README.md)

## 运行结果与归档

- [原运行结果（含失败状态）](../../../../results/state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0/summary.json)
- [保存审核归档说明](../../../../results/backup/07_state_port/01_state_port_synthetic_saved_audit.DESCRIPTION.md)

登记文件与配置

[方案登记](../../../../configs/proposals/state_port_synthetic_v1.json) · [登记 state_port_saved_review_v1](../../../../configs/proposals/state_port_saved_review_v1.json) · [登记 state_port_saved_sources_v1](../../../../configs/proposals/state_port_saved_sources_v1.json)

## 执行协议

<a id="doc-e881ee12a327"></a>

<a id="doc-e881ee12a327-a1-s0-失败运行的独立保存证据审核"></a>

### A1-S0 失败运行的独立保存证据审核

2026-11-11。本协议只审核已生成的证据，不执行新的科学计算。

唯一来源为运行 `state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0`，源提交 `992738b0c4bfbf26e073f40a29f67cea4c4ac429`。其四个范围已关闭，数值记录已写出；随后写汇总 JSON 时遇到 NumPy 布尔值序列化错误，运行保持失败。原授权消费关闭，失败日志、不完整的 90 字节 summary 和所有原文件保持原样。

新授权 `state_port_saved_review_v1_20261111` 仅允许一次保存证据审核。机器计划绑定审核脚本、此协议和外置来源清单；准确审核收据完成后方可执行。入口先核授权与哈希，再在共享 Git 目录预留一次性预算，不重试、不覆盖已有审核。解码范围严格为本运行的 `inputs.npz`、`algebra.npz`、`jacobians.npz`、`paths.npz`，各一次、`allow_pickle=False`；其余 NPZ 只核文件哈希。可读取本运行 JSON、日志及冻结计划，校验清单中全部 22 个文件的哈希。不解析损坏的 summary，不要求不存在的成功 stage evidence。

审核复算保存量的有限差分、loss、归一化误差、矩阵投影关系和阈值，检查记录数量、形状、有限性与索引完整性。核验 15 个点态记录、4 组正确 Jacobian、16 条长路径、2 条名义短路径、144 条差分短路径；两个人工错误须被原定阈值检出。核对原失败记录的 attempted/returned、四范围日志的 165 对操作及日志区间哈希。不得调用模型、RHS、积分、求导、SVD、expm、LS、拟合或新科学算子。

审核报告只写入 `reports/generated/state_port_synthetic_20261111/`。原工作流不得因审核而前进，原运行也不得回填成功汇总或改为成功。报告分开记录 `process_status=FAIL_PRESERVED` 和保存证据的科学检查结论；只有全部算术与完整性检查通过，后者才可记为 `PASS_FROM_SAVED_EVIDENCE`。该结果至多支持另行准备真实 train 机制比较，保持 `KEEP_BASELINE`。

<a id="doc-1716ff046240"></a>

<a id="doc-1716ff046240-a1-s0状态依赖端口合成正确性协议"></a>

### A1-S0：状态依赖端口合成正确性协议

日期：2026-11-10<br>
状态：待独立批准。允许范围仅限本协议列出的 A1-S0；批准收据须绑定实现、预算和本协议。workflow 必须设置 synthetic_correctness 显式审批门，未取得对应批准收据不得执行。


<a id="doc-1716ff046240-目的和范围"></a>

#### 目的和范围

以一组固定已知参数检查常值端口和状态依赖端口的能量结构、COI、数值实现及完整梯度。零真实载荷、零拟合、零优化更新、零初始化求解。通过仅表示固定实现通过本协议的结构与梯度检查，可另行设计 A1 train；不代表真实数据收益，也不授权其他范围。


<a id="doc-1716ff046240-固定参数"></a>

#### 固定参数

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


<a id="doc-1716ff046240-检查与阈值"></a>

#### 检查与阈值

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


<a id="doc-1716ff046240-预算和执行边界"></a>

#### 预算和执行边界

成功完成的固定预算：NumPy RHS 69,017 次；PyTorch RHS 9,769 次；5 次 Jacobian、110 次 VJP、2 次标量反传；162 条 rollout、3 次 probe（setup、algebra、Jacobian）。另计 614 次 model construction 和 158 次 public rollout checker，共 772 次 PassiveLatentClosure validation，每次 3 次 eigvalsh，即 2,316 次参数校验；这不是可观测性或谱研究。机器计划与独立批准收据绑定这些上限，科学执行后不得更改。

保存全部正负 RHS、144 条短路径正负 loss、2 条 torch 名义路径及梯度；失败时保存 last_rhs。各 scope 失败即停止、保留证据，不续跑。按项记录 attempted/returned、计数和失败状态；反传及重计算不得记零。SVD、expm、LS 均为零。

完成后允许一次独立保存证据核验：只解码本次运行的 `inputs.npz`、`algebra.npz`、`jacobians.npz`、`paths.npz` 各一次及其 JSON 索引，核对有限差分、loss、误差与门槛的算术和调用记录；可检查本次完整 artifact manifest 的文件哈希。不得重新调用模型、RHS、积分器、求导或其他科学算子，核验结果另外保存。运行本身的失败不得由核验覆盖。

只执行本协议固定确定性参数，不读真实或已保存轨迹数组、标签、val/test，不进行拟合、优化、初始化求解或 A1-S0 外附加计算。需独立批准收据及 workflow 的 synthetic_correctness 显式审批门。接受结论限于：“固定 A1 实现通过本范围的结构与梯度检查，可另行设计下一次比较。”A1 train、数据访问、模型升级和部署均需另行登记批准。

理论依据：[用户提供的研究讨论](chatgpt-conversation://6af25e1f-db88-83e9-a3ae-d02442950a25)、[研究方案](../../05_research_governance/cross_campaign_plan_20261110/README.md#doc-c2d77b9f9462)、[A16 闭环理论](../../04_kiki_a03_a17_pipeline/a16/README.md#doc-7de4989d1a4c)、[能量屏障补充推导](../../04_kiki_a03_a17_pipeline/a16/README.md#doc-8abac79dff34)。

## 报告与结果

<a id="doc-fc70cc3f9d2d"></a>

<a id="doc-fc70cc3f9d2d-状态依赖端口结构与完整梯度检查"></a>

### 状态依赖端口：结构与完整梯度检查

2026-11-11。研究范围为 A1-S0：一个固定十节点、两隐藏状态的合成系统，零拟合、零真实数据读取。独立保存证据审核的 **200 项检查全部通过，0 项不一致**，支持本轮结构、完整梯度和数值一致性结论。原运行在完成数值检查后因汇总 JSON 序列化失败而关闭，其失败状态保留。基线保持 `KEEP_BASELINE`。


<a id="doc-fc70cc3f9d2d-比较设计"></a>

#### 比较设计

基线为固定正弦势能闭环模型。常值修正与状态依赖修正分别采用

\[
C_{\rm const}=C_0+N\Delta,\qquad
C_{\rm state}(q)=C_0+P\operatorname{diag}(\sin(Qq))N\Delta,
\quad \Delta=s_C\tanh\Theta.
\]

两组均有 18 个参数；此次只检查预先给定参数，不学习参数。互连两侧使用同一个端口，能量函数沿用原模型。检查原端口、两种零增量端口、两种非零增量端口，并在相同完整初态下比较实现、步长与梯度。[协议](#doc-1716ff046240)固定全部参数、时间表和判据。


<a id="doc-fc70cc3f9d2d-保存结果"></a>

#### 保存结果

表中是全部注册条件的最大绝对差或最大归一化残差；它们是实现与数值检查指标，不是相对真实电网轨迹的预测误差。

| 检查 | 保存记录中的最大值 | 固定要求 |
|---|---:|---:|
| 零增量与原基线路径 | 0 | ≤1e−12 |
| NumPy 与 PyTorch 的 3 s 路径 | 1.78e−15 | ≤1e−10 |
| 0.005 s 与 0.0025 s 步长路径 | 1.09e−9 | ≤1e−6 |
| 归一化能量恒等式残差 | 5.88e−15 | ≤1e−11 |
| 质量 COI 残差 | 2.66e−15 | ≤1e−10 |
| 完整 Jacobian 与中心差分 | 5.15e−10 | 每组按 1e−8+2e−6·maxabs(FD) |
| 短传播参数梯度与中心差分 | 2.27e−11 | 每组按同一公式 |

单侧改动端口的反例留下 0.02459 的归一化功率残差，高于检出门槛。切断端口角度梯度的反例产生 0.008981 的 Jacobian 差异，也超过对应容差十倍。检查因此同时覆盖正确实现与两类故意错误。

全部固定工作已完成：14 条 NumPy 长路径、2 条 PyTorch 长路径、144 条差分短路径、2 条名义短路径；69,017 次 NumPy RHS、9,769 次 PyTorch RHS、5 次 Jacobian 和 2 次标量反传。没有拟合、优化更新、初始化求解，也没有读取 train、val 或 test。


<a id="doc-fc70cc3f9d2d-运行状态与后续含义"></a>

#### 运行状态与后续含义

原运行 ID：`state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0`，源提交 `992738b0c4bfbf26e073f40a29f67cea4c4ac429`，执行前 [CI](https://github.com/LLzijian/kron-reduce/actions/runs/42852275799) 通过。各数值范围关闭后，最终汇总中的 NumPy 布尔值不能直接写入严格 JSON，导致 `TypeError`。该问题已通过显式布尔转换修复，并增加独立人工 fixture 的序列化回归；未重跑本轮参数矩阵或补写原汇总。

原运行状态始终为 `FAIL_PRESERVED`。[外置来源清单](../../../../configs/proposals/state_port_saved_sources_v1.json)封存其全部 22 个文件，包括不完整 summary 和失败日志。独立审核依据[保存证据协议](#doc-e881ee12a327)，将四个指定 NPZ 各解码一次，只核已有数组、算术和账目；没有新增模型、RHS、积分或求导调用。[审核结果](../../../../results/backup/07_state_port/01_state_port_synthetic_saved_audit.DESCRIPTION.md)记录 `PASS_FROM_SAVED_EVIDENCE`，原工作流不因审核而推进。

[Astra 结果审核](#doc-21110cda5059)接受上述有限科学结论，记为 `ACCEPT`。[证据收据](_sealed/state_port_synthetic_20261111_evidence.json)绑定原运行、独立审核与结果审核；两项一次性授权均已关闭。审计脚本的 Linux 可执行标记已补齐，文件内容及其冻结哈希保持不变。

这轮研究的作用是检查状态端口实现及其训练所需梯度。它没有证明状态依赖端口能够改善真实预测，也没有识别既有误差根因。下一项研究仍是独立登记的 A1 train 配对机制比较：相同完整起报态、相同 18 参数预算、冻结骨架，并同时与常值修正组和原 A06 比较。B 初始化联合训练、A2 势能修改和 C 时间目标继续分开。

## 复核与历史版本

<a id="doc-daba172fb79e"></a>

<details>
<summary>A1-S0 状态端口：准确实施审核</summary>

<a id="doc-daba172fb79e-a1-s0-状态端口准确实施审核"></a>

### A1-S0 状态端口：准确实施审核

2026-11-11，Astra Ultra 科学审核。模型与推理配置依据协调者任务分派记录为 `gpt-6-astra`、`ultra`。

**准确实施审核通过。** 批准本次固定合成正确性检查在独立一次性范围登记、显式 `synthetic_correctness` 审批记录、干净提交已推送、适用 CI 和实际加载环境守卫通过后执行一次。用户“调整项目研究方案并开始自主研究”的明确指令提供操作授权；本收据给出准确科学范围与实现批准。研究结果须由冻结门槛和保存证据决定。

| 绑定对象 | SHA256 |
| --- | --- |
| [计划](../../../../configs/proposals/state_port_synthetic_v1.json) | `2f0919965b08f56678d3bc8d0ac1ed30d1e1e75971c77f4074e256578a6dc545` |
| [协议](#doc-1716ff046240) | `9449b2e62a89c0ec2fb99c98d6a68fa6985bd7221e3ee4ec43ea2bd98bd2d526` |
| [模型](../../../../src/kron_reduce/models/state_port.py) | `099495d11945af25dee3f03780cf9c14692716fc38ab35c3d5d1b8f101e0f3f4` |
| [评估核心](../../../../src/kron_reduce/evaluation/state_port_correctness.py) | `4b9473fa54aba5dcb6101f8c43f3dbbdab85b7ecd43ce1371c7a7b02401dccc8` |
| [独立入口](../../../../src/kron_reduce/workflows/state_port_synthetic.py) | `bc8a7209d4db2989212e182d2e0bc7125f6b95df3fad78af0f503fbb982a303c` |

已读取最终协议、核心代码和守卫，逐字节核对计划的 **26 项实现绑定，全部匹配**。常值与状态端口均使用同一 18 参数映射 `s_C*tanh(Theta)`；状态端口两侧互易，质量 COI 投影与 Helmert 力基一致。能量函数不依赖端口，能量恒等式无需新增端口时间导数；完整状态与参数 Jacobian 保留 `C'(q)` 及参数映射的梯度。零增量基线、两后端、两步长、两档中心差分、短传播梯度与两个隔离负例承担不同检查职责，接受规则与协议相符。

固定调用树为 4 个范围、3 次 probe、162 条 rollout；NumPy/PyTorch RHS 分别为 **69,017 / 9,769**，5 次 Jacobian 对应 110 次向量反传，另有 2 次标量反传。614 次模型构造和 158 次公共 rollout 参数验证合计 2,316 次参数矩阵特征值检查。SVD、expm、LS、真实数组读取、拟合和优化更新均为零。预算覆盖预留、实际调用及返回；门槛失败先保存记录再停止，不重试或转移余额。保存的差分正负 RHS、路径、loss、梯度和能量原量足以进行约定的算术复核。

独立入口先核显式阶段审批，再核 active policy、准确计划及批准收据；Git common directory 中的一次性预留在科学核心执行前消费。实际加载源码路径、SHA 和固定依赖版本受守卫约束。原模型默认路径与已消费授权保持原状。此次审核未执行候选、RHS、梯度或传播，也未读取 NPZ。

工程测试依据协调者和独立工程代理的工具记录：两个测试文件 **12 passed（0.18 s）**，新增三个模块和两个测试文件 Ruff 通过；未另存本地测试日志。本审核读取了测试代码与其准确哈希，没有为补日志重复运行。提交后的 CI 是执行前必须通过并保存的独立工程证据。

运行完成后，允许按计划对本次 `inputs.npz`、`algebra.npz`、`jacobians.npz`、`paths.npz` 各解码一次，并核本次 JSON、完整工件哈希和已保存量的算术。该核验不得调用模型、RHS、积分器、autograd 或新科学算子，不能覆盖原运行失败。

通过的主张限于固定 A1 实现通过本范围的结构与梯度检查。真实数据收益、训练后的优势、误差根因、稳定性证书和模型升级均不由此成立；A1 train 需要独立设计、登记和准确实施审核。结论保持 `KEEP_BASELINE`。

</details>

<a id="doc-21110cda5059"></a>

<details>
<summary>A1-S0 状态端口：结果科学审核</summary>

<a id="doc-21110cda5059-a1-s0-状态端口结果科学审核"></a>

### A1-S0 状态端口：结果科学审核

2026-11-11，Astra Ultra 科学审核。

**有限科学结论接受：`ACCEPT`，`PASS_FROM_SAVED_EVIDENCE`；基线保持 `KEEP_BASELINE`。** 固定常值端口和状态依赖端口实现通过预定的结构、后端、步长与梯度检查，两个隔离错误均被检出。原科学运行在最终汇总序列化时失败，过程状态始终为 `FAIL_PRESERVED`；本结论来自另行批准的保存证据审核，不将原运行改记成功。

唯一数值来源是运行 `state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0`，源提交 `992738b0c4bfbf26e073f40a29f67cea4c4ac429`。其 22 个文件由[外置来源清单](../../../../configs/proposals/state_port_saved_sources_v1.json)封存，SHA256 为 `9c06a82592879750ef9b4a93b9c8d339f5728905dfd24346571eb577f1631841`。[独立保存审核](../../../../results/backup/07_state_port/01_state_port_synthetic_saved_audit.DESCRIPTION.md)的准确 SHA256 为 `7b4b8040622fb003be67243552bde2bae60e28b09d95ed86ca9540b3eaf7f29c`，**200 项检查通过、0 项失配**；共享一次性账本的完成记录绑定同一结果 SHA。

原[协议](#doc-1716ff046240)的全部固定记录齐全：15 条点态记录，4 组正确 Jacobian，14 条 NumPy 与 2 条 PyTorch 长路径，144 条差分短路径和 2 条名义短路径。NumPy/PyTorch RHS 分别为 69,017/9,769，5 次 Jacobian、2 次标量反传，模型构造/公共 rollout 参数验证分别为 614/158；六类 attempted 与 returned 均等于冻结预算。四个范围完整关闭，165 对外层调用及其日志区间哈希通过核验。

保存审核确认的主要结果如下；这些量描述固定合成实现，不是实际电网预测误差。

| 检查 | 最大差异或残差 | 判定 |
| --- | ---: | --- |
| 零增量与原基线路径 | 0 | 通过 1e−12 门槛 |
| NumPy/PyTorch 的 3 s 路径 | 1.7764e−15 | 通过 1e−10 门槛 |
| 两档步长的全部状态路径 | 1.0897e−9 | 通过 1e−6 门槛 |
| 归一化能量恒等式 | 5.8842e−15 | 通过 1e−11 门槛 |
| 质量 COI | 2.6645e−15 | 通过 1e−10 门槛 |
| 完整状态/参数 Jacobian 与中心差分 | 5.1534e−10 | 每组、每档差分均通过原相对与绝对组合门槛 |
| 短传播参数梯度与中心差分 | 2.2735e−11 | 每组、每档差分均通过原组合门槛 |

单侧端口修改留下归一化功率残差 `0.02458575`，超过 `1e−10` 检出门槛；切断角度梯度的反例产生 Jacobian 差异 `0.00898146`，超过对应十倍容差 `9.8726e−5`。因此检查同时确认了正确实现的一致性，以及对这两类错误的识别能力。

保存审核各解码四个获准 NPZ 一次，没有调用模型、RHS、积分器或自动微分；已有差分、loss、能量原量、路径比较和计数均按原定义核对。本次结果审核进一步只读 JSON 与报告，没有再解码数组、重算科学量或重复运行。原失败日志、不完整 summary 和所有原文件保持原样；原 workflow 不推进。序列化工程修复及独立人工回归与本轮科学证据分开记录。

接受的主张是：在已冻结的状态、参数和时间范围内，两种互易端口的实现保留预期结构，完整状态/参数梯度及短传播梯度与有限差分一致。两步长一致性不构成连续 ODE 误差证书，有限点态检查也不构成全域训练可靠性或稳定性证书。结果没有检验真实数据收益、训练后的两组优劣或既有误差根因。[结果报告](#doc-fc70cc3f9d2d)中的数值与上述范围一致。

下一步是独立的 A1 train 配对机制设计：固定共同完整预测初态和动力学骨架，以相同 18 参数预算比较常值端口与状态端口，并保留原 A06 参照。数据范围、训练预算、评价口径和准确实现须另行登记审核。本收据不授权真实读取、训练或模型升级；B 初始化联合训练、A2 势能修改与 C 时间目标保持独立。

</details>

<a id="doc-c5209b6b8905"></a>

<details>
<summary>A1-S0 失败运行：保存证据审核范围</summary>

<a id="doc-c5209b6b8905-a1-s0-失败运行保存证据审核范围"></a>

### A1-S0 失败运行：保存证据审核范围

2026-11-11，Astra Ultra 科学审核。

**批准准备并独立登记一次纯保存证据审核；不重跑 A1-S0。** 本范围用于判定已经完成的科学检查是否得到完整保存证据支持。执行须绑定准确审核脚本和外置失败工件清单，并消费新的只读一次性许可。原科学授权保持 consumed；本文件不批准任何新模型或科学算子调用。

唯一来源为 `results/state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0`。原科学计划 SHA256 为 `2f0919965b08f56678d3bc8d0ac1ed30d1e1e75971c77f4074e256578a6dc545`，源提交为 `992738b0` 所指的准确已批准提交；正式收据绑定完整提交 SHA 和失败工件清单。

本次只读 JSON 审查确认：异常发生在最终 `summary.json` 的 JSON 序列化；文件留下 90 字节不完整内容。`failure.json` 和 `RUN_FAILED.json` 保留 `TypeError`，`failed-operation.json` 的六类逻辑调用 attempted 与 returned 均等于全部冻结预算。setup、algebra、jacobians、paths 四个 journal 范围已关闭；三个科学记录的 `pass` 均为 true。这支持检查现存数值证据，不足以替代独立算术审核，也不改变运行失败事实。

来源清单为 22 个文件、1,305,816 原始字节，包括 16 个 JSON、1 个 JSONL 和 5 个 NPZ。允许生成和核验全部文件的 SHA256；不完整 summary 只按原始字节保存与识别，不补写或作为成功摘要解析。允许解码 `inputs.npz`、`algebra.npz`、`jacobians.npz`、`paths.npz` 各一次，合计压缩文件 1,188,578 字节。`failed-operation.npz` 只核哈希，不解码。原始字节哈希和科学数组解码分开计数；不读取其他运行或真实数据。

审核沿原协议执行以下固定核验：

- 来源文件集合、尺寸、哈希、准确源提交，以及外置失败清单；原目录所有字节保持原样。
- 15 条点态记录、4 组正确 Jacobian、16 条长路径、2 条名义短路径与 144 条差分短路径的集合、形状和有限性；每组中心差分的索引不重不漏。
- 从保存的正负 RHS 重算差分 Jacobian，从保存短路径重算 loss 与参数差分；按原容差比较保存的自动微分结果，核对两个隔离负例。
- 从保存的 energy_rate、supply_minus_dissipation 和分母重算能量残差；核对质量 COI、零增量基线、后端一致性与两步长误差。这里只做已有量的算术，不重新求能量梯度或向量场。
- 核对六类逻辑 attempted/returned、四范围限额和关闭状态、165 对外层调用记录及 journal 字节区间哈希。禁止把未记录反传或失败调用计为零。

模型、RHS、积分器、autograd、SVD、expm、LS、拟合和新路径调用数均为零。检查程序不得导入科学实现以重新生成证据。每个允许的 NPZ 解码一次后在内存内完成检查；异常保存部分审核记录并终止，不自动重试。原运行文件、失败账本和 workflow 状态均不得写入。新清单、审核结果与审核自身账本以新文件写至 `reports/generated/` 等原运行目录之外的注册位置。

结果采用两个独立字段：原运行始终为 `process_status=FAIL_PRESERVED`；全部审核通过时，才可记录 `scientific_checks=PASS_FROM_SAVED_EVIDENCE`，其含义是原协议规定的有限结构与梯度检查得到保存证据支持。若任何数值门槛、来源完整性或预算核对失败，则保留具体失败或未决项，不签科学通过。外置审核不回填 summary，不创建原运行成功凭证，不推进失败 workflow。

后续修复 JSON 布尔类型及使用独立人工 fixture、不重跑 S0 参数矩阵的序列化回归属于独立工程变更，不用于改写本次结果。该回归使用均匀质量、单位刚度与零状态/输入，调用两个点态 RHS，没有调用正式 fixture、轨迹、梯度或执行入口。保存证据审核本身仍为零模型和零 RHS 调用。无论审核结果如何，均保持 `KEEP_BASELINE`；真实 train 比较仍需另行设计和登记。

</details>

<a id="doc-992d4fd27e63"></a>

<details>
<summary>A1-S0 保存证据审核：准确实施批准</summary>

<a id="doc-992d4fd27e63-a1-s0-保存证据审核准确实施批准"></a>

### A1-S0 保存证据审核：准确实施批准

2026-11-11，Astra Ultra 科学审核。

**批准一次独立保存证据审核。** 用户已明确授权自主推进研究；本收据把后续工作限定为已完成数值记录的算术核验。新许可 `state_port_saved_review_v1_20261111` 登记并通过准确绑定守卫后，只执行指定审核脚本一次。原科学许可保持 consumed，原运行继续为 `FAIL_PRESERVED`。

| 对象 | SHA256 |
| --- | --- |
| [审核计划](../../../../configs/proposals/state_port_saved_review_v1.json) | `4f7544efa707ad136a631eae7d0e314d4b11d498e64a0a15f58cc98accde602a` |
| [审核协议](#doc-e881ee12a327) | `99f11d32095ab876292e6c22f4fb2b45707bdb3e03432c3d10e45778b98364f3` |
| [外置来源清单](../../../../configs/proposals/state_port_saved_sources_v1.json) | `9c06a82592879750ef9b4a93b9c8d339f5728905dfd24346571eb577f1631841` |
| [审核脚本](../../../../scripts/audit_state_port_saved.py) | `b337cd31bec4457a731c467947387ec4ea62b1b033604f015b6b47ba8afae5ed` |
| [原科学计划](../../../../configs/proposals/state_port_synthetic_v1.json) | `2f0919965b08f56678d3bc8d0ac1ed30d1e1e75971c77f4074e256578a6dc545` |

已只读核对这些准确文件、审核算术和一次性入口。唯一来源运行是 `state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0`，源提交 `992738b0c4bfbf26e073f40a29f67cea4c4ac429`。来源清单固定 22 文件、1,305,816 字节；文件集合、尺寸与 SHA 在数组解码前核验。指定四个 NPZ 各解码一次，共 1,188,578 压缩字节；另一个失败诊断 NPZ 只核哈希。审核不会导入模型或科学执行模块，也没有模型、RHS、积分、求导、谱分解、LS 或拟合调用。

审核复算原阈值下的差分 Jacobian、路径 loss、传播参数梯度、能量原量归一和路径比较，核对固定记录集合、有限性、两个隔离负例、完整调用预算与日志区间哈希。日志的 `rollout` 类别在局部范围计数中明确映射为账本限额的 `rollouts`；这项静态审查发现的口径错误已在任何真实证据解码前修正。原 JSON 描述确认 16 条长路径均为 301×22，并有 14 组 NumPy 路径能量原量。

入口核 active policy、准确计划、脚本、来源、协议、原计划与批准收据，然后在共享 Git 目录创建排他的已消费预留。新结果以排他写入方式保存至原运行之外；异常保留审核失败账本，不重试。原运行文件、90 字节不完整 summary、原失败账本及 workflow 均不写入。

工程依据为协调者报告的三个测试文件 **18 passed（0.45 s）** 及 Ruff 通过。审核器测试使用临时小数组、授权拒绝情形和人工 330 行日志；165 对 attempted/returned、范围哈希与单复数类别映射均通过，未读取本次运行 NPZ。另一个序列化回归使用独立人工参数，调用两个点态 RHS，没有重跑 S0 参数矩阵；它与本次零科学调用的保存审核分开记录。本准确审核自身只读代码、JSON 和绑定文件哈希，没有解码 NPZ 或调用科学算子。

只有全部保存算术、来源和预算检查通过，才可在外置报告记 `scientific_checks=PASS_FROM_SAVED_EVIDENCE`；否则记录失败或未决。该字段不覆盖 `process_status=FAIL_PRESERVED`，不产生原运行成功摘要或成功 stage evidence。科学主张仍限于 A1-S0 的固定结构与梯度检查，结论保持 `KEEP_BASELINE`。

</details>

## 回执与附件

- [state_port_synthetic_20261111_evidence.json](_sealed/state_port_synthetic_20261111_evidence.json)
- [state_port_execution_review_20261111.json](_sealed/state_port_execution_review_20261111.json)
- [state_port_result_review_20261111.json](_sealed/state_port_result_review_20261111.json)
- [state_port_saved_review_approval_20261111.json](_sealed/state_port_saved_review_approval_20261111.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/state_port_saved_review_20261111_zh.md](_sealed/state_port_saved_review_20261111_zh.md)
- [docs/protocols/state_port_synthetic_v1_zh.md](_sealed/state_port_synthetic_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/research/state_port_result_review_20261111_zh.md](_sealed/original/state_port_result_review_20261111_zh.md)
- [docs/reports/07_state_port/_sealed/state_port_synthetic_20261111_zh.md](_sealed/original/state_port_synthetic_20261111_zh.md)
- [docs/reports/07_state_port/state_port_synthetic_20261111_zh.md](_sealed/state_port_synthetic_20261111_zh.md)
- [docs/research/state_port_execution_review_20261111_zh.md](_sealed/state_port_execution_review_20261111_zh.md)
- [docs/research/state_port_result_review_20261111_zh.md](_sealed/state_port_result_review_20261111_zh.md)
- [docs/research/state_port_saved_recovery_scope_review_20261111_zh.md](_sealed/state_port_saved_recovery_scope_review_20261111_zh.md)
- [docs/research/state_port_saved_review_approval_20261111_zh.md](_sealed/state_port_saved_review_approval_20261111_zh.md)

</details>
