# A14：固定模型的积分初态将历史重建误差降低约一半

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/04_kiki_a03_a17_pipeline/KIKI_a14.DESCRIPTION.md)

登记文件与配置

[方案登记](../../../../configs/proposals/kiki_a14_integral_history_reads.json)

## 研究方案

<a id="doc-3727f048e593"></a>

<a id="doc-3727f048e593-a14固定模型的积分隐藏初态与闭环历史机制审计"></a>

### A14：固定模型的积分隐藏初态与闭环历史机制审计

2026-10-24，Astra Ultra。状态：科学方案冻结草案，**尚无真实执行授权**。A12 需先完成并经独立审核；A14 还需独立登记、实现审核、固定源码与单次预算。不得复用任何已消费授权。本方案不改变 A13 已冻结的 seed、人口、算法、数值阈值或 V/I 条件。


<a id="doc-3727f048e593-1-唯一问题"></a>

#### 1. 唯一问题

在固定 A06 step240 的三个二阶隐藏模型上，利用原 record100ms 的 11 个真实 q/v 点与原功率输入策略，能否构造数值有效的隐藏起点估计，并改善闭环历史重建？只替换各段历史的条件初态；共享骨架、隐藏动力学、encoder、质量、尺度与所有参数保持原样。

唯一对照是已经封存的原 encoder 闭环历史；唯一候选是四点 Gauss 求积所得积分最小二乘初态。二点求积只作数值检查。新参数 0、拟合 0、优化更新 0、未来预测/未来评分 0、标签/LOS 计算 0。不测试 20ms 初始化，不读另外 10 条不合格事件的原始数组，不比较候选网格。

该问题由 A13 固定合成检验支持：15 个病例及物理检查通过，反例同时证明满秩、小残差、被动性不能单独保证真实隐藏状态恢复。A13 方案 SHA 为 `d9462c02244032bf1a130502c7bb11a8851ef29beccfcbef85ad757fa54b472a`，结果审查 SHA 为 `05c89aec4c84c4f4d446ba1bb0d3b4bc7fe39e431d11a1880b52d6471c356fad`；已存数组独立审计 SHA 为 `465212171daf54cfc92353d25fcdc18fa400b39fee3242067c2a7a9d03983141`。后者只复算保留证据支持的部分，不能称全部 15 病例完整算子重放。


<a id="doc-3727f048e593-2-来源人口与顺序"></a>

#### 2. 来源、人口与顺序

源模型固定为 `paper_kiki_a06_refinement-joint_refinement-20261024T025925447927Z-9aa5ee65-524be996` 中 step240、seed 324/325/326。源 manifest SHA 为 `d1c4cc402dd30207c4ba639a9d4d1f1e86e3051ce0ce51e72ac4d045be17cc01`。

| checkpoint 相对路径 | SHA256 |
|---|---|
| `checkpoint-240-324/model.npz` | `a7f094d52da465b0fe8214e7e74fe408594dbe06efe78e659163a469351623b8` |
| `checkpoint-240-325/model.npz` | `14d58fd01f438230bd9aa05a5305e07bf5595f5a589d875f08d4a420428450b2` |
| `checkpoint-240-326/model.npz` | `2ab76998e1295b9fb456de111cd8a2d5ae7bcbe9e8a8c8dac770abb639e6b5bd` |

原 allocation 固定为 `paper_kiki_p2_a03_seed324-kiki_rerun-20261022T095918Z-21a5a23b/role_allocation.json`，SHA `b103651cd33cb60dbdbad71c9d258545602cd6070a8860becc4fe9b2a6ac2b6d`。仅其角色/身份/已声明长度元数据用于恢复原 268 条 train：N≥12 的 258 条参加历史审计，其余 10 条固定记录为旧合同不合格；不补样、不改角色。

缓存行顺序按照原 provider：来源字典序，再保持 allocation 中该来源行的原顺序，筛选 role=train、source_role=train、N≥12。不是按本轮误差、标签或新规则排序。登记时由这些已绑定元数据导出完整 268 槽清单、258 行缓存索引和 774 个 seed/事件槽；绑定其 SHA 后才能读取缓存。source/line/ID 仅用于对应与汇总，不进入估计器。

原权重保持
\[
w_i=\frac1{N_{source}}\frac1{N_{line(source_i)}}
\frac1{N_{event(source_i,line_i)}},\qquad\sum_{i=1}^{258}w_i=1.
\]
物理线路跨来源仍按原层次计算。三个 seed 是同一人口上的算法重复，不能当成 774 个独立真实样本，也不产生置信区间。


<a id="doc-3727f048e593-3-已核验的缓存布局与许可读取"></a>

#### 3. 已核验的缓存布局与许可读取

这里的布局结论来自保存源码；尚未解码真实成员检验其内容。A09/A10 运行分别为：

- A09：`paper_kiki_a09_train_mechanism-train_mechanism-20261024T054046076318Z-ff5ae772-e3d8db5f`，manifest SHA `ae87ac95d337a9c8a1d4fdadd305701190e704cf9a85923f0ab9e27dbbfd4573`。
- A10：`paper_kiki_a10_train_refinement-train_refinement-20261024T062534123221Z-161cae52-515f5625`，manifest SHA `41741c4a3ab0973be2053cb23e29efd06ceaac3af02239e72f2cda140dccbc86`。

| 文件（相对对应 run） | 唯一许可成员 | shape；类型 |
|---|---|---|
| A10 `real_train_arrays.npz` | `history` | (258,11,20)；float64；真实 [q,v] |
| 同上 | `history_time` | (258,11)；float64；末点为零的原浮点时钟 |
| 同上 | `constant_power`、`equilibrium` | 各 (258,10)；float64 |
| 同上 | `mass`、`target_scale` | (10,)、(20,)；float64 |
| A09 `seed-{324,325,326}-h005/paths.npz` | 各文件仅 `history` | 各 (258,11,22)；float64；旧闭环 [q,v,z] |
| A10 `seed-{324,325,326}-h0025/paths.npz` | 各文件仅 `history` | 各 (258,11,22)；float64；旧闭环 [q,v,z] |
| A06 三个 `model.npz` | `mass,edge_stiffness,damping,coupling,latent_skew,latent_dissipation,encoder,feature_scale,target_scale` | 依次 (10,)、(45,)、(10,10)、(10,2)、(2,2)、(2,2)、(2,80)、(80,)、(20,)；float64 |
| 同上 | `metadata` | 三个零维 Unicode 字符串；schema/settings/fit_record/backbone 身份 |

A10:183–200、271 明确以独立关键字保存六个真值输入成员；A09:356 与 A10:340 分开保存 `history`、`future`、`offsets`。因此许可历史可单独解码，**不能先解码整段或全部 NPZ 再切片**。A09 的真实输入缓存缺独立 `history_time` 与 `constant_power`，本轮不使用它来补齐输入。

所有 NPZ 以 allow_pickle=False 按逐成员白名单读取，每个成员最多一次；模型加载不能在显式预读后再调用旧 load 重复解码。禁止 `.values()`、`dict(np.load(...))` 或成员遍历解码。只可枚举 ZIP/NPY 头部核对名字/shape/dtype，未许可成员只识别名字，不能解压其数组值。输入成员形状、类型、有限性或数值先验不合格则停止新科学运算；不回 HDF 修补。

明确禁止：真实输入文件的 `future/forecast_time/offsets/omega_s`，模型路径文件的 `future/offsets`，旧 `event_diagnostics.json/audit_summary.json/refinement_checks.json`、评分和 LOS 子树，任何 A11 结果、标签、内部 val、source val/test、HDF、CSV、splits。源 manifest 可读取文件名与 SHA 元数据；源物理证书只校验原字节 SHA，不解码其中评分或未来统计。

`source_inventory_metadata.json` 的 SHA 为 **`82ade950c1475bb1781997491e409413404e9635e3f9b0e52caead2d7dbe586c`**，逐项列出上述十个 NPZ 的原清单 SHA 和 stat 大小。生成程序 SHA 为 `74da4dbf15f7ade7c13c9bb3d9e09d2ac2cd7ff7ec9f4df74394fc512aeaf786`。该登记步骤只读三份已封存 manifest 并 stat 文件，NPZ 成员解码、NPZ 内容 hash、新模型计算均为 0。


<a id="doc-3727f048e593-31-缓存读取预算"></a>

##### 3.1 缓存读取预算

十个 NPZ 合计 87,732,555 字节；运行前后各顺序校验一次，固定 **20 次原文件字节校验、175,465,110 字节**，与逻辑成员解码分账。校验不得多做 EOF 探测或遇失败重试。三个源 manifest、allocation、登记文档与证书的元数据完整性读取另外逐文件记账，不混作数组解码。

最多打开十个 NPZ 解码上下文。六个真值成员共 64,788 个 float64，六份旧模型 history 共 374,616 个，三个 checkpoint 数值数组共 1,329 个：**39 个数值成员、440,733 个 float64、3,525,864 个逻辑字节**。另有三个 metadata 文本成员，总成员数 42。文本按固定来源 SHA 单列实际字符/解码字节，不把它算成数值元素；单个只允许零维 Unicode，字符数硬上限 1,048,576，禁止 object/pickle。此上限是格式读取防护，不是调参；超界拒绝本轮，不截断文本或重读其他版本。


<a id="doc-3727f048e593-32-旧物理证据的继承"></a>

##### 3.2 旧物理证据的继承

源物理资格只继承已归档 A10 科学审核 `docs/reports/04_kiki_a03_a17_pipeline/a10/_sealed/kiki_a10_astra_result_review_20261024_zh.md`，SHA `43f3f6f5add1f2d5e3bf46fc53c9ccc202aa4c2fcb2453aea03090902756313b`，及独立收据 `docs/reports/04_kiki_a03_a17_pipeline/a10/_sealed/kiki_a10_train_refinement_20261024_evidence.json`，SHA `1337b2f8c4cc9b8827a38a37cf47a15c916ed4a66cce05ca61314d4a1bb235ab`。两文件分别 7,901、23,492 字节；各前后一次字节校验共 62,786 字节，不解析其诊断正文。

A09 `path_physics` 对 history+future 联合取最大误差，A10 复用同函数；已通过的联合上界蕴含其中 history 的上界。这里准确称为继承既有证据，不称新历史独立物理检查，也不额外调用旧路径 energy_identity。A09 原 .01/.005 精化失败仍保留；本轮依赖 A10 完成的 .005/.0025 验证，不改写 A09 结论。


<a id="doc-3727f048e593-4-输入策略和前置核对"></a>

#### 4. 输入策略和前置核对

真实历史是记录首点后名义 100ms，实际输入 `history_time=clock[:11]-clock[10]`；传播用该数组减去首值，保留每段浮点间隔。不是物理清除时刻，也不从 forecast_time 重建时钟。各区间以原 `ceil(h/max_step−1e−12)` 规则核对两个最大步长分别为 20/40 次历史更新；不满足便在新力、指数和 RHS 之前停止。

Pm[10] 派生的 `constant_power` 在整个模型历史里保持常值，这是已声明的输入策略，**不证明原始机械功率过去 100ms 实际恒定**。积分方法可以把这项输入误差吸收到隐藏初态；A13 S10 已构造零残差但初态错误的反例。因此下述残差或历史改善不得称真实物理隐藏状态恢复。

三个 checkpoint 必须为 n=10、r=2、sine、frames=3、原 encoder_mode；九数组及 metadata 来源 SHA 固定。质量和 target_scale 与 A10 六成员缓存逐数组相同，三个源模型各自 feature_scale 不重估。全部数组有限，质量与尺度为正；保留原结构校验。

按原 `_features`/encoder 仅做每槽一次特征和线性映射：`z_enc=E*(features/feature_scale)`，共 774 次。两份对应缓存 history 的首个隐藏状态与它按绝对 1e−12 校验；可见初态按原 `_state`/COI 投影规则核对。不能重跑 encoder 历史来“核实缓存”。两份旧历史的可见归一化步长差、隐藏相对步长差可用已许可成员直接计算，沿用第 8 节阈值；不调用 RHS。

对照只复用旧历史；候选从同一真实首点、唯一估计初态和同一常功率开始，使用原闭环 `closure.rollout`。不把观测 v 插入闭环中间时刻，不调用会重新覆盖 z 的原 `model.initialize`。结束后可另存 `[q_obs(T),v_obs(T),z_cl(T)]` 重置状态，q/v 必须逐数组等于最后真实观测、z 必须等于各自闭环末态；只组装和保存，不运行未来。


<a id="doc-3727f048e593-5-唯一估计方程"></a>

#### 5. 唯一估计方程

令 \(M=diag(m)\)、\(Q=I-\mathbf1m^T/(\mathbf1^Tm)\)、\(P=Q^T\)、\(A=J-R\)。保留源方程
\[
\dot q=v,\quad M\dot v=P[p-F(q)-Dv-Cz],\quad\dot z=C^Tv+Az.
\]
这里 \(F(q)=B\operatorname{diag}(k)[\sin(B^Tq+B^Tq_{eq})-\sin(B^Tq_{eq})]\)，B 为源模型同序 complete incidence；平衡角 \(q_{eq}\) 沿用许可静态先验，不把位移 q 当绝对角。两项相位分别计算并相加，保持源 RHS 的运算次序。

数学时钟明确为 \(\tau_j=history\_time[j]-history\_time[0]\)、\(\tau_0=0\)、\(h_j=\tau_{j+1}-\tau_j\)、\(T=\tau_{10}\)、\(I_j=[\tau_j,\tau_{j+1}]\)。不得将末点归零的负历史时钟直接代入矩阵指数。q/v 在实际历史点间分别线性插值，\(\Phi(\tau)=e^{A\tau}\)、\(w'=Aw+C^Tv_{obs},w(0)=0\)。逐区间定义
\[
g_j=\int_{I_j}P[p-F(q_{obs})-Dv_{obs}]dt-M(v_{j+1}-v_j),
\quad b_j=g_j-PC\int_{I_j}w(t)dt,
\quad L_j=PC\Phi(\tau_j)\int_0^{h_j}e^{Au}du.
\]
唯一候选求 \(\hat z_h=\arg\min_z\|\widetilde Lz-\widetilde b^{(4)}\|_2^2\)，其中
\[
S=diag(1/(m_i s_{v,i})),\quad c_j=\sqrt{h_j/T}/h_j,
\quad\widetilde L_j=c_jSL_j,\quad\widetilde b_j=c_jSb_j.
\]
\(s_v\) 为冻结 target_scale 的速度部分，不重新归一化。

每区间唯一 8×8 增广矩阵指数为 \(\exp(h_j\mathcal A)\)，其第一块行为 \([\Phi_h,F_1,F_2,F_3]\)，增广矩阵 \(\mathcal A\) 为
\[
\begin{pmatrix}A&I&0&0\\0&0&I&0\\0&0&0&I\\0&0&0&0\end{pmatrix}.
\]
\(u_j=C^Tv_j,d_j=C^T(v_{j+1}-v_j)/h_j\) 时，
\[
w_{j+1}=\Phi_hw_j+F_1u_j+F_2d_j,
\qquad\int_{I_j}w=F_1w_j+F_2u_j+F_3d_j.
\]
\(\Phi(\tau_0)=I\)，后续前缀按 \(\Phi(\tau_{j+1})=\Phi_{h_j}\Phi(\tau_j)\) 复用上面已算块，不增加指数调用。不使用 A 的逆或伪逆，不按 A 大小切换公式。正弦力固定二点与四点 Gauss–Legendre，阻尼用线性插值解析积分，功率用常值积分；四点是主解，二点只作诊断。

每槽一次薄 SVD，同一分解解两个右端，无 ridge。数值秩阈值仍为
\[
64\epsilon_{64}\max(100,2)\max\{\sigma_1(\widetilde L),\|SPC\|_2\}.
\]
\(\|SPC\|_2\) 用 A13 已验证的 2×2 Gram 闭式计算，不暗加 SVD。每 seed 的 PC 与 \([PC;.01PCA]\) 各只分解一次，投影/静态分解缓存供 258 槽复用；原合成内核若逐槽分解 C，必须在独立实现中显式改为传入这些缓存并做合成等价检验，不能沿用旧调用次数后超预算。任何 rank<2 的槽标 unresolved，不传播最小范数解，不退回 encoder 冒充候选成功。


<a id="doc-3727f048e593-6-必须分开的诊断"></a>

#### 6. 必须分开的诊断

1. 积分力空间残差 \(r_{force,j}=(I-PC(PC)^\dagger)g_j\)，是动量单位；另报 \(r_{force,j}/h_j\) 的平均力单位。投影先在物理空间完成，之后才计算 S 归一化值。
2. 动态一致性残差 \(r_{dyn}=b-L\hat z_h\)。即使各区间的 \(b_j\) 都在 range(PC) 内，也未必存在同一初态解释全部区间。一般有 \((I-\Pi_{PC})r_{dyn,j}=r_{force,j}\)，两项不是独立或正交的误差分解；不能把它们相加作为总误差。该最小二乘目标的降低是构造结果。
3. 运动学残差 \(r_{kin,j}=q_{j+1}-q_j-h_j(v_j+v_{j+1})/2\)，分别保存 rad 与除以 s_q 的量，不归因给隐藏状态。
4. 全部奇异值、固定阈值/rank、条件数及绝对增益 \(1/\sigma_{min}\)。已知固定 L 时的 \(\|\Delta z\|\le\|\Delta\widetilde b\|/\sigma_{min}\) 只作条件敏感度；没有真实噪声及骨架误差上界，不签发真实误差证书。
5. 两求积初态与 \(z_T^{obs}=\Phi(T)\hat z_h+w(T)\)、两个步长闭环末态、彼此差值、\(\|z_h\|^2/2\) 与相对 encoder 储能。分母零的比值为 None，不填 0 或无限大的 JSON 数值。不得把观测驱动末态替换为候选闭环起报状态。

保存逐区间物理值、归一化值与最大值；汇总采用同一事件权重/全体分母。可对 force/dynamic 以 \(n^{-1}\sum_j\|c_jSr_j\|^2\) 汇总，对 kinematic 以 \(n^{-1}\sum_j(h_j/T)\|r_{kin,j}/s_q\|^2\) 汇总；这些是描述统计，不新增选择门槛。


<a id="doc-3727f048e593-7-固定新计算预算"></a>

#### 7. 固定新计算预算

按 seed 324、325、326 顺序，每个 seed 按固定缓存行顺序处理全 258 条；每槽先共同矩阵/两个右端，再 .005、.0025 历史。高层源模型加载/构造各 3 次；原 rollout 内的参数校验器依原实现保留，不构成新拟合模型。

| 项目 | 全成功计划值/硬上限 |
|---|---:|
| seed/事件槽 | 774 |
| 新模型参数、参数拟合、优化更新 | 0/0/0 |
| 原 encoder 特征与线性映射 | 各 774 |
| 8×8 增广 expm | 7,740 |
| 独立正弦力求值（Gauss2+Gauss4） | 46,440 |
| L 薄 SVD | 774 |
| PC / [PC;.01PCA] 静态 SVD | 各 3，总显式 SVD 780 |
| 复用 SVD 的两右端求解 | 1,548 |
| 新候选闭环历史路径 | 1,548 |
| 新 RK4 完整状态更新 | 46,440 |
| 历史积分 RHS | 185,760 |
| 新历史物理 RHS（每路径 11 输出点） | 17,028 |
| **新真实 RHS** | **202,788** |
| 新未来路径、未来评分、标签/LOS | **0/0/0** |

独立 force 计数不含已计 RHS 内部的力表达式；原 energy_identity 另外计算势能梯度并调用一次 RHS。这些内部运算保持源实现，不能将独立 force/RHS 两列误称所有标量正弦运算数量。每次物理检查只调用原 energy_identity 一次。

所有预算在调用前记录 attempted，返回后另记 returned；每条独立 rollout/物理段都有上下文。仅正常完成的 rollout 才记完成更新数，失败后的 returned//4 只能叫返回四阶段组，不能冒充成功状态更新。数值失败保留已耗算子量，预留槽不转给重试；正常情况下上述值应精确一致，失败时实际消耗较少并保留未尝试项。


<a id="doc-3727f048e593-8-事前接受条件"></a>

#### 8. 事前接受条件

**V：数值与计算有效性。** 来源、成员白名单、全人口对应、原权重、身份和预算全部通过；774 槽均完成且 L 的 rank=2；两求积的初态差及观测驱动末态差分别不超过该量的 \(10^{-6}(1+\|z^{(4)}\|)\)。两步长历史的可见状态除以 target_scale 后的最大绝对差≤1e−5；隐藏状态差≤\(10^{-5}(1+\max_t\|z^{(.0025)}(t)\|)\)。候选每输出点原能量恒等式**绝对**误差≤1e−8、质量加权 q/v COI 绝对残差≤1e−10。旧对照继承冻结物理证书，许可 history 成员自身的有限性/初态/步长核对必须通过。任何失败/缺失/非法访问使 V=false。

**I：闭环历史重建收益。** 对两个步长、三个 seed 各自计算非初始 10 点的单事件值
\[
\ell_i=\frac1{10\cdot20}\sum_{j=1}^{10}\|
([q^{cl}_{ij},v^{cl}_{ij}]-[q^{obs}_{ij},v^{obs}_{ij}])/s\|_2^2.
\]
q/v 分通道分母分别为 10×10。候选相对缓存 encoder 的 \(\sum_iw_i\ell_i\) 在全部六个 seed/步长组合严格下降，两个分通道均不增，原权重最差 10% 的 CVaR 均不增。CVaR 按单事件 \(\ell_i\) 降序取质量 0.1，边界事件只取所需分数质量，最后除以 0.1；同值并列顺序按冻结行号，无标签参与。严格下降使用直接 `<`，不追加事后容差或删除异常事件。

这里 I 独立于 LS 的构造目标，但仍使用同一段已观测历史，不能称未见未来泛化证据。失败时总体值及 I 标未完成/false，不对成功子集重新归一化后宣称改善。

执行完成、V、I 与后续授权分开保存：完整结果若 V∧I 则本轮机制问题 `ACCEPT`，否则 `REJECT`。V∧I 只支持提出一项新的、有限的未来初始化比较，科学审核还须检查残差分解与储能是否暴露骨架/输入混淆；不自动批准下一轮。始终 `new_training_authorized=false`、`future_evaluation_authorized=false`、`model_promoted=false`、`promotion_decision=KEEP_BASELINE`、`certified=false`。


<a id="doc-3727f048e593-9-失败保存入口与实施检查"></a>

#### 9. 失败保存、入口与实施检查

采用新的唯一 AUTH 和 Git 共用一次性账本，在任何科学输入 NPZ 字节校验/成员解码前原子预留；不得打开 A12 或 A06–A11 旧预算。专用 CLI 只做登记元数据/源码/干净 HEAD/流程锁检查后进入不可变 runner 内层；禁止通用 verify_data、原 provider 或旧 A09/A10 工作流入口，因为它们会扩大原始/未来读取。通用入口必须对该特定受限执行模式提前拒绝。

全 774 槽自开始存在。来源或读取权限/保存/预算错误是整轮停止；已经发生的操作和其余 not_attempted 槽保留。局部 rank/矩阵/数值问题不更换方案，记录 unresolved/failed 并继续其他独立预定槽；任一失败使 V=false。所有局部矩阵、两右端、SVD、状态估计、残差、缓存对照历史、新候选两步长历史及组装 reset 状态都保存为独立成员；为避免 A13 证据缺口，本轮不得只保存检查布尔值。保留源数组指纹、全部访问顺序/算子/RHS 账本，支持零新 RHS 的独立回放。

实施前合成检查必须覆盖：真实 future 成员一旦被解码便报错的 NPZ 读白名单、源哈希/成员缺失和全人口顺序错误；PC 静态 SVD 缓存与原合成内核等价且计数为6；10段非完全相同浮点间隔；rank不足不传播；算子或RHS失败不重试/不假记完整更新；错误动力学与输入吸收反例；同源 encoder 起点与闭环末态/可见 reset；成功与 REJECT 产物都满足严格 artifact/stage contract。没有这些工程与科学复核，不启动真实审计。


<a id="doc-3727f048e593-10-后续解释边界"></a>

#### 10. 后续解释边界

V 失败先解决数值、来源或条件性问题；不能调整 ridge、加采样点或换步长继续本轮。V 通过而 I 不成立，就拒绝本次积分初态的闭环历史收益假设，不按同一 val 结果继续加训练轮数或换温度。V∧I 也不等于真实 220 状态恢复，更不等于 FN 改善或全 27 失稳覆盖。

若力空间外残差占主要部分，不能只归因于 encoder；若动态一致性残差明显但力空间残差小，固定 A/C、输入近似和骨架误差仍需区分。状态依赖 C(q) 只有在这些解释被独立排查、状态与残差关系在事前保留的 train 线路上重现后，才可成为另一个科学假设；相关性不是因果证据。本轮不估计 C(q)、不选择基函数、不增加隐藏阶数。


<a id="doc-3727f048e593-源码依据"></a>

#### 源码依据

- A09 保存/物理检查：`src/kron_reduce/workflows/kiki_train_mechanism.py`，SHA `4becae0bf8a651362b45f4e17d9fd3a2762cfd07ece286ccb4991f8ee4336a64`。
- A10 独立历史成员与输入扩充：`src/kron_reduce/workflows/kiki_train_refinement.py`，SHA `ff068dca8c096fb0c46980d56dcee0c30cb11fbc6c818c9e681dc6ea9504ec2b`。
- 原 record100ms 输入变换：`src/kron_reduce/data/kiki_history.py`，SHA `c9bed1a04f6c924976a3799cfb9cc5e558954c61c8d97bd0e76eb726e2ab715a`。
- 原非线性模型/RK4：`src/kron_reduce/models/nonlinear_latent.py`，SHA `766991fa340f5d4176694fc5ad78eded074cf0e906d0c944b49057cbd26697c4`。

本方案的推导与文献适用条件沿用 A13：积分形式避免显式求导的原始方法背景见 [Dattner–Klaassen](https://doi.org/10.1214/15-EJS1053)，结构系统初态校准见 [Günther–Jacob–Totzeck](https://arxiv.org/abs/2301.02019)，可观测性与采样边界见 [Hermann–Krener](https://doi.org/10.1109/TAC.1977.1101601) 和 [Zeng 等](https://arxiv.org/abs/2204.14021)。这些文献不替本方案证明真实 kiki 的隐藏初态可恢复。

<a id="doc-255db39467d5"></a>

<a id="doc-255db39467d5-a14-数学模块接口与合成验证"></a>

### A14 数学模块：接口与合成验证

2026-10-24，Astra Ultra。模块位于 `integral_initializer_kernel.py`，SHA256 `5bc7ba32a9a6646dbb119810f0e67f90f11cff20e6deaf356a43424517c59182`。它只接收调用者给定的数值数组和固定 closure，不含数据读取、模型加载、拟合、RHS 或 rollout。真实执行仍须独立登记和整条实现审核；本记录不授予执行权限。


<a id="doc-255db39467d5-固定接口"></a>

#### 固定接口

`prepare_static(closure, velocity_scale, *, owner_key, ledger) -> StaticCache`：要求 n=10、r=2、sine，速度尺度为十维正数。`owner_key` 是模型/seed 身份字符串，不参与方程。返回冻结的模型、尺度与派生矩阵缓存，恰执行 PC 和 `[PC;.01PCA]` 两次 SVD。`cache.data` 是只读 mapping，保存 P/Q/A/PC/S、O、两个 SVD、投影矩阵和 incidence；`model_sha256` 绑定全部六个 closure 数值数组、backbone 与 velocity_scale，`payload_sha256` 绑定缓存全部内容。每次使用前重新检查 owner、模型数组指纹和缓存内容指纹，拒绝跨模型、跨 owner 或被修改缓存。

`build(q, v, time, closure, p, eq, *, cache, owner_key, ledger) -> ArrayRecord`：q/v 为 (11,10)，time 为 (11,)，p/eq 为 (10,)。time 减首值形成数学时钟，保留每段实际浮点间隔。输出沿用 A13 的 L/Ltilde、两套 g/b/btilde、SVD、zh/zTobs、force/dynamic/kinematic residual 等名字；额外返回 `candidate_zh`、静态绑定、逐区间平均力残差。缓存元数据与结果中的嵌套 mapping 应由保存器递归展开，所有数值数组独立保存；不要把整个 mapping 当 object 数组写入 NPZ。

成功事件恰执行 10 次 8×8 expm、1 次 L-SVD、60 次独立正弦力求值和 2 次复用 SVD 的 LS。无 A 逆、ridge、可替换投影入口、额外 sampling 或自选阈值。PC 的 rank 仅用于投影/诊断，不是接受门槛；PC rank=1 时动态 L 仍可能 rank=2。

L rank<2 时 `status='unresolved'`，矩阵、两个右端、力空间残差与运动学残差保留；`zh2/zh4/zTobs2/zTobs4/candidate_zh`、动态残差和隐藏能量为 None，LS 调用为 0。调用者必须把该槽两个候选 rollout 标成 not_attempted，V=false、I 未完成；不能传播最小范数解或回退 encoder。该情况的 10 expm、1 L-SVD、60 force 已消费，其余预定槽可以继续。


<a id="doc-255db39467d5-算子计数与失败语义"></a>

#### 算子计数与失败语义

`Ledger(limits=None, event_sink=None, context='')` 可由工作流传入类别上限和持久日志回调。类别固定为 expm/svd/force/ls。`attempted` 在日志及调用之前登记，`dispatched` 在 attempted 日志成功后、真实 callable 前登记，`returned` 记录实际返回，`failed` 记录 callable 异常；三者在正常完成时相同。日志失败前已预留的 attempted 不能冒称实际 dispatched。

event_sink 在每个 attempted/returned/failed 转换被调用。日志失败抛 `OperatorJournalError`；预算拒绝抛 `OperatorBudgetError`；两者将账本标记 poisoned，后续算子均拒绝。工作流须整轮停止并保留已耗与未尝试槽，不把这类错误当局部数值问题继续。一般数值算子失败保留调用计数和错误，是否继续其他独立槽按冻结协议处理。模型/缓存身份错误属于来源完整性问题，应停止，不能换缓存重试。


<a id="doc-255db39467d5-已完成的合成正确性证据"></a>

#### 已完成的合成正确性证据

6 项小型测试通过，ruff 通过，未重复 A13 的 15 病例运行。测试覆盖：A13 与 A14 的 L/g/b/初末态/残差逐项在绝对 1e−12 内一致；负起点、十个非均匀间隔原样使用；静态缓存复用与精确算子数；跨 owner/改变模型在新算子前拒绝；rank 不足无候选、无 LS；PC rank1 而动态 L rank2 的有效正例；日志前后失败及预算拒绝均不重试。

这次持久验证实际完成的纯合成算子总数为：60 expm、15 SVD、10 LS、360 独立 force。该数包括测试中一次 A13 参照调用及 A14 小型调用；用于日志失败检查的两个普通 Python callable 探针单列，不算数学力求值。真实数据读取、模型加载、RHS、rollout 均为 0。

验证目录：`kernel-validation-20261024T111429513988Z/`。`receipt.json` SHA256 为 `e805c9998e63358540303db7ca32fccc5fe8ecab0af01c2e4725fdd522c97650`，包含全部源码/日志 SHA，状态 PASS。`synthetic_operator_ledgers.json` 保存每条算子记录，SHA256 `a1bcbaa30eb8d1a2c60070709780dc2d2286a6076fcd5c90de0a53519ec87025`。

测试 `test_integral_initializer_kernel.py` SHA256 `19ae89f09433ae9ebaa881ff9d2864abc24051cf80269112f6663896fea6dd33`；持久验证脚本 `validate_kernel.py` SHA256 `f7562c2bdb11d0796d63c5f9008bd12e8e861e9f173a1a65db2b0f77a903f475`。冻结 A13 参照模块 SHA 保持 `e9a1060f8a06f57e8ca4669f5010f437e9ff4791073cafd5aad28be827d86e2a`，未修改其方案、运行或结果。

这证明 A14 数学模块正确复现冻结的估计方程及数值拒绝边界。它不证明真实 kiki 初态可恢复，也不证明闭环历史或未来轨迹会改善；这些必须由独立 A14 及后续明确授权的证据分别回答。

## 执行协议

<a id="doc-485585d3d0c3"></a>

<a id="doc-485585d3d0c3-a14-有界历史机制比较授权"></a>

### A14 有界历史机制比较授权

用户持续指令为“继续改进算法和设计，一定要降到0，持续工作，得到最终结果后向我汇报”，并要求核心算法和科学接受判断由 Astra Ultra 完成。依据这项持续授权，登记一次 A14；它不复用、重启或增加任何已消费预算。

仅固定 A06 step240、seed324/325/326，以及原 source train 中 record100ms 合格的258条历史缓存；全268条已分配事件保留。唯一候选为冻结四点 Gauss 积分隐藏初态，原 encoder 缓存为对照。二点求积与两种固定步长仅作数值检查。774个seed/事件槽，零拟合和优化更新，零未来路径/评分/标签/LOS，内部val及source val/test关闭。

读取仅限登记的10个NPZ、42个成员。运行前后各一次完整字节校验，175465110字节；39个数值成员共440733个float64，另3个Unicode标量。源物理资格只继承A10冻结证书的字节身份，不解析旧诊断。

计算硬上限：7740 expm、780 SVD、1548 LS、46440独立力求值、1548候选历史、46440 RK4更新、185760积分RHS和17028物理RHS，合计202788新RHS。科学方案、来源清单、人口、权重、预算、实现与Astra审核均冻结后才允许执行。专用入口在任何NPZ字节校验前原子消费一次性共享预算，失败不重试。

V与I逐项遵守design_zh.md；数值有效且全部六组历史指标满足固定改善条件才ACCEPT，否则REJECT。结论只回答闭环历史机制，仍属未经独立确认的train内证据。始终KEEP_BASELINE，不授权新训练、未来比较、模型晋升、认证或交付预测。后续比较须另行登记及Astra审核。

工程日志遵循journal_decision_zh.md：每槽运算前持久预留全部额度，逐调用前后write/flush，日志和数组同步后才写持久关闭标记。崩溃槽unknown/spent，保留读取前缀与上下界，不续跑、不重试。日志或来源错误停止整轮。

## 报告与结果

<a id="doc-2ccc8ee870ee"></a>

<a id="doc-2ccc8ee870ee-a14固定模型的积分初态将历史重建误差降低约一半"></a>

### A14：固定模型的积分初态将历史重建误差降低约一半

对三个固定参数模型，积分初态方法将固定模型的历史重建归一化均方误差降低50.43%–51.58%，最差10%事件的原权重平均误差降低45.46%–46.29%。三个随机种子、两个积分步长的六组比较全部满足事前改善条件，774个seed/事件槽的数值与物理条件全部通过。两份独立只读审计确认结果。


<a id="doc-2ccc8ee870ee-比较范围与指标"></a>

#### 比较范围与指标

使用kiki原source train中固定分配的268条事件；其中258条符合原100ms输入合同，另10条保持原合同不合格记录。三个固定模型（随机种子324/325/326）在同一258条事件上比较。表中均为记录起点后100ms的历史重建，采用原source→线路→事件权重，只计算非初始的10个观测点。归一化均方误差为[q,v]20通道按冻结target_scale归一化后的平方差均值；最差10%指标按原权重质量取0.1，边界事件只取所需比例。

原encoder对照与积分候选采用相同模型、观测、功率近似和自由闭环方程。候选仅利用这段已观察历史估计隐藏起点，之后重建该历史；没有在闭环中间注入观测。下表为最大步长0.0025s，0.005s结果一致。

|随机种子|原encoder均方误差|积分初态均方误差|下降|原最差10%误差|新最差10%误差|下降|
|---|---:|---:|---:|---:|---:|---:|
|324|0.040263247|0.019532820|51.49%|0.197284630|0.106871842|45.83%|
|325|0.038367438|0.019019832|50.43%|0.193019128|0.105271006|45.46%|
|326|0.040699470|0.019708160|51.58%|0.202384570|0.108694443|46.29%|


<a id="doc-2ccc8ee870ee-科学判断"></a>

#### 科学判断

本轮回答了一个具体问题：对固定的二阶隐藏动力学，利用观测历史重新估计条件初态，能够显著改善自由闭环历史重建。全部事件的求积一致性、步长一致性、能量恒等式和质量中心坐标检查均通过；样本范围、权重与归一化尺度保持冻结。

Astra的机制复核同时指出，新的模型内部隐藏储能增加，且仍有当前隐藏作用范围之外的力残差。历史误差下降尚不能解释为真实隐藏状态恢复，也不能替代未来预测检验。下一项研究将冻结该初始化方法，单独检验未来轨迹与失稳漏报表现。


<a id="doc-2ccc8ee870ee-记录与执行边界"></a>

#### 记录与执行边界

本轮完成7740次矩阵指数、780次SVD、1548次复用分解求解、46440次独立力求值，以及202788次RHS；全部与固定预算一致。拟合、优化更新、未来路径和标签读取均为0。source val/test和内部val未打开。

科学结论为ACCEPT，模型决策为KEEP_BASELINE。A14授权已消费，不重复运行；任何后续比较另行登记。本轮没有计算新方法的未来漏报率，不发布预测或晋升模型。

机器可读收据 `kiki_a14_integral_history_20261024_evidence.json` 已绑定完整run、源码、结果清单及两份独立审计的SHA；Astra科学审核见 `docs/reports/04_kiki_a03_a17_pipeline/a14/_sealed/astra_result_review_zh.md`。

## 复核与历史版本

<a id="doc-94d9a4d3a788"></a>

<details>
<summary>A14 积分隐藏初态历史审计：实施审核</summary>

<a id="doc-94d9a4d3a788-a14-积分隐藏初态历史审计实施审核"></a>

### A14 积分隐藏初态历史审计：实施审核

2026-10-24，Astra Ultra。

**裁定：实施审核通过，批准在登记、提交推送、CI 与干净 HEAD 检查全部完成后，仅执行一次已冻结的 A14 有界历史机制审计。** 已发现的科学实现和失败证据问题均已修复，最终源码与合成验证足以支持该次执行。本批准不授权未来预测或评分、标签读取、训练、模型晋升、认证或部署；所有既往已消费授权继续关闭。


<a id="doc-94d9a4d3a788-审核绑定"></a>

#### 审核绑定

本审核者实际读取最终核心源码、读取边界、登记配置与验证收据，并逐项核对下列清单中的 **48 个文件原始字节，差异为 0**。核对范围不包含真实科学输入的字节校验或成员解码；本次审核的真实模型构造、force、RHS、拟合均为 0。

| 对象 | SHA256 |
|---|---|
| 科学方案 `design_zh.md` | `eed74751bbf1f53761ab8da71ee7d31b31baa586f391d3a344bcb1f06ba88831` |
| 日志持久化规则 `journal_decision_zh.md` | `cec78681a92649f46a56aaebc9f27eb3fdf2bbb1c0fd3410ea7e3400ebdf4b9a` |
| 最终 48 文件清单 `implementation_bindings_v2.json` | `cb58ab52a64c627273f2372d266761f3c95c02bafa947dab38441021dbf4ef45` |
| 验证索引 `implementation_validation.json` | `18ff57103536bfdf7a1c49c208ec835075246b4f3919e47cb756331e9a08ef1c` |
| 工作流 `kiki_integral_history_audit.py` | `da36e6d3ed5df4c3be3c2231739d34d7a43ad1e73689ccaa6a1a420721cbde58` |
| 数学内核 `integral_initializer.py` | `517ced3d272d601e78ec0e445d57f35361c19e04b5282288d8eab63e63542372` |
| 许可成员读取器 `limited_npz_audit.py` | `e96e939f9b7df72a4895906ca444e3c85c3bcd144360b413ba201ad5c1e57031` |
| 缓存前置核对 `history_cache_preflight.py` | `42fc18bb6ad7bcc3be18f6c9cda1d4249d586aafe08b55fe3b78c0efeea0e5b7` |
| 分段日志 `scoped_audit_journal.py` | `cb6537d5f68380c2187b48b8db2fb9ce4fff5faff0045d9a88a9001529fb7a6a` |
| 机制统计 `history_mechanism.py` | `c75cb00b729036d79cfa6823e0b0f5ffd76c5d19d580fb8e0297399cbcad9bba` |
| 比较指标 `history_metrics.py` | `7bd641bae5b66dcb76445615c722cce17bc23f7bace679a4fd3e80197313c9bc` |

其他依赖、测试、配置和协议以 v2 清单为准。v1 清单保留为过程证据，不作为执行绑定。本稿签署时 policy 仍为 `pending_implementation_review`；本审核已确认其配置字段与实验 TOML 一致，单次执行上限为 1，禁止重试。协调者随后应绑定本稿与 v2 清单，将独立授权激活；这不是本审核已经执行真实审计的声明。


<a id="doc-94d9a4d3a788-读取与计算范围"></a>

#### 读取与计算范围

来源固定为 A06 step240 三个 seed 324/325/326，原 record100ms 合同。保留 268 条原 train 分配，其中固定 258 条合格历史进入 774 个 seed/事件槽；另外 10 条只记录原合同不合格。人口顺序、source→line→event 权重及六份缓存对照保持原样，seed 不当作独立真实样本。

专用入口在任何许可 NPZ 内容校验前原子占用 Git 共用的一次性预算。通用 runner 在完整数据校验之前拒绝该执行模式。原 allocation 位于不可变结果目录，按绑定 SHA 核对其元数据，不要求不存在的 Git blob；已登记的 tracked 配置、清单、源码和审核文件仍需工作区与 HEAD 字节双重核验。三个源 manifest 仅用于来源身份确认。不得调用原 provider 或读取 HDF、CSV、splits、未来成员、旧诊断/评分/LOS 子树、内部 val 或 source val/test。

读取器只打开固定十个 NPZ 的 42 个许可成员，39 个数值成员合计 440,733 个 float64，另三个零维 Unicode metadata 成员；未许可成员仅识别名字。header 检查与一次 payload 读取分别记账，错误后停止，不回源补读。原 NPZ 前后各一次校验共 175,465,110 字节；两份既有物理证书前后校验共 62,786 字节，证书正文不解码。证书只继承既有联合上界对历史部分的保证，不称本轮重新计算旧路径物理量。

| 新计算 | 全成功计划值／硬上限 |
|---|---:|
| 源模型加载、构造 | 各 3 |
| 原特征、encoder 线性映射 | 各 774 |
| 8×8 expm | 7,740 |
| 显式 SVD | 780，其中静态 6、逐槽 L 为 774 |
| 复用 SVD 的右端求解 | 1,548 |
| 独立正弦 force | 46,440 |
| 候选闭环历史路径 | 1,548 |
| RK4 完整更新 | 46,440 |
| 积分 RHS／物理 RHS | 185,760／17,028 |
| **总新 RHS** | **202,788** |
| 拟合、优化更新、未来路径、标签读取 | **全部 0** |

积分估计沿用冻结的增广矩阵指数、Gauss4 主解与 Gauss2 检查、固定 SVD 秩阈值和两步长；没有 ridge、阈值搜索或新增参数。历史时钟保留原浮点间隔，子步核对采用原 `max(1, ceil(...))` 规则。候选只替换历史首隐藏状态，随后调用原闭环 rollout；中途不注入观测。末尾 reset 只组装真实可见末态与闭环隐藏末态，不执行未来。


<a id="doc-94d9a4d3a788-失败保存与比较规则"></a>

#### 失败保存与比较规则

新版在候选计算前独立保存全部许可真值输入及六份缓存对照。每个完整积分结果先保存，再计算描述统计；计算输出非有限抛出局部数值异常，已算矩阵、指数返回、SVD、区间力、两右端和完成的解通过 `partial_record` 保留，不重算。某 seed 的静态分解纯数值失败时，其 258 槽标为 `prerequisite_failed`，候选为 `not_attempted`，保存证据后继续其他 seed。PC 秩不足本身不阻止动态 L 满秩的合法情形；L 秩不足不产生最小范数候选，也不退回 encoder。

来源/缓存身份、保存、日志或预算错误停止整轮。每槽先持久预留全部额度，逐调用前后记录 attempted 与 returned/failed 并 flush，槽末同步日志后，从持久字节重新核验配对和预算、绑定区间 SHA，再写关闭标记。RHS 的配对键包含 seed、event、step、phase、sequence，同一上下文禁止重复。未闭合槽保持 unknown/spent，不能重试、恢复计算或把剩余额度转给新尝试。只有正常完成的 rollout 才计完整状态更新；失败后的四次返回组不冒称成功更新。

V 要求全部 774 槽完成、L 秩为 2，以及全部来源、读取、预算、求积精度、两步长精化和物理条件通过。求积初态与观测驱动末态分别满足冻结的 `1e−6 × (1 + 四点解范数)` 上界；可见归一化步长差不超过 1e−5，隐藏差满足冻结的相对上界；每输出点能量恒等式绝对误差不超过 1e−8，质量加权 q/v COI 残差不超过 1e−10。

I 在每个 seed、每个步长的完整 258 条历史上比较非初始十点：原权重主 MSE 必须严格降低，q/v 两分量及最差 10% 权重的分数边界 CVaR 均不增加。六组全部满足才 I=true。任何缺失/失败都使总体比较不完整，不删除失败或重归一化成功子集。力空间、动态一致性、运动学残差分别保存和汇总；`S r` 与 `c S r` 明确区分，不能把互有重叠的残差相加为总误差。


<a id="doc-94d9a4d3a788-验证证据及其适用范围"></a>

#### 验证证据及其适用范围

最终本地收据位于 `scratch/one_off/kiki-a14-initializer-20261024/local-ci-20261024T115909695275Z/receipt.json`，SHA `e2e44f024c70a73b281bef2610fce5f707c58062f194f550a11177d1ef920379`。本审核已核对收据及八份原始日志 SHA；八条 CI 同款命令全部返回 0，346 项既有测试和 72 项 A14 测试通过。核心修复过程中，本审核者另实际运行过 15 项内核合成测试及 ruff，A13 多量 1e−12 等价、秩边界、数值异常、部分结果与调用次数检查通过。

完整的前一版合成 `_execute` 已完成全 774 槽：V=true、I=false，正确输出 REJECT；精确消耗上表全部算子和 202,788 个**合成** RHS。产物清单 SHA 为 `9a825f2a246967079097b4599c595afb91e2e5546ec82af9336761698c0b817d`。独立已存产物审计收据 SHA 为 `1428a68ef9afc8b38e33c7ae8043939e8b67bb3097ffd72774fbaaa69d54ac6b`；其记录核验 10,098 个工件、778 个连续日志区间 SHA、42 个许可成员及完整预算，差异为 0。该审计没有重新执行算子或 RHS。

完整合成外层脚本最后的断言失败仍保留：它把输出 NPZ 写入误计为输入成员读取。工作流工件已完整生成，但不将整次外层脚本改称 PASS。前一版 RHS 日志尚未逐行包含最终路径身份；其独立审计仅证明汇总和分段计数。最终身份绑定由双步长 262 RHS 的短合成、错误路径身份/重复序号反例验证；静态失败隔离、物理检查异常前保存路径、描述统计溢出及部分结果保存也由最终定向测试验证。**没有声称最终源码逐字节重跑了完整 202,788 RHS 流水线。** 这些更改不改变正常方程、人口、选择条件或预算，定向验证足以闭合其新增行为。


<a id="doc-94d9a4d3a788-执行与科学结论边界"></a>

#### 执行与科学结论边界

协调者完成本稿绑定、独立 A14 授权激活、commit/push、远端 CI 和干净 HEAD／原字节核对后，可执行这一轮。任何绑定变化或前置失败均不得绕过检查。一次性预算一经占用即消费，真实失败也不重跑；结果必须保留并独立复核。

本次批准只确认实现能够回答冻结的历史机制问题。只有 V∧I 才接受该问题；它使用同一段已观测历史，不能当作未见未来收益、真实隐藏状态恢复、FSR 降低或全 27 个失稳样本覆盖证据。Pm[10] 回放策略的误差仍可能被初态吸收。即使 V∧I 成立，也只允许提出下一项独立、有限的未来比较方案，不能自动执行。所有结果均保持 `KEEP_BASELINE`、`new_training_authorized=false`、`future_evaluation_authorized=false`、`model_promoted=false`、`certified=false`。

</details>

<a id="doc-82c14511d5e8"></a>

<details>
<summary>A14：动量积分初始化的真实历史结果审查</summary>

<a id="doc-82c14511d5e8-a14动量积分初始化的真实历史结果审查"></a>

### A14：动量积分初始化的真实历史结果审查

审查日期：2026-10-24。科学负责人：Astra Ultra。

**裁定：历史机制 ACCEPT；模型保持 KEEP_BASELINE。** 固定 A06 的三个模型，使用同一 100 ms 真实历史，通过动量积分估计两个隐藏状态，在不增加模型参数、不更新骨架的条件下，将闭环历史归一化 MSE 降低 50.43%–51.58%，最差 10% 原权重质量的历史 CVaR 降低 45.46%–46.29%。全部 774 个 seed—事件槽、两档步长及冻结的数值、物理、读取和预算条件通过。两份独立保存数组审计均复现该结论。

这项结果支持提出一次独立、有限的未来迁移检验。A14 没有生成未来轨迹、读取标签或检验漏判；既有 A06 的每 seed `FN=1/14、FP=0/74` 没有因此改变。A14 的 ACCEPT 不授权未来计算、训练、模型晋升或对外预测。


<a id="doc-82c14511d5e8-1-结果身份与审查范围"></a>

#### 1. 结果身份与审查范围

唯一 run：`paper_kiki_a14_integral_history-history_mechanism-20261024T121439628465Z-34de0389-4cf0648f`。

| 证据 | SHA256 |
| --- | --- |
| 执行源码 commit | `34de0389a78ce3876765997e2029be020191cac9` |
| run 的 `artifact_hashes.json` | `7438242d358244dece184a393ef871374322d04415237fff733ecd7c01767938` |
| run 的 `summary.json` | `9cf1ab27c4d0935f6f013c14e143c76e7ea324d5b44057bd7266c30a62f91ba8` |
| 冻结科学设计 `design_zh.md` | `eed74751bbf1f53761ab8da71ee7d31b31baa586f391d3a344bcb1f06ba88831` |
| 最终实施审查 `astra_implementation_review_zh.md` | `208a4cf60987d3faaaa4b8a7e01f0c1fcd17166c96d49b1a4dfb8ce842bd3b4c` |

本次审查读取 A14 封存产物、注册元数据及独立审计收据。补充统计只使用已保存的数值数组；没有重新读取原缓存或原数据 payload，没有执行模型力、RHS、矩阵指数、SVD、最小二乘或模型预测。


<a id="doc-82c14511d5e8-2-比较回答的问题"></a>

#### 2. 比较回答的问题

三个来源均固定为 A06 step240，seed 为 324、325、326。人口固定为原 internal train 分配的 268 个事件，其中 258 个满足原 100 ms 历史合同；10 个未覆盖事件保留在分配记录中。A12 的 20 ms 输入审计没有改变这次人口。

对照使用原 encoder；候选使用已冻结的四点 Gauss 求积动量积分最小二乘，估计历史起点的隐藏状态。两者都从真实历史首点可见状态出发，按原动力学闭环积分到历史末点，中途不注入观测。候选历史末点的可见状态随后重置为真实末点，隐藏状态保留闭环积分值；该重置状态在 A14 中保存而不向未来积分。观测驱动隐藏系统的末态用于诊断，不能替换闭环末态。

历史主指标为首点之后 10 个真实时刻、20 个可见通道的归一化 MSE，使用原 target scale 及 source→line→event 层次权重。CVaR 按同一事件 MSE 从大到小累计原权重质量，截取最差 10%，边界事件按所需部分质量计入。三个 seed 是同一事件人口上的模型复现，不是 774 个独立实验样本。

| seed | 步长 / s | 对照历史 MSE | 候选历史 MSE | 对照历史 CVaR10% | 候选历史 CVaR10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 324 | 0.005 | 0.040263245293 | 0.019532819239 | 0.197284619947 | 0.106871837964 |
| 325 | 0.005 | 0.038367436557 | 0.019019831656 | 0.193019117507 | 0.105271001301 |
| 326 | 0.005 | 0.040699467746 | 0.019708159094 | 0.202384560045 | 0.108694438685 |
| 324 | 0.0025 | 0.040263247176 | 0.019532819848 | 0.197284630430 | 0.106871842469 |
| 325 | 0.0025 | 0.038367438359 | 0.019019832195 | 0.193019127939 | 0.105271005501 |
| 326 | 0.0025 | 0.040699469533 | 0.019708159717 | 0.202384570096 | 0.108694443327 |

六组的角度与转速分项 MSE 都不增。例如 0.005 s 下，三个 seed 的角度 MSE 分别由 0.00103169、0.00098106、0.00104335 降至 0.00046227、0.00045009、0.00046593；转速 MSE 分别由 0.07949480、0.07575382、0.08035559 降至 0.03860337、0.03758957、0.03895039。

最小二乘会构造性地降低自身线性积分目标；本次有信息的实证是另行闭环积分后的历史轨迹、分通道误差和尾部误差同时改善。它证明该固定模型的初始化仍有可利用的历史信息，不等于恢复真实系统全部内部状态。


<a id="doc-82c14511d5e8-3-数值物理与预算"></a>

#### 3. 数值、物理与预算

全部 774 槽完成，无求积、步长精化或物理失败。主要数值余量如下。

| 检查 | 实测最大值 | 冻结要求 |
| --- | ---: | --- |
| 两步长可见路径的最大归一化差 | `1.3371251e-7` | `≤1e-5` |
| 两点/四点求积起始隐藏状态差除以其门槛 | 0.00241733 | `≤1` |
| 两点/四点求积观测驱动末态差除以其门槛 | 0.00257024 | `≤1` |
| 两步长隐藏路径差除以其门槛 | 0.00096648 | `≤1` |
| 能量恒等式绝对残差 | `2.2737368e-13` | `≤1e-8` |
| 质量加权 COI 绝对残差 | `3.1086245e-15` | `≤1e-10` |

本轮精确完成 3 次模型加载与构造、774 次特征提取与 encoder 调用、7,740 次矩阵指数、780 次 SVD、1,548 次最小二乘、46,440 次可见力求值；1,548 条历史路径使用 46,440 次 RK4 更新。积分 RHS 为 185,760 次，物理检查 RHS 为 17,028 次，合计 202,788 次，attempted 与 returned 均一致。拟合、优化更新、未来路径和标签读取均为零。

V 与 I 按冻结定义全部为真。未改变 `1e-5` 精化门槛，没有删样本、补跑、缩小人口或改阈值。上述能量恒等式与 COI 检查支持数值实现符合注册结构，不构成真实系统稳定域证书。


<a id="doc-82c14511d5e8-4-残差条件数与隐藏储能说明了什么"></a>

#### 4. 残差、条件数与隐藏储能说明了什么

**注册最小二乘在这批历史上没有出现秩不足或严重条件恶化。** 加权设计矩阵的条件数分别为 3.3934、3.3830、3.2606，最小奇异值为 3.2980、3.4610、3.3452。它们约束的是固定模型、固定缩放和固定观测区间内的线性求解敏感性；没有测量噪声幅度证据时，不能把这些数值改写为真实隐藏状态的误差证书。

**历史改善伴随更高的模型隐藏储能。** 使用模型定义的 `H_z=||z_h||²/2`，原权重均值如下。

| seed | encoder 起始隐藏储能 | 积分估计起始隐藏储能 | 储能增加事件的原权重质量 |
| --- | ---: | ---: | ---: |
| 324 | 4.36964 | 11.47250 | 0.84366 |
| 325 | 4.72245 | 10.64375 | 0.83987 |
| 326 | 4.45977 | 10.87904 | 0.86569 |

该量属于约化模型的储能坐标，不能与真实 220 维系统内部能量直接等同。它给出下一步必须做未来检验的具体理由：能较好解释历史的隐藏状态，也可能向未来释放更多能量。应直接检验冻结估计器，不能根据这些结果临时裁剪、缩放隐藏状态或添加岭项。

**固定端口不能解释全部动量残差。** 令 `g_j` 为可见动量平衡积分，`r_F=(I-proj(range(PC)))g`，`r_D=b-L z_h`。使用物理欧氏范数及原时间系数 `c_j`，再按原事件权重汇总平方量，`r_F` 占 `r_D` 的比例在三个 seed 分别为 46.514%、45.939%、47.152%。因此，在这套骨架、输入和历史插值固定时，相当一部分动量残差位于端口子空间之外，调整两个初始隐藏状态本身无法消除。

这个比例不识别残差来源：骨架误差、实际输入与 Pm10 回放策略的差异、观测误差和未表达动力学都可能参与。原报告另存的 `S` 缩放残差与上述物理范数不是同一度量；`S` 缩放一般不保留投影的正交性，不能把其两个平方均值相除，声称得到正交分解比例。该现象也尚不足以支持直接改为状态依赖端口 `C(q)`。

运动学残差的最大角度值为 `2.1187623e-4 rad`，最大归一化值为 `5.6148280e-4`，用于描述真实历史与离散 `q̇=v` 的兼容程度。它不检验隐藏状态真实性。闭环隐藏末态与观测驱动末态的加权平均距离约为 0.0227–0.0237；两者并不相同，下一次未来比较必须从已保存的闭环隐藏末态开始。


<a id="doc-82c14511d5e8-5-独立复核与证据边界"></a>

#### 5. 独立复核与证据边界

全产物审计收据为 `scratch/one_off/kiki-a14-result-audit-20261024/real-v1.json`，SHA256 `64d06e691f24d02ca0552fde1f7d740476c7b1755d1bacb08bbe1e0112870262`。它核对 10,098 个工件、3,870 份打包记录、774 槽和 778 个 scope 的连续范围 SHA 及逐调用身份，独立复算 V/I/ACCEPT，mismatch 为零。

独立指标收据为 `scratch/one_off/kiki-a14-metrics-audit-20261024/real-v1/audit.json`，SHA256 `6b20088472e1177a210c470155cff405cc2f61fac4e220a63c34735f36e904e1`。它完成 6,192 项数组身份检查，复核 observed/baseline/reset 及全部原权重、MSE、CVaR，结果 PASS。两项审计均无新 source 读取、模型加载、预测或 RHS。

本审查的附加描述统计保存在 `scratch/one_off/kiki-a14-initializer-20261024/astra_saved_diagnostics.json`，SHA256 `259a61286c5070a9748cb6751745af37b3e485a4efc4c5a8f916696fc02b38d5`；对应脚本 SHA256 `fffdd0291bb68ec9fbf99efd23623def5b21f71724855cbe1687ec29c6ba314d`。它绑定 1,551 个实际读取文件的 SHA，只对已存数组计算描述统计，不增加科学模型算子。


<a id="doc-82c14511d5e8-6-合并与下一步裁定"></a>

#### 6. 合并与下一步裁定

**A14 具备合入 main 的实际收益。** 收益是固定骨架下得到可复核的历史闭环精度提升，以及支持严格成员白名单、一次性预算、逐调用身份、失败中间量保全和独立重放的研究能力。可按正常代码审阅和 CI 流程集成 A14 的研究模块、测试与结果档案；本裁定不把它设为部署默认初始化器，也不晋升任何模型。

下一项科学问题应为：同一个积分初始化器的历史收益，能否在不再注入真实观测的未来路径中保持，并满足原来的轨迹、数值、物理及误报约束。先固定来源、估计器、比较人口、有限计算预算和停止条件，再独立登记；不再以同一 val 反复延长训练或调整温度。A14 的隐藏能量与残差诊断用于解释这项问题，不能用于挑事件、调初始化器或按标签路由。

A14 授权已经 consumed，不得重跑、追加槽或以本 ACCEPT 自动打开未来。原 100 ms 合同下 val 的 27 个失稳样本中，合格范围为 14 个，另 13 个不覆盖；即使后续观察到 `FN=0/14`，也不是全 27 个或总体风险为零。20 ms 覆盖问题与新独立 kiki 确认继续保持各自的输入合同和授权边界。

</details>

<a id="doc-ebe11c183abf"></a>

<details>
<summary>A14 算子日志的持久化边界</summary>

<a id="doc-ebe11c183abf-a14-算子日志的持久化边界"></a>

### A14 算子日志的持久化边界

2026-10-24，Astra Ultra。**允许按预定事件槽持久预留完整额度，逐调用写入并 flush，在槽关闭时统一 fsync。无需每次算子调用 fsync。** 本补充只规定工程记账，不改变 A14 方程、数据、人口、阈值、算子上限或科学接受条件；正式实现仍需审核后才能执行。

原要求中的“调用前 attempted、返回后 returned”继续逐调用保留。flush 确保调用前日志已交给操作系统；它不保证断电后仍存在。可靠性由事先落盘的整槽预算预留与失败后不重试保证，不能把这种方式描述为逐调用均已持久化。

1. **先预留再运算。** 在该槽第一个算子前，持久记录唯一 slot ID、固定次序、输入与模型绑定、每类完整硬额度和日志起始位置，执行 write、flush、fsync。单次 campaign 的共享授权账本也必须已原子预留。失败时不调用算子。
2. **逐调用记录。** attempted 必须写入并 flush 成功后才 dispatch；返回或异常后立即写入 returned/failed 并 flush。沿用 attempted/dispatched/returned 的不同含义。任何日志 write、flush 或 fsync 错误都停止整轮，不能作为可继续的局部数值异常。
3. **先证据落盘，再关闭槽。** 槽关闭前，对本槽全部算子与 RHS 日志 flush+fsync，核对顺序、配对、预算和已保存数组。需保留的数组与结果文件也应保存完成并同步。最后写入并同步关闭标记，其中绑定日志范围/哈希、实际计数、结果文件哈希和状态。成功、局部失败、rank 不足应分别标记；“日志已完整关闭”不等于科学通过。不得先写 complete 再同步其证据。
4. **不确定就消费预留额度。** 崩溃或缺少有效关闭标记时，该槽为 unknown/spent。保留原日志和完整记录前缀；可能截断的末条也保留，不回写修补。只能报告经一致性核验的调用下界及预留上界，不能凭缺失返回记录推断没有执行，也不能用 returned//4 推断完整状态更新。该槽全部预留额度不可复用；当前一次性 campaign 不得重启续跑，其余计划槽保留 not_attempted。
5. **覆盖静态和失败路径。** 每个 seed 的两次静态 SVD 使用单独、同样受限的预留/关闭范围。正常数值失败在日志仍可靠时可按既定协议关闭为 failed/unresolved，继续其他预定槽；不转移未用额度、不补跑。来源读取仍遵循独立的受限读取账本，本补充不扩大文件或成员权限。
6. **正常结束仍精确核数。** 全部成功槽保留完整逐调用记录，最终精确核对 7,740 expm、780 SVD、1,548 LS、46,440 独立 force 和 202,788 RHS 等冻结预算，封存完整日志 SHA。有失败时实际量、预留量和未知量分开，V=false；保存成功不能把失败变成通过。

此方案以有限且不可重复的最坏情况预算控制崩溃风险，同时保留正常运行的精确审计能力。它不增加重试、恢复续算或新科学计算权限。

</details>

## 回执与附件

- [kiki_a14_integral_history_20261024_evidence.json](_sealed/kiki_a14_integral_history_20261024_evidence.json)
- [implementation_bindings_v1.json](_sealed/implementation_bindings_v1.json)
- [implementation_bindings_v2.json](_sealed/implementation_bindings_v2.json)
- [implementation_validation.json](_sealed/implementation_validation.json)
- [source_inventory.json](_sealed/source_inventory.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_a14_development_authorization_20261024_zh.md](_sealed/kiki_a14_development_authorization_20261024_zh.md)
- [docs/reports/04_kiki_a03_a17_pipeline/kiki_a14_integral_history_20261024_zh.md](_sealed/kiki_a14_integral_history_20261024_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/research/kiki_a14_integral_history/design_zh.md](_sealed/original/design_zh.md)
- [docs/research/kiki_a14_integral_history/astra_implementation_review_zh.md](_sealed/astra_implementation_review_zh.md)
- [docs/research/kiki_a14_integral_history/astra_result_review_zh.md](_sealed/astra_result_review_zh.md)
- [docs/research/kiki_a14_integral_history/design_zh.md](_sealed/design_zh.md)
- [docs/research/kiki_a14_integral_history/journal_decision_zh.md](_sealed/journal_decision_zh.md)
- [docs/research/kiki_a14_integral_history/kernel_api_and_validation_zh.md](_sealed/kernel_api_and_validation_zh.md)

</details>
