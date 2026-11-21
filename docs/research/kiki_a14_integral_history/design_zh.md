# A14：固定模型的积分隐藏初态与闭环历史机制审计

2026-10-24，Astra Ultra。状态：科学方案冻结草案，**尚无真实执行授权**。A12 需先完成并经独立审核；A14 还需独立登记、实现审核、固定源码与单次预算。不得复用任何已消费授权。本方案不改变 A13 已冻结的 seed、人口、算法、数值阈值或 V/I 条件。

## 1. 唯一问题

在固定 A06 step240 的三个二阶隐藏模型上，利用原 record100ms 的 11 个真实 q/v 点与原功率输入策略，能否构造数值有效的隐藏起点估计，并改善闭环历史重建？只替换各段历史的条件初态；共享骨架、隐藏动力学、encoder、质量、尺度与所有参数保持原样。

唯一对照是已经封存的原 encoder 闭环历史；唯一候选是四点 Gauss 求积所得积分最小二乘初态。二点求积只作数值检查。新参数 0、拟合 0、优化更新 0、未来预测/未来评分 0、标签/LOS 计算 0。不测试 20ms 初始化，不读另外 10 条不合格事件的原始数组，不比较候选网格。

该问题由 A13 固定合成检验支持：15 个病例及物理检查通过，反例同时证明满秩、小残差、被动性不能单独保证真实隐藏状态恢复。A13 方案 SHA 为 `d9462c02244032bf1a130502c7bb11a8851ef29beccfcbef85ad757fa54b472a`，结果审查 SHA 为 `05c89aec4c84c4f4d446ba1bb0d3b4bc7fe39e431d11a1880b52d6471c356fad`；已存数组独立审计 SHA 为 `465212171daf54cfc92353d25fcdc18fa400b39fee3242067c2a7a9d03983141`。后者只复算保留证据支持的部分，不能称全部 15 病例完整算子重放。

## 2. 来源、人口与顺序

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

## 3. 已核验的缓存布局与许可读取

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

### 3.1 缓存读取预算

十个 NPZ 合计 87,732,555 字节；运行前后各顺序校验一次，固定 **20 次原文件字节校验、175,465,110 字节**，与逻辑成员解码分账。校验不得多做 EOF 探测或遇失败重试。三个源 manifest、allocation、登记文档与证书的元数据完整性读取另外逐文件记账，不混作数组解码。

最多打开十个 NPZ 解码上下文。六个真值成员共 64,788 个 float64，六份旧模型 history 共 374,616 个，三个 checkpoint 数值数组共 1,329 个：**39 个数值成员、440,733 个 float64、3,525,864 个逻辑字节**。另有三个 metadata 文本成员，总成员数 42。文本按固定来源 SHA 单列实际字符/解码字节，不把它算成数值元素；单个只允许零维 Unicode，字符数硬上限 1,048,576，禁止 object/pickle。此上限是格式读取防护，不是调参；超界拒绝本轮，不截断文本或重读其他版本。

### 3.2 旧物理证据的继承

源物理资格只继承已归档 A10 科学审核 `docs/research/kiki_a10_astra_result_review_20261024_zh.md`，SHA `43f3f6f5add1f2d5e3bf46fc53c9ccc202aa4c2fcb2453aea03090902756313b`，及独立收据 `docs/reports/task3_model_refinement/kiki_a10_train_refinement_20261024_evidence.json`，SHA `1337b2f8c4cc9b8827a38a37cf47a15c916ed4a66cce05ca61314d4a1bb235ab`。两文件分别 7,901、23,492 字节；各前后一次字节校验共 62,786 字节，不解析其诊断正文。

A09 `path_physics` 对 history+future 联合取最大误差，A10 复用同函数；已通过的联合上界蕴含其中 history 的上界。这里准确称为继承既有证据，不称新历史独立物理检查，也不额外调用旧路径 energy_identity。A09 原 .01/.005 精化失败仍保留；本轮依赖 A10 完成的 .005/.0025 验证，不改写 A09 结论。

## 4. 输入策略和前置核对

真实历史是记录首点后名义 100ms，实际输入 `history_time=clock[:11]-clock[10]`；传播用该数组减去首值，保留每段浮点间隔。不是物理清除时刻，也不从 forecast_time 重建时钟。各区间以原 `ceil(h/max_step−1e−12)` 规则核对两个最大步长分别为 20/40 次历史更新；不满足便在新力、指数和 RHS 之前停止。

Pm[10] 派生的 `constant_power` 在整个模型历史里保持常值，这是已声明的输入策略，**不证明原始机械功率过去 100ms 实际恒定**。积分方法可以把这项输入误差吸收到隐藏初态；A13 S10 已构造零残差但初态错误的反例。因此下述残差或历史改善不得称真实物理隐藏状态恢复。

三个 checkpoint 必须为 n=10、r=2、sine、frames=3、原 encoder_mode；九数组及 metadata 来源 SHA 固定。质量和 target_scale 与 A10 六成员缓存逐数组相同，三个源模型各自 feature_scale 不重估。全部数组有限，质量与尺度为正；保留原结构校验。

按原 `_features`/encoder 仅做每槽一次特征和线性映射：`z_enc=E*(features/feature_scale)`，共 774 次。两份对应缓存 history 的首个隐藏状态与它按绝对 1e−12 校验；可见初态按原 `_state`/COI 投影规则核对。不能重跑 encoder 历史来“核实缓存”。两份旧历史的可见归一化步长差、隐藏相对步长差可用已许可成员直接计算，沿用第 8 节阈值；不调用 RHS。

对照只复用旧历史；候选从同一真实首点、唯一估计初态和同一常功率开始，使用原闭环 `closure.rollout`。不把观测 v 插入闭环中间时刻，不调用会重新覆盖 z 的原 `model.initialize`。结束后可另存 `[q_obs(T),v_obs(T),z_cl(T)]` 重置状态，q/v 必须逐数组等于最后真实观测、z 必须等于各自闭环末态；只组装和保存，不运行未来。

## 5. 唯一估计方程

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

## 6. 必须分开的诊断

1. 积分力空间残差 \(r_{force,j}=(I-PC(PC)^\dagger)g_j\)，是动量单位；另报 \(r_{force,j}/h_j\) 的平均力单位。投影先在物理空间完成，之后才计算 S 归一化值。
2. 动态一致性残差 \(r_{dyn}=b-L\hat z_h\)。即使各区间的 \(b_j\) 都在 range(PC) 内，也未必存在同一初态解释全部区间。一般有 \((I-\Pi_{PC})r_{dyn,j}=r_{force,j}\)，两项不是独立或正交的误差分解；不能把它们相加作为总误差。该最小二乘目标的降低是构造结果。
3. 运动学残差 \(r_{kin,j}=q_{j+1}-q_j-h_j(v_j+v_{j+1})/2\)，分别保存 rad 与除以 s_q 的量，不归因给隐藏状态。
4. 全部奇异值、固定阈值/rank、条件数及绝对增益 \(1/\sigma_{min}\)。已知固定 L 时的 \(\|\Delta z\|\le\|\Delta\widetilde b\|/\sigma_{min}\) 只作条件敏感度；没有真实噪声及骨架误差上界，不签发真实误差证书。
5. 两求积初态与 \(z_T^{obs}=\Phi(T)\hat z_h+w(T)\)、两个步长闭环末态、彼此差值、\(\|z_h\|^2/2\) 与相对 encoder 储能。分母零的比值为 None，不填 0 或无限大的 JSON 数值。不得把观测驱动末态替换为候选闭环起报状态。

保存逐区间物理值、归一化值与最大值；汇总采用同一事件权重/全体分母。可对 force/dynamic 以 \(n^{-1}\sum_j\|c_jSr_j\|^2\) 汇总，对 kinematic 以 \(n^{-1}\sum_j(h_j/T)\|r_{kin,j}/s_q\|^2\) 汇总；这些是描述统计，不新增选择门槛。

## 7. 固定新计算预算

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

## 8. 事前接受条件

**V：数值与计算有效性。** 来源、成员白名单、全人口对应、原权重、身份和预算全部通过；774 槽均完成且 L 的 rank=2；两求积的初态差及观测驱动末态差分别不超过该量的 \(10^{-6}(1+\|z^{(4)}\|)\)。两步长历史的可见状态除以 target_scale 后的最大绝对差≤1e−5；隐藏状态差≤\(10^{-5}(1+\max_t\|z^{(.0025)}(t)\|)\)。候选每输出点原能量恒等式**绝对**误差≤1e−8、质量加权 q/v COI 绝对残差≤1e−10。旧对照继承冻结物理证书，许可 history 成员自身的有限性/初态/步长核对必须通过。任何失败/缺失/非法访问使 V=false。

**I：闭环历史重建收益。** 对两个步长、三个 seed 各自计算非初始 10 点的单事件值
\[
\ell_i=\frac1{10\cdot20}\sum_{j=1}^{10}\|
([q^{cl}_{ij},v^{cl}_{ij}]-[q^{obs}_{ij},v^{obs}_{ij}])/s\|_2^2.
\]
q/v 分通道分母分别为 10×10。候选相对缓存 encoder 的 \(\sum_iw_i\ell_i\) 在全部六个 seed/步长组合严格下降，两个分通道均不增，原权重最差 10% 的 CVaR 均不增。CVaR 按单事件 \(\ell_i\) 降序取质量 0.1，边界事件只取所需分数质量，最后除以 0.1；同值并列顺序按冻结行号，无标签参与。严格下降使用直接 `<`，不追加事后容差或删除异常事件。

这里 I 独立于 LS 的构造目标，但仍使用同一段已观测历史，不能称未见未来泛化证据。失败时总体值及 I 标未完成/false，不对成功子集重新归一化后宣称改善。

执行完成、V、I 与后续授权分开保存：完整结果若 V∧I 则本轮机制问题 `ACCEPT`，否则 `REJECT`。V∧I 只支持提出一项新的、有限的未来初始化比较，科学审核还须检查残差分解与储能是否暴露骨架/输入混淆；不自动批准下一轮。始终 `new_training_authorized=false`、`future_evaluation_authorized=false`、`model_promoted=false`、`promotion_decision=KEEP_BASELINE`、`certified=false`。

## 9. 失败保存、入口与实施检查

采用新的唯一 AUTH 和 Git 共用一次性账本，在任何科学输入 NPZ 字节校验/成员解码前原子预留；不得打开 A12 或 A06–A11 旧预算。专用 CLI 只做登记元数据/源码/干净 HEAD/流程锁检查后进入不可变 runner 内层；禁止通用 verify_data、原 provider 或旧 A09/A10 工作流入口，因为它们会扩大原始/未来读取。通用入口必须对该特定受限执行模式提前拒绝。

全 774 槽自开始存在。来源或读取权限/保存/预算错误是整轮停止；已经发生的操作和其余 not_attempted 槽保留。局部 rank/矩阵/数值问题不更换方案，记录 unresolved/failed 并继续其他独立预定槽；任一失败使 V=false。所有局部矩阵、两右端、SVD、状态估计、残差、缓存对照历史、新候选两步长历史及组装 reset 状态都保存为独立成员；为避免 A13 证据缺口，本轮不得只保存检查布尔值。保留源数组指纹、全部访问顺序/算子/RHS 账本，支持零新 RHS 的独立回放。

实施前合成检查必须覆盖：真实 future 成员一旦被解码便报错的 NPZ 读白名单、源哈希/成员缺失和全人口顺序错误；PC 静态 SVD 缓存与原合成内核等价且计数为6；10段非完全相同浮点间隔；rank不足不传播；算子或RHS失败不重试/不假记完整更新；错误动力学与输入吸收反例；同源 encoder 起点与闭环末态/可见 reset；成功与 REJECT 产物都满足严格 artifact/stage contract。没有这些工程与科学复核，不启动真实审计。

## 10. 后续解释边界

V 失败先解决数值、来源或条件性问题；不能调整 ridge、加采样点或换步长继续本轮。V 通过而 I 不成立，就拒绝本次积分初态的闭环历史收益假设，不按同一 val 结果继续加训练轮数或换温度。V∧I 也不等于真实 220 状态恢复，更不等于 FN 改善或全 27 失稳覆盖。

若力空间外残差占主要部分，不能只归因于 encoder；若动态一致性残差明显但力空间残差小，固定 A/C、输入近似和骨架误差仍需区分。状态依赖 C(q) 只有在这些解释被独立排查、状态与残差关系在事前保留的 train 线路上重现后，才可成为另一个科学假设；相关性不是因果证据。本轮不估计 C(q)、不选择基函数、不增加隐藏阶数。

## 源码依据

- A09 保存/物理检查：`src/kron_reduce/workflows/kiki_train_mechanism.py`，SHA `4becae0bf8a651362b45f4e17d9fd3a2762cfd07ece286ccb4991f8ee4336a64`。
- A10 独立历史成员与输入扩充：`src/kron_reduce/workflows/kiki_train_refinement.py`，SHA `ff068dca8c096fb0c46980d56dcee0c30cb11fbc6c818c9e681dc6ea9504ec2b`。
- 原 record100ms 输入变换：`src/kron_reduce/data/kiki_history.py`，SHA `c9bed1a04f6c924976a3799cfb9cc5e558954c61c8d97bd0e76eb726e2ab715a`。
- 原非线性模型/RK4：`src/kron_reduce/models/nonlinear_latent.py`，SHA `766991fa340f5d4176694fc5ad78eded074cf0e906d0c944b49057cbd26697c4`。

本方案的推导与文献适用条件沿用 A13：积分形式避免显式求导的原始方法背景见 [Dattner–Klaassen](https://doi.org/10.1214/15-EJS1053)，结构系统初态校准见 [Günther–Jacob–Totzeck](https://arxiv.org/abs/2301.02019)，可观测性与采样边界见 [Hermann–Krener](https://doi.org/10.1109/TAC.1977.1101601) 和 [Zeng 等](https://arxiv.org/abs/2204.14021)。这些文献不替本方案证明真实 kiki 的隐藏初态可恢复。
