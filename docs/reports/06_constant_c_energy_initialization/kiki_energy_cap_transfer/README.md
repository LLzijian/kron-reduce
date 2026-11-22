# encoder 参照能量球：固定 A06 的 train 迁移结果

[返回主题页](../README.md)

## 运行结果与归档

- [运行结果（包内 `summary.json`）](../../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer_runs.DESCRIPTION.md#run-1)
- [归档说明](../../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.DESCRIPTION.md)

登记文件与配置

[方案登记](../../../../configs/proposals/kiki_energy_cap_transfer_v1.json) · [登记 kiki_energy_cap_saved_sources_v1](../../../../configs/proposals/kiki_energy_cap_saved_sources_v1.json) · [登记 kiki_energy_cap_truth_window_audit_v1](../../../../configs/proposals/kiki_energy_cap_truth_window_audit_v1.json)

## 研究方案

<a id="doc-1b7e68adc5d3"></a>

<a id="doc-1b7e68adc5d3-encoder-参照能量球迁移科学设计审核"></a>

### encoder 参照能量球迁移：科学设计审核

2026-11-09，Astra Ultra。**设计结论：研究范围、来源和预算已冻结，可进入最终准确实施审核与独立登记；本设计审核不提供科学执行批准。** 状态：`EXECUTION_SCOPE_FROZEN_APPROVAL_BY_BOUND_RECEIPT`。

[冻结协议](#doc-a3dca8fc03e4)把下一项工作固定为真实 train 的零拟合机制比较：保持 A06 三模型、原 258 个合格事件、Pm10 输入、100 ms 历史和最多 3 s 的真实可用未来，只将 A14 积分初态增加同历史 encoder 起点能量球约束。E 是原 encoder 质量参照，I 是已拒绝的无约束积分流程；二者保存路径复用，不重跑旧研究。

选择这一问题的依据是现有证据已经完成合成结构及数值检查，而真实历史收益尚未迁移到未来。继续增加匹配合成病例不能回答这个缺口。固定能量球是否有效没有预设：新候选必须相对 E 达到事前工程收益与规则保护，不能因优于 I 就进入训练。


<a id="doc-1b7e68adc5d3-已核查的数学与数据条件"></a>

#### 已核查的数学与数据条件

1. 半径 `rho=||z_enc,h||` 可由预测时已经取得的合法历史确定。当前模型 `initialize` 将 encoder 输出置于历史首点，随后闭环传播并只在起报点重置可见状态。A14 已保存 baseline history 首点、Ltilde/btilde4/SVD、原模型参数和两档 reset；源码与 JSON 结构支持冻结读取方案。本审核未读取 NPZ；正式运行按准确来源清单核验载荷身份。
2. 这个球是原模型的逐例经验参照，不是实际隐藏能量物理证书，也不是已标定的总体覆盖概率。保留 encoder 作先验，不能声称已经完成无 encoder 初始化器。
3. 冻结球 solver 的秩阈值复用、SVD 重建、原方程 KKT、零半径、非活跃精确复用及有限求根规则与本问题一致。球内有 encoder 这一可行点，所以代数拟合优于 encoder 不是独立收益证据。真实未来改善才是决策依据。
4. 初始能量、闭环隐藏末态能量和可见 reset 的总能量跳变分开。半径不再施加于预测起点或未来；不存在用末态裁剪暗加第二个算法的空间。
5. 新规则诊断只由新 scope 明文许可的保存可见轨迹推导。保持原五点规则、精化和余量检查，不读取标签，不将真实可用段触发计数写成 8 s 失稳分类或新的 FNR。


<a id="doc-1b7e68adc5d3-决定是否继续训练的门"></a>

#### 决定是否继续训练的门

主指标保持 A15 的 source→line→event 原权重、每事件真实可用 1 s/3 s 等权归一化 MSE。三个 seed、两个步长分别要求相对 E 至少 1% 主指标改善，改善超过预定粗细 MSE 差异代理的十倍及 1e-12；六项分时域/通道/尾部/完整3s保护不增，两类轨迹触发错误整数计数不增。1% 是继续投入训练的工程收益尺度，不是统计显著性；两档差异不是严格连续误差界。

接受条件为 `V and mechanism_exercised and G_E and D`，G_I 独立完整报告。历史闭环精度用于解释，不另加收益门，也不能抵消未来失败。至少一槽实际新传播才表示约束机制受到检验；非活跃相等不算主动改善。全人口与所有分支保留，不以均值掩盖保护项、未决或未覆盖记录。

若通过，可提出同一合法历史映射的可微实现、隐式梯度和训练/使用一致性设计；这仍是新算法，需要新的预算与准确审核。若只优于 I、或相对 E 未达门，拒绝这个固定模型直接替换方案。若数值有效且约束活跃仍有预测退化，足以调整该具体经验上界路线；不能据此宣布所有上界、所有积分初始化或联合训练不可能。端口、势能、时间目标与覆盖问题保持各自待验证状态。


<a id="doc-1b7e68adc5d3-冻结预算与准确实施门禁"></a>

#### 冻结预算与准确实施门禁

[机器计划](../../../../configs/proposals/kiki_energy_cap_transfer_v1.json)与[来源清单](../../../../configs/proposals/kiki_energy_cap_saved_sources_v1.json)已列明 268/258/774 人口映射、原权重和时钟、逐 scope 算子上限、来源成员及实现绑定。正式运行总读取 8,528 文件、21,699 成员、418,279,836 原始 bytes、239,827,112 数组逻辑 bytes。初始化/历史、未来时钟、未来评分三阶段分别为 5,430/2/3,096 文件和 13,956/3/7,740 成员；全部候选历史冻结先于时钟读取，全部未来候选冻结先于 E/I/truth 未来评分载荷读取。

令 A 为 active 或零半径的新传播槽数、S_A 为其未来点数和、R 为正乘子活跃槽数。新历史/未来各 2A 段；积分 RHS 为 240A+24S_A，逐输出点物理 RHS 为 24A+2S_A，合计上限 5,831,802。scope 为 3,873+2A、最多 5,421，probe 总上限 37,199；根更新/标量评估分别≤64R/80R。固定模型构造三次，其内部 PassiveLatentClosure 验证为 3+4A 次、eigvalsh 为 9+12A 次。这些内部检查与模型加载、SVD、通用 LS 是不同操作，分别列账。

来源读取器须在载荷读取前核对 NPY shape、dtype、顺序和总字节，头部≤4,096 bytes、含格式前缀≤4,108 bytes；匹配后单次读取声明大小的载荷。原始文件读取、NPY 头部和数组逻辑读取分账。run 保存完整执行计划、来源清单、模型六参数与必要输入；旧 E/I 历史和 E/I/truth 未来可以使用包含 path/SHA/member/步长/身份的准确引用。

设计允许两次独立保存证据审核，每次读取新 run 的 manifest 及列出产物一次，原来源文件/成员各最多一次，原始 bytes 与成员上限分别为 418,279,836 和 21,699。审核与执行分账，禁止新 solver、模型构造、RHS、SVD、expm、拟合或传播。这个许可用于复算已保存代数和评分，不扩大候选比较。

不得为方便复用导入拒绝的 A15/A17 代码；可独立读取其明确获准的保存结果。工程测试只用人工数组和 mock，不能提前求真实候选或传播。准确登记与批准前保持零新科学数据读取、零新模型/求解/RHS/训练。

本审核依据协议、当前模型/求解器/来源读取/执行与评分代码，以及已归档 Markdown/JSON。已审查调用树、因果冻结顺序、复用分支、预算公式与失败保存；最终源码字节、完整工程测试日志、运行时依赖和准确绑定仍由独立实施收据核定，本文件不写 `approved=true`。正式运行还须独立授权登记、clean pushed commit、对应 CI 和运行前哈希/路径守卫。用户持续授权覆盖完成这些步骤后的固定一次比较，无需重复请求宽泛许可；旧 consumed 授权及默认部署保持原样。

## 执行协议

<a id="doc-a3dca8fc03e4"></a>

<a id="doc-a3dca8fc03e4-encoder-参照能量球固定模型的-train-未来迁移协议"></a>

### encoder 参照能量球：固定模型的 train 未来迁移协议

2026-11-09，Astra Ultra 科学设计。状态：`EXECUTION_SCOPE_FROZEN_APPROVAL_BY_BOUND_RECEIPT`。本文件冻结研究设计；实际批准状态由准确实施收据与独立授权记录共同决定。

本轮回答一个真实机制问题：在固定 A06 三个模型上，将积分初态限制在同一合法历史的 encoder 起始能量范围内，能否改善原 encoder 的未来预测，并避免无约束积分初始化的迁移退化。唯一新增候选为这个逐例能量球；模型参数、输入策略、历史长度、评分和原人口不变。

用户继续推进的指令支持自主完成设计、独立登记、准确实现审核及随后一次有限比较。来源清单与预算按本协议及机器计划冻结；准确实施收据、独立登记、clean pushed commit、对应 CI 和运行前守卫全部通过后，才能启动一次科学读取与计算。本文件本身不提供 active 批准。既有 A06、A14、A15、A17 及合成研究的 consumed 授权保持关闭。本轮不训练、不修改默认初始化器，终态始终 `KEEP_BASELINE`。


<a id="doc-a3dca8fc03e4-1-候选参照与问题范围"></a>

#### 1. 候选、参照与问题范围

对 seed s、事件 i，使用已保存的原 encoder 闭环历史首点隐藏状态 z_E,h，固定

\[
\rho_{si}=\|z_{E,h}^{si}\|_2,\qquad \bar H_{si}=\tfrac12\rho_{si}^2.
\]

半径来源只能是该模型对同一 record100ms 合法历史产生的历史起点状态。读取 A14 保存的 baseline history 首行，不重新调用 encoder、不重新拟合尺度、不取预测起点 z_E,T。两档 baseline 历史的首个隐藏状态必须逐元素相同，且继承 A14 已完成的 encoder 身份核验。新的执行仍须核验来源、成员和首点对应。

| 方法 | 定义 | 本轮计算方式 |
| --- | --- | --- |
| E | 原 A06 encoder 初始化、闭环历史及原未来 | 复用准确对应的保存路径 |
| I | A14 Gauss4 无约束积分初态、闭环历史及 A15 未来 | 复用准确对应的保存路径 |
| C | 同一积分方程，增加半径 rho 的原点球约束 | 每槽一次球求解；按分支复用或新传播 |

rho 是逐例、因果可用的 **encoder 经验参照**，不是实际隐藏能量的物理上界，也不是已完成跨事件分布覆盖标定的分位数。本方案保留 encoder 作为冻结先验，不能称为已经去掉 encoder 参数。真实系统没有可读取的 z 真值，不报告真实隐藏恢复误差。

半径不乘可调系数，不搜索分位数、上界、种子或事件，不由未来误差、标签、残差比例或新的规则诊断改变。球 LS 同时改变初态模长和谱方向；其收益只归于这一具体初始化流程，不唯一归因于能量大小。

E 的初态本身在球内，因此 C 在同一积分方程上的最优代数目标不高于 E 是构造性质，不能作为新的科学成功门。闭环历史和未来才检验其实际作用。


<a id="doc-a3dca8fc03e4-2-固定人口模型与可读来源"></a>

#### 2. 固定人口、模型与可读来源

使用 A06 step240、seed 顺序 324、325、326，n=10、r=2、原 sine 势能、质量、阻尼、C、J、R、平衡先验及 target_scale。原分配 268 个 train 事件全部保留，其中 258 个满足 record100ms 合同，共 774 个 seed/事件槽；另外 10 个明确记为不覆盖，不补样、不视作正确预测。三个 seed 是同一人口的算法重复，不当作 774 个独立真实事件。

原事件顺序和 source→line→event 权重直接继承冻结人口：

\[
w_i=\frac1{N_{source}}\frac1{N_{line(source_i)}}
\frac1{N_{event(source_i,line_i)}},\quad \sum_iw_i=1.
\]

保存原浮点权重，不为消除舍入差重新归一化。ID、来源、线路和已声明长度只服务身份、分母和分组，不进入估计器。

来源身份采用已归档记录，逐文件/逐成员范围冻结于[来源清单](../../../../configs/proposals/kiki_energy_cap_saved_sources_v1.json)：

- A06：`paper_kiki_a06_refinement-joint_refinement-20261024T025925447927Z-9aa5ee65-524be996`，manifest `d1c4cc402dd30207c4ba639a9d4d1f1e86e3051ce0ce51e72ac4d045be17cc01`。优先读取 A14 已保存的同源六参数，避免重复解码 checkpoint。
- A14：`paper_kiki_a14_integral_history-history_mechanism-20261024T121439628465Z-34de0389-4cf0648f`，manifest `7438242d358244dece184a393ef871374322d04415237fff733ecd7c01767938`。提供原历史、输入、尺度、静态参数、积分方程/SVD、E/I 历史及对应 reset。
- A15：`paper_kiki_a15_future_transfer-future_transfer-20261024T145313076879Z-f23e9b23-c2066d4e`，manifest `e67dbb7c2d62ac82017e9d1049fd7a01c492c6b9c3587aa841d7bb89d247f1c4`。只复用原保存 E/I/truth、future clock、人口和所需物理静态量。其归档提交为 `5ac8b571751467f677fb569df3c0a1f4c323d065`；不得导入或执行该拒绝分支代码。

同一数组有多个保存副本时，精确清单只选一个数值载荷来源，其余用已绑定身份验证。清单冻结成员数量、shape、dtype、元素数、字节、读取次数、阶段、descriptor 键和来源 SHA；机器计划绑定清单及实现的准确字节。文件存在和 JSON 描述不等于已完成载荷核验；准备与实施审核阶段不打开来源 NPZ。

正式运行只允许清单中的 train 保存成员，禁止 pickle，按成员读取；不得解码全部 NPZ 再切片。每个源文件只读一次原始 bytes 并核 SHA，缓存 bytes 用于后续成员读取。每个成员先读取 NPY 标识、版本、长度和头部，头部≤4,096 bytes，含格式前缀的读取≤4,096+12 bytes；先核对 shape、dtype、C/F 顺序和成员总字节数，再单次读取已声明大小的数值载荷，并检查有限值。原始文件 bytes、NPY 头部 bytes、解码数组的逻辑 bytes 分账，不能把缓存解码再次记为源文件读取，也不能遗漏解压后的读取。

明确排除 HDF、CSV、原 provider、splits、源标签、事件时间、Pe、电压、真实隐藏状态、未来 Pm、内部 val、source val/test、A11 和 A17 数值来源。旧标签或旧诊断结果不能作为新方法的输入。未来 q/v 只用于冻结后的评价。


<a id="doc-a3dca8fc03e4-3-唯一估计方程秩与-kkt"></a>

#### 3. 唯一估计方程、秩与 KKT

L 明确指 A14 的 **Ltilde**，b 指 **btilde4**，不是未加权 L 或原始动量右端。保留 A14 的实际历史时钟、质量/速度尺度 S 和时间系数 c_j：S_ii=1/(m_i s_v,i)，c_j=sqrt(h_j/T)/h_j。直接读取其薄 SVD；只做键映射 `U→u, s→s, Vh→vh`，不重新分解、不改变方向符号、不重新做 Gauss 求积。

\[
z_C=\arg\min_{\|z\|_2\le\rho}\tfrac12\|Lz-b\|_2^2,
\quad L\in\mathbb R^{100\times2},\quad b\in\mathbb R^{100}.
\]

调用冻结的 `solve_energy_ball(L,b,svd,rho,unconstrained=zh4)`。新执行器只组合既有纯代数求解器、当前 main 的模型和独立纯评价函数；不改冻结 solver 或 integral_initializer。A14 的二/四点求积资格只继承已保存证据，不在本轮额外求一个 Gauss2 球解。

必需检查如下：

1. 所有输入有限、实数、形状准确；rho 有限且非负；U 为 100×2、s 为 2、Vh 为 2×2，奇异值非负且降序。保存阈值必须等于原定义 `64*eps64*100*max(s[0],natural_norm)`；不改原阈值。rank 必须与 `count(s>threshold)` 一致。
2. U/Vh 正交误差最大绝对值≤1e-10；SVD 重建 Frobenius 误差≤`1e-10*max(tiny,||L||F)`。tiny 仅是冻结 solver 的重建数值尺度，不加入能量、误差或收益分母。检查旧 zh4 的同方程正规残差，尺度与冻结 solver 相同。
3. rho=0 先于秩分支：唯一可行状态为零，可新传播；`multiplier=null, multiplier_kind=zero_radius_limit`。无穷乘子只可作数学极限解释，不伪造有限 KKT 证书。
4. rho>0 且 rank<2：`unresolved`、不传播、不退回 encoder 或最小范数解。这是本轮政策，不是凸球问题必然无解。来源 rank、阈值或身份相互矛盾则是完整性失败。
5. 满秩且旧 zh4 范数≤rho：逐元素复制旧解，lambda=0；完整复用 I 的两档历史与未来。不得为同一路径重算 RHS。
6. 满秩且球外：使用冻结谱式及原 lambda 单位的保护 Newton/二分；端点外向舍入、上端最多 8 次修复、最多 64 次根更新/80 次标量评估，只从可行上端返回。无新搜索或失败重试。

对 rho>0 的成功解，在原 L/b 上验证 lambda≥0、相对可行性≤1e-12；正乘子时边界范数相对误差≤1e-12、归一化互补性≤3e-12，且

\[
\frac{\|L^T(Lz-b)+\lambda z\|_2}
{\|L\|_F(\|L\|_F\|z\|_2+\|b\|_2)+\lambda\|z\|_2}
\le10^{-10}.
\]

分母为零时分子须精确为零；能量比的零分母记 null，不加 epsilon。能量可行性按同一范数容差解释为 H_C,h≤Hbar(1+1e-12)^2。保存 `constraint_active`、`regularization_active`、`constraint_binding` 三个不同字段；lambda=0 的恰在边界解不记为主动正则收益。保存全部区间、评估 trace、更新数、目标、残差及 KKT；`EnergyBallNumericalError.diagnostics` 必须在失败记录中原样保全。


<a id="doc-a3dca8fc03e4-4-历史重置与未来链"></a>

#### 4. 历史、重置与未来链

每个新状态从原真实历史首点 q/v 和 z_C,h 出发，以原 Pm10 派生 constant_power、原 equilibrium 自由闭环传播整个 100 ms。q/v 不在历史中间注入，观测驱动隐藏末态不能代替闭环末态。

最大步长固定 .005/.0025 s。历史使用原保存 history_time 减首值，未来使用原保存 forecast_time；保持当前模型的 RK4、原子步规则及 COI 处理。先按冻结时钟验证每历史分别 20/40 次更新，每未来分别 2N_i/4N_i 次更新；不满足则停止，不更改分派或缩步修复。不得新建批量积分器。

两步长分别得到自己的 z_C,T，再以 `[q_obs(T),v_obs(T),z_C,T]` 重置，未来仍使用原 Pm10 常值。未来中间不重置、不裁剪 z。每条候选未来首行必须与该步长 reset 一致；E/I 对照首行与其自身已存 reset 一致，三者前 20 维与真实历史末点一致。

每方法保留 H_z,h、H_z,T、相对 E/I 的差与有定义的比值。另存历史末态 x^-、重置 x^+、总储能 H(x^-)/H(x^+) 和 DeltaH_reset；隐藏分量在重置前后逐元素相同。初始能量约束不要求 H_z,T≤Hbar；重置总能量变化不混入连续功率恒等式。无真实隐藏状态误差或“真实能量被恢复”字段。

因果冻结顺序：

1. 验证来源清单和合法历史成员；构造全部 rho、球解、两档历史/reset。非活跃槽冻结其 I 历史/reset 的源 SHA 和键。
2. 持久保存全部 774 槽初态/分支/reset 及身份清单。之后才解码独立 future clock；N_i 可提前从已绑定人口元数据读取，但不读取未来振幅。实现可从冻结 N_i 生成原 `np.arange(N_i+1)*.01` 网格，但必须与随后获准读取的保存 forecast_time 逐元素相等后才传播；只通过 allclose 不足以替代原时钟身份。该生成不增加观测或使用目标振幅。
3. 按固定 seed/事件/.005/.0025 顺序生成全部需要的新未来。非活跃未来提前冻结为 I 的不可变源引用；完成新未来及引用清单的持久冻结后，才解码 E/I/truth 未来载荷用于评价。
4. 评分不会反馈到半径、路径、输入、步长、分支或停止参数。


<a id="doc-a3dca8fc03e4-5-轨迹指标完全保持-a15-口径"></a>

#### 5. 轨迹指标：完全保持 A15 口径

未来每事件含共同起点和 N_i 个非起点真实可用点，1≤N_i≤300；起点不评分。源元数据应给出 sum_i N_i=72,147，完整 3 s 事件 237 个；这些值须由新清单逐项核验，不能以旧汇总替代对应检查。

设 s_k 为原 20 通道 target_scale，误差为

\[
e^g_{ij}=\frac1{20}\sum_{k=1}^{20}
\left(\frac{x^g_{ijk}-x^{true}_{ijk}}{s_k}\right)^2,
\quad J_{i,H}=\{j\ge1:\tau_j\le H+10^{-12}\},
\]
\[
\ell_i^g=\tfrac12\left[
\frac1{|J_{i,1}|}\sum_{j\in J_{i,1}}e^g_{ij}
+\frac1{|J_{i,3}|}\sum_{j\in J_{i,3}}e^g_{ij}\right],
\quad M_g=\sum_iw_i\ell_i^g.
\]

两个集合须非空；每个窗口按各自真实可用点数平均。M 是归一化 MSE，非 RMSE。展示 sqrt(M) 时单列为归一化 RMSE，不能将阈值改用其平方根。q/v 分项分别按各 10 通道平均，两者平均等于 M。

对每 seed/步长/方法报告 M、纯 horizon1/horizon3 总体 MSE、q/v 主分项、原权重 fractional CVaR10%、完整 3 s 子集纯 horizon3 条件 MSE。CVaR 按 ell 降序、同值按原行号；截取原权重质量 0.1，边界取所需分数质量后除以 0.1。完整 3 s 子集只将原 w 除以其总质量，不重算 source/line 权重。另报 N_i<100、100≤N_i<300、N_i=300 三组的数量、原权重质量和使用同总体分母的贡献，不删除短事件。

历史指标与 A14 一致：排除首点，10 个非初始时刻×20 通道归一化 MSE及 q/v 分项、CVaR10%。历史指标用于解释，不另设改善硬门；不能以历史误差下降抵消未来失败。


<a id="doc-a3dca8fc03e4-6-真实可用段的五点规则"></a>

#### 6. 真实可用段的五点规则

新 scope 明文许可仅从已批准的 real_history、E/I/C/true future 可见轨迹及已保存 equilibrium、mass、omega_s 派生规则；不读取源标签或事件时间。白名单复用 main 的纯数组 `train_mechanism_audit` 及配套规则精化函数，准确函数/文件 SHA 由机器计划绑定并经最终实施收据核验。不得调用旧 workflow 或 provider。

路径固定为 `concat(real_history,future[1:])`，共同起点只出现一次。每点取

\[
a_j=\operatorname{ptp}(q_j+q_{eq})/\pi-1,
\quad b_j=\max_k|(Qv_j)_k|/(0.02\omega_s)-1,
\quad m_j=\min(a_j,b_j).
\]

五点窗余量是该窗 m 的最小值，域余量是窗余量的最大值；严格大于零才触发。历史有 7 个窗；含未来域取末点索引≥11，允许跨起报点，共 N_i 个窗；full 域有 N_i+7 个窗。保存最早确认末点和最大余量代表窗，最大值并列取最早窗。rho_rule=0 为 False 并保留 arithmetic_boundary；没有窗口为 None，不转 False。规则余量 `rho_rule` 与初始化半径 `radius` 分开命名。

保留 history、含未来、full 和真实未来固定代表五点窗四类诊断。粗细档使用同一个真实代表窗；逐项核查索引、原子界、向量界及

\[
|\rho_p-\rho_t|\le E_{atom}+\epsilon,
\quad E_{atom}\le E_{vector}+\epsilon,
\quad\epsilon=10^{-12}\max(1,|\rho_t|,|\rho_p|,E_{vector}).
\]

同方法粗细未来触发布尔值必须一致；两者触发时首次五点确认末点索引差≤2。这些是数值有效性要求。源规则阈值不以 epsilon 改写。首次逐点联合违规与五点确认分别报告，不把确认末点减四当作真实首次违规。

记 T_i 为真实含未来域触发，P_i^g 为方法 g 的相应触发。各 seed/步长分别计整数

\[
N_{miss}^g=\sum_i1[T_i=True\land P_i^g\ne True],
\quad N_{extra}^g=\sum_i1[T_i=False\land P_i^g=True].
\]

两者分别叫“真实正触发未保留数”和“真实未触发而预测触发数”，不命名为标签 FNR/FSR 或 8 s 失稳分类。未知/缺失使规则保护未完成，不从分母移除。


<a id="doc-a3dca8fc03e4-7-事前决定规则"></a>

#### 7. 事前决定规则


<a id="doc-a3dca8fc03e4-v来源数值与完整性"></a>

##### V：来源、数值与完整性

全 774 槽必须有确定成功分支，来源/白名单/人口/权重/预算/冻结顺序与保存审计通过；不存在失败、未知或正半径 rank 未决。成功 solver 的上述检查全通过。每条新历史及未来的每个保存输出点检查原能量恒等式绝对残差≤1e-8、q/v 质量 COI 绝对残差≤1e-10；纯能量、状态及所有派生量有限。

新路径粗细两档可见最大绝对差除以原 target_scale≤1e-5；隐藏最大二范数差≤`1e-5*(1+细档全路径最大隐藏范数)`，历史与未来分别检查。未来比较必须保留各自历史末态。E/I 继承原物理证书，同时在许可保存路径上核验有限性、初态和粗细差；不重做其 RHS 物理检查。第 6 节所有规则精化和界核验通过。数值有效不等于预测改善或稳定域证明。


<a id="doc-a3dca8fc03e4-g_e相对-encoder-的实际未来收益"></a>

##### G_E：相对 encoder 的实际未来收益

对每个 seed 和两个步长 h，要求

\[
M_{C,h}\le0.99M_{E,h}.
\]

此外对每 seed 定义固定精化差异代理

\[
d_s=\max(|M_{C,.005}-M_{C,.0025}|,
|M_{E,.005}-M_{E,.0025}|),
\]
\[
\min_{h\in\{.005,.0025\}}(M_{E,h}-M_{C,h})
>\max(10^{-12},10d_s).
\]

1% 是本轮预定的最低工程收益要求，决定是否值得继续投入同构训练，不是从新结果挑出的百分比或统计显著性。d_s 是两档结果差异代理，不是精确 ODE 的严格误差上界。若 E 的 MSE 为零，正收益条件无法通过，百分比展示为 null，不加 epsilon。

六个 seed/步长组还须各自满足六项保护：horizon1、horizon3、q 主分项、v 主分项、CVaR10%、完整 3 s 子集纯 horizon3，均 `C≤E+1e-12`。不跨 seed 平均抵消，不挑最佳 seed 或多数表决。


<a id="doc-a3dca8fc03e4-g_id-与最终裁定"></a>

##### G_I、D 与最终裁定

G_I 按与 G_E 完全相同的主改善、两步长代理和六项保护计算，将 E 换成 I；这是相对被拒绝流程的机制对照，不能替代 G_E。两组全部标量、差值和逐项布尔值保留。

D 要求所有六个组分别 `N_miss^C≤N_miss^E` 且 `N_extra^C≤N_extra^E`。这是对原质量参照的规则保护；I 的同类计数完整报告，但不以只优于 I 通过 D。

M 表示至少一个槽实际进入 active 或 rho=0 的新传播分支，并报告每 seed 的事件数和原权重质量；非活跃复用不计主动收益。最终 `scientific_verdict=ACCEPT` 当且仅当 `V and M and G_E and D`。G_I 独立记录，不新增为替代主门。其余为 REJECT，并分别标清数值未完成、机制未受到检验、未来收益失败或规则保护失败。

| 结果 | 研究决定 |
| --- | --- |
| V、M、G_E、D 全通过 | 支持这个固定经验球初始化流程；可提出同一因果映射的可微训练设计、梯度检查和独立预算，不自动开始训练 |
| 只通过 G_I，未通过 G_E/D | 具体流程改善了被拒绝参照，但未达到原 encoder 的继续条件；拒绝直接替换 |
| V 通过、约束活跃而 G_E/D 失败 | 有效负结果；调整“该经验能量球可直接修复迁移”的具体路线，不外推为所有能量约束或联合训练不可能 |
| M=false、rank 未决或数值失败 | 分别记机制未检验或未完成；不改半径、补样、缩步或重试 |

任何结果都不晋升模型，不把当前 train 机制比较称为新独立确认，不结束尚未检验的端口、势能、时间目标或覆盖问题。若具体方案失败，后续目标调整须以完整结果为依据另行提出，不能立即把结构扩张当作已证修复。


<a id="doc-a3dca8fc03e4-8-冻结范围与有限预算"></a>

#### 8. 冻结范围与有限预算

令 A 为 774 槽中 active 或 rho=0、需生成新路径的槽数，S_A 为这些槽的 N_i 之和，R 为其中正乘子活跃槽数。非活跃复用不消耗新路径预算；预算预留按最坏分支，未用额度不转移。[机器计划](../../../../configs/proposals/kiki_energy_cap_transfer_v1.json)冻结身份、人口、逐 scope 上限、全局预算及准确实现绑定；最终实施收据须逐项核对这些字节与以下调用树。

| 项目 | 固定数或条件式 |
| --- | --- |
| 方法记录 | 774 槽×E/I/C，两个步长完整保留；另有 268 个分配覆盖记录 |
| ball solver | 每槽最多一次，共≤774；根更新≤64R，标量评估≤80R |
| 新历史 / 新未来路径 | 各 2A；总 4A≤3,096 |
| RK4 更新 | 60A+6S_A |
| 积分 RHS | 240A+24S_A |
| 每输出点物理 RHS | 24A+2S_A |
| 全部新 RHS | 264A+26S_A；若 A=774、S_A=3×72,147，上限为 5,831,802 |
| 新 SVD / expm / 通用无约束 LS / Gauss 力求积 / encoder | 均为 0；球求解内矩阵代数另记，不能称零计算 |
| 模型构造与内部结构检查 | 三个固定模型；PassiveLatentClosure 验证共 3+4A≤3,099 次，其中 eigvalsh 共 9+12A≤9,297 次 |
| scope | 3,873+2A≤5,421；固定部分含来源、初始化、历史、历史审核、时钟、评分及汇总 |
| probe | 全局 37,199：源文件/成员 30,227；科学与汇总 6,972。每项按固定调用记录，未用额度不转移 |
| 拟合 / 优化更新 / val/test / 标签读取 | 均为 0 |

来源读取按因果冻结顺序分成三阶段；每文件、每获准成员最多读取一次：

| 阶段 | 文件数 | 成员数 | 原始文件 bytes | 数组逻辑 bytes |
| --- | ---: | ---: | ---: | ---: |
| 来源根记录、全部初始化与历史 | 5,430 | 13,956 | 189,856,779 | 13,067,688 |
| 全部初始化/历史冻结后的 future clock | 2 | 3 | 11,047,690 | 583,376 |
| 全部未来预测冻结后的 E/I/truth 评分输入 | 3,096 | 7,740 | 217,375,367 | 226,176,048 |
| 总计 | 8,528 | 21,699 | 418,279,836 | 239,827,112 |

文件总数包括 4,653 个 JSON 和 3,875 个 NPZ；成员数只计获准 NPZ 成员。原始文件读取上限 418,279,836 bytes，数值载荷逻辑读取上限 239,827,112 bytes，NPY 头部及格式前缀另限每成员 4,108 bytes。完成时核对实际源文件、成员、逻辑字节与原始字节计数；中途失败只保留已用计数，不补读。

准确实施审核与运行前门禁使用以下已建立的记录：

| 记录 | 绑定内容与条件 |
| --- | --- |
| [机器计划](../../../../configs/proposals/kiki_energy_cap_transfer_v1.json) | 新唯一 scope/authority、268/258/774 人口与原权重、N_i/时钟/offset、逐 scope 与全局算子上限、条件完成公式及实现绑定 |
| [来源清单](../../../../configs/proposals/kiki_energy_cap_saved_sources_v1.json) | 逐文件 SHA、逐成员 shape/dtype/字节、阶段、来源 descriptor 键及不可变引用；不得在运行中扩展 |
| [workflow](../../../../configs/workflows/kiki_energy_cap_transfer_v1.json)与[experiment](../../../../configs/experiments/kiki_energy_cap_transfer_v1.toml) | 独立入口及一次性共同账本；旧 consumed 授权保持关闭 |
| `implementation_bindings/runtime/tests` | 由机器计划绑定新 kernel/reader/workflow/评价 helper、冻结 solver/模型和依赖；Python 3.12、实际模块路径及纯人工工程测试最终核验后才能签收据 |
| `review/activation` | 尚须 Astra 准确实施收据、独立授权登记、clean pushed commit、该提交 CI 及运行前路径/哈希守卫；设计冻结不替代这些条件 |


<a id="doc-a3dca8fc03e4-9-失败保存与独立审核"></a>

#### 9. 失败、保存与独立审核

正式入口在任何许可来源 NPZ 校验/解码或科学调用前原子消费新共同一次性预算。范围预留 durable，逐调用 attempted/returned 留存，完成后核身份与范围摘要；未决/失败/日志或预算错误停止全轮，保留其余 not_attempted 槽。中断范围为 unknown/spent，不恢复、不重读、不重试。不因指标未达阈提前跳过后续正常独立槽；科学分数在固定路径全部冻结后统一判定。

新路径返回后先保存，再执行物理检查或评分；失败中间态、最后一次 RHS、solver diagnostics、已经读到的必要输入和实际预算不丢失。run 保存本次执行计划、完整来源清单、三个固定模型的六参数及尺度。保留完整 observed history、L/b/SVD/zh4、半径来源、C 历史和未来、各 reset、能量及全部评分/规则中间量。旧 E/I 历史、E/I/truth 未来及对应时钟允许采用新来源清单内的不可变引用，逐项保留 path、SHA、member、seed、event、步长和角色；不能仅凭旧病例名称定位。

输出至少包含 `process_status,V,mechanism_exercised,G_E,G_I,D,scientific_verdict,promotion_decision`，逐槽 solver 分支和每 seed/步长门槛明细。固定 `promotion_decision=KEEP_BASELINE, selected_model=null, certified=false, new_training_authorized=false, val_evaluation_authorized=false`。process PASS 与科学 REJECT 可以同时成立。

新 scope 明文允许两次独立的保存证据审核，用于核对来源、预算、KKT、reset、能量和指标。每次审核只可读取新 run 的 manifest 及其列出的全部产物一次，并读取同一来源清单内原文件/成员各最多一次；原来源每次上限为 8,528 文件、21,699 成员、418,279,836 原始 bytes、239,827,112 数组逻辑 bytes，NPY 头部限制同上。新 run 产物读取按其 manifest 独立记账，不能转移为额外来源额度。两次审核各自记录读取账本，与正式运行分账；旧 E/I/truth 引用的解码仅在这一新许可内进行。

审核禁止新 solver、模型构造、RHS、SVD、expm、拟合和路径传播，只做已保存数组的指标与代数复核。准确实施审核和最终科学结果审核分别留收据；本文件冻结设计，只有绑定收据、独立授权及运行前门禁共同成立才可运行。


<a id="doc-a3dca8fc03e4-依据"></a>

#### 依据

- [100 ms 准备稿 §5](../constant_c_100ms_synthetic/README.md#doc-d306b4c8a553)：经验能量先验、训练同构、端口/势能及时间目标的边界。
- [A14 设计](../../04_kiki_a03_a17_pipeline/a14/README.md#doc-3727f048e593)与[结果审核](../../04_kiki_a03_a17_pipeline/a14/README.md#doc-82c14511d5e8)：原人口、积分方程、能量和历史收益。
- A15 归档提交 `5ac8b571751467f677fb569df3c0a1f4c323d065` 中 `docs/research/kiki_a15_future_transfer/design_zh.md`、`astra_result_review_zh.md`：原 future 口径、规则和有效负结果；只读文本参考，不导入其代码。
- [冻结球求解准备协议](../constant_c_energy_initializer/README.md#doc-9ca7305815de)及[真实机制前的合成确认审核](../constant_c_energy_confirmation/README.md#doc-78265c298ea7)：数值求解与有限合成证据。

<a id="doc-bab170f31860"></a>

<a id="doc-bab170f31860-代表真值窗口的独立保存证据补核"></a>

### 代表真值窗口的独立保存证据补核


<a id="doc-bab170f31860-原因与范围"></a>

#### 原因与范围

原数值审核已保存为 `reports/generated/kiki_energy_cap_transfer_20261110/numeric-audit.json`，SHA256 `c419c61a743dcf116bae7049f7d93fcd338751cc4264f1efe63f85533c2ddafd`。它从粗步长候选路径选代表窗口，而冻结实验要求从真实路径选窗。该口径错误涉及 774×3 个方法记录的八个字段及固定窗口 bound，共 20,898 项检查；其中 15,407 项失配。原报告原样保留，不改写为通过。

新范围 `kiki_energy_cap_truth_window_audit_v1_20261110` 仅补核上述 20,898 项。复用原审核中不受影响的 221,204 项；不能把错误口径下碰巧通过的检查重复计入。正式科学运行、源授权和原两次审核均不重开。


<a id="doc-bab170f31860-计算与读取"></a>

#### 计算与读取

对每个事件，将真实 100 ms 历史与真实未来第 1 点起的路径拼接。仅在末点索引≥11的五点窗口中，以冻结规则余量最大者作为代表窗口；平局取末点最早者。固定该窗口后，对 E/I/C 各方法的粗细步长路径计算窗口索引、两种余量、两侧固定窗规则、E_atom、E_vector 和 bound。比较容差沿用原审核，不能为减少失配放宽。

[机器计划](../../../../configs/proposals/kiki_energy_cap_truth_window_audit_v1.json)逐文件绑定哈希、大小和数组成员。新 run 仅读取已保存的历史、平衡点、质量、omega_s、C 未来路径及评分 JSON；A15 保存来源仅读取两步长 E/I 未来，真实未来只读取粗步长那一份。每文件和每成员最多一次，缓存仅在本次进程内复用。

上限：4,991 文件、484,126,094 文件 bytes、5,206 成员、254,839,136 数组载荷 bytes；每成员 NPY 头部及前缀≤4,108 bytes。源 provider 载荷、标签、val/test 均不读取。新求解、模型构造、RHS、SVD、expm、拟合和路径传播均为零。不重算原审核的全局指标和科学判据。


<a id="doc-bab170f31860-授权与收尾"></a>

#### 授权与收尾

协调代理依据用户持续自主推进的授权登记此独立证据审核，Astra 对准确脚本、机器计划、读取边界和人工反例签收后方可执行。调用前独占创建新范围的消费记录，失败亦消费；不重试、不覆盖输出，不借用原范围的未用额度。

原实验实现和判据保持冻结。补核独立输出实际/期望索引、失配明细、读取账本、代码与计划哈希；随后将新范围标为 consumed。最终报告并列保留初审计 FAIL、错误原因及补核结果。科学决定以已冻结的主指标和规则判据为准，补核不产生新候选或新的实验成绩。

## 报告与结果

<a id="doc-4eaeefc4b469"></a>

<a id="doc-4eaeefc4b469-encoder-参照能量球固定-a06-的-train-迁移结果"></a>

### encoder 参照能量球：固定 A06 的 train 迁移结果

2026-11-10。固定三个 A06 step240 模型，在 258 个合格 train 事件、774 个 seed—事件组合上比较初始化方案。保留全部 268 个分配事件的覆盖记录。输入为原 100 ms 因果历史，未来仅使用实际可用的最多 3 s。


<a id="doc-4eaeefc4b469-结果与路线决定"></a>

#### 结果与路线决定

**拒绝用本次能量球初始化直接替换原 encoder，保持 KEEP_BASELINE。** 约束在 666 个组合中激活，控制了历史起点隐藏状态的模长；相对无约束积分初始化，未来主指标改善 44.54%–49.07%，但仍为原 encoder 的 7.86–8.10 倍。三个 seed 均出现 2 个额外规则触发，未通过规则保护。

本轮给出了新的真实 train 机制证据：在固定 A06 模型、固定积分方程和本次因果半径规则下，控制初始隐藏储能并改善历史拟合，仍不足以恢复 encoder 的未来表现。因此结束这条直接替换路线，取消以它为已验证前提继续推进训练的安排。联合训练、状态依赖端口、势能和时间目标仍是独立候选，须各自提出可证伪机制与协议；本结果没有判定它们不可能。


<a id="doc-4eaeefc4b469-比较口径"></a>

#### 比较口径

- **E：原 encoder。** 复用冻结的原历史与未来路径。
- **I：无约束积分初始化。** 复用获准的 A14 初始化及 A15 保存路径；旧执行授权保持 consumed。
- **C：encoder 参照能量球。** 令历史起点半径为同事件的 `rho = ||z_encoder,h||`，复用 I 的加权积分矩阵、右端和 SVD，求球约束最小二乘。半径只由合法历史决定，是经验参照，不是真实隐藏能量证书。

三方法使用同一模型、历史输入、可见末点重置和未来输入。0.005 s 与 0.0025 s 两种最大积分步长独立传播各自的历史隐藏末态。C 未激活的 108 个组合精确复用 I；其余 666 个组合生成新路径。先冻结全部初始化与历史，再读取未来时钟；再冻结全部预测，最后解码获准的 E/I/truth 未来评分载荷。

主指标 M 为沿用冻结尺度归一化后的均方误差：各事件分别计算真实可用点上的 1 s、3 s 均方误差，等权平均两个时域，再按原事件权重汇总。它是归一化 **MSE**，不是 RMSE。下表列 0.0025 s 结果；0.005 s 得出相同判定。

| seed | E：原 encoder | I：无约束积分 | C：能量球 | C 相对 I 改善 | C / E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 324 | 0.0610608 | 0.958147 | 0.487971 | 49.07% | 7.99 |
| 325 | 0.0600571 | 0.877574 | 0.486669 | 44.54% | 8.10 |
| 326 | 0.0609344 | 0.921125 | 0.479183 | 47.98% | 7.86 |

C 相对 E 的 1 s、3 s、q、v、最高误差 10% 权重质量的均值（CVaR10%）、完整 3 s 子集六项保护全部失败；相对 I 的相同保护全部通过。优于 I 的结果不能替代相对 E 的门槛。


<a id="doc-4eaeefc4b469-历史储能与失步规则"></a>

#### 历史、储能与失步规则

| seed | E 历史 MSE | I 历史 MSE | C 历史 MSE | 约束激活的原权重质量 |
| --- | ---: | ---: | ---: | ---: |
| 324 | 0.0402632 | 0.0195328 | 0.0276027 | 84.37% |
| 325 | 0.0383674 | 0.0190198 | 0.0261608 | 83.99% |
| 326 | 0.0406995 | 0.0197082 | 0.0270267 | 86.57% |

历史指标使用 100 ms 历史中起点之后的观测，尺度和事件权重均保持冻结。C 相对 E 的历史 MSE 降低 31.44%–33.59%，仍伴随显著更差的未来 MSE。上表最后一列是原权重质量，不是事件数量比例，也不是逐事件能量翻倍比例。

每个组合均保留历史起点与预测起点的隐藏储能、可见重置能量跳变、约束乘子和 KKT 残差。球约束只直接限制历史起点，不给传播后的隐藏储能设上界。真实系统没有提供隐藏状态真值，本轮不产生隐藏状态恢复精度结论。

下表为 258 个 train 可见轨迹的冻结五点规则比较，其中 21 个真值轨迹触发规则；两个步长计数一致。“漏掉”指真值触发而预测未触发，“额外”指相反情况。

| seed | E 漏掉 / 额外 | I 漏掉 / 额外 | C 漏掉 / 额外 |
| --- | ---: | ---: | ---: |
| 324 | 17 / 0 | 12 / 3 | 16 / 2 |
| 325 | 17 / 0 | 12 / 3 | 16 / 2 |
| 326 | 17 / 0 | 12 / 4 | 16 / 2 |

C 少漏掉一个规则触发，不能抵消两个新增额外触发，因此 D=false。这是获准 train 可见轨迹的机制诊断，不是源标签分类结果、val 成绩或新的 8 s 漏判率。


<a id="doc-4eaeefc4b469-判据与执行证据"></a>

#### 判据与执行证据

| 判据 | 结果 |
| --- | --- |
| V：来源、预算、物理、数值与配对有效性 | true |
| 约束实际发挥作用 | true，666 个组合激活 |
| G_E：相对原 encoder 的收益及六项保护 | false |
| G_I：相对无约束积分的独立收益及保护 | true |
| D：相对 encoder 的规则保护 | false |
| 科学结果 / 模型决定 | REJECT / KEEP_BASELINE |

实际生成 2,664 条路径，使用 4,690,800 次积分 RHS 和 393,564 次物理 RHS，共 5,084,364 次；37,199 次 probe。球求解完成 24,132 次根更新、25,464 次标量评估。5,205 个范围完成关闭，全部调用返回，预算没有重试或转移。

本轮读取白名单内 8,528 个保存文件、21,699 个数组成员，原文件 418,279,836 bytes、数组载荷 239,827,112 bytes、NPY 头部及前缀 2,777,472 bytes。原始 provider 载荷、内部 val、源 val/test 和标签均未读取；拟合和优化更新均为零。默认模型与初始化器不变。

70 项专项工程测试通过；源提交的 CI 为 666 passed、1 deselected。独立来源与预算审核、独立保存数组审核及准确哈希记录见[证据收据](_sealed/kiki_energy_cap_transfer_20261110_evidence.json)。最终科学决定见[Astra 结果审查](#doc-6c9728c858a8)。

独立数值审核最终保留 221,204 项不受影响的检查，加上另行登记的 20,898 项代表真值窗口补核，合计 242,102 项有效检查，零剩余失配。初审核误从粗步长候选路径选窗，产生 15,407 项失配；初始 FAIL 原样保留，补核替换整个错误口径下的检查族，未重新求解或传播。元数据审核的六项文件、来源、人口和算子检查全部通过；其授权状态项在政策收尾后单独闭合，保留了运行时 `user_approved` 有效且共同预算已消费的完整记录。

运行：`kiki_energy_cap_transfer_v1-train_transfer-20261109T150903551189Z-89135853-cf169575`；源提交：`891358534869dea5810893083bb651b95c378a32`。23,061 个文件的清单 SHA256 为 `6abadbc49d115e297fff59c2be018fa936d9f281ceeb9e9208157f5404cb751d`。候选实现与证据保留在 `codex/encoder-energy-cap-transfer`，不合入默认模型路径。

[冻结协议](#doc-a3dca8fc03e4)与[准确实施批准](#doc-31d26f640ba4)限定本次一次性范围。结果支持调整这一具体替换方案，不把初始能量、端口外残差或共同能量屏障单独认定为以往失败的已证实根因。

## 复核与历史版本

<a id="doc-31d26f640ba4"></a>

<details>
<summary>encoder 参照能量球迁移：准确实施审核</summary>

<a id="doc-31d26f640ba4-encoder-参照能量球迁移准确实施审核"></a>

### encoder 参照能量球迁移：准确实施审核

2026-11-09，Astra Ultra（任务分派配置 `gpt-6-astra`、`ultra`）。

**准确实施审核通过。** 依用户持续推进的授权，批准新范围 `kiki_energy_cap_transfer_v1_20261109` 在独立登记、干净提交已推送、对应 CI 和实际运行环境守卫通过后，执行唯一一次固定 A06 模型的 train 机制比较。范围为三个 seed、258 个合格事件、774 槽，每槽保留 E/I/C 的两档结果；268 个原分配事件的覆盖记录全部保留。批准不预设科学结果。

| 对象 | SHA256 |
| --- | --- |
| [机器计划](../../../../configs/proposals/kiki_energy_cap_transfer_v1.json) | `3ca756b4be5c6c73a26864258a7554a34c27b87cd3f03bc75e0389c284a63bf6` |
| [冻结协议](#doc-a3dca8fc03e4) | `0e0cca43e50dbb9a6b03c9311b84c7c413ebe604bfba3d8ab06c9e76a1272219` |
| [来源清单](../../../../configs/proposals/kiki_energy_cap_saved_sources_v1.json) | `d0dcdfdcebf768ae29d2f1b7e87d40e773f1fa5ad29095b63f1f1ac1f80a3ed1` |
| `scratch/one_off/energy-cap-registration/engineering-tests.txt` | `8c947719d684e6c0ec506c43acef6ca248b6c4ed951bbf2fc6788b34cb2be891` |

已逐项读取并核对计划的 **40 项实现绑定**，全部匹配。最终工程记录为 **70 passed**，Ruff 通过；覆盖人工数组求解适配、评分与裁定、白名单读取、因果冻结、失败保存以及临时 Git 一次性执行门禁。最终修订补齐 NPY 头部和载荷读取失败时的实际字节计数，以及完成时逻辑字节与头部上限断言；不改变候选、容差、判据或预算。

实现保留唯一因果半径 `rho=||z_encoder,h||`，直接使用同槽 A14 的 Ltilde/btilde4/SVD/zh4。冻结球求解器执行零半径优先、正半径秩门、SVD 一致性、KKT 和有限求根检查；非活跃分支精确复用 I。两档历史各自闭环、各自保留隐藏末态，只把可见状态重置为同一获准观测。全部 774 槽初始化与历史冻结后才读取未来时钟，全部候选未来或非活跃不可变引用冻结后才解码 E/I/truth 未来评分载荷。半径与预测不读取未来目标。

读取范围为清单内已保存的 train 证据：8,528 文件、21,699 成员、418,279,836 原始 bytes、239,827,112 数组逻辑 bytes；每文件、每成员一次。三阶段文件数为 5,430/2/3,096。NPY 头部与格式前缀每成员最多 4,108 bytes，先核 shape/dtype/大小，再读取声明载荷。scope 中 `source_payload_reads=0` 保留原 provider 数据禁读含义；本收据明确批准上述独立清单内的保存载荷，不表示本轮没有真实保存数据读取。

令 A 为 active 或零半径的新传播槽数、S_A 为其未来点数之和、R 为正乘子活跃槽数。新路径 `4A`，积分 RHS `240A+24S_A`，物理 RHS `24A+2S_A`，合计≤5,831,802；scope 为 `3,873+2A`、最多 5,421，probe 为 37,199。根更新/评估≤64R/80R。内部 closure 验证 `3+4A`、eigvalsh `9+12A` 单独列明；新 SVD、expm、通用无约束 LS、Gauss 力求积、encoder 与拟合均为零。实际分支计数须与保存账本一致。

公共一次性预算在科学读取或调用前消费，逐 scope 先持久化预留，再记 attempted/returned。数值、物理、秩、预算、日志或来源错误停止全轮，保留已返回数组、最后 RHS、求解诊断和未运行槽，不重试、不缩步补救。run 保存计划、来源清单、模型六参数及必要结果；旧 E/I/truth 使用准确不可变引用。允许两次分账的独立保存证据审核，各自至多读取同一来源文件/成员一次及新 run manifest 所列产物一次，禁止新求解、模型构造、RHS、SVD、expm、拟合或传播。

科学接受仍要求 `V and mechanism_exercised and G_E and D`：六个 seed/步长组相对原 encoder 达到预定 1% 主指标收益、超过粗细差异代理，并通过六项保护和规则保护。G_I 完整独立报告，优于已拒绝参照不能代替 G_E。任何结果均 `KEEP_BASELINE`；不自动授权训练、val/test、模型晋升、证书或结构扩展。所有来源研究的 consumed 授权保持关闭。

本审核仅读代码、Markdown、JSON 和工程日志并核验文件哈希，没有打开来源 NPZ，也没有调用模型、球求解器、RHS 或路径传播。运行批准由[准确 JSON 收据](_sealed/kiki_energy_cap_implementation_review_20261109.json)与新独立授权共同绑定；执行前条件不能由本文审核结论替代。

</details>

<a id="doc-6c9728c858a8"></a>

<details>
<summary>encoder 参照能量球迁移：最终科学结果审核</summary>

<a id="doc-6c9728c858a8-encoder-参照能量球迁移最终科学结果审核"></a>

### encoder 参照能量球迁移：最终科学结果审核

2026-11-10，Astra Ultra。**科学裁定：REJECT；模型决定：KEEP_BASELINE。** 数值与证据审核已闭合，拒绝的是在固定 A06 模型中直接用本次能量球积分初态替换原 encoder 的方案。

本轮比较三个 A06 step240 模型、同一 258 个合格 train 事件，共 774 槽；全部 268 个原分配事件的覆盖记录保留。E 为原 encoder，I 为无约束积分初态，C 为使用同历史 encoder 首点范数作半径的球约束积分初态。774 槽均完成，666 槽约束活跃、108 槽精确复用 I；三个 seed 的活跃槽数为 221/219/226，对应原权重质量 84.37%/83.99%/86.57%。机制已受到实际检验。

以下为步长 0.0025 s、原 source→line→event 权重下的归一化 MSE。未来主指标 M 先对每事件真实可用的 1 s/3 s 窗口等权，再按事件权重汇总；历史指标使用 100 ms 中除首点外十个观测点。两者均不是 RMSE。

| seed | 历史 E | 历史 I | 历史 C | 未来 E | 未来 I | 未来 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 324 | 0.04026325 | 0.01953282 | 0.02760268 | 0.06106081 | 0.95814668 | 0.48797057 |
| 325 | 0.03836744 | 0.01901983 | 0.02616079 | 0.06005712 | 0.87757441 | 0.48666908 |
| 326 | 0.04069947 | 0.01970816 | 0.02702673 | 0.06093439 | 0.92112545 | 0.47918289 |

C 相对 I 的未来 M 降低 49.07%/44.54%/47.98%，但仍为 E 的 7.99/8.10/7.86 倍。每个 seed、每档步长，相对 E 的六项保护全部失败：1 s、3 s、q、v、CVaR10% 和完整 3 s 子集；相对 I 的这六项全部通过。0.005 s 与 0.0025 s 结论一致。C/E 主指标粗细差异代理为 1.92e-10 至 3.40e-10，远小于 C−E 的 0.418 至 0.427；拒绝不是接近 1% 门槛的边缘结果。E 对照保存的 `effect_resolved=false` 表示规定的正收益未成立，不能解释为数值上无法辨认这次退化。

规则保护也失败。六个 seed/步长组均为 E 漏触发/额外触发 17/0，C 为 16/2；减少一个漏触发不能抵消两个新增额外触发。I 的计数为 12/3、12/3、12/4，C 相对 I 也存在两类错误的权衡。这是获准 train 可见轨迹、真实可用未来段上的五点规则诊断，不是新标签分类或 8 s FNR。

因此冻结判据为 `V=true, mechanism_exercised=true, G_E=false, G_I=true, D=false`。接受条件 `V and mechanism_exercised and G_E and D` 不成立。球约束、KKT、重置及数值过程有效，与候选科学拒绝可以同时成立；本轮初始能量控制不能被写成未来精度保证或真实隐藏状态恢复。


<a id="doc-6c9728c858a8-具体路线调整"></a>

#### 具体路线调整

本次新增证据表明：在这些固定模型与事件上，把积分初态限制到同事件 encoder 的起始模长范围，可以缓解无约束方案的迁移退化，但不足以恢复 encoder 的未来表现。C 的历史闭环误差仍低于 E，未来却显著更差，说明该具体历史拟合与初始能量控制流程未满足未来目标。它没有识别出方向误差、模型失配、端口或势能中的哪一项是唯一根因。

- 保留原 encoder、模型参数及默认初始化流程，结束这条固定模型直接替换分支；不通过补调半径、挑选事件或增加容易通过的合成病例改变本轮裁定。
- 保留“能量球改善 I 的连续误差”这一有限结果，撤回把它作为“已修复迁移、可以直接替换 E”的依据。G_I 的收益不能代替 G_E 和 D。
- 取消以本方案已经有效为前提自动进入同构训练的安排。联合训练若继续，须作为新假设说明训练/使用映射一致化如何改变未来目标，并独立冻结梯度验证、训练目标和预算。联合训练、状态依赖端口、势能及时间目标候选未在本轮执行，也未被本轮证伪。

这满足用户长期目标中“经验证发现具体方案需要调整”的完成条件；不等于原四阶段目标全部完成，也不等于其他候选不可能。后续应从修订后的研究问题重新确定工作范围。


<a id="doc-6c9728c858a8-审核闭合与证据"></a>

#### 审核闭合与证据

原数值审核保留为 **FAIL**：242,102 项中 15,407 项失配，全部来自代表窗检查族。审核器错误地从 coarse 预测选窗，冻结科学实现从原真值选窗。修正未改变实验代码、保存路径、阈值或科学判据，而是登记新独立范围 `kiki_energy_cap_truth_window_audit_v1_20261110`，仅对 774×3 个方法记录重核八个字段及固定窗 bound。

补核 **20,898 项全部通过、零失配**，另有 **12 项读取账本检查通过**。它替换旧审核整个 20,898 项检查族，包括错误口径下碰巧通过的项；其余 221,204 项原检查保留。有效数值检查为 **221,204+20,898=242,102**，没有重复计数。补核仅读 4,991 文件、5,206 成员，484,126,094 原始 bytes、254,839,136 数组 bytes、666,368 头部 bytes，独立一次性预算已消费；没有新求解、模型构造、RHS 或传播。

原元数据审核也保留为 FAIL：六项检查通过，授权状态收尾项未通过。运行时 `user_approved` 是合法状态，公共预算已持久消费；独立 closeout 核对当前 consumed 状态和共同 completed 收据后，该项通过。原六项以引用保留，七项有效元数据检查完成，没有重扫运行工件或源数组。

| 保存证据 | SHA256 |
| --- | --- |
| [原数值审核 FAIL](../../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.DESCRIPTION.md) | `c419c61a743dcf116bae7049f7d93fcd338751cc4264f1efe63f85533c2ddafd` |
| [代表窗补核 PASS](../../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.DESCRIPTION.md) | `f14a9930021eb95567d02b1ae0765bacbbd39231ee777adabff301ea503b519e` |
| [原元数据审核 FAIL](../../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.DESCRIPTION.md) | `e99728eaa3724edc8ab4bfb689e451326a3f8338e116ea022073c4a322c8a1b5` |
| [元数据收尾 PASS](../../../../results/backup/06_constant_c_energy_initialization/04_kiki_energy_cap_transfer.DESCRIPTION.md) | `328ba207ed3e7cb5fd9a0879bb5c2207220b006a717675508f4aa1454ab5b6d4` |

原科学运行实际生成 2,664 条路径，积分 RHS 4,690,800 次、物理 RHS 393,564 次，合计 5,084,364；probe 37,199，求根更新/评估 24,132/25,464，attempted 与 returned 一致。补核预算与它分账。默认模型、训练状态及旧 consumed 授权保持原样。

run 为 `kiki_energy_cap_transfer_v1-train_transfer-20261109T150903551189Z-89135853-cf169575`；source commit 为 `891358534869dea5810893083bb651b95c378a32`，封存 manifest SHA256 为 `6abadbc49d115e297fff59c2be018fa936d9f281ceeb9e9208157f5404cb751d`。本科学审核只读代码与封存 JSON，未打开 NPZ 或调用科学算子。完整比较见[结果报告](#doc-4eaeefc4b469)，机器裁定见[本审核 JSON](_sealed/kiki_energy_cap_result_review_20261110.json)。

</details>

## 回执与附件

- [kiki_energy_cap_transfer_20261110_evidence.json](_sealed/kiki_energy_cap_transfer_20261110_evidence.json)
- [kiki_energy_cap_implementation_review_20261109.json](_sealed/kiki_energy_cap_implementation_review_20261109.json)
- [kiki_energy_cap_result_review_20261110.json](_sealed/kiki_energy_cap_result_review_20261110.json)
- [kiki_energy_cap_truth_window_approval_20261110.json](_sealed/kiki_energy_cap_truth_window_approval_20261110.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_energy_cap_transfer_v1_zh.md](_sealed/kiki_energy_cap_transfer_v1_zh.md)
- [docs/protocols/kiki_energy_cap_truth_window_audit_20261110_zh.md](_sealed/kiki_energy_cap_truth_window_audit_20261110_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/research/kiki_energy_cap_result_review_20261110_zh.md](_sealed/original/kiki_energy_cap_result_review_20261110_zh.md)
- [docs/reports/06_constant_c_energy_initialization/_sealed/kiki_energy_cap_transfer_20261110_zh.md](_sealed/original/kiki_energy_cap_transfer_20261110_zh.md)
- [docs/reports/06_constant_c_energy_initialization/kiki_energy_cap_transfer_20261110_zh.md](_sealed/kiki_energy_cap_transfer_20261110_zh.md)
- [docs/research/kiki_energy_cap_design_review_20261109_zh.md](_sealed/kiki_energy_cap_design_review_20261109_zh.md)
- [docs/research/kiki_energy_cap_implementation_review_20261109_zh.md](_sealed/kiki_energy_cap_implementation_review_20261109_zh.md)
- [docs/research/kiki_energy_cap_result_review_20261110_zh.md](_sealed/kiki_energy_cap_result_review_20261110_zh.md)

</details>
