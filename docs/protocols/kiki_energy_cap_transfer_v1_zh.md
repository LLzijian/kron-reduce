# encoder 参照能量球：固定模型的 train 未来迁移协议

2026-11-09，Astra Ultra 科学设计。状态：`EXECUTION_SCOPE_FROZEN_APPROVAL_BY_BOUND_RECEIPT`。本文件冻结研究设计；实际批准状态由准确实施收据与独立授权记录共同决定。

本轮回答一个真实机制问题：在固定 A06 三个模型上，将积分初态限制在同一合法历史的 encoder 起始能量范围内，能否改善原 encoder 的未来预测，并避免无约束积分初始化的迁移退化。唯一新增候选为这个逐例能量球；模型参数、输入策略、历史长度、评分和原人口不变。

用户继续推进的指令支持自主完成设计、独立登记、准确实现审核及随后一次有限比较。来源清单与预算按本协议及机器计划冻结；准确实施收据、独立登记、clean pushed commit、对应 CI 和运行前守卫全部通过后，才能启动一次科学读取与计算。本文件本身不提供 active 批准。既有 A06、A14、A15、A17 及合成研究的 consumed 授权保持关闭。本轮不训练、不修改默认初始化器，终态始终 `KEEP_BASELINE`。

## 1. 候选、参照与问题范围

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

## 2. 固定人口、模型与可读来源

使用 A06 step240、seed 顺序 324、325、326，n=10、r=2、原 sine 势能、质量、阻尼、C、J、R、平衡先验及 target_scale。原分配 268 个 train 事件全部保留，其中 258 个满足 record100ms 合同，共 774 个 seed/事件槽；另外 10 个明确记为不覆盖，不补样、不视作正确预测。三个 seed 是同一人口的算法重复，不当作 774 个独立真实事件。

原事件顺序和 source→line→event 权重直接继承冻结人口：

\[
w_i=\frac1{N_{source}}\frac1{N_{line(source_i)}}
\frac1{N_{event(source_i,line_i)}},\quad \sum_iw_i=1.
\]

保存原浮点权重，不为消除舍入差重新归一化。ID、来源、线路和已声明长度只服务身份、分母和分组，不进入估计器。

来源身份采用已归档记录，逐文件/逐成员范围冻结于[来源清单](../../configs/proposals/kiki_energy_cap_saved_sources_v1.json)：

- A06：`paper_kiki_a06_refinement-joint_refinement-20261024T025925447927Z-9aa5ee65-524be996`，manifest `d1c4cc402dd30207c4ba639a9d4d1f1e86e3051ce0ce51e72ac4d045be17cc01`。优先读取 A14 已保存的同源六参数，避免重复解码 checkpoint。
- A14：`paper_kiki_a14_integral_history-history_mechanism-20261024T121439628465Z-34de0389-4cf0648f`，manifest `7438242d358244dece184a393ef871374322d04415237fff733ecd7c01767938`。提供原历史、输入、尺度、静态参数、积分方程/SVD、E/I 历史及对应 reset。
- A15：`paper_kiki_a15_future_transfer-future_transfer-20261024T145313076879Z-f23e9b23-c2066d4e`，manifest `e67dbb7c2d62ac82017e9d1049fd7a01c492c6b9c3587aa841d7bb89d247f1c4`。只复用原保存 E/I/truth、future clock、人口和所需物理静态量。其归档提交为 `5ac8b571751467f677fb569df3c0a1f4c323d065`；不得导入或执行该拒绝分支代码。

同一数组有多个保存副本时，精确清单只选一个数值载荷来源，其余用已绑定身份验证。清单冻结成员数量、shape、dtype、元素数、字节、读取次数、阶段、descriptor 键和来源 SHA；机器计划绑定清单及实现的准确字节。文件存在和 JSON 描述不等于已完成载荷核验；准备与实施审核阶段不打开来源 NPZ。

正式运行只允许清单中的 train 保存成员，禁止 pickle，按成员读取；不得解码全部 NPZ 再切片。每个源文件只读一次原始 bytes 并核 SHA，缓存 bytes 用于后续成员读取。每个成员先读取 NPY 标识、版本、长度和头部，头部≤4,096 bytes，含格式前缀的读取≤4,096+12 bytes；先核对 shape、dtype、C/F 顺序和成员总字节数，再单次读取已声明大小的数值载荷，并检查有限值。原始文件 bytes、NPY 头部 bytes、解码数组的逻辑 bytes 分账，不能把缓存解码再次记为源文件读取，也不能遗漏解压后的读取。

明确排除 HDF、CSV、原 provider、splits、源标签、事件时间、Pe、电压、真实隐藏状态、未来 Pm、内部 val、source val/test、A11 和 A17 数值来源。旧标签或旧诊断结果不能作为新方法的输入。未来 q/v 只用于冻结后的评价。

## 3. 唯一估计方程、秩与 KKT

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

## 4. 历史、重置与未来链

每个新状态从原真实历史首点 q/v 和 z_C,h 出发，以原 Pm10 派生 constant_power、原 equilibrium 自由闭环传播整个 100 ms。q/v 不在历史中间注入，观测驱动隐藏末态不能代替闭环末态。

最大步长固定 .005/.0025 s。历史使用原保存 history_time 减首值，未来使用原保存 forecast_time；保持当前模型的 RK4、原子步规则及 COI 处理。先按冻结时钟验证每历史分别 20/40 次更新，每未来分别 2N_i/4N_i 次更新；不满足则停止，不更改分派或缩步修复。不得新建批量积分器。

两步长分别得到自己的 z_C,T，再以 `[q_obs(T),v_obs(T),z_C,T]` 重置，未来仍使用原 Pm10 常值。未来中间不重置、不裁剪 z。每条候选未来首行必须与该步长 reset 一致；E/I 对照首行与其自身已存 reset 一致，三者前 20 维与真实历史末点一致。

每方法保留 H_z,h、H_z,T、相对 E/I 的差与有定义的比值。另存历史末态 x^-、重置 x^+、总储能 H(x^-)/H(x^+) 和 DeltaH_reset；隐藏分量在重置前后逐元素相同。初始能量约束不要求 H_z,T≤Hbar；重置总能量变化不混入连续功率恒等式。无真实隐藏状态误差或“真实能量被恢复”字段。

因果冻结顺序：

1. 验证来源清单和合法历史成员；构造全部 rho、球解、两档历史/reset。非活跃槽冻结其 I 历史/reset 的源 SHA 和键。
2. 持久保存全部 774 槽初态/分支/reset 及身份清单。之后才解码独立 future clock；N_i 可提前从已绑定人口元数据读取，但不读取未来振幅。实现可从冻结 N_i 生成原 `np.arange(N_i+1)*.01` 网格，但必须与随后获准读取的保存 forecast_time 逐元素相等后才传播；只通过 allclose 不足以替代原时钟身份。该生成不增加观测或使用目标振幅。
3. 按固定 seed/事件/.005/.0025 顺序生成全部需要的新未来。非活跃未来提前冻结为 I 的不可变源引用；完成新未来及引用清单的持久冻结后，才解码 E/I/truth 未来载荷用于评价。
4. 评分不会反馈到半径、路径、输入、步长、分支或停止参数。

## 5. 轨迹指标：完全保持 A15 口径

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

## 6. 真实可用段的五点规则

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

## 7. 事前决定规则

### V：来源、数值与完整性

全 774 槽必须有确定成功分支，来源/白名单/人口/权重/预算/冻结顺序与保存审计通过；不存在失败、未知或正半径 rank 未决。成功 solver 的上述检查全通过。每条新历史及未来的每个保存输出点检查原能量恒等式绝对残差≤1e-8、q/v 质量 COI 绝对残差≤1e-10；纯能量、状态及所有派生量有限。

新路径粗细两档可见最大绝对差除以原 target_scale≤1e-5；隐藏最大二范数差≤`1e-5*(1+细档全路径最大隐藏范数)`，历史与未来分别检查。未来比较必须保留各自历史末态。E/I 继承原物理证书，同时在许可保存路径上核验有限性、初态和粗细差；不重做其 RHS 物理检查。第 6 节所有规则精化和界核验通过。数值有效不等于预测改善或稳定域证明。

### G_E：相对 encoder 的实际未来收益

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

### G_I、D 与最终裁定

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

## 8. 冻结范围与有限预算

令 A 为 774 槽中 active 或 rho=0、需生成新路径的槽数，S_A 为这些槽的 N_i 之和，R 为其中正乘子活跃槽数。非活跃复用不消耗新路径预算；预算预留按最坏分支，未用额度不转移。[机器计划](../../configs/proposals/kiki_energy_cap_transfer_v1.json)冻结身份、人口、逐 scope 上限、全局预算及准确实现绑定；最终实施收据须逐项核对这些字节与以下调用树。

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
| [机器计划](../../configs/proposals/kiki_energy_cap_transfer_v1.json) | 新唯一 scope/authority、268/258/774 人口与原权重、N_i/时钟/offset、逐 scope 与全局算子上限、条件完成公式及实现绑定 |
| [来源清单](../../configs/proposals/kiki_energy_cap_saved_sources_v1.json) | 逐文件 SHA、逐成员 shape/dtype/字节、阶段、来源 descriptor 键及不可变引用；不得在运行中扩展 |
| [workflow](../../configs/workflows/kiki_energy_cap_transfer_v1.json)与[experiment](../../configs/experiments/kiki_energy_cap_transfer_v1.toml) | 独立入口及一次性共同账本；旧 consumed 授权保持关闭 |
| `implementation_bindings/runtime/tests` | 由机器计划绑定新 kernel/reader/workflow/评价 helper、冻结 solver/模型和依赖；Python 3.12、实际模块路径及纯人工工程测试最终核验后才能签收据 |
| `review/activation` | 尚须 Astra 准确实施收据、独立授权登记、clean pushed commit、该提交 CI 及运行前路径/哈希守卫；设计冻结不替代这些条件 |

## 9. 失败、保存与独立审核

正式入口在任何许可来源 NPZ 校验/解码或科学调用前原子消费新共同一次性预算。范围预留 durable，逐调用 attempted/returned 留存，完成后核身份与范围摘要；未决/失败/日志或预算错误停止全轮，保留其余 not_attempted 槽。中断范围为 unknown/spent，不恢复、不重读、不重试。不因指标未达阈提前跳过后续正常独立槽；科学分数在固定路径全部冻结后统一判定。

新路径返回后先保存，再执行物理检查或评分；失败中间态、最后一次 RHS、solver diagnostics、已经读到的必要输入和实际预算不丢失。run 保存本次执行计划、完整来源清单、三个固定模型的六参数及尺度。保留完整 observed history、L/b/SVD/zh4、半径来源、C 历史和未来、各 reset、能量及全部评分/规则中间量。旧 E/I 历史、E/I/truth 未来及对应时钟允许采用新来源清单内的不可变引用，逐项保留 path、SHA、member、seed、event、步长和角色；不能仅凭旧病例名称定位。

输出至少包含 `process_status,V,mechanism_exercised,G_E,G_I,D,scientific_verdict,promotion_decision`，逐槽 solver 分支和每 seed/步长门槛明细。固定 `promotion_decision=KEEP_BASELINE, selected_model=null, certified=false, new_training_authorized=false, val_evaluation_authorized=false`。process PASS 与科学 REJECT 可以同时成立。

新 scope 明文允许两次独立的保存证据审核，用于核对来源、预算、KKT、reset、能量和指标。每次审核只可读取新 run 的 manifest 及其列出的全部产物一次，并读取同一来源清单内原文件/成员各最多一次；原来源每次上限为 8,528 文件、21,699 成员、418,279,836 原始 bytes、239,827,112 数组逻辑 bytes，NPY 头部限制同上。新 run 产物读取按其 manifest 独立记账，不能转移为额外来源额度。两次审核各自记录读取账本，与正式运行分账；旧 E/I/truth 引用的解码仅在这一新许可内进行。

审核禁止新 solver、模型构造、RHS、SVD、expm、拟合和路径传播，只做已保存数组的指标与代数复核。准确实施审核和最终科学结果审核分别留收据；本文件冻结设计，只有绑定收据、独立授权及运行前门禁共同成立才可运行。

## 依据

- [100 ms 准备稿 §5](../research/constant_c_100ms_preparation_20261109_zh.md)：经验能量先验、训练同构、端口/势能及时间目标的边界。
- [A14 设计](../research/kiki_a14_integral_history/design_zh.md)与[结果审核](../research/kiki_a14_integral_history/astra_result_review_zh.md)：原人口、积分方程、能量和历史收益。
- A15 归档提交 `5ac8b571751467f677fb569df3c0a1f4c323d065` 中 `docs/research/kiki_a15_future_transfer/design_zh.md`、`astra_result_review_zh.md`：原 future 口径、规则和有效负结果；只读文本参考，不导入其代码。
- [冻结球求解准备协议](constant_c_energy_initializer_v1_zh.md)及[真实机制前的合成确认审核](../research/constant_c_energy_confirmation_result_review_20261109_zh.md)：数值求解与有限合成证据。
