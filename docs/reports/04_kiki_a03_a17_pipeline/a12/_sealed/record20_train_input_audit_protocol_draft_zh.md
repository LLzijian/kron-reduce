# 记录 20 ms 三点历史：单次 train 输入与时钟审计协议草案

2026-10-24，Astra Ultra。状态：可供独立登记的方案草案，**不是执行授权**。本稿不依据 A11 中间结果，未读取新增真实 payload，未修改任何 tracked 文件。新的登记、实现审核与冻结完成前，不执行下述真实读取。

## 1. 唯一问题和结论范围

本次只回答：在原冻结的全部 268 条内部 train 中，记录 20 ms 的三个真实 q/v 观测及 Pm[2] 是否可用、索引是否全部不晚于登记的起报时刻，并至少留下两个真实未来点；实际记录时钟、已有违规信息和源标签规则是否一致。这是索引层面的因果性检查，不证明这些量在真实在线装置中可获得。

不创建或加载模型，不拟合、不优化、不预测、不算模型误差或分类分数，不选模型，不修改 LOS 阈值，不审计初始化器的准确率。积分隐藏初态方案不并入本次审计。通过只支持本次已审计 train 的输入可用性与所列规则一致性，不证明 val 覆盖、实际清除状态可用、真实隐藏状态可恢复、全生成器质量复核或最终零风险。

固定合同描述：记录网格 0、0.01、0.02 s 的 q/v，起报时刻为记录 0.02 s；外部功率读取第 2 行，采用原单位与 COI 变换，未来若另有模型比较才按单独注册的 ZOH 策略使用。这里不验证真实 Pm 在历史或未来恒定，因此不能由本次通过声称已验证现有 `CausalStateHistory` 的恒定历史功率条件；后续若用 Pm[2] 统一驱动历史回放，必须另外明确这项输入策略及其误差边界。

固定计算预算：0 模型构造/加载，0 未来预测路径，0 优化器更新，0 拟合，0 积分 RHS，0 物理检查 RHS，0 模型力计算。允许的科学计算只有所列源数组检查、坐标转换与真值规则统计。

## 2. 固定来源和人口

沿用同一个 source train 内部 allocation，不重新划分，不按标签、误差或历史资格抽样：

- 原 allocation：`E:/03-nn_Transfer_Func/kron-reduce/results/paper_kiki_p2_a03_seed324-kiki_rerun-20261022T095918Z-21a5a23b/role_allocation.json`。
- 固定 SHA256：`b103651cd33cb60dbdbad71c9d258545602cd6070a8860becc4fe9b2a6ac2b6d`。
- 从该文件仅按既有 `source_role="train"` 且 `role="train"` 形成 268 条白名单，包含原 100 ms 不合格的 10 条。原 258 条 100 ms 合格身份由同一文件的既有长度条件重建，不能重新挑选。
- 来源仅 `ieee39_postfault`、`ieee39_expansion`；数据根由原登记 `source_root`/环境解析方式确定，真实执行前记录解析后的绝对路径。每个 HDF 必须与 allocation 中 `source_bindings` 及静态 registry 的原 SHA 一致。
- 静态 registry 为 `E:/03-nn_Transfer_Func/kron-reduce/data/kiki_ieee39_v1/manifest.json`，SHA256 `688d564f381f5ba472633cdcc3258f84baf7a3861616c5230432dba9c5e9ecb3`。

原 allocation 的角色/身份元数据可用于确认白名单与不重分组；内部 val 的 HDF dataset、事件属性和标签均不读、不检查、不评价。source val/test 同样关闭。本次不打开任何 `dataset_index.csv` 或 `splits.json`，不重新运行 `allocate_roles`，不复用会自行扫描这些文件的现有 provider。source train 成员资格来自上述已经登记并绑定的 allocation，不重新读取其他角色。

执行顺序固定为 `(source_dataset, raw_id)` 字典序。白名单每行核验完整 event_id、raw_id、source、source_role、role、fault_line、declared_length；禁止重复身份、未知 source、路径分隔符/越界 raw_id。只按白名单直接定位 `events/<raw_id>`，不遍历 `events` 的其他键。每条分配事件从运行开始即在槽位账本中存在，失败也不移除。

## 3. 精确字段与读取边界

### 3.1 每个来源的静态读取

每个 HDF 只读取以下九个 root 属性各一次：`andes_version`、`case_sha256`、`config`、`git_commit`、`mva_base`、`omega_s`、`postfault_signature`、`raw_units`、`state_mapping`。不得一次转换整个 `attrs` 字典。

每个来源只读取以下三个静态数组各一次：`inertia[:]`、`equilibrium/delta[:]`、`equilibrium/Pm[:]`。解码前先核验各自 shape 恰为 `(10,)`、dtype 为可安全转为 float64 的数值浮点类型；不合格则不读取数组。返回后要求每个均为 10 个有限值；惯性严格为正。保持原 60 Hz、100 MVA、单位字符串、机组顺序及源码/配置身份检查。质量为 inertia/omega_s，两来源质量按照原合同的绝对 1e-12 容差核验；各来源平衡量分别保存并按同一公式使用，不临时更改或重新拟合先验。

历史源码身份、完整配置、case 与恢复系统签名由同目录 `kiki_source_only_evidence.json` 绑定（SHA256 `dbc0dc0a1215a8ece8086e0765f08dcb0bc3742ca0c680f1b91bc794ccfb59cf`）；根属性应与该静态证据对应来源一致。`state_mapping` 只作单位/顺序核对，不读取任何隐藏状态值，不把 220 个状态位置当成主模型特征。

### 3.2 第一阶段：全部 train 的数组

对每个白名单事件，先读取四个 dataset 的 shape/dtype 元数据。设 allocation 给出的正整数长度为 \(N_i\)。要求 t 为 \((N_i,)\)，delta、omega、Pm 均为 \((N_i,10)\)，数值浮点类型可安全转换到 float64，且 \(1\le N_i\le1501\)。不匹配则记录失败，不改用实际长度、不截断到 1501、不补点、不从其他文件补读。

通过形状与上界检查后，唯一允许的数组 selections 是：

| dataset | 唯一 selection | 用途 |
|---|---|---|
| `events/<raw_id>/t` | `[0:N_i]` | 全记录网格、真实时长 |
| `events/<raw_id>/delta` | `[0:N_i, 0:10]` | 三点输入及独立真值规则 |
| `events/<raw_id>/omega` | `[0:N_i, 0:10]` | 三点输入及独立真值规则 |
| `events/<raw_id>/Pm` | `[2, 0:10]` | 起报时因果功率 |

每个 selection 最多尝试一次；Pm[2] 只有索引存在时才尝试。若 \(N_i<3\)，其余真实数组仍可按固定 selection 保存，但 Pm[2] 标为不可用，不能改读 Pm[0]。本阶段不读取任何事件属性，包括 label、event time、valid length、故障元数据或求解器终止信息。

不读取 Pe、电压、完整未来 Pm、full_state、algebraic_state 或任何其他数组。完整 t/q/v 是本次预先声明的真值诊断范围，不是模型输入。用于输入审计的独立数组对象严格只有原 `t[:3]-t[2]`（名义值 `[-.02,-.01,0]`）、三点 q/v、Pm[2] 派生功率及已声明静态先验。必须保留真实时钟的浮点差，不能替换成新生成的名义常数。event ID、line、标签、真实长度、未来点数、mask、资格与违规统计都留在审计元数据中，不进入该对象；该对象只存数组，不构造任何模型类。

### 3.3 第二阶段：同一全部 train 的事件属性

第一阶段全体 268 个槽位、原始允许数组、输入对象、全部无标签统计、访问日志、预算账本和 SHA 清单封存后，才对**原全部 268 条**读取下列 11 个属性，各最多一次：

`label`、`event_time_s`、`termination_reason`、`trajectory_end_s`、`valid_length`、`clearing_time_s`、`record_start_delay_s`、`solver_success`、`failed_steps`、`postfault_signature`、`fault_line`。

不只读第一阶段通过者，不因标签补充或重新读取数组，不扫描 CSV 作回退。缺失/非法属性保留原值类型与错误原因；不得把缺失、NaN、未知字符串或失败转换为稳定、零时刻或“未触发”。没有成功封存第一阶段，就不进入本阶段。

## 4. 读取预算和计数语义

精确每事件 \(N_i\) 已由固定 allocation 字节决定，不在运行中按观测结果选择。登记时应从该文件的静态整数导出 `planned_reads.json`，列明白名单、每个 selector、\(S=\sum_iN_i\)、两来源文件大小和逐槽位预算，绑定其 SHA；这个纯元数据步骤不打开真实数组。本稿不为填数字额外读取 allocation 或数据。授权正文须包含导出的 S，未完成该绑定不许启动。

| 预算项 | 全部成功的计划值/硬上限 |
|---|---:|
| 原分配事件槽位 | 268 |
| 每事件数组 selection | 4，合计 1072 |
| 静态数组 selection | 6 |
| 数组 selection 总数 | **1078** |
| 解码后 float64 逻辑标量 | \(21S+2740\)，最多 **8,450,368** |
| 解码后逻辑数组字节 | \(8(21S+2740)\)，最多 **67,602,944 bytes** |
| 四 dataset 的 shape/dtype 元数据检查 | 1072 个固定 dataset 描述符 |
| 三静态 dataset 的 shape/dtype 元数据检查 | 6 个固定 dataset 描述符；描述符总计 1078 |
| root 属性读取 | 18 |
| 事件属性读取 | 2948 |
| 属性读取总数 | **2966** |
| 原始标签解码 | 268 个固定事件的 label；只在第二阶段 |
| 真值规则描述 | 最多 268 个完整真实记录，不调用模型 |
| 模型/拟合/更新/预测/RHS | **全部为 0** |

逻辑数组预算不等于压缩 HDF 的磁盘读取字节。两个来源 HDF 各在开始前和结束后校验一次原始文件 SHA，校验顺序读完整文件但不解码任何 dataset；单独登记校验读取量 \(2(B_1+B_2)\)，不将其混作角色 payload 访问。CSV/splits 连校验读取也不在本次范围内，因为本次不使用它们。

HDF 数据访问最多两阶段各打开每个来源一次，共四次只读 HDF 打开。第一次负责静态与白名单数组，第二次只读白名单事件属性；不在运行中因失败反复打开或补读。文件校验的底层只读打开另计。内存中已有数组的重复统计和保存不增加源 selection；应计入统计日志而不伪造新读取。

所有计数均在实际调用前记 `attempted`，成功返回后另记 `returned` 与实际逻辑元素数；预计划、尝试、返回、失败、未尝试分开。失败时不能声称已读满 1078 次，也不能把未用预算转给其他字段/事件。授权只供这一轮，无自动重试或续跑到通过。

## 5. 第一阶段固定检查与统计

### 5.1 无标签输入和时间检查

原始时钟要求从零开始、有限、严格递增，且逐点与 `arange(N_i)*.01` 的绝对差不超过 1e-10 s。不得插值、重新采样或修正原时间。q/v 与 Pm[2] 都按原单位/COI 公式转换并保留前后数组 SHA。COI 只核验数值转换与质量加权约束，不调用能量或模型 RHS。

具体使用现有坐标变换的运算顺序：令 \(m=\mathrm{inertia}/\omega_s\)，\(\mathcal Q_m(x)=x-\mathbf1(m^Tx)/(\mathbf1^Tm)\)，\(\mathcal P_m(p)=p-m(\mathbf1^Tp)/(\mathbf1^Tm)\)。则 \(q_k=\mathcal Q_m(\delta_k)-\mathcal Q_m(\delta_{eq})\)、\(v_k=\mathcal Q_m((\omega^{pu}_k-1)\omega_s)\)、\(p_{20}=\mathcal P_m(Pm_2-Pm_{eq})\)。q 的单位为 rad，v 为 rad/s；LOS 描述仍按下一节原始 pu 速度计算。必须核验变换后值有限，质量加权 q/v 及 p 的总和绝对残差不超过原坐标检查的 1e-10；计算与原值一并保存。该检查不能证明真实隐藏状态已被这三个点确定。

每条分开给出：

- `history_available`：前三个真实 t/q/v 点及 Pm[2]、静态先验均合法。
- `at_least_two_real_future_points`：第 3、4 个索引点的真实 t/q/v 存在、有限且网格合法。
- `record20_input_supported`：上述条件和所有输入/来源检查均通过。
- `full_visible_record_valid`：整个已声明 t/q/v 记录均满足形状、时钟、有限性；它服务于后续真值描述，不伪装成模型特征。

真实长度、未来点数与这些布尔值只作支持检查与记账。不能将它们输入任何 encoder、模型或以后分类器。本次本来也没有模型调用。

### 5.2 固定真值规则

在标签尚未读取时，对完整合法记录按原始源码方式计算

\[
 A_k=\max_i\delta_{ki}-\min_i\delta_{ki},\quad
 S_k=\max_i|\omega^{pu}_{ki}-\overline\omega^{pu}_{I,k}|,
 \quad b_k=\mathbf1\{A_k>\pi\land S_k>.02\}.
\]

其中 \(I\) 为原 `inertia` 数组，\(\overline\omega^{pu}_{I,k}=\sum_i I_i\omega^{pu}_{ki}/\sum_i I_i\)。实现沿用源 `coi_project(raw_omega, inertia)` 的运算顺序，不能改用已投影 v 再除 omega_s，也不换成质量数组后声称逐位相同。保持源惯性权重、原始绝对 pu 速度、严格不等号及连续五点；不添加阈值容差，不改变 hold，不寻找使标签更一致的新规则。保存逐点 A、S、两个分量布尔值和 b。

固定输出：首次角度越界点、首次速度越界点、首次联合违规点 \(k_{first}\)、首次五点窗口起点 \(k_{run}\)、其末点 \(k_*\)。没有窗口/触发用 `None` 并给出原因，绝不改成 0。任何全记录非法导致的规则未完成标为 `invalid`，不当作未触发。

对记录 20 ms 起点 o=2，另保存前三点已有联合违规情况、在 o 处连续违规点数 \(c_o\in[0,3]\)、确认是否已发生，以及
\(L_{first}=t_{k_{first}}-t_2\)、\(L_{run}=t_{k_{run}}-t_2\)、\(L_{confirm}=t_{k_*}-t_2\)，并另存名义网格差 \((k-2).01\)。实际时间差保留原浮点值；缺少对应索引时提前量为 None。首次联合违规在记录第零点时，标记其记录前真实起始时间未知。

对旧共同人口也从同一已读取数组描述记录 o=10 的相同统计；不读取 Pm[10]，不重新构造旧模型输入，不运行旧预测或重算旧评分。两种 o 只是已观察信息与真值时钟的对照，不是模型预警质量比较。

### 5.3 稳定记录条件也在标签之前计算

保存完整记录是否无五点触发、是否达到 15 s、最后 100 点是否同时满足角差严格小于 π 和 COI 速度严格小于 0.005 pu。1501 点时最后 100 点为索引 **1401…1500，即 14.01…15.00 s**，不改成含 14.00 s 的 101 点。N<100 时末段检查为不可完成，不能把少于 100 点的全记录当成完整末段。短记录不够原稳定条件，标记“不满足源稳定接受条件”，不能由此自动称为失稳。

## 6. 第一阶段封存与后置标签核对

`prelabel_freeze.json` 应绑定：完整 268 槽位、静态数组、允许原始数组、三个点输入对象、逐点真值统计、全部错误、计划/实际访问日志和预算、源码/配置/来源 SHA；明确 `labels_decoded=0`、internal val/source val/test 数据与标签均未访问。冻结失败则整个审计停止，后置标签不读。

第二阶段只做原定义一致性核对：

- `unstable` 必须有五点触发，且 \(k_*=N_i-1\)；reason 为 `physical_loss_of_synchronism`；event_time_s 与该末点、trajectory_end_s 与记录最后点相差不超过 1e-10 s。
- `stable` 必须 N=1501、完整无五点触发、达到 15 s、最后 100 点两个稳定条件通过；reason 为 `stable_horizon_reached`；event_time_s 为源约定的空值，不能把未知/非法数值当空值。
- 两类共同要求 trajectory_end_s 为有限数，且与原 `t[-1]` 的绝对差不超过 1e-10 s；不能只核对失稳样本的记录末点。
- valid_length 为同一个严格整数 N；solver_success 为真且 failed_steps 为整数 0；事件 postfault_signature 与固定来源签名一致；fault_line 与 allocation 一致。
- clearing_time_s 和 record_start_delay_s 为有限值，并符合正清除时间/正 delay 的源约定。记录绝对起点可由二者相加表示，但本次没有读取原始 ANDES 绝对时间或开关执行日志，因此只核对元数据自洽，不独立证明开关在该时刻实际执行。不得把 delay 固定为 0.005 s。

数值/布尔属性按 HDF 原生标量类型校验：整数不接受布尔值或字符串截断，solver_success 不用 Python 对任意非空字符串的真值转换，stable 的空 event_time_s 只接受源定义的空字符串（字节串仅可严格 UTF-8 解码）。配置与字符串属性按源格式解析，非法类型不能通过强制转换变成合格值。

若源标签、时间或规则不一致，原值和独立计算值同时保存；不改标签、不补数据、不用该事件重设阈值。第二阶段不得修改第一阶段的输入、统计、cohort 或失败状态；收尾逐文件重新检查第一阶段 SHA。

## 7. 固定分组与报告

报告始终以原 268 条为分母，给出全部槽位状态。无模型预测，**不产生 FN、FP、FSR、NRMSE、模型准确率或“零漏判”指标**。源标签一致性报告用“一致/不一致/无法核对的条数”，不包装为分类器准确率。

主要分组在标签之前由 allocation 长度固定：

1. 原 100 ms 共同人口：既有 \(N_i\ge12\) 的 258 条，逐条保留身份；分别报告其 20 ms 输入通过/失败和两种观察起点的已见违规描述。
2. 原 100 ms 不合格人口：其余 10 条。单列其中新合同通过者、仍失败者及原因；不得预填“10 条全部新增”。理论预期的 \(5\le N_i\le11\) 是待核验条件。
3. 全体 268 条汇总：输入通过数、完整可见记录检查通过数、规则/元数据一致数、失败/未决数，以及覆盖的物理 Line 数。失稳/稳定的描述分组只能在第二阶段以后添加，不能重算第一阶段选择。

已见违规描述至少区分“前三点完全未联合违规”“已有违规但最后点未处于连续段”“连续 1/2/3 点违规”。对有确认者分别列出首次逐点违规、首次成立窗口起点、确认末点的相对时间。符合源失稳规则的最短五点记录在起报时已见三点违规；仅余 20 ms 的确认提前量不得称为在首次违规前预见物理失稳。

所有计数与 line 分组仅用于审计展示，不产生预测路由或事件权重。内部 val 的 27 个失稳样本不在本次读取范围，报告必须明确“未实测其 20 ms 覆盖”，不能从 train 或源码证明填成 27/27。

## 8. 失败行为与停止条件

来源/授权/计划 SHA、白名单、角色隔离或硬预算检查失败时，在下一次真实解码前停止；已经发生的读取如实保存，全部剩余槽位标 `not_attempted`，不自动重跑。禁止原文件写入。

单事件形状、时钟、非有限值、输入不足、属性缺失或规则不一致，不影响其他已预留事件的唯一尝试；该事件记录全部已发现原因，保留未完成项，不从分母删除。不增加读取、不改变 selection、不补点。不得把 `attempted`、`returned` 或失败的零长度数组当成完成证据。

输出保存/封存失败、非法范围访问尝试或计数器失效属整体失败，立即停止后续读取和标签阶段，尽力写入独立失败记录及未尝试槽位。任何失败都保留，不覆盖旧 run，不以同一授权重试。每次访问日志须包含阶段、source、event、role、dataset/attr、selection、planned/attempted/returned、逻辑数量及错误；顺序号足以复核标签读取晚于第一阶段全部冻结。

允许记录多个原因，但至少有独立代码：`source_binding_mismatch`、`allocation_mismatch`、`forbidden_role_or_identity`、`source_static_prior_invalid`、`missing_event_or_dataset`、`declared_length_or_shape_mismatch`、`length_out_of_registered_bound`、`clock_invalid`、`nonfinite_visible_record`、`history_unavailable`、`two_future_points_unavailable`、`origin_power_unavailable_or_invalid`、`attribute_missing_or_invalid`、`source_rule_or_time_mismatch`、`source_solver_or_signature_mismatch`、`budget_or_access_order_violation`、`artifact_freeze_failure`。未计算的指标为 None，不填零。

## 9. 科学验收与下一步边界

运行是否完成、审计科学结论、未来训练授权分开：

- `execution_completed`：所有计划槽位均得到终态并完整封存；可以在存在科学不一致时完成，不能由此自动接受。
- `scientific_verdict=ACCEPT` 仅当来源/角色/单位与先验、268 条三点输入和两未来点、完整可见记录时钟、原规则与后置元数据、所有预算/顺序/文件不变检查全部通过。此时 `record20_train_contract_supported=true`。
- 任一上述必需检查失败或未完成即 `scientific_verdict=REJECT`、`record20_train_contract_supported=false`。通过子集的实际数目照报，不改成删例后的全覆盖。
- 无论结果如何，`new_training_authorized=false`、`model_selection_performed=false`、`model_promotion_allowed=false`、`internal_val_coverage_verified=false`、`zero_missed_instability_established=false`。

通过后才可据明确的输入与已见违规范围提出独立的初始化/动力学研究；仍需新的固定预算、数据边界和实现审核。若失败，先解决本次已指出的来源、时钟或输入问题，不能直接训练规避失败。该结果不替代最终独立新 kiki 确认。

## 10. 正式运行前的必要合成审核

实现至少证明：只从固定 268 白名单定位事件；shape/长度超界在 payload 读取前被挡住；Pm[2] 与未来 Pm/旧 Pm[10] 严格分离；五点全违规最短例、持续性中断、分量不同步、阈值相等及非有限值处理正确；100 点末段端点正确；旧共同人口固定 258 且不合格 10 条保留；任何第一阶段冻结失败均阻止标签读取；所有第二阶段面向原全人口；未决不转成稳定；selection/字节/属性预算与 attempted/returned 语义一致；源或冻结工件篡改导致拒绝；所有模型/预测/RHS入口在该流程均不可达。

这份草案只冻结上述一个审计问题与预算，没有候选网格、没有新模型对照，也没有任何由 A11 结果触发的自动执行分支。
