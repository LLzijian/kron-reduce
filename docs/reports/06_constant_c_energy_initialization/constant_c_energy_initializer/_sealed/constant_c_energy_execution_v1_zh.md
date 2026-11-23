# 能量约束积分初始化器：一次性合成对照执行协议

2026-11-09。状态：`EXECUTION_SCOPE_FROZEN_APPROVAL_BY_BOUND_RECEIPT`。是否获准执行由绑定本协议和准确实现的审查收据及研究政策共同决定。

用户明确继续按已准备计划推进本阶段，覆盖固定 9 病例×3 半径的一次合成对照及保存证据复核。新范围仍须独立注册、绑定准确实现并通过科学审核后执行，不需要重复取得已授权步骤的宽泛确认。本协议定义执行边界，不代替准确实现批准收据。

## 1. 冻结依据和范围

继承[准备协议](constant_c_energy_initializer_v1_zh.md) SHA `c93c6e492d6153a165297c1ae8da227993ad22ce556ae0cb65c47e3da900cda6`、冻结求解器 SHA `71e6be51569ce43eb67bce466ddfa6e222a0cf32338c2a33a10ae5ed76f9ef74`、[保存输入清单](../../configs/proposals/constant_c_energy_saved_sources_v1.json) SHA `441ac8924c1e027fe79587d5866076834a8060debca06844845d3928751bd4be`。三者不修改。当前文件补齐准备协议原先留待完整实现确定的调用树、保存与新授权规则。

新授权 ID 为 `constant_c_energy_initializer_v1_20261109`，experiment/workflow 为 `constant_c_energy_initializer_v1`，stage 为 `synthetic_comparison`，data_policy 为 `saved_synthetic_only`，max_executions=1、retry_allowed=false。阶段一授权保持 consumed；基线及部署初始化器不变。

病例顺序固定为 `L100_clean,L100_alt,L100_weakdir,S100_clean,S100_alt,W100_clean,W100_weakdir,K100_mismatch,U_hidden_null`。每例依次计算 `zero,oracle_radius,fixed_radius`。半径、球心、秩规则、求根及 KKT 容差与准备协议一致，不按结果更改。预计 27 条方案记录，其中 null 的两个正半径记录为政策未决；未决不传播替代状态。数值过程通过不要求科学指标全部达阈。

## 2. 最小执行器与功能复用

科学核心是 `kron_reduce.evaluation.energy_initializer_audit.execute(*, plan, root, result_dir)`；其导入不读取数组或计算候选。专用工作流为 `kron_reduce.workflows.energy_initializer_synthetic`，独立实现新范围的 authorize、共享预算消费、run 和 CLI。

为落实本轮执行，新范围准确允许以下只读功能复用，旧文件原样并纳入实现绑定：

- `constant_c_observability` 的 `Operations,save_record,write_json,errors,rk_path,exact_path,check_path,model_parameters`；内部被这些函数直接调用的纯序列化/评分功能随同绑定。
- `constant_c_synthetic` 的 `_read_json,_write_new_json,_verify_loaded_runtime` 三个纯工程函数。
- 公共 `NonlinearPassiveClosure`、`ScopedAuditJournal` 及其物理/数据契约依赖。

这项明确范围只允许所列辅助功能，不调用阶段一的 execute、authorize、run、预算消费或 CLI，也不调用旧参数生成、矩阵指数、积分重建、SVD 或 LS 过程。不使用 `checked_exact`，因为本轮白名单不包含直接端点矩阵；线性路径使用已核验来源的同一准确转移。准备阶段没有启用上述执行功能；当前新增范围通过准确实现审核后方能启用。

## 3. 保存输入、因果链与引用

`SavedEnergyInputs(root, manifest, ops).verify()/load(stem)/report()` 是唯一来源读取接口。source-inputs 范围逐个读取白名单 25 文件原始字节并核 SHA，在同一缓存字节上解析 14 个 JSON；11 个 NPZ 记录共解码 189 个允许成员。NPZ 容器可枚举成员名，只打开白名单成员；名称、shape、dtype、载荷大小与有限性一致，数组只读。不得通过 JSON 中未解码的描述符触发额外数组读取。

每个文件只读取一次，每个允许数组只解码一次；load 的缓存复用不增加读数。执行时先 verify，再按 `linear-fixtures,sine-truth` 和病例顺序加载 11 组记录。文件读取/校验各计 1 次 probe，数组校验/解码各计 1 次 probe，总计 214；JSON 解析及 report 为缓存元数据操作，不新增 probe。

setup 按 main、weak、mismatch、sine、null 构造五个公共 closure 对象，参数、常输入和线性转移全部来自已保存来源；平衡参考为零。K100 使用主真值与 mismatch 候选参数。null 使用自身参数、功率、转移和 `first[10:21]` 历史、`first[20:]` 真值未来。

初始化只向冻结求解器传入 L、b、保存 SVD/秩阈值、原无约束解和半径。oracle 的调用方从真实历史初态获得范数，不传入真值方向。候选状态与未来不进入半径选择。

inactive 分支逐元素复用无约束解并引用同一旧完整路径；不重新传播或调用物理 RHS。zero/active 分支从获准观测首点 q/v 与该候选 z_h 出发，历史自由闭环 100 ms，在预测起点只以获准观测末点替换 q/v，保留该路径自己的 z_T^cl，再预测 8 s。正弦粗细两档分别保留自己的历史末态，禁止交叉借用。

每条新路径全部输出检查 finite 和 COI，首/中/末三点检查能量恒等式。每组历史/未来保存 reset 前后总能量，离散重置跳变不计为连续被动性失败。正弦仍使用名义步长 `.00125/.000625`，dispatch=`nominal*(1+1e-10)`，预检每 10 ms 为 8/16 个 RK4 更新，保存实际时钟、步数及 min/max 步长。

## 4. 准确静态调用树与条件预算

共 11 个范围：`source-inputs`、`setup` 及 9 个病例。每个范围先持久保留预算再调用，逐调用记录 attempted/returned/failed。成功范围标记 `completed_within_budget`，完整性另外通过下面的精确条件式验收，不将未用额度转给其他范围。

| 范围 | probe 上限 | rollout 上限 | physical_rhs 上限 | integration_rhs 上限 |
| --- | ---: | ---: | ---: | ---: |
| source-inputs | 214 | 0 | 0 | 0 |
| setup | 5 | 0 | 0 | 0 |
| 每个普通线性病例（共 6 个） | 17 | 6 | 18 | 0 |
| 每个正弦病例（共 2 个） | 23 | 12 | 36 | 233,280 |
| U_hidden_null | 8 | 2 | 6 | 0 |
| 全局 | **375** | **62** | **186** | **466,560** |

probe 是登记的功能块调用，不是互斥底层数值原语：

- 固定部分为来源 214、模型构造 5、病例输入摘要 9、约束初始化器 27、普通病例无约束未来基准评分 8、25 个可传播/引用候选的两种未来参考评分 50、状态/因果/能量代数指标 25；合计 338。
- 每个新线性候选再计 1 次 reset 能量对 probe；每个新正弦候选计两次 reset 能量对及一次粗细误差 probe。
- 每例 solver 计三次 probe，包括 zero、inactive、active 或未决。新通用 LS 为零不意味着没有调用能量约束求解器。求根内部至多 16×64 次更新、16×80 次标量评估，冻结求解器内部上限与返回 trace 双重核对。

令 k_l、k_s 为实际新增的线性、正弦候选数。成功结束必须精确满足：

\[
7\le k_l\le19,\quad2\le k_s\le6,\qquad
N_{probe}=338+k_l+3k_s,
\]
\[
N_{rollout}=2k_l+4k_s,\quad
N_{physical}=6k_l+12k_s,\quad
N_{integration}=77760k_s.
\]

每普通线性病例 probe=`14+该例新候选数`；每正弦病例 probe=`14+3×该例新候选数`；null 固定 8。source-inputs/setup 固定 214/5。全部 attempted 与 returned 相等、failed=0，且 11 个范围完整关闭。正弦时钟记录恰为 4k_s 段，四倍 RK4 更新总数等于实际积分 RHS。

内部已知调用另列：线性保存转移作用 `810k_l≤15,390` 次；纯 reset 总能量求值 `2k_l+4k_s≤62` 次；公共参数验证的 `eigvalsh` 为 `3(5+4k_s)≤87` 次，分别嵌于五次模型构造及正弦公共 rollout 的输入检查，不是额外 SVD。独立力求值、新 SVD、新矩阵指数、新通用 LS、新真值生成、参数拟合和优化更新均为零。RHS 内的力及能量恒等式内部计算属于对应调用，不另称独立 force 操作。

## 5. 指标、保存与失败

未来评分复用原 errors 公式，分别与干净真值及同病例 true-z 对照比较。q_rms 的原单位为 rad，v_rms 为 rad/s；同时声明固定参考尺度 q_ref=1 rad、v_ref=1 rad/s，visible_rms 和 visible_max_abs 在这些单位参考下合并 20 通道。它们不使用数据方差或路径幅度归一化，不称为真实 kiki 的归一化 RMSE。3 s 和 8 s 分别使用 300/800 个非起点输出。

新增历史指标明确命名为 `history_observed_visible_mse_11points`：全部 11 个历史点、20 通道相对获准观测的平方误差均值。它不同于阶段一其他历史误差口径，不直接报告两者增减。

隐藏恢复及未来绝对阈值沿用准备协议；非劣仍以同一时域的 visible_max_abs 差≤`1e-8` 判定。三种半径全部报告，分别保留状态恢复、能量、3/8 s 达阈及非劣结果。零分母能量比为 null。科学指标未达阈不会停止其他登记病例，也不改变阈值或择优汇报。

每病例保存获准观测、真实历史/未来、L/b/保存 SVD、无约束解与原路径、true-z 路径、功率/转移、三个求解器结果/trace/残差及新路径；五组模型参数独立保存。inactive 的完整原路径及复用标记保留，不冒称新传播。来源 SHA、读取回执、实际环境、逐调用日志、条件计数、时钟、结果和工件清单组成同一不可变运行。

任一来源、数值、KKT、时钟、物理或预算失败立即停止全轮。求解器异常须在 journal 的失败记录前将 diagnostics 放入 partial，保留已返回算子结果和 last_RHS；写失败不得触发重试。无解码、未计算、未传播或未完成的项目明确保留，不用伪造状态补齐。日志关闭后才生成最终工件哈希清单。

## 6. 注册、一次性门禁与事后审查

新计划 `configs/proposals/constant_c_energy_execution_v1.json` 绑定本协议、准备协议、冻结求解器、输入清单、核心/reader/workflow、公共辅助依赖、测试、manifest/workflow/config、运行环境约束和准确科学批准收据。批准收据绑定同一 scope、plan SHA 与全部实现 SHA；旧授权记录不变。

实际执行提交必须干净、已推送且适用 CI 通过，加载模块的实际路径/SHA 和依赖版本通过守卫。专用入口在 Git common directory 下原子消费新授权，跨 worktree 共享一次性预算；消费先于源数组解码、模型构造或求解。通用 runner 拒绝该模式。执行后重新核对源 HEAD/绑定及结果清单，异常、中断和写失败均保持预算已消费。

独立事后审计从本轮保存数组与原 25 文件/189 成员白名单核对分支、KKT、状态/未来指标及实际计数，不调用求解器、模型、RHS、SVD、矩阵指数或新路径。审计读取是另列的保存证据复核，不能混入正式运行的 214 个来源 probe。科学结论在保存证据审计后逐轴裁定；本轮不产生真实 train/val/test 结论、失稳漏判率、稳定域证书或模型晋升。
