# A17 整个实现的最终精确审核

2026-10-25，Astra Ultra。审核提交 `6babf354fa3986fd3220ebd53ff3d86dc6ada64b`，分支 `codex/kiki-a17-input-consistency`。

**结论：APPROVE_FOR_INDEPENDENT_BOUNDED_REGISTRATION。** 当前实现可以进入唯一 A17 的独立有界登记；本报告不激活 policy、不消费预算、不授权本轮读取真实载荷或执行科学计算，也不是对尚未产生的 A17 科学结果给出 ACCEPT。没有发现阻止这一登记准备的实现问题。正式登记的配置、权限依据和执行提交仍须按下述条件复核。

本轮只读检查源码、冻结控制 JSON、合成验证日志和工程收据，并核对工作文件与 HEAD blob 的字节身份。没有打开或 hash 真实旧 NPZ/HDF，没有运行模型、LS、RHS、拟合或新的真实比较。独立辅助审核同样没有执行科学计算。

**证据与精确范围。** 统一验证收据 `engineering_full_validation_v4.json` 的 SHA256 为 `49fbb883a8ef9d30ef36fc7d2e4d8128e09acfd858c8a14758b5020f9f178733`，记录 509 个合成测试通过及 Ruff 通过；两条 warning 来自预先设置的加权溢出拒绝反例。该收据的 48 项绑定、日志 SHA 和验证前后无改动记录均已独立核对。另补 8 项实际依赖/导入链绑定，共 56 项工作文件与 HEAD blob 一致，完整清单见 `whole_implementation_review_bindings.json`。新增依赖没有冒称为 canonical 收据原列项目。原 v3 的 7 个失败及早期审核问题完整保留；最后的 OutputIO 修复恢复具体 Path 类原有的属性槽或继承关系，4 种定向情况和统一回归覆盖后续父类 hook，不再留下跨测试的全局状态污染。

协调任务报告该精确提交已经推送，两项 CI 成功：push run `40172621914`、PR run `40172625937`。此处依据协调任务提供的 GitHub API 核对结果；本审核自行核对的是本地 HEAD、文件/blob 和保存日志。后续登记提交仍须有自己的精确 CI 证据。

科学流程与冻结设计一致。唯一候选先用同槽保存分解求解 `btilde4+d_w`，严格继承原 rank/tol，依次检查分解、正规方程残差与条件敏感度代理。唯一 `PhiT z0+wT` 只作继承求积检查，不替代闭环隐藏末态。历史 RK4 四阶段使用原 Pm11 分段线性输入，端点原值、阶段顺序及 callback 实际返回/实际使用的功率均保存；未来保持原 Pm10 常值输入。两个步长分别从同一新 z0 回放历史，分别保留自己的 zT，并只在 T 重置观察到的 q/v。完整重置前后状态与能量差可核对；可见重置造成的能量变化不被误称为耗散失败。没有新分解求解器、额外初始化候选、encoder、fit 或训练更新；构造固定模型时的 9 次结构 eigvalsh 单独计数。

三阶段读取与保存严格对应固定人口。先冻结全部 1,548 个 origin 或失败记录，才读取未来 clock；全部 1,548 条候选未来及精化结果冻结后，才解码 I/E/truth。原 268 个 train 分配、258 个 eligible 事件、774 个 seed-event 槽及 source→line→event 权重不变。I 是已被拒绝的 A15 机制参照，E 是 A09/A10 保存的质量参照，六个 seed/步长组分别判断；只改善 I 不能支持替换 E。联合输入一致性流程的比较不能分离 LS 右端修正与历史 PL 回放各自的贡献，也不能把 A16 非零扰动当成未来因果证据。

冻结预算及其逐 scope 依赖已用元数据整数求和重新核对：

| 项目 | 固定上限/全成功精确数 |
| --- | ---: |
| 科学/身份 scopes，三阶段分别 | 10,356；3,100 / 2,580 / 4,676 |
| 固定 closure / 结构 eigvalsh / 唯一新 LS | 3 / 9 / 774 |
| 新历史 / 新未来 | 1,548 / 1,548 |
| 积分 RHS / 物理检查 RHS / 全部 RHS | 5,380,344 / 451,458 / 5,831,802 |
| 实际 RK4 更新 / 历史 input_eval | 1,345,086 / 185,760 |
| 总体指标组 / 配对验收组 | 18 / 12 |
| NPZ 文件 / context / 独立成员读取 | 3,358 / 3,616 / 20,406 |
| 解码 f8 元素 / 逻辑载荷字节 | 21,486,372 / 171,890,976 |
| 两轮 opaque hash 动作 / 字节 | 6,716 / 641,698,014 |
| 来源 JSON 元数据文件 / 字节 | 3,376 / 72,922,783 |
| 每成员 NPY header 上限 / 全部 header 上限 | 10,000 / 204,060,000 字节 |

6,049,612 是含嵌套真实过程调用的类别总和，不是互不重叠的算术原语数。诊断矩阵乘法 4,644、范数调用 15,480、矩阵范数 3,870、向量范数 1,325,736；LS 内两个矩阵向量乘法另有 1,548 次，观测映射已包含在求积诊断中。RK4 更新由实际成功更新的记录核对，没有空 callback 充数。输出文件压缩字节按真实返回量记录；上表 opaque hash 字节不冒充全部压缩读取字节，header 上限不冒充实际已读量。

预算目录位于 git-common-dir，按 AUTH 独占创建，预订先于任何真实来源 hash/解码；关联 worktree 共享同一预算。每 scope 先持久预订，实际 attempted/returned/failed 日志范围同步、读取重建并 hash，结果 JSON/NPZ 保存后才持久关闭。正常局部数值失败保留全部已返回前段并可继续独立槽；依赖失败槽记录为 prerequisite_failed、实际工作为零。失败 scope 再尝试 callback、来源/身份/权限/预算/日志/保存错误均停止整个对象；不重试、不转移额度、不从新目录恢复。缺失权重仍进入固定总体及配对报告，任何失败使 V=false。

输出与封存边界成立。OutputIO 保存实际文件生命周期、返回字节、短写与未知操作，并在关闭自身日志时原生 fsync；失败后仅独立失败保存器工作。SealLedger 对已有 N 个文件 hash，加一次 source_access 计行，保存 N+1 个真实动作的前段。最后四个 summary/manifest 写入与 hash 的实际记录位于共享预算的外部 completed 收据，避免自 hash 循环；异常携带实际前段和失败阶段，硬中断未保存部分保持 unknown/spent。新 run 内审核器明确不验证这个外部收据。completed 写入又早于 harness 的 strict manifest 检查及 record_stage，因此单有 completed 文件不能证明整个工程流程完成；若其后出现 RUN_FAILED，必须按失败保留，不补跑。

独立结果审核器只读取显式指定的新 run。它核对封存文件与逻辑数组 SHA、全部 scope/依赖/实际计数、冻结人口及科学前段链、来源摘要与本次保存元数据的连接、原参数前后指纹、PL 调度、重置与未来起点。保存数值门的非负性及标量公式、对应路径的精化诊断、质量 COI、LS 保存右端和既有未来指标均有独立证据重放；不调用模型或重求 LS。原权重贡献保持 NumPy float64 左到右累加，避免 Python 3.12 补偿 sum 改变冻结结果。230 个审核器合成用例及实际小型控制流程联调包含在本次 509 项中。生产公开入口保留 268/258/774/10,356 的严格约束，没有 toy 人口开关。小型集成不冒称已执行过全规模真实或合成科学过程。

本轮科学判定固定为 V∧G_I∧G_E：每个参照的六组都须满足主 MSE 严格降低及其余预定守门指标不升高，沿用 1e−12 比较容差。任何结果均 KEEP_BASELINE，不改变默认初始化，不给出新的 FSR/LOS、稳定域证书、部署或模型晋升结论，更不保证零漏判。

**登记后的必要复核条件：**

1. 独立登记 AUTH=`kiki_a17_development_20261025`、experiment=`paper_kiki_a17_input_consistency`、workflow=`kiki_a17_input_consistency_v1`、stage=`input_consistency`、module=`kron_reduce.workflows.kiki_input_consistency`、dataset=`kiki_ieee39_v1`、execution_mode=`bounded_history_cache_audit_cli_only`、contract=`measured_history_pm11_linear_100ms_zoh_v1`。manifest、TOML、workflow、input-contract、policy 和授权正文逐项一致。policy 的用户授权依据须来自真实会话授权；本报告不能替代或伪造 explicit_user_reply。已有授权的覆盖范围由协调任务核实，不以状态标签自证。
2. 原 design、read plan、compute plan 字节保持不变，分别为 `61d50be6…`、`ee8a7aa5…`、`efa025a2…`，完整 SHA 见机器收据。冻结控制中的 prepared/inactive 制备标记不需要为登记而改写；执行权限来自独立 policy。新 protocol、TOML、workflow、input_contracts 和 manifest 的最终字节，连同本审核正文、完整 56 项实现绑定，必须再次核对工作文件与 HEAD blob。当前 authorization 只强制绑定清单含 controller，并不会自动证明清单完整，因此不得删减其余必要绑定。policy 自身的快照/提交另外登记，避免要求一个文件包含自己的 SHA。
3. 登记必须原样给出完整 read/compute 限额、原人口和失败规则、一次执行、零 fits/updates、禁止 retry/resume、labels/internal val/source val/test、HDF、未来 Pm、旧 I/E 新 rollout 和模型晋升。新 protocol 对“初始化新分解为零”与“结构 eigvalsh=9”、actual/maximum 字节、内部诊断与逐标量审计的区别应明确，不重新简化预算。
4. workflow 使用 evidence_contract_version=2，唯一下一科学阶段是 input_consistency，该阶段终结本轮；不能添加自动训练、重跑或换步长转移。stage evidence 区分工程 process PASS 和科学 ACCEPT/REJECT，强制 KEEP_BASELINE 及 false 的 certification/deployment/promotion/新漏判声明。专用 CLI 仍先检查授权与绑定；通用 runner 必须在 generic data verification 前拒绝。登记校验仅用控制元数据和合成拒绝路径，不通过真实运行或 generic data-check 探测配置。
5. 启动前另行核验已选 Python 3.12.14 解释器及 import 来源，使用被冻结 constraints 的 numpy 2.5.2、scipy 1.18.0、h5py 3.16.0、torch 2.13.0、filelock 3.32.2；测试工具同受约束。现代码只强制 Python 3.12，environment.json 是事后记录，不能说成所有版本的自动前置门。直接调用已核解释器，不使用 uv run。
6. 登记及任何获准激活的最终提交须 clean、精确推送、该 HEAD 的 CI 成功，并完成配置/权限与 SHA 的独立复核。除已审核登记文件外，不得夹带核心实现改变；若核心、读取、计数、冻结、评分或依赖字节变化，先重新审核相应范围。运行期间保持代码、环境、policy 与控制不变；不得通过其他 clone/worktree、改名或删除预算取得第二次执行。
7. 将来唯一获准执行若完成，结果验收需同时核共享预算 completed 的完整 final_seal/末四动作/manifest SHA、harness strict 封存检查及新 run 独立审核结果。新结果审核可读本次保存的 JSON/NPZ，不重读旧来源、不新增 RHS/LS。之后才做正式结果科学审核与工程集成判断。

这些条件是登记与执行前需要闭合的具体事项，不是授权命令。本次对实现的肯定不改变 A04–A16 已消费状态和 A15 REJECT，也不启动 A17。
