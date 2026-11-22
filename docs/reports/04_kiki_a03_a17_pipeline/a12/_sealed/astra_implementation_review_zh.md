# A12 record20 输入审计：实现审核

2026-10-24，Astra Ultra。审核对象为尚未执行真实读取的 A12 实现候选 v7；最终针对性测试及其日志 SHA、两份测试文件的纯行尾规范化均已核验。

## 科学裁定

**ACCEPT：批准 v7 实现用于已冻结的一次性 A12 train 输入审计。** 完整可见记录与功率失败的两种耦合问题已修复，目前没有剩余科学阻断。该批准不是实际 train 输入支持已经成立。协调代理须把本审核、全部 26 个文件及原方案/协议绑定到活动授权，提交并验证工作区与 HEAD 原字节一致、干净状态及工程检查后，才可通过专用入口执行一次。

真实结果仍按原方案判断：全体 268 条所需输入、完整可见记录、来源/规则/后置属性及全部读取门槛同时通过，才得到该 train 合同的 `ACCEPT`。任一必要项失败即 `REJECT`，失败也消耗本次预算；不得补读、补跑或更换人口。

本审核没有读取真实 HDF、CSV、splits 或轨迹 payload，没有构造模型、拟合、预测、模型力或 RHS。A13 合成初始化研究独立于本次审计，没有进入 A12 的读取、计算或接受条件。

## 身份与批准字节

冻结设计 SHA256 为 `3ccfd537eaf3a8896ccf713d5757798ecf10807ce37770d13ae0b33447b2f6e4`；正式授权协议 SHA256 为 `c9ca9547d32c72534f9e84a3f5addd5ba791d48259aa337b506209954d11b661`；读取计划 SHA256 为 `4f0cb7552e31c310822e34e05d668ad6a1f07be5ab398f97bcd7db21de51a054`。

完整实施绑定来自 `scratch/one_off/kiki-a12-implementation-20261024/implementation_candidate_bindings_v7.json`，SHA256 **`4d2454cef234a3e85cb1299c6a329303db7b5458d7155d68f409a5e624a443c6`**。我已独立对照当前文件字节核验其中全部 26 项，数量 26、SHA 不匹配 0。

| 核心文件 | SHA256 |
|---|---|
| `src/kron_reduce/workflows/kiki_record20_input_audit.py` | `79f0cf267c623a39f4769490a5352e694cc5c8cea42ab3fdc979a977e420d3b3` |
| `src/kron_reduce/data/limited_hdf_audit.py` | `4d1125da99e47cb43653b8ca39689f696e24e13994ae3df8c78c896f96f035e0` |
| `src/kron_reduce/data/audit_access_ledger.py` | `6c0333cb9e1f5c56ba309334a52f84655c2b14cc606862ab97997cf70835d1bc` |
| `src/kron_reduce/data/durable_audit_ledger.py` | `c1aa5889395f7d8e280a6d787c82c8f8247a79f2ee506974c09a2554dda87943` |
| `src/kron_reduce/evaluation/record20_input.py` | `1f9002a532519756e2e891d63f478bc954d7a052f7c70f55c65b8c1a7a026c7e` |
| `src/kron_reduce/evaluation/record20_coordinates.py` | `395a231f5b6653d5f751d7bcd9eec0f8b30954caf6cffd4f8ecbe54155567332` |
| `src/kron_reduce/evaluation/record20_metadata.py` | `5d829a8806dd86fa5b12eb11bf76ae55a29ae5b0774747d5f33b8c94d1eda85d` |
| `src/kron_reduce/physics/coordinates.py` | `9a738d46647ab631407f8e4b5814fe0a7868c717e31fda13963861c10e373089` |
| `project_harness/runner.py` | `108cf22a3cb3d9ccf0f33088034cbc03b77529e68bff67e23defd142bd614e4b` |
| `tests/test_record20_workflow.py` | `fbc28c82b07b8c814e3566837c200793c60c2edb552a20cd522c0c2ca6554158` |

其余配置、合同、harness 依赖和合成测试的逐项 SHA 以同一 v7 清单为完整记录。`configs/research_policy.json` 因需绑定本审核 SHA，不纳入该自引用清单；最终必须保持已审配置字段逐项相等，填写本审核 SHA 与全部 v7 绑定，再将状态从 `implementation_review_pending` 改为有效的单次授权。不能以此处理为由改变算法、读取范围、预算或验收。

## 读取边界和计数

专用入口先做仅涉及登记元数据的 policy/文档/源码检查、干净 Git 检查和流程锁，再调用既有不可变运行内层 `_run_experiment`。通用 `run_experiment` 在 `verify_data` 之前拒绝 `bounded_input_audit_cli_only`，避免通用校验打开本轮禁止的 CSV/splits。专用调用链没有使用原 provider，也没有事件目录枚举。

固定白名单只从已绑定 allocation 的 `source_role=role=train` 重建，并逐行与读取计划比较，保持 `(source_dataset,raw_id)` 顺序、原 268 条、既有 N≥12 的 258 条共同人口和其余 10 条。内部 val 与 source val/test 不进入 reader 白名单，其他事件键、属性和数组没有访问路径。

在任何源 HDF 校验或解码前，先在 Git 共用目录原子占用 A12 唯一预算目录。目录创建不允许复用；预留成功后发生任何错误，预算仍已消费，不因空结果或未完成阶段重新开放。

| 项目 | 冻结上限/全部成功值 |
|---|---:|
| 事件槽位 | 268 |
| HDF 只读打开 | 4，两个来源各两阶段一次 |
| 数组 selections / shape-dtype 描述符 | 各 1078 |
| root 属性 / 后置事件属性 | 18 / 2948 |
| 属性总数 | 2966 |
| float64 逻辑标量 | 7,501,168 = 21×357,068+2740 |
| float64 逻辑字节 | 60,009,344 |
| 两个 HDF 前后各一次原文件校验字节 | 6,870,747,346 |
| 模型、拟合、优化更新、预测、模型力、全部 RHS | 全部 0 |

原文件 SHA 校验是两次限定长度的顺序字节读取，不解码其他角色；它与逻辑数组读取分开计量。固定大小检查、实际返回字节和文件状态检查保留。读取事件数组前，shape/dtype 必须与计划精确吻合；不允许按实际长度改 selection，不补点，不改读其他功率行。Pm 只读索引 2 的十个值。

每个数组、描述符、属性的访问都先持久记录 attempted，再执行源回调，返回后另记数量。重复访问、非法阶段、预算越界或持久日志失败会使账本失效并停止源访问；`AccessLedgerError` 及其持久日志子类不会被普通事件失败捕获逻辑吞掉。普通缺失/形状错误保留失败或跳过状态，可以继续其余原计划槽，但全体成功门槛不再成立。

## 输入、时钟和源规则

输入对象只含 `t[:3]-t[2]`、三点 q/v、Pm[2] 派生功率和声明的静态先验，保留真实时间差。身份、线路、标签、长度、资格、未来点数与规则统计留在审计记录中；未创建模型或 `CausalStateHistory`。本轮不证明历史或未来功率恒定。

坐标转换沿用非均匀质量下不同的 Q/P 运算：q/v 使用质量加权状态投影，功率使用其对偶投影。LOS 描述另从原始 pu 速度、原 inertia 和源运算顺序计算，没有从模型速度转换回 pu 后冒充源规则。

四类布尔值保留各自含义。前三点与 Pm[2] 合法决定历史输入；前五点可见量决定两个真实未来点；完整记录另作可见量检查。审核发现的必修问题有两个分支，均属同一语义：完整可见记录不能连带要求功率合法或成功读取。Pm[2] 为 NaN、完整 t/q/v 合法时，保留 `full_visible_record_valid=True`，但输入失败、整轮不能接受；Pm selection 缺失或读取失败时，driver 仍独立检查已返回的 t/q/v，以显式 None 表示未取得功率，不制造功率向量、不产生输入对象、不在 NPZ 冒充原始 Pm。相反，索引 5 以后可见量非法时，也不抹掉此前合法的三点输入和两个未来点；完整记录失败仍使整轮拒绝。

规则使用角差严格大于 π、惯性 COI 速度严格大于 0.02 pu，二者同点成立并连续五点。首次单分量违规、首次联合违规、首次完整五点窗口起点/末点分别保存；缺少窗口是 None，非法记录是 invalid，不互换为“稳定”。起报点 2 与 10 的已有违规信息、实际/名义时间差分别记录。首次联合违规在索引 0 时保留记录前起始时间未知的标记。

稳定条件保留 1501 点、末点达到 15 s、无五点触发及末 100 点严格角差/速度条件；末 100 点为索引 1401…1500。属性类型严格校验，stable 的 event_time_s 只接受源空字符串，unstable 的触发末点必须是记录末点且与 event_time_s 一致。两类均核验 trajectory_end_s。`fault_line` 按 `Line_6` 这类完整字符串比较；不转成数字或按来源路由。

## 封存、失败和结论

第一阶段读取后保存全部允许原始/变换数组、输入对象及逐数组 shape/dtype/C-order 字节 SHA、268 个槽位、无标签统计、静态原生元数据和账本；持久日志关闭后，把这些工件及 io 日志前段封存。只有成功写出并绑定 `prelabel_freeze.json`，才能新建第二阶段日志、转换阶段并读取全部 268 条的 11 个事件属性。

标签只用于源规则一致性核对。第二阶段不修改第一阶段选择、输入或统计；收尾逐一重验第一阶段 SHA，并再次校验两个源 HDF。任何源/角色/授权/预算/封存失败停止后续访问；部分读取与未尝试选择由全计划及账本保留，不把未完成记录计成通过。输出失败尽力保留错误；不改变原结果、重试或补读。

最终接受要求全部计划访问成功且零失败、精确逻辑元素和校验字节、四次 HDF 打开、两份静态先验、每条输入和完整记录/规则合法、每条后置元数据一致。完成流程与科学结论分开，`REJECT` 也可以完整封存并交给既有 stage evidence 校验。没有 FN、FP、FSR、NRMSE 或模型选择结果。

## 合成检查与最终执行前条件

已审阅的合成测试覆盖非均匀质量、真实浮点时钟、五点规则/阈值等号/中断/分量不同步、非法值和短记录、100 点端点、Pm 单行隔离、非法选择在解码前拒绝、持久日志失败中止、冻结失败不读标签、全人口保留与唯一预算不复用。完整 268 条内存合成流程使用实际 reader、账本、validators 和 summary，验证四次打开、原计划计数及全部第一阶段 SHA；它另调用原严格工件校验和真实 A12 stage contract 校验，减少单次真实执行结束时才暴露接口不兼容的风险。

工程测试运行由协调代理执行，科学审核没有冒称重跑。v5 的 125 项 pytest、完整绑定 Python ruff 与 project audit 均有已读取的持久 PASS 收据；v6 仅变更 driver、坐标 helper 及其两份测试，最后补充针对性测试与全绑定 ruff。原 v5 成功流程和 v6 单 Pm 失败流程分别覆盖严格工件与 A12 stage contract 接口，旧收据不覆盖或改写。v7 相对 v6 仅把 `tests/test_audit_access_ledger.py` 最后两个空行及 `tests/test_record20_input.py` 最后一个空行的 CRLF 改为 LF；我把这三个行尾在内存中还原后，分别精确恢复 v6 登记 SHA，证明没有其他字节变化。生产源码和测试语义均未改变，无需据此重跑科学计算。本稿最终核对的是 v7 全部实际字节与上述科学边界。执行前必须完成同字节提交、工作区/HEAD 双 SHA 检查、活动单次授权及工程验证，不可跳过待定状态直接调用内部工作流。

以下收据位于 `scratch/one_off/kiki-a12-implementation-20261024/`；可逐字节归档到 tracked 的 `docs/research/kiki_a12_record20/checks/`，不修改原记录。我已读取五份收据，并独立核验各自绑定原始日志 SHA，全部相符。收据均声明 `real_workflows_executed=0`。

| 收据 | 结果 | 收据 SHA256 |
|---|---|---|
| `v5-pytest-20261024T102718672733Z.json` | 125 项通过 | `2daf11ad153f4f098633cc2bf7e4a164a5a2e4bb5d97ce23b0d39bb27161e112` |
| `v5-audit-20261024T102718881510Z.json` | project audit PASS | `8ddd256e01aa64165cf89cbcf555930e7a0648fadaa4326585d67f95588101b7` |
| `v5-ruff-20261024T102718788893Z.json` | 全绑定 Python ruff PASS | `3e902cf80d520f9f264a29c587cbe22245cc1e77f32d05b8d028116d77ca2e47` |
| `v6-pytest-20261024T103348089723Z.json` | 两份变更测试文件共 22 项通过 | `9000fae1a4c509d6192eef7db59a7cbcaaa2e0ff5967603bb893c093c65c05e6` |
| `v6-ruff-20261024T103348059821Z.json` | 全绑定 Python ruff PASS | `04a0ce2090d4b0ea75d98d14a3f5d1ced60bd57afa4f9549010909ce769fe369` |

无论真实结果如何，`new_training_authorized=false`、`model_selection_performed=false`、`model_promotion_allowed=false`、`internal_val_coverage_verified=false`、`zero_missed_instability_established=false` 必须保持。通过只能支持这 268 条 train 的 record20 输入与规则一致性，不能填成 val 的 27/27 覆盖，也不能将五点确认提前量解释为在首次逐点违规前预见失稳。下一项初始化机制审计需要独立登记；本审核没有扩大为 A13 真实计算授权。
