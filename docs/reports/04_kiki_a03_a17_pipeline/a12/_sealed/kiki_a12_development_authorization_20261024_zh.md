# A12：记录 20 ms 三点输入的单次 train 审计

用户于 2026-10-24 指示继续改进算法、持续工作并完整保留实验。依此登记一个由 Astra Ultra 设计和审核、协调代理核实证据的输入审计。授权 ID 为 `kiki_a12_development_20261024`。本文件与冻结方案共同约束范围；有效状态及实现审核 SHA 以 `configs/research_policy.json` 为准。实现审核未完成时禁止真实执行。

唯一目标是核实原分配 268 条内部 train 在第三个真实记录点起报时的三点输入、两个真实未来点、时钟和源规则是否可用且一致。`record20` 指首个故障清除后记录点之后名义 20 ms，不声称记录首点就是物理清除时刻。原 258 条 100 ms 共同样本及其余 10 条分别保留；不重新分组，不删除失败。

完整科学方案为 `docs/research/kiki_a12_record20/record20_train_input_audit_protocol_draft_zh.md`，SHA `3ccfd537eaf3a8896ccf713d5757798ecf10807ce37770d13ae0b33447b2f6e4`。字段、阈值、严格类型、坐标变换、已见违规描述、失败处理和科学验收均照该方案执行。源码与入口的补充审查在同目录的 `record20_entrypoint_boundary_review_zh.md` 和 `record20_entrypoint_inner_runner_addendum_zh.md`。后者允许在完整专用外层检查后调用只负责不可变运行与证据记录的 `_run_experiment`；禁止调用会扩大读取范围的通用 `run_experiment`/`verify_data`。

固定读取计划为 `configs/proposals/kiki_a12_record20_reads.json`，SHA `4f0cb7552e31c310822e34e05d668ad6a1f07be5ab398f97bcd7db21de51a054`。其 `execution_authorized=false` 表示生成计划时未执行，保留为原始设计事实；有效执行授权由本文件及活动 policy 决定。计划引用的 scratch 草案与已归档设计字节一致。

由已有 allocation 元数据得到 `S=357068`，不是重新读取轨迹后选择的预算：

| 项目 | 固定值／硬上限 |
|---|---:|
| 原分配内部 train | 268 |
| 原 100 ms 共同样本／不合格样本 | 258／10 |
| 数组 selections／shape-dtype 描述符 | 各 1078 |
| root 属性／后置事件属性 | 18／2948 |
| 总属性读取 | 2966 |
| float64 逻辑数组标量 | 7,501,168 |
| 逻辑数组字节 | 60,009,344 |
| 两个 HDF 各前后一次原文件 SHA 校验字节 | 6,870,747,346 |
| 只读 HDF 打开 | 最多 4 次 |
| 模型构造／加载／拟合／优化更新／预测／力求值／RHS | 全部 0 |

只从冻结白名单直接读取两个来源的内部 train。内部 val、source val/test 的数组及事件属性关闭；CSV、splits 连完整性校验也不读。开始与结束各顺序校验一次 HDF 原始字节，不解码其他角色。校验读取量与逻辑数组读取量分别记账。

第一阶段每事件只读完整 `t/delta/omega` 和单行 `Pm[2]`，另读源静态先验。输入对象仅含真实 `t[:3]-t[2]`、三点 q/v、起报功率及静态先验，不包含身份、标签、长度、资格或未来描述；不构造 `CausalStateHistory`，不证明历史或未来 Pm 恒定。所有 268 槽、输入、无标签统计和访问日志持久封存后，第二阶段才从同一全部 268 条读取已列明的 11 个属性。`fault_line` 按源码的线路身份字符串（例如 `Line_6`）完整比较，不能强转整数。

执行身份固定：experiment `paper_kiki_a12_record20_input`，workflow `kiki_a12_record20_input_v1`，stage `input_audit`，module `kron_reduce.workflows.kiki_record20_input_audit`。唯一入口为该模块的专用 CLI。通用项目运行器在任何数据校验前拒绝该 experiment。专用入口完成 policy、文档/实现 SHA、干净 Git 和流程锁检查；工作流再次检查后，先原子预留跨 worktree 共用的一次性预算，再校验或打开源 HDF。

每个选择在源调用前持久记录 `attempted`，返回后记录实际数量。第一阶段日志关闭、SHA 封存后，第二阶段使用另一新日志。日志/封存/范围或预算失败立即停止；单事件数据失败保留并继续其余原计划槽。失败仍消耗本次授权，不重试、不补读、不扩大范围。

运行完成与科学接受分开。只有全体 268 条必需输入、两个未来点、完整时钟、静态/规则/后置属性和预算全部通过才接受 train 输入支持；否则拒绝并保留每条原因。本审计不产生 FN、FP、FSR、NRMSE 或零漏判指标，不验证 val 覆盖，不创建新训练权限，不晋升模型，不建立认证范围，不开放部署预测。后续初始化研究或算法对照仍需各自登记及审核；已有 A04–A11 授权均不重新开启。
