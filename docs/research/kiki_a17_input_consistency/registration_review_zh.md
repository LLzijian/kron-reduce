结论：**APPROVE_FOR_ONE_BOUNDED_ACTIVATION**。本次独立登记复核通过，适用于提交 `25993bb7ceed9af857dd1895cc4becdc527788d5` 中的 A17 登记及其 63 项精确绑定；56 项实现、依赖、测试、科学设计与控制文件继续沿用已批准提交 `6babf354fa3986fd3220ebd53ff3d86dc6ada64b` 的原字节。当前 policy 仍为 `registration_review_pending`，共享预算未消费。本报告同意在下列条件满足后，由统筹执行一次独立激活；报告没有修改 policy 或启动真实计算。

授权依据是用户已经给出的持续推进、核心科学设计与防线判断委托，以及保留并推送全部方案和结果的要求。该委托覆盖这一次用于检验输入一致性的固定模型、原 train、有界比较，无须把它改写成不存在的新增逐预算回复，也无须重复索取同一授权。我的范围判断和审核结论不代替用户授权，更不把“持续降到 0”解释成自动训练、自动增加候选或重复执行的权限。

独立控制核对通过 282 项：TOML 与 policy 逐值、逐类型相同；manifest、输入合同和专用入口身份一致；63 个工作文件与本次 Git blob 的 SHA256 全部匹配；原 56 项与已审实现相同；其他授权、实验登记及输入合同保持不变。新 workflow 为 evidence contract v2，唯一终点为 `input_consistency`，工程 PASS 与科学 ACCEPT/REJECT 分开，22 个必需字段均有明确类型。流程完成但科学 REJECT 能正常归档，不触发新候选、重试或晋升。

初次登记提交 `dd1643cb0fd3d0805fa8e7d3cf019ecf72bddc5a` 的两项问题已闭合：workflow 的协议哈希现与 TOML、policy 和协议原字节一致；新增字段类型约束拒绝 0/false 混用和非字符串提交标识。保存的纯元数据验证包含 1 份合法记录及 22 个错型反例，全部按预期处理。初次 FAIL、修复中间版本和元数据验证程序失败均保留，未覆盖旧证据。期间工作文件已开始修复造成的旧提交/工作区差异不列作第三项科学缺陷。最终 workflow SHA256 为 `123cefa69c0ea0ba4ac0b16d3c6bf7c752b1c3e08f5df710686215936508d90a`。

唯一科学对象保持冻结：三个 A06 step240 模型、seed 324/325/326、原 268 个分配 train 事件及其中 258 个合格历史、774 个 seed–事件槽和原 source→line→event 权重。候选 C 使用保存分解的一次 b+d 初态求解和 Pm11 分段线性闭环历史；每个步长分别保留自己的隐藏末态，只重置观测 q/v。未来仍用原 Pm10 常值。先冻结全部历史预测起点，再读未来时钟；全部 C 未来和失败冻结后才读保存的 I/E/truth。I 保留 A15 已拒绝机制身份，E 保留 A09/A10 基线身份，优于 I 不能替代优于 E。三个 seed、两个步长分别验收，全部原分母和失败保留；不跨组平均抵消失败。

| 登记量 | 不得增加的固定量 |
| --- | ---: |
| 总范围／三阶段范围 | 10,356／3,100、2,580、4,676 |
| 嵌套过程回调类别总数 | 6,049,612 |
| 模型构造／结构 eigvalsh／唯一 LS | 3／9／774 |
| 新历史路径／新未来路径 | 1,548／1,548 |
| 全部 RHS／实际 RK4 更新 | 5,831,802／1,345,086 |
| NPZ 文件／context／成员 | 3,358／3,616／20,406 |
| 解码 f8 元素／来源 JSON 字节 | 21,486,372／72,922,783 |
| opaque 完整性 hash 字节 | 641,698,014 |
| 新 fit／优化更新 | 0／0 |

完整分类上限、诊断运算、实际返回计数及失败保留规则以已冻结 read/compute 文件和前次整体验收为准，表格不替代它们。局部数值失败保留已返回前段；来源、身份、预算、日志和保存错误停止整轮；中断为 unknown/spent。一次共享预算必须先于真实源 hash/解码消费，不重试、不恢复、不转移，也不能通过换工作树或 clone 再执行。

激活和执行前必须完成以下条件：

1. 将本报告与机器收据原字节归档，保存其 SHA 和本次登记提交；新增审核绑定，保留此前 whole implementation 审核字段。登记提交及最终激活提交都应精确推送并通过各自 CI。此复核不将尚未取得的 CI 状态写成成功。
2. 最终激活提交只包含下文允许的元数据变化。核对完整工作文件与该提交 blob、全部绑定及干净状态；任何科学源、依赖、测试、协议、TOML、workflow、输入合同、design/read/compute 的变化需另行审核。
3. 使用 `scratch/runtime/a07-py312/Scripts/python.exe`、Python 3.12.14 和约束文件中的七个包版本。旧运行环境收据验证了 29 个模块来源及七项版本，但采集时工作区有登记改动；它不能作为最终 clean HEAD 证据。须在最终激活 HEAD 再留存只导入、不构造模型或执行科学的运行环境、来源路径和干净状态记录，不使用 uv run 改建环境。
4. 确认同一 common Git 目录中的 A17 共享预算不存在、policy 消费数仍为整数 0，其他已消费授权保持关闭。上述条件满足后，只能使用已审专用 CLI 一次；不得用真实调用探测是否能通过权限门。

允许的激活差异精确限定为：A17 policy 的 `status` 从 `registration_review_pending` 改为 `user_approved`，`registration_status` 改为 `approved_for_one_bounded_activation`；新增本次登记审核正文和机器收据的路径、SHA、审核结论及已满足记录，并将这两份原字节文件加入 `implementation_bindings`。保留用户原话、authority/范围说明、whole implementation 历史字段和全部预算/关闭开关。`registration_review_required` 和 `actual_execution_commit_ci_required` 保留为 true，并用履行证据表明要求已满足。policy 自身的 SHA 或最终提交身份另存外部收据，不能把文件自身的 SHA 写回同一文件。

manifest 允许且仅允许 A17 的 `registration_status` 作同样状态替换；按现有字节只替换该字符串后，允许的 manifest SHA256 为 `db3352681c46f28be58c17fca7c7fa4c3eef15b115c1ece87a3014c318105068`，policy 中该文件的绑定随之更新。AGENTS.md 与本目录 README 可更新 A17 状态及指向本审核；.gitattributes 可为新增审核/收据加字节保留条目。新增登记修复、CI 和运行环境证据可归档。上述文档变化不得改变科学含义、人口、输入、预算、阈值或历史授权。协议正文和准备控制文件中的制备状态保留原字节，不为激活改写。

执行后的工程完成必须同时具备外部 completed 收据中最后四个封存动作的实际返回/字节/manifest SHA、harness 严格封存及阶段记录、独立新 run 保存证据审核；仅存在 completed 文件不足以判定工程成功。任何 RUN_FAILED、未闭合范围或 unknown/spent 均保留且不得重试。科学接受仍只按冻结 V∧G_I∧G_E，任何结果均 KEEP_BASELINE，不开放 labels、事件时刻、未来 Pm、Pe、电压、HDF、internal val 或 source val/test，不产生新 FSR/LOS、认证、部署、默认初始化改动或达到零漏判的结论。

本次审核只读取源码、配置、已归档文本/控制元数据及合成验证收据；真实旧载荷读取/hash、模型构造、LS、RHS、fit、policy 写入和预算消费均为 0。
