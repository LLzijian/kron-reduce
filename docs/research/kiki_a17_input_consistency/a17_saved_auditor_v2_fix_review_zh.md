# A17 独立保存结果审计器 v2 修订审核

2026-10-25，Astra Ultra。结论：**APPROVE_BOUNDED_READ_ONLY_AUDITOR_REVISION**。这只批准下述审计器缺陷修复的科学边界；协调者核准精确版本后可另启一次新 run 内的只读复核。本报告不表示该复核已通过，不改 A17 科学结果，不允许重跑真实比较或启动训练。

原独立审计器 SHA256 `6159a1a6c38f7b2bb39be18670f4dc5da11246f0d1a96a86d626e7ce9199bc73` 已保留在执行提交 `e736e26091a777f96025f70f865186eb02290d7c`。原 FAIL 日志 `a17_saved_result_audit_stdout_v1.log` 及对应 process receipt 原样保留；原审计退出码为 1，不能改写成通过。

## 已重现的缺陷

只读取已结束的新 run `paper_kiki_a17_input_consistency-input_consistency-20261025T083355586536Z-e736e260-f1ffd6d9` 内的 `read_plan.json`、`access_ledger.json`、`sources.json`。这些 JSON 内的旧来源路径始终作为数据字符串，未据此打开任何旧文件。用旧版本 `audit_saved_provenance` 对这三个已保存对象进行一次纯元数据局部重现，得到确定的代码位置：旧文件第 825 行遍历 static 参数并调用 fingerprint；第 738 行尝试对字符串按 `shape` 等键索引，抛出 TypeError。

生产 A14 static 保存格式是六个数组描述符加 `backbone: "sine"`。冻结 context 和来源 compact proof 则各自只含六个数组。原审计器错误地遍历了参数字典全部七项；原合成 fixture 只放六个数组，因此未触发该缺陷。局部重现走过此前 774 个同槽元数据绑定后在此失败；这不等于整个结果审计已完成。

## 修订与保留的检查

仅修改 `src/kron_reduce/evaluation/input_consistency_result_audit.py` 与 `tests/test_input_consistency_result_audit.py`，另一份 metrics 测试文件保持原字节。

1. 固定 static context 必须恰有六个唯一的 `parameters__...` 成员，对应 mass、edge_stiffness、damping、coupling、latent_skew、latent_dissipation。保存参数字典恰含这六项及 backbone，backbone 必须为 sine。
2. 每个必需参数仍必须是合法描述符，array_member 与冻结成员准确一致；原 shape、dtype、逻辑 SHA、A06 祖先指纹和 compact proof 全字段比较继续执行。不采用“跳过非字典”或放宽缺项的修法。
3. 审计版本标识改为 `A17_independent_saved_evidence_replay_v2`。失败收据增加当前审核阶段、最后代码位置及异常代码位置链；读取 traceback 元数据，不额外打开源文件。
4. V、G_I、G_E、人口权重、阈值、失败保留、冻结顺序、源读取范围和科学算子均未改变。真实 run、manifest、预算、模型、运行环境、控制计划与原审计失败证据均未改写。

## 合成验证及独立复核

两个 auditor 测试文件整套 **250 PASS，6.56 s**；三文件 Ruff **PASS**。pytest 有一条缓存目录写权限警告，测试全部通过；原日志完整保留，未为去掉警告重复测试。新增正例包含生产格式的 backbone 标量，而 compact proof 仍是六数组；19 类反例覆盖参数缺失/错型/错成员、context 缺失/重复/错前缀/额外成员、祖先 SHA/shape/dtype 错误、proof 缺失/额外/错位，以及 backbone 身份错误。另有 TypeError 阶段和代码位置回归。

子审阅人 `state_dependent_port_theory` 对最终 diff 只读复核，确认六参数身份检查未放宽、无科学判据变动，未发现该局部修订阻断项。它未读取真实 run 或另跑测试；测试结论来自上述实际日志。

精确新绑定：

- auditor：`560ad5ca1c04286b11a8cf8a12b0a37257a95ef254eb9ee30d564d2a58ef3bc3`。
- 主 audit tests：`28a3e58692fbde7e845d4a0945166e0dda2f9e2cf2735e1437bfab59dc651a9e`。
- 未改 metrics tests：`228833044fd1b32e62fbcd8d746cf520fa4ac59bead61c99bed816148c339138`。
- pytest 日志：`0b519fca750380142431946194b7c7552918a02a8f88e5ba544944ce1f72c4d3`。
- Ruff 日志：`a4443afdcfb6d7363adb285762515ccf7cf50473b1a05c20c1a50f6bed4d26b0`。

## 下一步权限边界

协调者应绑定 v2 源码与新独占 stdout/process receipt，只对同一不可变的新 run 执行保存证据审计。该复核可以按既有审核器读取新 run 的 JSON/NPZ/日志并重放保存指标，但不得沿来源路径读取旧 payload、构造模型、求 LS、执行 RHS、积分或做新的科学比较。旧 v1 FAIL 必须与新收据并存；如出现产物不一致，仍应报告 FAIL，不能为通过而修订真实产物或阈值。

当前尚未运行修订版真实审计。主程序报告的 V/G_I/G_E 和 REJECT 仍待完整独立复核；本修订不会将拒绝变成接受，也不赋予联合训练、C(q)、默认初始化改变或模型晋升权限。科学结果、工程归档及 main 集成继续分别判断。
