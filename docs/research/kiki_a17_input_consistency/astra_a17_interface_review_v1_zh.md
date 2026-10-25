# A17 接口只读审核 v1：暂不批准集成执行

2026-10-25，Astra Ultra。范围为新 kernel、metrics、journal、reader 及其合成测试源码，同时核对已审纯数组 checks 的衔接。**结论：存在阻断项；这不是完整工作流或精确实现绑定批准。** 未打开/hash 真实 NPZ/HDF，未运行真实数组、模型、求解、RHS 或新的合成测试。

本稿保留初读版本发现的问题。协调 agent 已收到并正在并行修复；以下按函数及语句定位，不声称后续工作区版本仍保留同一问题。修复必须有针对性的合成回归和重新只读核对，不能以已有测试数量代替关闭问题。

## B01　汇总派生值非有限时仍可能形成通过的比较

位置：`input_consistency_metrics.py::population_scores` 的 partial、CVaR、完整 3 s 条件均值；`paired_group` 的差值和比较。

输入事件值有限不保证加权乘加有限。纯合成反例：单事件各指标取最大有限 float64，counts=[300]，weights=[1+5e−13]。现有总权重容差接受它，加权指标可能变成 inf，仍标 complete；零损失候选可以相对这一无效参照通过。

修复：所有派生总体指标、时长组贡献、CVaR、条件均值和差值都检查有限性；非有限则保留原始返回/部分结果并标失败或未完成，不进入任何比较。配对入口也独立检查收到的指标，不能仅相信上游 complete。禁止 inf−inf、inf≤inf 或可用子集汇总成为正式通过。回归覆盖乘加、CVaR/条件均值和差值溢出。

## B02　最终门使用真值转换，且 V=false 时正式 G 仍可能为真

位置：`input_consistency_metrics.py::campaign_verdict` 的 `all(group['pass'])`。

完整提供 12 个键，但每组为 `{'complete':False,'pass':NaN}`、V=True，NaN 的真值为 True，原实现可返回 ACCEPT。另 `campaign_verdict(False, all_good_groups)` 原实现给 G_I=G_E=True，与协议的正式门未完成/false 不符。

修复：检查每组 schema、`complete` 与 `pass` 的严格布尔类型；只有 complete is True 且 pass is True 才可参与通过。缺字段、字符串、NaN、None 均不得通过。正式 G_I/G_E 还必须受 V 约束；可另外保存原始比较作为描述，不把它们标成有效总体门。回归同时覆盖 12 键完整但畸形的值、V=false、缺组、失败组和正常六组全通过。

## B03　“after”完整性校验没有真正限定为读后

位置：`input_consistency_reader.py::integrity`，初读版本只对 before 约束 phase；`finish` 只检查两个 hash 名称都出现。

按原接口可以在 phase1 调 after，甚至在 before 之前，随后完成读取；最终集合仍为 before/after。这样没有任何读后完整性校验。

修复：after 仅可在 phase3、before 已完成、全部注册 context 返回后执行；before 只能是读取前的唯一首次完整性遍历。越序必须在任何底层文件 open/read 前拒绝并 poison。回归覆盖 after-before、after-contexts、重复 after 和正常顺序。最终审计还应从持久顺序核验，不能只看集合。

## B04　元数据、冻结标记和 NPY 头部需要前置字节界

位置：reader 的 `read_metadata`、`advance`、`read_context`。

原 `read_metadata` 调 `read_bytes()` 后才核长度；若文件增大，拒绝前已读超登记字节。原 `advance` 可对 root 内任意文件执行同样无界读取/hash，且任意内容只要 SHA 对上就能充当“人口冻结”。

NPY 头部也不能只依赖 NumPy 默认 max_header_size：受约束运行时的 `numpy/lib/_format_impl.py` 初读第 634–638 行是先按 header_length 读取，再检查头部大小。原 reader 在该读取之后才核 ZIP 解压大小。

修复要求：

- 元数据先核 stat，读取量硬限定为原登记长度，读后再核长度/SHA/stat；不额外 EOF 探测，不因失败再读。
- 冻结路径只允许本次新 run 的两个指定标记，明确 phase/run/源码/人口/槽与失败状态/artifact SHA schema，并有已知长度和硬界；不能用旧 run 的任意 JSON 或任意文件代替。完整工作流应证明它绑定所有 origin 或所有候选 future 的已保存状态。
- descriptor 的底层 read 采用累计硬界代理；超过剩余头部额度的请求在实际读取前拒绝。先核 ZIP file_size 与计划 payload、头部上限的关系。保留原 NPY 解析器，避免修改历史实验代码；新增字节限额和实际计数写入 A17 计划。

回归使用显式合成文件：大于登记长度的 JSON、错误 run/phase 或空壳冻结标记、带异常头部长度的压缩成员；监控底层 callback 确认拒绝前没有超额读取。

## B05　已返回的原始计算/读取前段在后续异常时丢失

涉及 reader `_call/read_metadata/read_context`、journal `call`、kernel `solve_saved_decomposition/reset_visible_origin`。

具体触发包括：

1. 数组 callback 已返回，但 returned 日志 fsync 失败。原 `_call` 把 selector 改成 failed，数组与逻辑计数仅在外层正常返回后保存，导致真实已解码值及“函数已返回”的事实丢失。
2. 同一 context 的第一个成员或 metadata 列表的第一个文档已返回，后续成员失败；局部 arrays/documents 未返回，也没有绑定到异常或读者持有的部分记录。
3. LS callback 已完成，但 dispatcher 的 returned 日志失败；局部 rhs/projected_rhs/coefficients/z₀ 不再随异常交付。
4. reset 的第一次 energy 已返回，第二次 energy 或其日志失败；原代码尚未构造 record，丢掉第一次 energy。
5. 局部数值异常已含 partial_record，记录 failed 的日志又失败；新的日志异常不能隐藏原异常及其部分结果。

修复：明确区分 callback attempted、callback actually returned、durable terminal 三件事；日志失败后仍然全局停止、范围 unknown/spent，不伪造正常 returned。把每次实际返回值、局部数组/文档前段、两个 reset 状态及逐步 energy 字段放入可被最终失败保存器取得的部分记录；保留底层异常链，不覆盖已有原始部分记录。不得为恢复证据重复读取、求解、energy 或 RHS。

被 poison 的账本不能继续科学调用。全局失败保存由独立失败收尾路径写出已有内存证据；如果磁盘故障使它也无法保存，只能如实保留 unknown/spent，不能正常 closed。回归必须在“callback 已返回、terminal 日志失败”的准确位置注入一次故障，不能只在 context 的首次 attempted 上故障。

## B06　计划别名可变；关闭状态需要显式拒绝后续调用

位置：reader `self.plan=validate_saved_plan(plan)`、contexts/files 对原嵌套 row/member 的引用及 `_check/close`。

仅保留调用者的原 dict 会使构造后的 member、phase、path、shape 等改变读取边界。至少需要深复制所有嵌套行/成员并固定内部计划；不能仅复制顶层或向外返回可变内部对象。最终核对固定 plan SHA、派生限额和实际完整计数。

成功 close 后应有显式 closed 状态；后续请求在修改 used/selectors 或尝试 append 前拒绝。底层 journal 已关闭虽能阻止真正读取，也不应让关闭对象继续产生虚假的 attempted 状态。回归修改原 plan 的嵌套成员，确认读者边界不变；再检查 close 后所有访问入口都不改变计数。

## B07　局部数值失败后当前 scope 不应继续调用；闭合证据不能覆盖保留字段

位置：`input_consistency_journal.py::call/persist_and_finish`。局部失败后继续同 scope 的新 index，可借剩余限额重新调用或继续失败链；这项也由协调 agent 转述的独立工程复核发现。

修复：scope_failed 锁使局部数值失败后只可保存失败证据及 failed finish；下一独立 scope 的 begin 才重置。尝试同 scope 再执行在 dispatch 前拒绝并全局 poison。只读检查发现基础 ScopedAuditJournal 已对 status=complete 强制逐类全额度返回，应保留这项能力。

`persist_and_finish` 中辅助 evidence 还不能通过字典展开覆盖显式 status 或实际 artifact_hashes；这些字段应保留，status 限定为冻结枚举。未知状态不能绕过 complete 的精确计数。失败闭合必须有原因及相应依赖绑定；不能直接用任意 status 字符串把漏调用的 scope 关成成功。

## B08　kernel 前置检查中的派生 COI/功率和也须有限

位置：`input_consistency.py::path_preflight`。

q/v/power 的元素有限，并不保证 q@mass、v@mass 或 sum(power) 有限。正负巨大数的乘加可产生 inf−inf=NaN；原代码仅用 `max(abs(...))>1e−10` 或 `np.max(abs(sum))>1e−10`，NaN 可能使拒绝条件为 False，进而启动 RHS。

修复：先保存并检查这些派生量有限，再比较原阈值。不得依赖第一次 RHS 的失败来代替初态/输入预检。合成回归应构造有限元素但点积或求和 NaN 的例子，确认 RHS callback 从未执行。没有读取真实值来判断此问题是否会在真实样本触发。

## 接口衔接与完整工作流待证项

- 五个 checks 的 dispatch 类别固定为 `decomposition_check/ls_diagnostics/quadrature_proxy/path_refinement/physical_judgement`。原 journal 的旧名字需同步；不能出现一套 helper 名称和另一套预算名称。
- 唯一 observation_map 必须只算一次。建议用 quadrature_proxy 内部 matmul attempted/returned 的派生细计数对预留上限核对。不能为凑 journal 行补算映射，也不能 `call(observation_map, lambda:None)`。把整个 helper 外再包装为一次映射，在校验早退时会错记实际 map=0 的情况。若需要独立调用日志，只能把 callback 放到真实映射处。
- `input_consistency_checks.py` 的范围保持不变：它不承担来源绑定或原 rank/tol 门。workflow/kernel 必须先证明 rank=2、原 threshold 的身份与固定公式一致、奇异值严格大于阈值，再把同槽 A16 Gram 界传入相同的 B。
- reader 中 freeze SHA 本身不是科学冻结充分证据；三阶段 marker、全部 774 槽/1,548 状态、失败依赖与实际保存 artifacts 需要工作流及独立审计闭合。
- 输入变换、原 h 两次减法、20/40 和 2N_i/4N_i、ΣN_i、I/E/truth 来源链及未来共同 clock 仍需完整实现证明。metadata inventory 的校验通过不能替代未来运行的头部/成员/范围检查。
- read plan 的 `allowed_json_paths` 若只是描述“所用字段”，应明确其性质；若声称是运行时字段白名单，现有完整 `json.loads` 返回接口尚未实现这种限制。精确文件白名单、元数据字节预算和 JSON 语义范围不能混写。

本轮不重跑任何测试来证明以上反例，全部来自源码推理与独立理论复核。协调 agent 报告的已有 90 项合成测试及 inventory 核对收据保留原结果；它们未覆盖上述若干触发点。修复后应只补针对性合成回归，随后再进行完整工作流的精确绑定审核。A17 保持未授权执行。
