# 日志同步开销：只读代码评估

## 静态代码证据

- `durable_audit_ledger.AppendOnlyAuditJournal.append()` 对每条 JSONL 记录执行 `write → flush → os.fsync`。`DurableAccessLedger.call()` 在回调前写入并同步 `attempted`，回调成功后同步 `returned`；回调失败时尝试同步 `failed`。phase 轮换要求先关闭 phase 1 journal，关闭本身再执行一次 flush/fsync。由此，source 操作前的 durable reservation 和操作结果的 durable terminal 记录分属操作前后两个边界，不能用一次事后的 fsync 同时替代。
- `limited_npz_audit` 本身通过 `AccessLedger` 记账；默认 ledger 是内存实现，构造器可注入 ledger factory。只有注入 `DurableAccessLedger` 时，NPZ selector 转换才产生上面的逐条 fsync。NPZ 读取器每个已登记文件只打开一次，分别经 descriptor/payload selector 完成成员读取；该模块没有自己的 fsync 循环。
- `InputConsistencyReader` 使用同一 `AppendOnlyAuditJournal`。每个 `_call()` 先持久化 `attempted`，然后运行回调，成功后持久化 `returned`；失败时记录 `failed`，并永久 poison。attempt 写入失败时回调不会运行。phase freeze 和 reader close 是独立关闭边界；close 也 flush/fsync。按此状态机，尚未同步的 attempt 不能授权 source callback；尚未同步的 terminal 不能被算作已闭合返回。
- `OutputIO._append()` 对每条输出 I/O transition 写入并 flush journal，但不逐条 fsync。open 的 attempted/terminal 是该 journal 的主要 transition；read/write/seek/fsync 细节累计在内存 handle 记录。context exit 执行 journal flush/fsync 后关闭。业务结果文件自身的 fsync 是另一个被测量的操作，不能与 journal fsync 混为一谈。

## 未来可单独评估的方案

1. 优先保持每次 source callback 前的 reservation fsync。若未来想将多个 reservation 合并同步，必须先持久化完整、不可撤销的预留集合，再允许集合内操作；未执行的预留也会消耗预算。这会改变当前状态机和证据含义，需要新协议、测试与审查，不能应用到当前 A17。
2. 可单独研究将已完成回调的 terminal 记录暂存到有界批次，再在释放下一阶段、freeze、journal 轮换或报告完成前统一 flush/fsync 并 range-hash。批次中断时，所有“attempted 但 terminal 未持久化”的操作必须按 unknown/spent 处理，保留最后一个已同步前段，禁止重试或重新访问 source。消费者必须明确接受这种缺少 terminal 的失稳样本；若当前审计格式要求每个 operation 必须有闭合记录，就不能直接采用。
3. `OutputIO` 已按每条 transition flush、退出时 fsync 的方式工作；在这一边界上不应再假定逐条 fsync 是慢点。任何调整仍需保持 open/write 的实际字节、失败未知量、journal exit durability 和最终 range-hash 一致。

## 结论边界

这些判断来自指定源码的调用路径与状态字段；本次没有读取运行产物、common budget 或任何 payload，也没有计时或 benchmark。代码结构说明逐条 fsync 可能增加延迟，但未证明它是本次读取阶段变慢的原因，更没有量化收益。任何批处理都只是未来独立工程提案，不改变当前 A17 的记录、预算、重试或 source-access 规则；需 Astra 后续审查批准。
