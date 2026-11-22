# 20ms 输入审计入口与源属性：源码裁定

2026-10-24，Astra Ultra。只读源码审查；未打开真实 HDF、CSV、splits、allocation 或 A11 结果，未进行模型计算。本文补充 SHA256 为 `3ccfd537eaf3a8896ccf713d5757798ecf10807ce37770d13ae0b33447b2f6e4` 的协议草案，不扩大其读取范围，也不构成执行授权。

## 1. 选择专用 CLI，保留已有登记与证据门槛

本次采用独立登记的 A12 模块 CLI。暂不增加通用 integrity profile，也不允许以标准 runner 的默认校验为理由读取 CSV、splits 或其他事件。

实际入口存在明确冲突：`project_harness/runner.py:35–38` 在导入 workflow 前调用 `verify_data`。`data_check.py:70–86` 会按 shard 清单校验全部文件，并打开 HDF 统计全部 `events` 数量；`94–109` 还可能校验其他 ready 数据集。因此，即使 A12 的 `run()` 自身只读取白名单，沿标准入口启动仍不满足本协议。

专用 CLI 不得调用 `run_experiment`、私有 `_run_experiment`、`verify_data` 或 `run_one(data_verified)` 来启动该审计，也不通过猴子补丁替换这些函数。它应明确复用或逐项实现以下原有约束，所用辅助函数须确认没有额外数据读取：

1. 从同一个 project manifest、active policy、实验配置和 workflow 核对唯一实验/阶段/模块/数据集/输入合同身份；使用元数据范围内的 `validate_run_policy`、审批/设计/源码的 HEAD 与工作区字节绑定校验。不能伪装成 synthetic 或 historical reproduction 来跳过 real-data 规则。
2. 在真实数据校验或解码之前，确认 clean HEAD、固定 runtime、正确的下一阶段及必需审批，生成不可覆盖的 run ID。使用 Git common directory 下的共享锁和一次性 ledger 目录，原子预留唯一授权；不是各 worktree 自己的一份预算。任何终止保留预留记录，不自动重试。
3. 将唯一许可的两个 HDF 路径、原 SHA、文件大小、268 条白名单及精确 selector 预算绑定到计划。只对这两个 HDF 做协议允许的前后各一次原始字节校验；不迭代 allocation/source binding 中其他文件，不枚举 HDF 的其他事件。
4. 仍保存全部读取/失败/未尝试槽位、无标签封存、后置属性核对、实际预算、完整 artifact manifest 和 stage evidence。收尾使用严格工件校验及 `record_stage`，不另造一套宽松成功规则。完成但科学 REJECT 可以如实登记为完成过程加 REJECT；致命错误必须保留 `RUN_FAILED`，不能伪造 PASS。
5. 在收尾再次核对 HEAD、受保护文件及源字节身份，遵守原报告与授权消费流程。专用入口本身不获得模型调用或后续训练权利。

还需一个只拒绝、不新增读取权限的最小通用入口检查：为该实验注册例如 `execution_mode="bounded_input_audit_cli_only"`，标准 `run_experiment` 在 `verify_data` 之前遇到此模式立即报错。该改动只阻止误用标准入口，不实现可自由配置的完整性范围。否则错误地调用标准 runner 时，仍会在模块有机会拒绝之前产生越界读取。合成测试必须拦截 `verify_data`/文件打开，证明标准入口对 A12 零读取拒绝，原有实验路径不变。

所有 CLI 可变参数只能选择已经冻结的登记身份，不能接受任意事件列表、字段、数据根覆盖、跳过校验或临时预算。环境解析后的合法数据根也必须与登记路径和两个文件 SHA 一致。任何共享 ledger 已存在、授权 consumed、阶段不符或源码绑定不符都在源文件打开前拒绝。

## 2. 两版生成器均保存全部必需属性

已逐项核对保存的历史源码：postfault commit `62cc1ea9c999860a132e2b79945788f0715703e3`，expansion commit `9666105d3adfb44ffc45d416ff734cc187691d26`，路径均为 `experiments/generate_private_dataset.py`。

| root 属性 | postfault 行号 | expansion 行号 |
|---|---:|---:|
| config | 228 | 229 |
| andes_version | 230 | 231 |
| case_sha256 | 231 | 232 |
| git_commit | 232 | 233 |
| omega_s | 233 | 234 |
| mva_base | 234 | 235 |
| raw_units | 235 | 236 |
| postfault_signature | 249 | 250 |
| state_mapping | 250 | 251 |

两版事件 metadata 定义行号相同：`fault_line` 在 187；`clearing_time_s/record_start_delay_s/label` 在 189；`termination_reason/event_time_s` 在 190；`trajectory_end_s/valid_length` 在 191；`postfault_signature/solver_success/failed_steps` 在 192。全部 metadata 项由 postfault 259–260、expansion 260–261 逐项写入 `group.attrs`。没有需要依赖 CSV 才能取得的必需字段。

root 的 signature/mapping 在 `rows` 为空的分支写入，但该写入先于事件完整保存与 `rows.append`；仅看到这两个属性不等于已经证明首事件成功接受。真实文件是否保留全部属性仍由未来授权审计核查，本次没有预读 payload。

缺少 root 必需属性使来源检查失败并停止后续解码；缺少事件属性保留该槽位失败，并按原协议完成其余预留尝试。两种情形都不能从 CSV、其他事件或现代生成器默认值补齐。全体必需项未通过时科学结论为 REJECT。

## 3. 本次实际读取的核心源码字节

| 文件 | SHA256 |
|---|---|
| project_harness/runner.py | d7c74506feb915e99671907679ebf14ccce900cf84eefbd1b50cf2d541819caf |
| project_harness/data_check.py | 2ac5412637dc6ec9039a78329a22b1cef09a497304f5a8e61ed7900120504947 |
| project_harness/policy.py | ff1e45110707e6241a5d280e1812d299cb58e2a7d58257e426072f401dfcb150 |
| project_harness/loop.py | f8e214696d6392f8b2200af6b8972b29b5966a791695612b53bc29d70a2de090 |
| project_harness/evidence.py | bdd3b26e27f0edb0e6f89ba78dc1cbe0d32d9b15fc14506765d189cde2c207ae |
| src/kron_reduce/workflows/kiki_short_history.py | e50a8ab67864dac9df198e44e41fb74e777f27203443f226a20c8c6b65ac2722 |

两版历史生成器的 SHA 分别为 `f34b2cf1430f9c9666757436a04e65e193a2a65950c1402daa0f36f17171260f` 与 `4e82dba68b49543fea3e00086c499a362e9346d659450a8580916fd45a2dca12`，已由原 source-only evidence 绑定。当前裁定仅决定下一实现入口；最终执行仍须完整登记、合成测试、实际源码审核及冻结。A11 的结果审查优先，本文未改变 A11。
