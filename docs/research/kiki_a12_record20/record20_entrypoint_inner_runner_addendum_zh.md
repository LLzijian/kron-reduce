# A12 专用入口复用内层 runner 的补充裁定

2026-10-24，Astra Ultra。只读源码审核，无真实数据读取或计算。补充原入口审查 SHA256 `08df16e4fd51457aaf431696844b1edb43a5daef436278ec2809d214dea721d1`；原文件保留。本条只修正其中对私有 `_run_experiment` 的一概禁止，不改变字段、人口、阶段、预算或数据范围。

**同意专用 CLI 在完整执行外层门槛后，复用现有 `_run_experiment`。** 已读 `project_harness/runner.py` SHA256 `d7c74506feb915e99671907679ebf14ccce900cf84eefbd1b50cf2d541819caf`，其 96–166 行的内层入口只处理登记、阶段、不可覆盖目录、模块运行、严格工件验证与 stage evidence，不调用 `verify_data`。因此无须复制这套已有证据逻辑。

批准的待实现调用链为：固定身份的 A12 专用 CLI → 元数据范围的 `validate_run_policy` 与完整审批／设计／源码字节校验 → clean HEAD 与同一 workflow 锁 → `_run_experiment` → A12 模块严格核对自身全部授权字段并原子占用 Git common directory 的一次性账本 → 只按既定范围校验两个 HDF 与读取固定白名单 → 原严格工件核验和 `record_stage`。

外层不能省略原 policy、审批或并发门槛；内层动态导入的模块和唯一实验／阶段必须与冻结 manifest 精确一致。模块导入不得访问真实数据。A12 模块在任何 HDF 校验字节或 dataset/attr 读取之前完成自身授权、shared ledger 及固定计划检查；失败保留 run 与槽位，不自动重试。通用 `run_experiment` 仍须在 `verify_data` 前拒绝该专用 execution mode。对整个调用链做合成读取拦截测试，而不是仅测试最终 reader。

这种复用只替代重复的工程代码，没有建立通用完整性豁免。最终批准仍以实际专用 CLI、runner 前置拒绝、A12 模块和 synthetic tests 的冻结字节审核为准；现在没有真实执行授权。
