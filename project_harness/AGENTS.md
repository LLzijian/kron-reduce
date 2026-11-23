# Harness 使用与维护说明

`project_harness` 管理试验登记、Git 状态、数据校验、不可变运行记录、审计和阶段审批。数值模型、特征函数库和稳定性分类器放在 [`src/kron_reduce/`](../src/kron_reduce/)。

## 查询和推进工作流

1. 在仓库根目录运行 `python -m project_harness loop --workflow <名称> plan`，其中 `python` 使用共用的 `../.venv/Scripts/python.exe`。`loop plan`、`loop status` 和 `loop next` 保持只读，通过登记元数据和已保存状态查看下一阶段、证据要求、待审批项和运行历史。`list` 列出数据及试验登记。请显式指定工作流，默认值 `paper_v1` 对应历史论文流程。
2. 用 `loop run-one` 执行内置检查。获准的通用试验使用 `run <试验> --stage <阶段>`，专用审计按其协议运行登记的命令。规范数据的历史复现须带 `--historical-reproduction`，并匹配冻结配置、工作流哈希和原有审批。
3. 经操作人明确批准后，用 `loop approve <审批项>` 记录审批。用 `loop advance <阶段> --evidence-file <文件>` 提交已核验的外部证据。证据通过校验后进入下一阶段，科学判定可按登记规则进入验收或回退。
4. `audit` 检查仓库约束和已存证据，`git-check` 检查工作区是否干净。`data-check --dataset <名称>` 核验冻结元数据和源文件哈希，须在单独授权的数据读取范围内执行。

完整登记见[项目清单](../configs/project_manifest.json)、[工作流配置](../configs/workflows/)与[研究政策](../configs/research_policy.json)。历史 `paper_v1_full` 执行前须提交源代码基线、通过数据支持审计并明确批准 `full_run_approved`。运行器在打开 `test` 轨迹前冻结每组交叉验证和随机种子的选择记录。汇总、对抗性检查、LaTeX 表格和论文组装均按各自登记的阶段执行，论文发表由人工确认。

## 实现约束

- 审批、手动提交证据、内置检查和试验启动共用同一把工作流锁。从检查状态到写入证据一直持有该锁。推进阶段和审计回放历史时共用状态转换及证据校验逻辑。
- 在校验数据源或启动试验之前，先核对登记阶段、工作流审批和科学执行权限。一次性专用审计遵循其登记命令与限定读取流程。通用运行器在通用数据校验之前拒绝接手这些登记项。
- 每次试验分配新的运行 ID，保留失败证据。已用完的一次性授权保持关闭。
- 将简单且独立的工作交给轻量子代理，明确文件范围与验收要求。协调者负责复核和集成。见[工程分工](../docs/architecture.md#engineering-task-allocation)。
- Harness 改动用 `python -m project_harness audit` 和 Harness 回归测试核验。源文件校验和单独通过已授权的 `data-check` 检查。

<a id="git-archives" name="git-archives"></a>

## Git 归档

在试验的 `codex/` 分支上提交最终代码和证据回执，从干净的工作区执行归档。结论取 `completed`、`rejected`、`failed` 或 `abandoned`，说明中写明原因，有运行 ID 或回执路径时一并记录。

```powershell
& '../.venv/Scripts/python.exe' -m project_harness git-archive --outcome rejected --message '轨迹 RMSE 未达到晋升门槛' --push
```

带注释的标签 `attempt/<分支>/<完整提交哈希>` 记录分支、结论、说明和对应提交。`--push` 将标签推送至 `origin`。确认远端收到标签后再删除分支，已发布标签保持固定。推送失败时保留分支并用同一结论和说明重试。用 `git fetch --tags` 获取归档，`git tag --list 'attempt/*'` 查看列表，`git show <标签>` 查看记录。
