# kron-reduce

**IEEE 39 节点系统故障后动态等值建模**

项目保留十台发电机的空间分辨率，在统一的动力学模型中辨识恢复力、耗散与记忆效应，预测故障清除后的功角和转速。模型从清除瞬间的状态或固定长度的观测历史起报，用预测状态推进后续轨迹，并从轨迹判定稳定性。

[研究成果](docs/reports/README.md) · [项目架构](docs/architecture.md) · [科学约束](docs/scientific_contract.md) · [Harness 使用说明](project_harness/AGENTS.md)

## 代表性结果

| 研究 | 结果与评价范围 |
| --- | --- |
| [M5 独立盲测](docs/reports/10_postfault_structure/m5_blind/README.md#doc-a2fdc902d881) | 冻结的 M3b G2 使用 100 ms 历史，在同源新仿真的 94 个事件中检出 29/32 个失稳样本，62 个稳定样本均判稳。按冻结 TASK3 规则的全名单准确率为 96.81%，1.25 ms 与 2.5 ms 积分步长的判定逐例一致。 |
| [TASK3 方法对照](docs/reports/10_postfault_structure/task3_comparison/README.md) | 对齐起报输入、误差尺度和稳定性判定规则，逐名单报告轨迹误差、失稳检出及可判定覆盖率。 |
| [M4 物理一致性检查](docs/reports/10_postfault_structure/m4_m3b/README.md) | 在冻结的十机模型上核查故障后主模态和物理结构。起报储能分布见 [M4 v2](docs/reports/10_postfault_structure/m4_v2/README.md)。 |

## 研究范围

现行 KIKI 对照采用固定的 TASK3 名单，`train=346` 用于拟合，`val=107` 用于按预先声明的规则选择模型，`test=76` 用于模型冻结后的评价。已有使用记录的 `test` 名单用于固定对照。独立确认使用另行登记的数据。名单、100 ms 输入和评分细则见 [TASK3 对照规范](docs/reports/05_research_governance/task3_standard/README.md#doc-6c6486ac9f76)与[冻结名单](configs/proposals/kiki_task3_split_reference_v1.json)。

每套轨迹数据拟合一个共享模型。真实数据试验保留十机分辨率，另外用已知真值的合成数据核验隐状态、物理模块和稳定性证书。新轮次的执行权限见[研究政策](configs/research_policy.json)，结构辨识目标见[路线图](docs/structure_discovery_roadmap.md)。

## 快速开始

在仓库根目录使用项目共用的 Python 3.12。首次安装依赖时运行第一条带 `pip` 的命令，后续可直接查看登记和审计结果。

```powershell
$repoPython = '../.venv/Scripts/python.exe'
& $repoPython -m pip install -e '.[dev]' -c requirements/constraints-py312.txt
& $repoPython -m project_harness list
& $repoPython -m project_harness loop --workflow postfault_m4_v2 plan
& $repoPython -m project_harness audit
```

`list`、`loop plan` 和 `audit` 查看登记、阶段和已存证据。运行试验须按[工作流说明](project_harness/AGENTS.md)使用登记入口。CUDA 环境与逐机模型的性能实测见[架构文档](docs/architecture.md#machine-local-cuda)。

## 仓库导航

| 位置 | 内容 |
| --- | --- |
| [`src/kron_reduce/`](src/kron_reduce/) | 物理模型、轨迹推演、训练和评价 |
| [`project_harness/`](project_harness/) 与 [`configs/`](configs/) | 试验登记、运行管理、证据审计和审批 |
| [`data/`](data/) | 数据登记和校验信息。KIKI 原始 HDF5 文件保存在仓库外，通过 `KIKI_DATA_ROOT` 指向包含 `ieee39_postfault` 和 `ieee39_expansion` 的目录 |
| [`docs/reports/`](docs/reports/README.md) 与本地 `results/` | 研究方案、结果、证据回执和归档索引 |
| [`paper/`](paper/) | 论文源码与投稿检查项 |

<a id="git-archives" name="git-archives"></a>

### Git 归档

在试验的 `codex/` 分支提交最终代码和证据回执，按 [Harness 归档说明](project_harness/AGENTS.md#git-archives)发布带注释的归档标签。

## 使用许可

论文发表前，项目原创代码与文档保留全部权利。使用和再分发须依据现有授权或取得著作权人事先书面许可。详见 [LICENSE](LICENSE)。
