# 仓库协作说明

本仓库学习并核验电力系统扰动后的物理结构约化模型。先保证物理正确性、闭环轨迹精度和证据无数据泄漏，再评价分类表现。

## 从哪里开始

- 修改子系统前，先读距离目标文件最近的 `AGENTS.md`。
- [项目架构](docs/architecture.md)说明代码和运行产物分别由哪些目录负责。[科学约束](docs/scientific_contract.md)定义输入、轨迹推演、稳定性判定和模型命名。
- [项目清单](configs/project_manifest.json)登记数据集、试验和运行入口。`configs/workflows/` 定义各阶段、验收条件和证据要求。[研究政策](configs/research_policy.json)规定科学执行权限。
- 在[研究索引](docs/research/README.md)查看后续工作的范围及既有决定，在[报告索引](docs/reports/README.md)查找结果与证据回执。

## 工作流

- 先用 `python -m project_harness loop --workflow <name> plan` 查看当前和后续阶段。用 `loop status` 查看已存历史，用 `list` 查看登记项。
- 通过登记入口逐阶段推进。用 `loop run-one` 执行内置检查，用 `loop advance` 提交已核验的外部证据。操作人明确批准后，才用 `loop approve` 记录审批。科学试验须另行登记范围和预算。
- 用 `python -m project_harness audit` 核验元数据和证据，用 `git-check` 检查工作区是否干净。`data-check --dataset <name>` 仅在授权的数据读取范围内执行。

## 科学约束

- 新的真实数据试验使用已登记的 `kiki_ieee39_v1`，保留十台发电机的空间分辨率。空间聚类须经明确的政策变更。用已知真值的合成数据核验结构和稳定性。历史 canonical 数据复现须加 `--historical-reproduction`，并匹配冻结的配置与工作流哈希。
- 新的 KIKI 对照采用冻结的 TASK3 名单，`train=346` 用于拟合，`val=107` 用于预先登记的选择，`test=76` 用于最终评价。按[冻结名单](configs/proposals/kiki_task3_split_reference_v1.json)使用带来源的准确样本 ID。禁止在 `train` 内再次拆分。模型冻结后才评价 `test`，输入和评分须匹配 [TASK3 的 100 ms 对照规范](docs/protocols/kiki_official_train_test_v1_zh.md)。已有使用记录的 `test` 名单按固定对照报告，独立确认须使用独立数据。已执行的历史协议保留原有划分。
- 开始迭代拟合前，登记更新次数上限、在完整 `val` 上评价检查点的频次、停止规则和检查点选择规则。保留 `train` 损失与 `val` 曲线，报告实际更新次数和所选检查点对应的更新次数。
- 每轮新获授权的训练先冻结模型与评价约定，从首次评估起覆盖完整的 `train=346`、`val=107`、`test=76` 名单。只有用户明确提出时才做两例或其他小样本试跑。无论 RMSE 是否达到晋升门槛，都须提供三组完整的轨迹和稳定性结果。快速无附加噪声评估也覆盖全名单，随后完成冻结的完整评价。`test` 只用于报告，禁止用于调参或修改模型。指标缺失、出现非有限值或其他无效结果时，记为失败并报告覆盖情况，禁止补零或填入虚构值。
- 依据冻结的 TASK3 阈值和时间窗口从预测轨迹派生稳定性标签。真实 TASK3 数据沿用冻结的真值标签。合成已知真值数据分别从真值轨迹和预测轨迹派生标签。以失稳为正类，分别报告各组的准确率、TP/TN/FP/FN、失稳召回率、失稳漏判率和稳定误报率。分别记录未决、未覆盖和数值失败，未决不计为稳定。出现任一未决、未覆盖或数值失败时，全名单准确率填 `null`，并报告可判定样本准确率和覆盖率。任何指标分母为零时填 `null`。`test` 结果禁止用于选阈值或改模型。
- 遵循登记的因果输入约定及其起报点。主模型特征排除 `Pe`、电压、标签、事件时刻、有效长度、掩码、故障元数据及未来汇总量。每套轨迹数据拟合一个模型。按激励工况核查各模块的数据支撑，并裁剪缺少支撑的模块。不同来源和工况的预测共用同一模型。
- 对超出认证激励范围的输入拒绝预测。已登记的有界评价按其授权范围执行。从预测轨迹派生探索性的稳定性判定。各项数值证书检查通过后才使用 `M2A-LYAP`。使用 `stability-certified equivalent` 还须通过 [Level 6 及稳定区域和裕度检查](docs/structure_discovery_roadmap.md#level-6-stability-certificates-and-regions)。
- 已用完的授权保持关闭，保留失败和未覆盖的样本。后续研究先核对已存证据。新对照须另行登记并取得科学审批。模型晋升、修改默认部署模型、从已否决的分支提取内容均须单独审查。

## 工程约定

- 协调者负责运行代码改动、集成和证据核验。核心算法设计、科学验收与边界判断交给 Astra Ultra。独立且范围明确的工作交给能力足够、成本较低的子代理，简单编辑和测试通常交给 Luna。见[工程分工](docs/architecture.md#engineering-task-allocation)。本项目禁用 Claude Code 和 CC Orchestrator。
- 直接调用选定的 Python 3.12 解释器。项目共用环境可从仓库根目录通过 `../.venv/Scripts/python.exe` 调用。版本遵循 `requirements/constraints-py312.txt`。备用环境与临时程序放入已忽略的 `scratch/`，一次性程序放入 `scratch/one_off/<run-id>/`。共用工作区禁止使用 `uv run`。随试验证据记录实际运行环境。
- 每次执行分配新的不可变运行 ID。完成后在 `codex/` 分支提交并推送，经审查后合入 `main`。在 Git 中记录清单和证据回执。原始轨迹、结果正文、数据归档、凭据及本地环境保存在 Git 之外。
- 每次 `codex/` 尝试结束后，在退役分支之前归档，失败的尝试也须归档。见 [Harness 的 Git 归档说明](README.md#git-archives)。
- `AGENTS.md` 保存长期有效的约定和入口。研究推进和决定写入研究文档，执行协议放在 `docs/protocols/`，整理后的结果放在 `docs/reports/`。
