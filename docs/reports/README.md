# 研究成果索引

按研究问题进入主题页，先看当前结论，再查历次报告、核验回执和图件。每次运行有自己的目录，已归档的证据可从任务组总说明查阅。

| 研究主题 | 查阅范围 |
|---|---|
| [故障后统一结构与独立盲测](postfault_structure/README.md) | 从首摆损失和初值诊断推进到结构筛选、物理核对和封存盲测 |
| [拓扑记忆与扰动路径](topology_memory/README.md) | 已知真值下的记忆识别、历史初值和耦合强度，以及 TASK3 train 做功诊断 |
| [逐机隐状态与拓扑修正](machine_local_topology/README.md) | 模型实现、观察器、记忆基线，以及阻尼、耦合和恢复力对照 |
| [短历史初始化与能量约束](short_history_initialization/README.md) | 常数 C 可恢复性、能量上界及固定 A06 模型上的 KIKI train 迁移 |
| [状态端口结构与 KIKI train 对照](state_port/README.md) | 合成结构复核及真实 train 配对 |
| [KIKI A03–A17 方法改进](task3_model_refinement/README.md) | 从早期隐状态历史试验到固定 TASK3 口径下的逐轮改进 |
| [IEEE39 模块化能量](modular_energy/README.md) | P1/P2 命名物理模块的候选取得、校准与可辨识性诊断 |
| [图结构与 M0-AFSD 发现](afsd_structure_discovery/README.md) | 图同调对照、模块结构选择，以及数据来源与激励混杂 |
| [M0 故障后重积分与暂态电势](m0_reintegration_structure/README.md) | 图结构 M0、三阶暂态电势、阻尼恢复与图模态机械功率里程碑 |
| [早期物理基线](early_physics_baselines/README.md) | Paper v1 实验卡、完整流程与 Paper v2 积分弱形式对照 |
| [数据源与研究治理](research_governance/README.md) | 数据来源核验、十机研究范围和历史归档回执 |

## 如何新增报告

- 同一问题继续比较时，运行产物放入对应运行目录。影响结论的证据在主题页补一行，链接运行编号或回执。
- 形成单独引用的结论时，把里程碑报告放进对应主题目录。多个冻结版本并存时，文件名带日期。
- 主题页链接报告、回执和图件。图件与验证脚本按报告编号放到该主题的 `figures/` 下。
- 正式报告写清实验身份、数据范围、划分、随机种子、输入口径、证据状态和科学终点。定量结论指向运行编号或证据路径，并记录实际访问范围。

历史登记仍可用原路径和原 SHA256 核验。[报告路径迁移清单](../../configs/report_relocations.json)记录现址。正文需要更新链接时，冻结原件随主题存放在 `_sealed/`。

## Retired top-level archives

原 `experiments/` 和 `reports/` 的 69 个文件已按研究主题归入四个任务组。先读任务组总说明，再按小任务打开包外说明。按旧路径搜索[归档核验回执](research_governance/retired_layout_20261121t093836z_evidence.json)可定位 ZIP、包内条目和 SHA256。

| 任务组总说明 | 新归入的资料 |
|---|---|
| [早期物理基线](../../results/SUMMARY_01_early_physics_baselines.md) | Paper v1 实验卡、冒烟检查与完整运行报告，Paper v2 积分弱形式报告 |
| [逐机隐状态与拓扑修正](../../results/SUMMARY_03_machine_local_topology.md) | S0/S1、逐机初始化、KIKI 基线与拓扑修正的核验材料 |
| [短历史初始化与能量约束](../../results/SUMMARY_07_constant_c_energy_initialization.md) | 常数 C 合成实验、能量约束比较与确认、KIKI 能量上界迁移 |
| [状态端口合成与 KIKI train 对照](../../results/SUMMARY_08_state_port.md) | 合成试验和真实数据对照的保存证据核验 |

## Early M0 report archives

[M0 故障后重积分与结构发现总说明](../../results/SUMMARY_06_m0_reintegration_structure.md)串起数据质量审计、暂态电势验证和 M0-GMD-PM 里程碑，并链接三个小任务各自的包外说明。[归档核验回执](m0_reintegration_structure/m0_reintegration_structure_archive_20261121_evidence.json)可按原路径查找 52 个文件所在的 ZIP。
