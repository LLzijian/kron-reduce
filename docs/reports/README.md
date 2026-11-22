# 研究成果索引

按研究问题进入主题页，先看当前结论，再查历次报告、核验回执和图件。每次运行有自己的目录，已归档的证据可从任务组总说明查阅。

主题编号按首轮有记录的研究时间排序，同一主题的后续工作沿用原编号。

| 研究主题 | 首轮记录 | 查阅范围 | 归档总说明 |
|---|---|---|---|
| [00 · 早期物理基线](00_early_physics_baselines/README.md) | 2026-07 | Paper v1 实验卡、完整流程与 Paper v2 积分弱形式对照 | [总说明](../../results/SUMMARY_00_early_physics_baselines.md) |
| [01 · M0 故障后重积分与暂态电势](01_m0_reintegration_structure/README.md) | 2026-07-29 | 图结构 M0、三阶暂态电势、阻尼恢复与图模态机械功率里程碑 | [总说明](../../results/SUMMARY_01_m0_reintegration_structure.md) |
| [02 · 图结构与 M0-AFSD 发现](02_afsd_structure_discovery/README.md) | 2026-08 | 图同调对照、模块结构选择，以及数据来源与激励混杂 | [总说明](../../results/SUMMARY_02_afsd_structure_discovery.md) |
| [03 · IEEE39 模块化能量 P1/P2](03_modular_energy_p1p2/README.md) | 2026-08-31 | P1/P2 命名物理模块的候选取得、校准与可辨识性诊断 | [总说明](../../results/SUMMARY_03_modular_energy_p1p2.md) |
| [04 · KIKI A03–A17 方法改进](04_kiki_a03_a17_pipeline/README.md) | 2026-10-19 | 从早期隐状态历史试验到固定 TASK3 口径下的逐轮改进 | [总说明](../../results/SUMMARY_04_kiki_a03_a17_pipeline.md) |
| [05 · 数据源与研究治理](05_research_governance/README.md) | 2026-10-23 | 数据来源核验、十机研究范围和历史归档回执 | — |
| [06 · 短历史初始化与能量约束](06_constant_c_energy_initialization/README.md) | 2026-11-09 | 常数 C 可恢复性、能量上界及固定 A06 模型上的 KIKI train 迁移 | [总说明](../../results/SUMMARY_06_constant_c_energy_initialization.md) |
| [07 · 状态端口结构与 KIKI train 对照](07_state_port/README.md) | 2026-11-11 | 合成结构复核及真实 train 配对 | [总说明](../../results/SUMMARY_07_state_port.md) |
| [08 · 逐机隐状态与拓扑修正](08_machine_local_topology/README.md) | 2026-11-12 | 模型实现、观察器、记忆基线，以及阻尼、耦合和恢复力对照 | [总说明](../../results/SUMMARY_08_machine_local_topology.md) |
| [09 · 拓扑记忆与扰动路径](09_topology_memory/README.md) | 2026-11-16 | 已知真值下的记忆识别、历史初值和耦合强度，以及 TASK3 train 做功诊断 | — |
| [10 · 故障后统一结构与独立盲测](10_postfault_structure/README.md) | 2026-11-19 | 从首摆损失和初值诊断推进到结构筛选、物理核对和封存盲测 | — |

## 如何新增报告

- 同一问题继续比较时，运行产物放入对应运行目录。影响结论的证据在主题页补一行，链接运行编号或回执。
- 形成单独引用的结论时，把里程碑报告放进对应主题目录。多个冻结版本并存时，文件名带日期。
- 主题页链接报告、回执和图件。图件与验证脚本按报告编号放到该主题的 `figures/` 下。
- 正式报告写清实验身份、数据范围、划分、随机种子、输入口径、证据状态和科学终点。定量结论指向运行编号或证据路径，并记录实际访问范围。

历史登记仍可用原路径和原 SHA256 核验。[报告路径迁移清单](../../configs/report_relocations.json)记录现址。正文需要更新链接时，冻结原件随主题存放在 `_sealed/`。

归档任务组沿用主题编号。历史配置可用归档时的路径定位到现址。[归档路径迁移清单](../../configs/archive_relocations.json)

## Retired top-level archives

原 `experiments/` 和 `reports/` 的 69 个文件已按研究主题归入四个任务组。先读任务组总说明，再按小任务打开包外说明。按旧路径搜索[归档核验回执](05_research_governance/retired_layout_20261121t093836z_evidence.json)可定位 ZIP、包内条目和 SHA256。

| 任务组总说明 | 新归入的资料 |
|---|---|
| [00 · 早期物理基线](../../results/SUMMARY_00_early_physics_baselines.md) | Paper v1 实验卡、冒烟检查与完整运行报告，Paper v2 积分弱形式报告 |
| [06 · 短历史初始化与能量约束](../../results/SUMMARY_06_constant_c_energy_initialization.md) | 常数 C 合成实验、能量约束比较与确认、KIKI 能量上界迁移 |
| [07 · 状态端口结构与 KIKI train 对照](../../results/SUMMARY_07_state_port.md) | 合成试验和真实数据对照的保存证据核验 |
| [08 · 逐机隐状态与拓扑修正](../../results/SUMMARY_08_machine_local_topology.md) | S0/S1、逐机初始化、KIKI 基线与拓扑修正的核验材料 |

## Early M0 report archives

[01 · M0 故障后重积分与暂态电势总说明](../../results/SUMMARY_01_m0_reintegration_structure.md)串起数据质量审计、暂态电势验证和 M0-GMD-PM 里程碑，并链接三个小任务各自的包外说明。[归档核验回执](01_m0_reintegration_structure/m0_reintegration_structure_archive_20261121_evidence.json)可按原路径查找 52 个文件所在的 ZIP。
