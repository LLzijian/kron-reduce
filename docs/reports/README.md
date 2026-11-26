# 研究成果索引

从下表进入编号主题，再打开对应的实验页。同一页收录这轮的研究方案、执行协议、报告、核验回执和运行结果入口。

主题编号按首轮有记录的研究时间排序，同一主题的后续工作沿用原编号。

| 研究主题 | 首轮记录 | 查阅范围 |
|---|---|---|
| [00 · 早期物理基线](00_early_physics_baselines/README.md) | 2026-07 | Paper v1 实验卡、完整流程与 Paper v2 积分弱形式对照 |
| [01 · M0 故障后重积分与暂态电势](01_m0_reintegration_structure/README.md) | 2026-07-29 | 图结构 M0、三阶暂态电势、阻尼恢复与图模态机械功率里程碑 |
| [02 · 图结构与 M0-AFSD 发现](02_afsd_structure_discovery/README.md) | 2026-08 | 图同调对照、模块结构选择，以及数据来源与激励混杂 |
| [03 · IEEE39 模块化能量 P1/P2](03_modular_energy_p1p2/README.md) | 2026-08-31 | P1/P2 命名物理模块的候选取得、校准与可辨识性诊断 |
| [04 · KIKI A03–A17 方法改进](04_kiki_a03_a17_pipeline/README.md) | 2026-10-19 | 从早期隐状态历史试验到固定 TASK3 口径下的逐轮改进 |
| [05 · 数据源与研究治理](05_research_governance/README.md) | 2026-10-23 | 数据来源核验、十机研究范围和历史归档回执 |
| [06 · 短历史初始化与能量约束](06_constant_c_energy_initialization/README.md) | 2026-11-09 | 常数 C 可恢复性、能量上界及固定 A06 模型上的 KIKI train 迁移 |
| [07 · 状态端口结构与 KIKI train 对照](07_state_port/README.md) | 2026-11-11 | 合成结构复核及真实 train 配对 |
| [08 · 逐机隐状态与拓扑修正](08_machine_local_topology/README.md) | 2026-11-12 | 模型实现、观察器、记忆基线，以及阻尼、耦合和恢复力对照 |
| [09 · 拓扑记忆与扰动路径](09_topology_memory/README.md) | 2026-11-16 | 已知真值下的记忆识别、历史初值和耦合强度，以及 TASK3 train 做功诊断 |
| [10 · 故障后统一结构与独立盲测](10_postfault_structure/README.md) | 2026-11-19 | 从首摆损失和初值诊断推进到结构筛选、物理核对和封存盲测 |
| [11 · 量测噪声与坏数据压力测试](11_noise_robustness/README.md) | 2026-11-27 | 冻结 V4 与 ES-ROM 的压力测试，以及 [V2 因果初始化器](11_noise_robustness/v2/README.md) |

## 如何新增报告

- 在主题下按轮次建立一份 README，依次记录方案、执行协议、报告和本轮运行。各 seed、重试和同一轮的冻结复评共享这份页面，有范围差异就分别注明。归档后将运行入口指向对应小包说明。
- 在本轮页面链接回执、图件和复核记录。冻结原件随本轮存入 `_sealed/`，各版本保留其登记时的正文和文件名。共用的 TASK3 对照规范在 05 主题中维护。
- 正式报告写清实验身份、数据范围、划分、随机种子、输入口径、证据状态和科学终点。定量结论指向运行编号或证据路径，并记录实际访问范围。

历史登记仍可用原路径和原 SHA256 核验。[报告路径迁移清单](../../configs/report_relocations.json)记录现址。正文需要更新链接时，冻结原件随主题存放在 `_sealed/`。

归档任务组沿用主题编号。历史配置可用归档时的路径定位到现址。[归档路径迁移清单](../../configs/archive_relocations.json)

## 按旧文件名查归档

- 原 `experiments/` 和 `reports/` 的 69 个文件可在归档回执中按原路径找到 ZIP、包内条目及 SHA256。[归档核验回执](05_research_governance/retired_layout_archive_receipt_20261121/_sealed/retired_layout_20261121t093836z_evidence.json)
- 原 `results/reports/` 的 52 个 M0 文件按数据质量审计、暂态电势验证和 M0-GMD-PM 里程碑归档。[M0 归档核验回执](01_m0_reintegration_structure/m0_reintegration_archive_20261121/_sealed/m0_reintegration_structure_archive_20261121_evidence.json)
