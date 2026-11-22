# TASK3 逐机记忆模型的阻尼校准

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md)

登记文件与配置

[拟合方案](../../../../configs/proposals/kiki_machine_local_topology_damping_fit_v1.json) · [test 方案](../../../../configs/proposals/kiki_machine_local_topology_damping_test_v1.json)

## 执行协议

<a id="doc-3c60d2115ec7"></a>

<a id="doc-3c60d2115ec7-task3-逐机记忆模型的阻尼校准"></a>

### TASK3 逐机记忆模型的阻尼校准

从原始 T0M1 第 12 步检查点出发，检验合成试验中恢复失步判断的阻尼校准是否适用于 TASK3。沿用十机方程、原 train 归一化尺度、记忆半径 6.61055385191009 和历史拟合先验权重 $10^{-6}$。只调整现有可见阻尼的统一倍率 $D=e^\alpha D_{\mathrm{base}}$，所有边刚度、逐机端口和记忆时间常数保持冻结。


<a id="doc-3c60d2115ec7-拟合与冻结"></a>

#### 拟合与冻结

- 使用原 TASK3 的 346 个 train 事件。每个事件用 11 点、100 ms 的功角和转速历史确定隐藏初态，预测功率沿用末点零阶保持。按原训练尺度计算 1 s 与 3 s 的等权状态均方误差，短轨迹按各时域内实际可用点数归一，再对全体事件等权汇总。训练积分步长上限为 5 ms，批大小 64，CUDA float64。
- 在起始 $\alpha$ 的两侧各 $\log 2$ 范围内取 9 个等间隔点。网格最优值在内部时，在两侧相邻网格点之间做有界标量搜索，最多追加 24 次完整 train 目标计算，绝对容差 $10^{-5}$。有限结果中保留损失最低的点，同时保存每次计算的损失和失败情况。记录边界最优及数值失败。
- 在读取 val 前保存最终方程、归一化尺度和初始化规则。val 用于对照原 T0M1，随后通过独立授权对同一冻结模型评分 test。RMSE、阻尼校准有效性或 val 表现未过门槛时，仍评分全部 train、val 和 test。


<a id="doc-3c60d2115ec7-完整评价"></a>

#### 完整评价

评分使用 TASK3 的原始名单和冻结标签，覆盖 train 346、val 107、test 76。预测 8 s，积分步长上限分别为 2.5 ms 和 1.25 ms，批大小 32。保存逐事件轨迹及五点失步规则的判断，汇报 TASK3 各时域归一化 RMSE、稳定性准确率、混淆计数、召回率、漏判率、误报率和未决情况。两档步长的判断一致性和缺失值处理沿用原 TASK3 对照。只要存在未决、未覆盖或数值失败，完整名单准确率即为 `null`。

val 科学门槛要求阻尼拟合有效，8 s 稳定轨迹评分完整覆盖 95 例，事件平均归一化 RMSE 至多为原始 T0M1 的 1.05 倍，TASK3 六项综合误差至多为 0.5，正确至少 104 例，误报为零，未决至多 1 例，未覆盖与数值失败均为零。test 目标要求 76 例均可判定且无数值失败，至少正确 69 例。test 仅用于报告。任何部署变更另行审查，保留现有基线。


<a id="doc-3c60d2115ec7-单次执行预算"></a>

#### 单次执行预算

| 上限 | 拟合与 train、val 评分 | 冻结 test 评分 |
| --- | ---: | ---: |
| 缓存打开次数 | 3 | 77 |
| 拟合次数 | 1 | 0 |
| 完整 train 目标计算 | 33 | 0 |
| 检查点解码 | 2 | 2 |
| 积分调用 | 7938 | 1106 |
| 动力学右端调用 | 19889280 | 3261120 |
| 切线右端调用 | 1202400 | 189600 |
| QR、SVD、三角求解、矩阵乘各自上限 | 66150 | 1520 |

评分预算覆盖批次发生数值异常时逐事件隔离计算的上界。实际用量和访问顺序写入追加日志。拟合和 test 各消耗一次独立授权，按保存结果核验名单、冻结摘要、预算和逐事件评分。

## 报告与结果

<a id="doc-3a3b55f16410"></a>

<a id="doc-3a3b55f16410-task3-逐机记忆模型的阻尼校准"></a>

### TASK3 逐机记忆模型的阻尼校准

冻结模型在 TASK3 test 的 76 例中判对 67 例，判定 72 例，完整准确率为 `null`。未达到完整 test 准确率超过 90% 的目标。

从原始 T0M1 第 12 步出发，使用全部 346 例 train 的可见 1 s／3 s 归一化均方误差校准统一阻尼。共计算 25 次完整目标，倍率从 1.0739319 变为 1.862418，train 目标从 0.30550504 降至 0.24791726，降幅 18.85%。有界搜索收敛。对照固定边刚度、逐机端口、时间常数、100 ms 初始化规则和记忆半径。


<a id="doc-3a3b55f16410-稳定性"></a>

#### 稳定性

| 模型 | 划分 | 已判定 | TP/TN/FP/FN | 未决／未覆盖／数值失败 | 完整准确率 | 可判定准确率 | 可判定召回／漏判／误报 |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 原 T0M1 | train | 338/346 | 24/308/2/4 | 8/0/0 | `null` | 98.22% | 85.71%／14.29%／0.65% |
| 校准阻尼 | train | 325/346 | 9/309/0/7 | 21/0/0 | `null` | 97.85% | 56.25%／43.75%／0.00% |
| 原 T0M1 | val | 106/107 | 8/95/0/3 | 1/0/0 | `null` | 97.17% | 72.73%／27.27%／0.00% |
| 校准阻尼 | val | 100/107 | 1/95/0/4 | 7/0/0 | `null` | 96.00% | 20.00%／80.00%／0.00% |
| 原 T0M1 | test | 71/76 | 0/67/0/4 | 5/0/0 | `null` | 94.37% | 0.00%／100.00%／0.00% |
| 校准阻尼 | test | 72/76 | 0/67/0/5 | 4/0/0 | `null` | 93.06% | 0.00%／100.00%／0.00% |

完整准确率的分母为冻结名单。任一事件未决、未覆盖或数值失败时，记为 `null`。召回、漏判、误报的可判定口径仅使用完成判断的对应类别。


<a id="doc-3a3b55f16410-轨迹误差"></a>

#### 轨迹误差

表中为 TASK3 事件平均归一化 RMSE，括号内为合格评分事件数及对应时域的可评分事件数。

| 模型 | 划分 | 1 s | 2 s | 5 s | 8 s | 失稳可用时段 | 六项分数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 T0M1 | train | 0.30225（315/315） | 0.292563（311/311） | 0.267455（311/311） | 0.259891（311/311） | 0.902775（35/35） | 0.46708 |
| 校准阻尼 | train | 0.331878（315/315） | 0.304789（311/311） | 0.256397（311/311） | 0.240989（311/311） | 1.06206（35/35） | 0.49561 |
| 原 T0M1 | val | 0.295697（96/96） | 0.259058（95/95） | 0.223642（95/95） | 0.218248（95/95） | 0.850543（12/12） | 0.438282 |
| 校准阻尼 | val | 0.337656（96/96） | 0.297698（95/95） | 0.245733（95/95） | 0.23232（95/95） | 0.995925（12/12） | 0.481972 |
| 原 T0M1 | test | 0.304603（67/67） | 0.254362（67/67） | 0.213347（67/67） | 0.19067（67/67） | 1.02531（9/9） | 0.42423 |
| 校准阻尼 | test | 0.339472（67/67） | 0.290372（67/67） | 0.233292（67/67） | 0.204973（67/67） | 1.04647（9/9） | 0.445277 |

校准后，train 检出的真失稳从 24/35 降至 9/35，val 从 8/12 降至 1/12。val 未决从 1 例增至 7 例，95 例真实稳定的 8 s 事件平均归一化 RMSE 上升 6.45%。val 科学门槛未通过。模型在读取 val 前冻结，随后完成全量 test。三份保存预测独立复算，逐例轨迹评分、稳定性和汇总一致。保留部署基线。


<a id="doc-3a3b55f16410-运行证据"></a>

#### 运行证据

- 拟合与 train、val 评分 `kiki_machine_local_topology_damping_fit_v1-fit-20261115T090530382303Z-9389f3db-ddaaae58`。[阶段回执](_sealed/kiki_machine_local_topology_damping_fit_20261115_evidence.json)
- 冻结 test 评分 `kiki_machine_local_topology_damping_test_v1-test-20261115T091230749822Z-546af593-d23ff98b`。[阶段回执](_sealed/kiki_machine_local_topology_damping_test_20261115_evidence.json)
- [独立复算](_sealed/kiki_machine_local_topology_damping_saved_audit_20261115_evidence.json) · [执行协议](#doc-3c60d2115ec7)

## 回执与附件

- [kiki_machine_local_topology_damping_fit_20261115_evidence.json](_sealed/kiki_machine_local_topology_damping_fit_20261115_evidence.json)
- [kiki_machine_local_topology_damping_saved_audit_20261115_evidence.json](_sealed/kiki_machine_local_topology_damping_saved_audit_20261115_evidence.json)
- [kiki_machine_local_topology_damping_test_20261115_evidence.json](_sealed/kiki_machine_local_topology_damping_test_20261115_evidence.json)
- [kiki_machine_local_topology_damping_fit_result_review_20261115.json](_sealed/kiki_machine_local_topology_damping_fit_result_review_20261115.json)
- [kiki_machine_local_topology_damping_fit_v1_review_20261115.json](_sealed/kiki_machine_local_topology_damping_fit_v1_review_20261115.json)
- [kiki_machine_local_topology_damping_saved_audit_v1_review_20261115.json](_sealed/kiki_machine_local_topology_damping_saved_audit_v1_review_20261115.json)
- [kiki_machine_local_topology_damping_test_result_review_20261115.json](_sealed/kiki_machine_local_topology_damping_test_result_review_20261115.json)
- [kiki_machine_local_topology_damping_test_v1_review_20261115.json](_sealed/kiki_machine_local_topology_damping_test_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_damping_v1_zh.md](_sealed/kiki_machine_local_topology_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_damping_20261115_zh.md](_sealed/kiki_machine_local_topology_damping_20261115_zh.md)

</details>
