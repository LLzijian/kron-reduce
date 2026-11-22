# TASK3 动量辨识阻尼的完整评估

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/06_momentum_coupling.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/kiki_machine_local_topology_momentum_damping_test_v1.json)

## 执行协议

<a id="doc-a5a5ed6273a5"></a>

<a id="doc-a5a5ed6273a5-task3-动量辨识阻尼的冻结对照"></a>

### TASK3 动量辨识阻尼的冻结对照

完整 train 346 的动量诊断给出一个通过预登记残差门槛的方向。固定原 T0M1 第 12 步方程，将可见阻尼乘以 train 已确定的 0.3436767364716119。该倍率只来自实测路径上、投影掉逐例未知初始记忆后的有符号阻尼残差。合成已知真值验证了这个方向的符号和辨识公式。闭环实验检验修正能否改善预测。


<a id="doc-a5a5ed6273a5-冻结与评分"></a>

#### 冻结与评分

保留边刚度、端口幅值、时间常数、原历史尺度、原记忆半径 6.61055385191009 和二次先验权重 \(10^{-6}\)。同一正阻尼倍率应用于 100 ms 历史求解和未来传播。初始化保持五次 Gauss–Newton 径向裁剪和测量末点重置，功率按原 ZOH 合同。每次预测只使用 11 帧历史。动量辨识所用代数隐态不进入预测。

冻结单一候选后完整运行 TASK3 train 346、val 107、test 76。两档 GPU float64 积分步长 2.5 ms 与 1.25 ms，批大小 32，8 s 滚动。沿用 TASK3 标签、轨迹评分、稳定性规则和双步长资格。汇报各划分的轨迹误差、TP/TN/FP/FN、召回、漏判、误报、未决与覆盖率。独立复算三份保存预测。

train 匹配时段的失步窗口在两档步长上须各超过原 6/35，TP 至少 24、FP 至多 2、未决至多 8，无未覆盖和数值失败。val 须 107/107 判定、至少正确 104、无误报、六项分数至多 0.50、稳定 95 例的 8 s 事件平均归一化 RMSE 至多为原对照的 1.05 倍。各门限独立报告，三个划分全部完成。test 仅报告，目标为 76 例全部判定、至少正确 69 例且零数值失败。


<a id="doc-a5a5ed6273a5-单次预算"></a>

#### 单次预算

一次冻结比较，零额外参数拟合。静态与开发访问日志和 test 访问日志分离。缓存打开最多 80 次，模型检查点载入 2 次。双步长整批及逐例异常隔离预算为 7658 次传播、22580160 次 Torch 右端、1312800 次切线右端，QR、SVD、三角求解和矩阵乘各 10580 次。部署判定为 KEEP_BASELINE。

## 报告与结果

<a id="doc-97c1fbb8b406"></a>

<a id="doc-97c1fbb8b406-task3-动量辨识阻尼的完整评估"></a>

### TASK3 动量辨识阻尼的完整评估

本轮把 train 动量诊断确定的固定阻尼倍率 `0.3436767364716119` 应用于原 T0M1 第 12 步模型，刚度、端口、时间常数、历史尺度、半径和先验均保持冻结。候选和绑定的原始 T0M1 参照均完成完整 train=346、val=107、test=76 评分；倍率未在本轮重选。科学门未通过，保留 `KEEP_BASELINE`。


<a id="doc-97c1fbb8b406-轨迹误差"></a>

#### 轨迹误差

表中为 TASK3 事件平均归一化 RMSE。括号内为该时域的评分样本数。

| 划分 | 模型 | 1 s | 2 s | 5 s | 8 s |
|---|---|---:|---:|---:|---:|
| train | 原 T0M1 | 0.30225（315） | 0.29256（311） | 0.26745（311） | 0.25989（311） |
| train | 阻尼倍率候选 | 0.38772（315） | 0.48276（311） | 0.68324（311） | 0.91372（311） |
| val | 原 T0M1 | 0.29570（96） | 0.25906（95） | 0.22364（95） | 0.21825（95） |
| val | 阻尼倍率候选 | 0.35561（96） | 0.39637（95） | 0.42395（95） | 0.46365（95） |
| test | 原 T0M1 | 0.30460（67） | 0.25436（67） | 0.21335（67） | 0.19067（67） |
| test | 阻尼倍率候选 | 0.30890（67） | 0.29906（67） | 0.26848（67） | 0.23387（67） |

失稳事件实际可用时段的事件平均归一化 RMSE 为：train 35 例，原 T0M1 0.90277、候选 0.81482；val 12 例，0.85054、0.76381；test 9 例，1.02531、1.03413。


<a id="doc-97c1fbb8b406-稳定性和覆盖"></a>

#### 稳定性和覆盖

混淆计数顺序为 TP/TN/FP/FN；失稳召回、漏判率和稳定误报率均按已判定的对应类别计算。完整准确率在任何事件未决时记为 `null`，可判定准确率只用已判定事件。

| 划分与模型 | 已判定 / 名单 | TP/TN/FP/FN | 未决 | 数值失败 | 完整准确率 | 可判定准确率 | 覆盖率 | 失稳召回 | 漏判率 | 稳定误报率 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train 原 T0M1 | 338/346 | 24/308/2/4 | 8 | 0 | `null` | 98.22% | 97.69% | 85.71% | 14.29% | 0.65% |
| train 候选 | 341/346 | 28/305/5/3 | 5 | 0 | `null` | 97.65% | 98.55% | 90.32% | 9.68% | 1.61% |
| val 原 T0M1 | 106/107 | 8/95/0/3 | 1 | 0 | `null` | 97.17% | 99.07% | 72.73% | 27.27% | 0.00% |
| val 候选 | 107/107 | 11/94/1/1 | 0 | 0 | 98.13% | 98.13% | 100.00% | 91.67% | 8.33% | 1.05% |
| test 原 T0M1 | 71/76 | 0/67/0/4 | 5 | 0 | `null` | 94.37% | 93.42% | 0.00% | 100.00% | 0.00% |
| test 候选 | 74/76 | 4/67/0/3 | 2 | 0 | `null` | 95.95% | 97.37% | 57.14% | 42.86% | 0.00% |

候选 test 判对 71/74 个已判定事件，完整名单仍有 2 例未决，故完整准确率为 `null`。三个划分的未覆盖和数值失败均为 0。


<a id="doc-97c1fbb8b406-预登记门槛"></a>

#### 预登记门槛

train 的 35 个真实失稳时段均纳入匹配诊断。原 T0M1 在 2.5 ms 与 1.25 ms 两档均复现 6/35 个窗口；候选两档均复现 10/35。train 门要求 TP≥24、FP≤2、未决≤8、无未覆盖或数值失败，并要求两档窗口数均大于 6。候选 TP=28、未决=5，两档窗口数均为 10，均达到相应条件；FP=5 超过上限 2，因此 train 门未通过。

val 门要求 107/107 可判定、至少判对 104 例、零误报、六项分数≤0.50，且稳定样本 8 s 事件平均归一化 RMSE 不超过原参照的 1.05 倍。候选 107/107 可判定、判对 105 例；但 FP=1，六项分数为 0.53098，且稳定样本 8 s 误差为 0.46365，高于上限 `1.05×0.21825=0.22916`。这三项使 val 门未通过。

test 目标要求 76/76 可判定、至少判对 69 例且数值失败为零。候选判对数为 71，数值失败为零，但只有 74/76 可判定，因此 test 目标未达到。test 仅作报告用途。


<a id="doc-97c1fbb8b406-复核与范围"></a>

#### 复核与范围

候选使用 100 ms、11 帧因果历史初始化和原 ZOH 输入合同。阻尼倍率来自先前 train 实测路径动量诊断，本次未拟合或重选。test 结果未用于调整模型。三份保存预测已由独立复核逐例重算，529 例评分无差异。[保存结果复核](_sealed/kiki_machine_local_topology_momentum_damping_saved_audit_20261115_evidence.json)

运行 `kiki_machine_local_topology_momentum_damping_test_v1-test-20261115T104620540787Z-c909dba9-6d32e340`。完整评估在 RTX 4060 Ti 上以 CUDA float64 运行，两档积分步长为 2.5 ms 和 1.25 ms。[执行回执](_sealed/kiki_machine_local_topology_momentum_damping_20261115_evidence.json) · [冻结协议](#doc-a5a5ed6273a5)

## 回执与附件

- [kiki_machine_local_topology_momentum_damping_20261115_evidence.json](_sealed/kiki_machine_local_topology_momentum_damping_20261115_evidence.json)
- [kiki_machine_local_topology_momentum_damping_saved_audit_20261115_evidence.json](_sealed/kiki_machine_local_topology_momentum_damping_saved_audit_20261115_evidence.json)
- [kiki_machine_local_topology_momentum_damping_saved_audit_review_20261115.json](_sealed/kiki_machine_local_topology_momentum_damping_saved_audit_review_20261115.json)
- [kiki_machine_local_topology_momentum_damping_test_result_review_20261115.json](_sealed/kiki_machine_local_topology_momentum_damping_test_result_review_20261115.json)
- [kiki_machine_local_topology_momentum_damping_test_v1_review_20261115.json](_sealed/kiki_machine_local_topology_momentum_damping_test_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_momentum_damping_v1_zh.md](_sealed/kiki_machine_local_topology_momentum_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/_sealed/kiki_machine_local_topology_momentum_damping_20261115_zh.md](_sealed/original/kiki_machine_local_topology_momentum_damping_20261115_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_momentum_damping_20261115_zh.md](_sealed/kiki_machine_local_topology_momentum_damping_20261115_zh.md)

</details>
