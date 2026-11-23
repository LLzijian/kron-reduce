# TASK3 功角阻尼完整评估

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md)

登记文件与配置

[拟合方案](../../../../configs/proposals/kiki_machine_local_topology_angle_damping_fit_v1.json) · [test 方案](../../../../configs/proposals/kiki_machine_local_topology_angle_damping_test_v1.json)

## 执行协议

<a id="doc-15a9d4b3a97f"></a>

<a id="doc-15a9d4b3a97f-task3-train-功角阻尼辨识"></a>

### TASK3 train 功角阻尼辨识

合成配对已通过后，固定原 T0M1 第 12 步模型，从完整 TASK3 train 346 的实测路径辨识阻尼。沿用原 100 ms 历史与起报功率零阶保持，实测段延至起报后最多 3 s。共享刚度、逐机记忆及原初态半径保持固定。四点 Gauss 力积分经过八点核查，允许的最大差为 \(10^{-10}\)。每例沿同一观测路径做一次 SVD 投影，消去自由记忆初态。

完整 train 的投影残差分别确定一个常值阻尼倍率和一个功角阻尼系数。常值倍率下限为 \(10^{-6}\)，功角系数范围为 \([0,2/3]\)。保存两个系数及全 train、稳定 311 例和失稳 35 例的动量残差平方和。三个组的功角阻尼残差均下降时，记为通过残差门。合格的唯一功角系数冻结用于另行登记的完整预测评估。实测未来用于共享系数辨识，真实预测仍只使用 100 ms 历史。

单次训练辨识包括 346 次 SVD、2 次标量拟合和至多 107260 个观测区间，每区间完成一次八点复核。载入冻结检查点两次，打开 static 与 development 缓存各一次。无 val 或 test 轨迹读取。

<a id="doc-72b7788e7ba1"></a>

<a id="doc-72b7788e7ba1-task3-功角阻尼的完整冻结评估"></a>

### TASK3 功角阻尼的完整冻结评估

将完整 train 已辨识的唯一功角阻尼系数应用于原 T0M1 第 12 步模型，历史初始化与未来使用同一个阻尼函数。沿用原静态系数、记忆半径、二次先验、10 ms 观测和起报功率零阶保持。评分尺度和真值标签使用冻结的 TASK3 口径。[辨识协议](#doc-15a9d4b3a97f)

冻结后用 CUDA float64 两档步长 2.5 ms 和 1.25 ms 完成 train 346、val 107、test 76。各划分完整汇报轨迹误差、稳定性混淆计数、准确率和覆盖率。任何开发门失败仍完成全部评分。对照为保存的原 T0M1。同一冻结规则适用于三个划分，test 只报告，保存预测独立复算。

沿用固定阻尼对照的 train 与 val 门槛。train 的匹配失步窗口在两档步长须分别超过原 6/35，TP 至少 24、FP 至多 2、未决至多 8，无未覆盖和数值失败。val 须 107/107 可判定、至少 104 例正确、零稳定误报、六项分数至多 0.50、稳定 95 例的 8 s 归一化 RMSE 至多为原对照的 1.05 倍。test 目标为 76 例全判定、至少 69 例正确且零数值失败。部署保持 `KEEP_BASELINE`。

单次冻结比较，零再拟合。缓存打开上限 80 次，检查点载入 2 次。使用上一轮相同的双步长预测和异常隔离算子预算。

## 报告与结果

<a id="doc-648afba6d2b5"></a>

<a id="doc-648afba6d2b5-task3-功角阻尼完整评估"></a>

### TASK3 功角阻尼完整评估

功角阻尼在完整 val 上判对 106/107，稳定误报为零，轨迹误差通过预登记门槛。完整 test 判对 69 例，3 例未决，完整准确率记为 `null`。冻结候选完成 train 346、val 107、test 76 的双步长评分。部署决定为 `KEEP_BASELINE`。


<a id="doc-648afba6d2b5-冻结设置与证据"></a>

#### 冻结设置与证据

原 T0M1 第 12 步检查点加入功角阻尼系数 \(\rho=2/3\)。保留记忆半径 6.61055385、二次先验 \(10^{-6}\)、100 ms 的 11 帧历史及零阶保持起报功率。预测使用 CUDA float64 的 2.5 ms 和 1.25 ms 步长。三个划分使用冻结的 TASK3 名单和标签，test 为已曝光的固定比较基准。

运行 `kiki_machine_local_topology_angle_damping_test_v1-test-20261115T113623593775Z-9f7cc490-7e35d1af`。独立审核复算 529 条保存预测，评分零差异。[运行回执](_sealed/kiki_machine_local_topology_angle_damping_20261115_evidence.json) · [冻结协议](#doc-72b7788e7ba1) · [独立复算](_sealed/kiki_machine_local_topology_angle_damping_saved_audit_20261115_evidence.json)


<a id="doc-648afba6d2b5-train-路径拟合"></a>

#### train 路径拟合

用完整 train 346 的 98,341 个实测区间拟合。动量方程投影消去自由记忆初态，346 例的算子均满秩，四点和八点求积的最大差为 \(4.16\times10^{-17}\)。无约束功角系数为 1.16852，按预登记范围截为 \(2/3\)。同口径的常值阻尼倍率为 0.3436767。[拟合回执](_sealed/kiki_machine_local_topology_angle_damping_fit_20261115_evidence.json)

| train 分组 | 样本数 | 原始 T0M1 残差平方和 | 常值阻尼候选 | 功角阻尼候选 |
|---|---:|---:|---:|---:|
| 全部 train | 346 | 21752.10 | 14727.02 | 17041.59 |
| 稳定 | 311 | 12204.86 | 9898.33 | 11578.56 |
| 失稳 | 35 | 9547.23 | 4828.69 | 5463.02 |


<a id="doc-648afba6d2b5-轨迹归一化-rmse"></a>

#### 轨迹归一化 RMSE

下表为 TASK3 口径的事件平均归一化 RMSE。括号内是该时长实际参与统计的合格样本数；失稳样本按真实可用时段单独列出。

| 划分 | 模型 | 1 s | 2 s | 5 s | 8 s |
|---|---|---:|---:|---:|---:|
| train | 原 T0M1 | 0.30225（315） | 0.29256（311） | 0.26745（311） | 0.25989（311） |
| train | 功角阻尼 | 0.30920（315） | 0.31404（311） | 0.41739（311） | 0.56767（311） |
| val | 原 T0M1 | 0.29570（96） | 0.25906（95） | 0.22364（95） | 0.21825（95） |
| val | 功角阻尼 | 0.29944（96） | 0.26180（95） | 0.22566（95） | 0.21966（95） |
| test | 原 T0M1 | 0.30460（67） | 0.25436（67） | 0.21335（67） | 0.19067（67） |
| test | 功角阻尼 | 0.30068（67） | 0.25165（67） | 0.21189（67） | 0.18960（67） |

失稳样本实际可用时段的事件平均归一化 RMSE 为：train 35 例，原 T0M1 0.90277、功角阻尼 0.85042；val 12 例，0.85054、0.78008；test 9 例，1.02531、1.01569。


<a id="doc-648afba6d2b5-稳定性与覆盖"></a>

#### 稳定性与覆盖

TP/TN/FP/FN 按失稳为阳性统计。train、val、test 的真实失稳/稳定分母分别为 35/311、12/95、9/67。完整准确率在存在未决样本时记为 `null`；可判定准确率只在判定样本上计算。未覆盖和数值失败在所有划分均为 0。

| 划分与模型 | 已判定/名单 | 覆盖率 | TP/TN/FP/FN | 未决 | 完整准确率 | 可判定准确率 | 失稳召回率 | 漏判率 | 稳定误报率 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| train 原 T0M1 | 338/346 | 97.69% | 24/308/2/4 | 8 | `null` | 98.22% | 85.71% | 14.29% | 0.65% |
| train 功角阻尼 | 342/346 | 98.84% | 27/308/3/4 | 4 | `null` | 97.95% | 87.10% | 12.90% | 0.96% |
| val 原 T0M1 | 106/107 | 99.07% | 8/95/0/3 | 1 | `null` | 97.17% | 72.73% | 27.27% | 0.00% |
| val 功角阻尼 | 107/107 | 100.00% | 11/95/0/1 | 0 | 99.07% | 99.07% | 91.67% | 8.33% | 0.00% |
| test 原 T0M1 | 71/76 | 93.42% | 0/67/0/4 | 5 | `null` | 94.37% | 0.00% | 100.00% | 0.00% |
| test 功角阻尼 | 73/76 | 96.05% | 2/67/0/4 | 3 | `null` | 94.52% | 33.33% | 66.67% | 0.00% |

两档步长各复现 train 的 7/35 个实测失步窗口，原 T0M1 各复现 6/35。train 的 3 个稳定误报超过上限 2。这 3 例贡献了 train 稳定组 8 s 事件平均归一化 RMSE 增量的 99.55%，两种模型在这 3 例的初始记忆均触顶，起报记忆储能差均小于 0.016。val 的六项分数为 0.42819，稳定 95 例的 8 s 归一化 RMSE 为 0.21966，相对原 T0M1 的 0.21825 上升 0.65%，通过 5% 门槛。

test 的 3 个未决均为真实失稳病例，其余 6 个真实失稳病例有 2 个检出、4 个判为稳定。完整准确率目标仍待验证。

## 回执与附件

- [kiki_machine_local_topology_angle_damping_20261115_evidence.json](_sealed/kiki_machine_local_topology_angle_damping_20261115_evidence.json)
- [kiki_machine_local_topology_angle_damping_fit_20261115_evidence.json](_sealed/kiki_machine_local_topology_angle_damping_fit_20261115_evidence.json)
- [kiki_machine_local_topology_angle_damping_saved_audit_20261115_evidence.json](_sealed/kiki_machine_local_topology_angle_damping_saved_audit_20261115_evidence.json)
- [kiki_machine_local_topology_angle_damping_fit_result_review_20261115.json](_sealed/kiki_machine_local_topology_angle_damping_fit_result_review_20261115.json)
- [kiki_machine_local_topology_angle_damping_fit_v1_review_20261115.json](_sealed/kiki_machine_local_topology_angle_damping_fit_v1_review_20261115.json)
- [kiki_machine_local_topology_angle_damping_saved_audit_review_20261115.json](_sealed/kiki_machine_local_topology_angle_damping_saved_audit_review_20261115.json)
- [kiki_machine_local_topology_angle_damping_test_result_review_20261115.json](_sealed/kiki_machine_local_topology_angle_damping_test_result_review_20261115.json)
- [kiki_machine_local_topology_angle_damping_test_v1_review_20261115.json](_sealed/kiki_machine_local_topology_angle_damping_test_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_angle_damping_fit_v1_zh.md](_sealed/kiki_machine_local_topology_angle_damping_fit_v1_zh.md)
- [docs/protocols/kiki_machine_local_topology_angle_damping_v1_zh.md](_sealed/kiki_machine_local_topology_angle_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_angle_damping_20261115_zh.md](_sealed/kiki_machine_local_topology_angle_damping_20261115_zh.md)

</details>
