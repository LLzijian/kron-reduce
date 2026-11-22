# TASK3 逐机功角阻尼完整评估

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/05_damping_corrections.DESCRIPTION.md)

登记文件与配置

[拟合方案](../../../../configs/proposals/kiki_machine_local_topology_spatial_damping_fit_v1.json) · [test 方案](../../../../configs/proposals/kiki_machine_local_topology_spatial_damping_test_v1.json)

## 执行协议

<a id="doc-adf76c91166b"></a>

<a id="doc-adf76c91166b-task3-train-逐机阻尼辨识"></a>

### TASK3 train 逐机阻尼辨识

合成配对通过后，以原 T0M1 第 12 步方程和全局功角阻尼为参照，固定强度 \(\rho=2/3\) 与尺度 1 rad。用完整 train 346 的实测路径拟合共享系数 \(\lambda\in[0,1]\)。每例使用真实可用、至多起报后 3 s 的实测段，沿用起报功率零阶保持。刚度、记忆系数与历史初始化配置保持固定。[配对合成协议](../synthetic_spatial_damping/README.md#doc-05f14349c042)

每例的全局与逐机力积分共用一次动量 SVD 投影，四点 Gauss 积分由八点复核。完整名单均须满秩且有效，最大求积差至多 \(10^{-10}\)。保存原始与截断系数和边界导数，报告全部、稳定 311 例、失稳 35 例的全局、逐机和迹匹配残差。逐机残差在三个组均低于原全局模型时通过残差门。有效的正系数冻结后完成完整预测，零系数对应既有全局方程。

单次辨识，真实读取限定 train，打开 static 和 development 各一次。载入冻结检查点两次，构造阻尼平方根一次，执行 346 次 SVD 和 1 次标量拟合，每种求积至多 107260 个区间。无 val 或 test 轨迹读取。

<a id="doc-a1acccecfa26"></a>

<a id="doc-a1acccecfa26-task3-逐机阻尼的完整冻结预测"></a>

### TASK3 逐机阻尼的完整冻结预测

固定原 T0M1 检查点、100 ms 观测输入、记忆初始化半径和先验。沿用强度 \(\rho=2/3\) 和尺度 1 rad，只应用完整 train 冻结的逐机分配系数。历史与预测共用逐机阻尼方程。[辨识协议](#doc-adf76c91166b)

CUDA float64 的 2.5 ms 和 1.25 ms 双步长完整评估 train 346、val 107、test 76。两档都使用冻结起报功率零阶保持、标签及 TASK3 评分尺度。每个划分汇报轨迹误差、五点失步规则的混淆计数、准确率和覆盖率。各门独立评分，完整 test 只作报告。保存预测单独复算。

train 的匹配失步窗口须超过原 T0M1 的 6/35，TP 至少 24、FP 至多 2、未决至多 8、无未覆盖或数值失败。val 须完整可判定、至少 104 例正确、零稳定误报、六项分数至多 0.50、稳定 95 例的八秒均值至多为原 T0M1 的 1.05 倍。test 目标为 76 例全判定、至少 69 例正确、零数值失败。部署保持 `KEEP_BASELINE`。

冻结比较预算沿用功角阻尼的完整名单、双步长和异常隔离设置。最多打开缓存 80 次，载入检查点 2 次，闭环评分期间拟合次数为零。

## 报告与结果

<a id="doc-08e5190828ec"></a>

<a id="doc-08e5190828ec-task3-逐机功角阻尼完整评估"></a>

### TASK3 逐机功角阻尼完整评估

逐机阻尼在 val 判对 106/107，稳定误报为零，轨迹误差通过预登记门槛。test 在 76 例中判对 70 例，2 例未决，完整准确率为 `null`。冻结候选完成 train 346、val 107、test 76 的双步长评分。部署决定保持 `KEEP_BASELINE`。


<a id="doc-08e5190828ec-冻结设置与依据"></a>

#### 冻结设置与依据

候选在原 T0M1 第 12 步模型上使用逐机阻尼。固定 \(\rho=2/3\)、尺度 1 rad、记忆半径 6.61055385 和先验 \(10^{-6}\)。使用 11 帧 100 ms 历史、零阶保持起报功率和 CUDA float64 的 2.5 ms、1.25 ms 双步长。比较使用保存的原 T0M1 与全局功角模型。test 为已曝光的固定 TASK3 比较基准。

运行 `kiki_machine_local_topology_spatial_damping_test_v1-test-20261115T122859502183Z-62f8d5b9-c3e57543`。独立复核复算了 529 例保存预测，评分零差异。[执行回执](_sealed/kiki_machine_local_topology_spatial_damping_20261115_evidence.json) · [冻结协议](#doc-a1acccecfa26) · [独立复算](_sealed/kiki_machine_local_topology_spatial_damping_saved_audit_20261115_evidence.json)


<a id="doc-08e5190828ec-train-观察路径拟合"></a>

#### train 观察路径拟合

完整 train 346 的算子均满秩，保留 98,341 个实测区间，四点与八点求积最大差为 \(3.84\times10^{-14}\)。原始估计 1.13205194 经预登记区间截为 \(\lambda=1\)。下表为动量残差平方和。[拟合回执](_sealed/kiki_machine_local_topology_spatial_damping_fit_20261115_evidence.json)

| train 分组 | 事件数 | 全局功角候选 | 逐机功角候选 | 迹匹配控制 |
|---|---:|---:|---:|---:|
| 全部 train | 346 | 17041.59 | 16917.66 | 17259.89 |
| 稳定 | 311 | 11578.56 | 11625.04 | 11570.93 |
| 失稳 | 35 | 5463.02 | 5292.62 | 5688.95 |


<a id="doc-08e5190828ec-轨迹归一化-rmse"></a>

#### 轨迹归一化 RMSE

表中为 TASK3 口径的事件平均归一化 RMSE。括号内为对应时长参与评分的样本数。

| 划分 | 模型 | 1 s | 2 s | 5 s | 8 s |
|---|---|---:|---:|---:|---:|
| train | 原 T0M1 | 0.30225（315） | 0.29256（311） | 0.26745（311） | 0.25989（311） |
| train | 全局功角 | 0.30920（315） | 0.31404（311） | 0.41739（311） | 0.56767（311） |
| train | 逐机功角 | 0.30763（315） | 0.31716（311） | 0.45207（311） | 0.64004（311） |
| val | 原 T0M1 | 0.29570（96） | 0.25906（95） | 0.22364（95） | 0.21825（95） |
| val | 全局功角 | 0.29944（96） | 0.26180（95） | 0.22566（95） | 0.21966（95） |
| val | 逐机功角 | 0.29920（96） | 0.26251（95） | 0.22625（95） | 0.22020（95） |
| test | 原 T0M1 | 0.30460（67） | 0.25436（67） | 0.21335（67） | 0.19067（67） |
| test | 全局功角 | 0.30068（67） | 0.25165（67） | 0.21189（67） | 0.18960（67） |
| test | 逐机功角 | 0.29395（67） | 0.24672（67） | 0.20933（67） | 0.18779（67） |

失稳样本的统计使用各事件实际可用时段。事件平均归一化 RMSE 为：train 35 例，原 T0M1 0.90277、全局功角 0.85042、逐机功角 0.81614；val 12 例分别为 0.85054、0.78008、0.76979；test 9 例分别为 1.02531、1.01569、1.01852。


<a id="doc-08e5190828ec-稳定性与名单覆盖"></a>

#### 稳定性与名单覆盖

真实失稳/稳定分母分别为 train 35/311、val 12/95、test 9/67。TP/TN/FP/FN 以失稳为阳性；未决不视为稳定。完整准确率在存在未决事件时为 `null`，可判定准确率仅按已判定样本计算。各模型在三个划分均无未覆盖或数值失败。

| 划分与模型 | 已判定/名单 | TP/TN/FP/FN | 未决 | 覆盖率 | 完整准确率 | 可判定准确率 | 失稳召回率 | 漏判率 | 稳定误报率 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| train 原 T0M1 | 338/346 | 24/308/2/4 | 8 | 97.69% | `null` | 98.22% | 85.71% | 14.29% | 0.65% |
| train 全局功角 | 342/346 | 27/308/3/4 | 4 | 98.84% | `null` | 97.95% | 87.10% | 12.90% | 0.96% |
| train 逐机功角 | 342/346 | 27/308/3/4 | 4 | 98.84% | `null` | 97.95% | 87.10% | 12.90% | 0.96% |
| val 原 T0M1 | 106/107 | 8/95/0/3 | 1 | 99.07% | `null` | 97.17% | 72.73% | 27.27% | 0.00% |
| val 全局功角 | 107/107 | 11/95/0/1 | 0 | 100.00% | 99.07% | 99.07% | 91.67% | 8.33% | 0.00% |
| val 逐机功角 | 107/107 | 11/95/0/1 | 0 | 100.00% | 99.07% | 99.07% | 91.67% | 8.33% | 0.00% |
| test 原 T0M1 | 71/76 | 0/67/0/4 | 5 | 93.42% | `null` | 94.37% | 0.00% | 100.00% | 0.00% |
| test 全局功角 | 73/76 | 2/67/0/4 | 3 | 96.05% | `null` | 94.52% | 33.33% | 66.67% | 0.00% |
| test 逐机功角 | 74/76 | 3/67/0/4 | 2 | 97.37% | `null` | 94.59% | 42.86% | 57.14% | 0.00% |

val 的逐机候选六项分数为 0.42674，全局功角候选为 0.42819。逐机候选在稳定 95 例的 8 s 平均归一化 RMSE 为 0.22020，比原 T0M1 的 0.21825 高 0.90%，通过 5% 门槛。

两档步长分别复现 train 的 7/35 个实测失步窗口。train 门因 3 个稳定误报超过上限 2 未通过。test 的 9 个真实失稳样本中，3 个判为失稳、4 个判为稳定、2 个未决。

## 回执与附件

- [kiki_machine_local_topology_spatial_damping_20261115_evidence.json](_sealed/kiki_machine_local_topology_spatial_damping_20261115_evidence.json)
- [kiki_machine_local_topology_spatial_damping_fit_20261115_evidence.json](_sealed/kiki_machine_local_topology_spatial_damping_fit_20261115_evidence.json)
- [kiki_machine_local_topology_spatial_damping_saved_audit_20261115_evidence.json](_sealed/kiki_machine_local_topology_spatial_damping_saved_audit_20261115_evidence.json)
- [kiki_machine_local_topology_spatial_damping_fit_result_review_20261115.json](_sealed/kiki_machine_local_topology_spatial_damping_fit_result_review_20261115.json)
- [kiki_machine_local_topology_spatial_damping_fit_v1_review_20261115.json](_sealed/kiki_machine_local_topology_spatial_damping_fit_v1_review_20261115.json)
- [kiki_machine_local_topology_spatial_damping_saved_audit_review_20261115.json](_sealed/kiki_machine_local_topology_spatial_damping_saved_audit_review_20261115.json)
- [kiki_machine_local_topology_spatial_damping_test_result_review_20261115.json](_sealed/kiki_machine_local_topology_spatial_damping_test_result_review_20261115.json)
- [kiki_machine_local_topology_spatial_damping_test_v1_review_20261115.json](_sealed/kiki_machine_local_topology_spatial_damping_test_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_spatial_damping_fit_v1_zh.md](_sealed/kiki_machine_local_topology_spatial_damping_fit_v1_zh.md)
- [docs/protocols/kiki_machine_local_topology_spatial_damping_v1_zh.md](_sealed/kiki_machine_local_topology_spatial_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_spatial_damping_20261115_zh.md](_sealed/kiki_machine_local_topology_spatial_damping_20261115_zh.md)

</details>
