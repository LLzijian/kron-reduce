# 单参数因果阻尼拟合结果

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_delayed_damping_v1.json)

## 执行协议

<a id="doc-bcab1b0f4e0c"></a>

<a id="doc-bcab1b0f4e0c-以因果历史估计为约束的单参数阻尼拟合"></a>

### 以因果历史估计为约束的单参数阻尼拟合

复用延迟失步的固定 train 12、val 6、test 6 名单和轨迹目标第 20 步方程。原保存预测作为对照。参数替换试验在同一真值起报隐态下确认了阻尼影响，现从原方程仅开放一个正阻尼尺度，固定刚度、端口幅值和时间常数。数值初值及尺度由原方程和 train 预处理给定。

在初始对数阻尼尺度左右各取 log 2 的范围，按对数尺度等距评估 9 点，包含原始尺度。每个候选从新估计历史隐态，使用原因果 100 ms 输入、零初始猜测、五次 Gauss–Newton、半径 1、零先验权重。预测在 GPU 上以 5 ms 运算，沿用事件等权 1/3 秒损失。失稳样本的损失计算到真值首次五点确认。拟合只见完整 train 的可见轨迹。

按 train 损失选取最佳网格点，并在其相邻区间以确定性有界搜索细化，至多新增 24 次评估，对数尺度容差为 1e-5。最佳网格点在边界时停在该点。数值失败记入试算记录，缺少有效细化时保留已测网格点并记录原因。整个搜索只修改共享阻尼尺度。

保存冻结权重后完整评价 train、val、test。候选依然只用因果隐态估计，沿用 GPU 2.5/1.25 ms 双步长、8 秒轨迹和稳定性规则。train 和 val 各自要求完整数值覆盖、匹配失步全部恢复、8 秒全体正确且无误报和未决。两划分失稳余量 RMSE 各降低至少 5%，整体及稳定组的 3/8 秒轨迹 RMSE 至多为对照的 105%。test 只报告。部署决定为 `KEEP_BASELINE`。

一次执行、一项标量拟合。预算为三次 NPZ 解码、33 次完整 train 目标评估、581 次传播、1,127,040 次 Torch RHS、73,200 次切线 RHS，QR、SVD、三角求解和对应乘法各 2,460 次。无梯度和 Adam 更新。包含候选最终评分的逐例数值隔离预算。真实数据读取为零。[真值参数替换的依据](../synthetic_delayed_origin/README.md#doc-f20b87ad9d23)

## 报告与结果

<a id="doc-f783ac8a331d"></a>

<a id="doc-f783ac8a331d-单参数因果阻尼拟合结果"></a>

### 单参数因果阻尼拟合结果

完整合成 train 的 19 次目标评估将共享阻尼尺度从 1.323438 拟合到 0.997339。train 轨迹损失从 0.0014102 降至 5.68399e-06。刚度、端口幅值和时间常数保持固定。合成门通过。

train、val、test 的固定名单分别为 12、6、6。每次目标评估从 100 ms 历史重新估计隐态，只使用完整 train 的可见轨迹计算损失。冻结阻尼后，完整三个划分均按 2.5 和 1.25 ms 双步长预测到 8 秒。

| 模型 | 划分 | 匹配失步 | 3 秒池化 NRMSE | 8 秒池化 NRMSE | TP/TN/FP/FN | 未决／未覆盖／数值失败 | 完整准确率 | 可判定准确率 |
|---|---|---:|---:|---:|---|---|---:|---:|
| 原因果模型 | train | 0/6 | 0.506048 | 1.0687 | 4/6/0/0 | 2/0/0 | `null` | 100.00% |
| 原因果模型 | val | 0/3 | 0.832798 | 1.85045 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |
| 原因果模型 | test | 0/3 | 0.944521 | 1.48821 | 2/3/0/0 | 1/0/0 | `null` | 100.00% |
| 拟合阻尼模型 | train | 6/6 | 0.140245 | 0.272669 | 6/6/0/0 | 0/0/0 | 100.00% | 100.00% |
| 拟合阻尼模型 | val | 3/3 | 0.123744 | 0.214814 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |
| 拟合阻尼模型 | test | 2/3 | 0.0669729 | 0.135242 | 3/3/0/0 | 0/0/0 | 100.00% | 100.00% |

候选三个划分的 8 秒稳定性覆盖率与失稳召回率均为 100%，漏判率和稳定误报率均为 0%。合成 test 的 3 个失稳全部在 8 秒内检出，其中 2 个在真值匹配时段内确认。train 6/6 和 val 3/3 个匹配失步均恢复。val 的匹配失稳余量 RMSE 从 0.0311416 降至 0.00273198。

本轮用原因果隐态估计通过了合成门。真实 TASK3 将用原 T0M1 第 12 步检查点和原半径，在完整 train 上重新拟合阻尼。真实 val 和 test 仍按 107 和 76 例完整报告。部署决定为 `KEEP_BASELINE`。

运行 `machine_local_topology_delayed_damping_v1-delayed_damping-20261115T083744695248Z-02325f0b-ec203ad9`，用时 17.8 秒。
[执行协议](#doc-bcab1b0f4e0c) · [证据回执](_sealed/machine_local_topology_delayed_damping_20261115_evidence.json) · [结果审核](_sealed/machine_local_topology_delayed_damping_result_review_20261115.json)

## 回执与附件

- [machine_local_topology_delayed_damping_20261115_evidence.json](_sealed/machine_local_topology_delayed_damping_20261115_evidence.json)
- [machine_local_topology_delayed_damping_result_review_20261115.json](_sealed/machine_local_topology_delayed_damping_result_review_20261115.json)
- [machine_local_topology_delayed_damping_review_20261115.json](_sealed/machine_local_topology_delayed_damping_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_delayed_damping_v1_zh.md](_sealed/machine_local_topology_delayed_damping_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/machine_local_topology_delayed_damping_v1_zh.md](_sealed/original/machine_local_topology_delayed_damping_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_delayed_damping_20261115_zh.md](_sealed/machine_local_topology_delayed_damping_20261115_zh.md)

</details>
