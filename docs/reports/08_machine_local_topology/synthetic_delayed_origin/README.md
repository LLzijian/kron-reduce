# 延迟失步的起报隐态与动力学参数对照

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_topology_delayed_origin_v1.json)

## 执行协议

<a id="doc-26c72e7bc4f6"></a>

<a id="doc-26c72e7bc4f6-延迟失步的起报隐态与动力学参数替换对照"></a>

### 延迟失步的起报隐态与动力学参数替换对照

复用延迟失步试验的固定真值、共同可见起报状态、轨迹目标第 20 步方程和保存的因果预测。名单为 train 12、val 6、test 6，保留各自全部稳定与失稳事件。三份 NPZ 各解码一次，绑定已验收的归档摘要、真值数组、检查点和原始预测哈希。共用原 train 尺度、真实匹配时段和 8 秒评分规则。

| 条件 | 共享系数 | 起报隐态 |
|---|---|---|
| 保存对照 | 第 20 步系数 | 已保存的因果估计 |
| 只替换起报隐态 | 第 20 步系数 | 同一事件的真值 |
| 刚度对照 | 第 20 步系数中只换真值刚度 | 同一事件的真值 |
| 阻尼对照 | 第 20 步系数中只换真值阻尼 | 同一事件的真值 |
| 记忆对照 | 第 20 步系数中只换真值端口幅值及时间常数 | 同一事件的真值 |
| 真值方程与历史估计 | 全部真值系数 | 用相同历史重新估计 |
| 全真值对照 | 全部真值系数 | 同一事件的真值 |

真值起报隐态取独立细步长轨迹的第 11 个点。五个直接预测条件的两档步长都使用同一份真值起报隐态。先核对保存对照的可见起报原点和历史末端隐态确实相符。真值方程重新估计隐态时，沿用五次 Gauss–Newton、半径 1 和先验权重 0。零拟合、零原始数据读取、零新真值生成。以 2.5 ms 和 1.25 ms GPU 双步长评分完整三个划分。直接预测条件的历史估计指标不适用。

以 train 和 val 解释结果。全真值对照须在两档上通过数值资格、复现全部匹配失步窗口、全体稳定性判断正确，且每例 8 秒归一化 RMSE 不超过 (10^{-5})。真值方程与其全真值对照的差反映正确方程下的初始化误差。第 20 步系数下替换隐态的结果衡量给定方程下的条件性效果。各单项参数替换与同一真值隐态下的第 20 步系数相比，衡量固定起报状态后的系数影响。

逐事件统计新增触发及丢失的窗口。train、val 都新增触发且没有丢失，在全覆盖条件下记录该干预有恢复证据。若还同时恢复两划分全部匹配窗口、完整稳定性判断正确、失稳余量 RMSE 至多为对应对照的 95%，且整体及真实稳定组 3/8 秒池化归一化 RMSE 至多为对照的 105%，则记录该条件下完整恢复。test 用于报告。交叉干预提供诊断证据，部署决定为 `KEEP_BASELINE`。

一次执行。一次因果预测组和五次直接预测组，每组每档最多一个完整批次，失败后至多逐例隔离一次。预算为三次 NPZ 解码，至多 600 次积分传播、5,832,000 次 Torch RHS、60,000 次切线 RHS，QR、SVD、三角求解和对应乘法各 480 次。闭环预测与历史初始化都计入额度。[前一轮延迟对照](../synthetic_delayed/README.md#doc-81598fbdd211)

## 报告与结果

<a id="doc-f20b87ad9d23"></a>

<a id="doc-f20b87ad9d23-延迟失步的起报隐态与动力学参数对照"></a>

### 延迟失步的起报隐态与动力学参数对照

固定同一真值起报隐态后，只替换阻尼恢复了合成 train 6/6、val 3/3 的匹配失步窗口。val 的 8 秒池化归一化 RMSE 从 1.902 降至 0.216，val 6/6 正确。刚度和记忆参数的单项替换均未恢复窗口。

七个条件覆盖固定合成 train 12、val 6、test 6，零拟合。五个直接预测条件共用独立细步长真值的第 11 点隐态。数值评分使用 2.5 和 1.25 ms。test 仅用于报告。

| 条件 | train 匹配失步 | val 匹配失步 | val 判对／总数 | val 8 秒 NRMSE | test 匹配失步 | test 判对／总数 | test 8 秒 NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 保存的因果对照 | 0/6 | 0/3 | 5/6 | 1.85045 | 0/3 | 5/6 | 1.48821 |
| 只替换起报隐态 | 0/6 | 0/3 | 5/6 | 1.90245 | 0/3 | 5/6 | 1.51083 |
| 只替换刚度 | 0/6 | 0/3 | 5/6 | 1.8001 | 0/3 | 5/6 | 1.45451 |
| 只替换阻尼 | 6/6 | 3/3 | 6/6 | 0.215562 | 3/3 | 6/6 | 0.137104 |
| 只替换记忆系数 | 0/6 | 0/3 | 5/6 | 2.24766 | 0/3 | 5/6 | 1.60812 |
| 真值系数与因果隐态 | 6/6 | 3/3 | 6/6 | 4.49888e-11 | 3/3 | 6/6 | 5.17203e-11 |
| 全真值对照 | 6/6 | 3/3 | 6/6 | 4.38047e-12 | 3/3 | 6/6 | 3.9993e-12 |

保存对照及只替换隐态、刚度或记忆系数的条件，在 val 和 test 都各有 1 例失稳未决，稳定误报为零。它们的可判定准确率为 100%，完整准确率为 `null`。其余三个条件在 val 和 test 全体可判且判对。各划分详细计数、轨迹和匹配余量见证据回执。

全真值方程用相同因果初始化便恢复全部失步窗口。学习方程仅换真值隐态仍未恢复匹配窗口，说明起报隐态饱和与漏判的因果贡献需要在给定动力学下解释。阻尼替换的作用已在固定真值隐态下分离。随后仅用可见 train 轨迹拟合阻尼尺度，并在因果初始化下检验。

运行 `machine_local_topology_delayed_origin_v1-delayed_origin-20261115T082734914398Z-30265e36-1b1b3627`，耗时 25.9 秒。
[执行协议](#doc-26c72e7bc4f6) · [证据回执](_sealed/machine_local_topology_delayed_origin_20261115_evidence.json) · [结果审核](_sealed/machine_local_topology_delayed_origin_result_review_20261115.json)

## 回执与附件

- [machine_local_topology_delayed_origin_20261115_evidence.json](_sealed/machine_local_topology_delayed_origin_20261115_evidence.json)
- [machine_local_topology_delayed_origin_result_review_20261115.json](_sealed/machine_local_topology_delayed_origin_result_review_20261115.json)
- [machine_local_topology_delayed_origin_review_20261115.json](_sealed/machine_local_topology_delayed_origin_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_delayed_origin_v1_zh.md](_sealed/machine_local_topology_delayed_origin_v1_zh.md)
- [docs/reports/05_research_governance/_sealed/docs/protocols/machine_local_topology_delayed_origin_v1_zh.md](_sealed/original/machine_local_topology_delayed_origin_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_topology_delayed_origin_20261115_zh.md](_sealed/machine_local_topology_delayed_origin_20261115_zh.md)

</details>
