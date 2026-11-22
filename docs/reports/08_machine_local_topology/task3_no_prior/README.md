# TASK3 零先验径向初始化

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/07_restoring_delayed_history.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/kiki_machine_local_topology_no_prior_test_v1.json)

## 执行协议

<a id="doc-2610789b0409"></a>

<a id="doc-2610789b0409-task3-固定径向历史求解的零先验对照"></a>

### TASK3 固定径向历史求解的零先验对照

合成配对证据支持在原径向历史求解中去掉二次先验。冻结原始 T0M1 第 12 步检查点和原 train 归一化尺度，历史起点半径维持 6.61055385191009。仅将二次先验权重从 $10^{-6}$ 改为零。使用相同的五次 Gauss–Newton 径向裁剪、100 ms 十一点历史、可见末点重置、预测起点功率零阶保持。无系数拟合。


<a id="doc-2610789b0409-全量评分"></a>

#### 全量评分

同一候选完整评价 TASK3 train 346、val 107、test 76。三份原 T0M1 评分由原冻结回执给出。预测仍用 CUDA float64、2.5 ms 与 1.25 ms 两档步长、批大小 32、8 s 完整滚动。采用 TASK3 正式评价和冻结真值。保存逐事件稳定性、两步长资格、轨迹误差、历史隐态储能和未决覆盖。完整准确率、可判定准确率、召回、漏判和误报按原口径报告。

train 的 35 个真实失稳事件额外核查最长 3 s 的匹配实测时段。各例从 11 帧历史与真实可用长度构造五点见证窗口，至少一帧位于预测起点之后。要求两个步长各复现超过原 6/35 例，TP 至少 24、FP 至多 2、未决至多 8，未覆盖和数值失败均为零。val 要求 107/107 判定、至少正确 104、零误报、TASK3 六项分数至多 0.5，稳定 95 例的 8 s 事件平均归一化 RMSE 至多为原 T0M1 的 1.05 倍。两个门独立报告，即使未通过也完成全部三个划分。

test 目标为 76/76 可判定且无数值失败、至少正确 69 例。test 只报告。独立复算三份保存预测后核对目标。部署判定为 KEEP_BASELINE。


<a id="doc-2610789b0409-单次预算"></a>

#### 单次预算

一次运行，零拟合。缓存打开上限 80 次，检查点解码 2 次。传播 7658 次，Torch 右端 22580160 次，切线右端 1312800 次，QR、SVD、三角求解和矩阵乘各 10580 次。包含每个批次发生数值异常时逐事件隔离一次的上界。train、val 与 test 使用分离的缓存访问日志。

## 报告与结果

<a id="doc-effc1602972f"></a>

<a id="doc-effc1602972f-task3-零先验径向初始化"></a>

### TASK3 零先验径向初始化

完整 TASK3 的判定结果仍受失稳漏判和未决限制。零先验候选将 test 的 8 s 事件平均归一化 RMSE 从 0.190670 降到 0.190526，67 个真实稳定事件均判对。9 个真实失稳事件中，5 个判为稳定，4 个未决。完整 test 准确率为 `null`。

本轮只把原 T0M1 径向历史求解的二次先验从 $10^{-6}$ 设为零。原第 12 步权重、半径 6.61055385191009、100 ms 历史、归一化尺度和闭环动力学固定。零拟合，CUDA float64 双步长，对同一冻结候选运行完整 train 346、val 107、test 76。[协议](#doc-2610789b0409)


<a id="doc-effc1602972f-稳定性"></a>

#### 稳定性

| 方法 | 划分 | 判定覆盖 | TP/TN/FP/FN | 未决／未覆盖／数值失败 | 完整准确率 | 可判定准确率 | 可判定召回／漏判／误报 |
|---|---|---:|---|---|---:|---:|---|
| 原 T0M1 | train | 338/346 | 24/308/2/4 | 8/0/0 | `null` | 98.22% | 85.71%／14.29%／0.65% |
| 零先验 | train | 338/346 | 24/308/2/4 | 8/0/0 | `null` | 98.22% | 85.71%／14.29%／0.65% |
| 原 T0M1 | val | 106/107 | 8/95/0/3 | 1/0/0 | `null` | 97.17% | 72.73%／27.27%／0.00% |
| 零先验 | val | 106/107 | 8/95/0/3 | 1/0/0 | `null` | 97.17% | 72.73%／27.27%／0.00% |
| 原 T0M1 | test | 71/76 | 0/67/0/4 | 5/0/0 | `null` | 94.37% | 0.00%／100.00%／0.00% |
| 零先验 | test | 72/76 | 0/67/0/5 | 4/0/0 | `null` | 93.06% | 0.00%／100.00%／0.00% |

train 的真值匹配时段内，零先验组的两个步长均复现 5/35 个失步窗口，原 T0M1 为 6/35。train 的匹配窗口门和 val 的完整判定门均未通过。两个门的结果独立于 test 报告。


<a id="doc-effc1602972f-轨迹"></a>

#### 轨迹

表中为 TASK3 事件平均归一化 RMSE，括号中为合格评分事件数。完整 8 s 可评分事件均为真实稳定样本。

| 方法 | 划分 | 1 s | 2 s | 5 s | 8 s | 六项分数 |
|---|---|---:|---:|---:|---:|---:|
| 原 T0M1 | train | 0.302250（315） | 0.292563（311） | 0.267455（311） | 0.259891（311） | 0.467080 |
| 零先验 | train | 0.301228（315） | 0.291301（311） | 0.266718（311） | 0.259431（311） | 0.466984 |
| 原 T0M1 | val | 0.295697（96） | 0.259058（95） | 0.223642（95） | 0.218248（95） | 0.438282 |
| 零先验 | val | 0.294905（96） | 0.258174（95） | 0.223141（95） | 0.217943（95） | 0.439115 |
| 原 T0M1 | test | 0.304603（67） | 0.254362（67） | 0.213347（67） | 0.190670（67） | 0.424230 |
| 零先验 | test | 0.304014（67） | 0.253759（67） | 0.213123（67） | 0.190526（67） | 0.425514 |

实际运行 kiki_machine_local_topology_no_prior_test_v1-test-20261115T100255429278Z-e5125686-7f041476。保存预测独立复算覆盖全部 529 例，逐例与汇总评分一致。部署决定为 KEEP_BASELINE。[运行回执](_sealed/kiki_machine_local_topology_no_prior_20261115_evidence.json) · [保存预测复算](_sealed/kiki_machine_local_topology_no_prior_saved_audit_20261115_evidence.json)

## 回执与附件

- [kiki_machine_local_topology_no_prior_20261115_evidence.json](_sealed/kiki_machine_local_topology_no_prior_20261115_evidence.json)
- [kiki_machine_local_topology_no_prior_saved_audit_20261115_evidence.json](_sealed/kiki_machine_local_topology_no_prior_saved_audit_20261115_evidence.json)
- [kiki_machine_local_topology_no_prior_saved_audit_review_20261115.json](_sealed/kiki_machine_local_topology_no_prior_saved_audit_review_20261115.json)
- [kiki_machine_local_topology_no_prior_test_result_review_20261115.json](_sealed/kiki_machine_local_topology_no_prior_test_result_review_20261115.json)
- [kiki_machine_local_topology_no_prior_test_v1_review_20261115.json](_sealed/kiki_machine_local_topology_no_prior_test_v1_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_topology_no_prior_v1_zh.md](_sealed/kiki_machine_local_topology_no_prior_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_topology_no_prior_20261115_zh.md](_sealed/kiki_machine_local_topology_no_prior_20261115_zh.md)

</details>
