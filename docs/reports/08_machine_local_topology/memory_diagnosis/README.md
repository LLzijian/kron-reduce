# 合成记忆核替换诊断

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_memory_diagnosis_v1.json)

## 执行协议

<a id="doc-13395d75b72b"></a>

<a id="doc-13395d75b72b-逐机记忆核的系数替换诊断"></a>

### 逐机记忆核的系数替换诊断

继承上一轮三类合成系统各 16 条 train 事件。独立 NumPy 真值生成器只积分这 48 条事件，使用原种子、名单和双步长。绑定原轮已选模型、冻结清单、train 评分记录及原运行身份。每类解码一次完整冻结模型张量，保持谐波与调制耗散的实际选中幅值。

每类运行四组。第一组保持冻结系数。第二组用真值替换全部局部端口幅值和时间常数。第三组用真值替换全部边刚度和九个原基底的阻尼系数。第四组同时替换两块。每组独立从相同的 11 帧历史重估三次 GN，并用实测可见状态重置后自由预测。

GPU float64 用 2.5 和 1.25 ms 两档积分到 3 秒，沿用 CUDA graph 和确定性 cuBLAS 配置。报告 0.5、1、3 秒合并归一化 RMSE、历史拟合误差、预测起点隐态误差与储能、记忆核 32 时延相对误差。用完整名单配对比较两块参数的作用。归一化最大步长差超过 \(10^{-5}\) 则标为无效。当一批求解发生数值异常时允许对整批逐例隔离一次。系数替换是机制对照，记忆核由端口幅值和时间常数决定，真实参数替换的作用通过轨迹与隐态变化判读。

预算为一次不可变运行、三次冻结模型解码、233280 次 NumPy RHS。24 批预测含最大 384 次逐例隔离的上限为 2040 次传播、3329280 次 Torch RHS、293760 次切线 RHS，QR、SVD、投影乘法和三角求解各 2304 次。拟合、参数更新、真实数据读取为零。诊断输出用于选择合成修正；真实 TASK3 由后续合成恢复门决定。

## 报告与结果

<a id="doc-d7906695b3c4"></a>

<a id="doc-d7906695b3c4-合成记忆核替换诊断"></a>

### 合成记忆核替换诊断

运行：`machine_local_memory_diagnosis_v1-diagnosis-20261114T123705106338Z-bd856994-79d63e8e`。仅使用合成 train 人群。


<a id="doc-d7906695b3c4-轨迹误差"></a>

#### 轨迹误差

以下为每个 16 例人群的 pooled NRMSE；四种方案均来自冻结系数替换。

| 人群 | 系数方案 | 0.5 s | 1 s | 3 s | 记忆核相对误差 | 隐态起点 RMSE 均值 | 隐态能量均值 | 重放一致性 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| null | selected | 9.00993e-05 | 0.000160144 | 0.000574362 | 0.0711831 | 0.011108 | 0.132318 | 通过，最大 ΔNMSE=0 |
| null | visible_truth | 0.000493994 | 0.000996421 | 0.00272865 | 0.0711831 | 0.00932512 | 0.124675 | — |
| null | memory_truth | 0.00051169 | 0.000994568 | 0.00268906 | 0 | 0.00953696 | 0.136086 | — |
| null | both_truth | 2.52146e-11 | 5.14791e-11 | 1.25313e-10 | 0 | 1.39377e-09 | 0.127126 | — |
| harmonic | selected | 0.000235031 | 0.000353931 | 0.0013766 | 0.120096 | 0.0230337 | 0.147612 | 通过，最大 ΔNMSE=0 |
| harmonic | visible_truth | 0.00108278 | 0.00237237 | 0.00688455 | 0.120096 | 0.0314127 | 0.127503 | — |
| harmonic | memory_truth | 0.00101582 | 0.00195031 | 0.00557787 | 0 | 0.0221999 | 0.155624 | — |
| harmonic | both_truth | 0.000712967 | 0.00168724 | 0.00490744 | 0 | 0.0186995 | 0.129807 | — |
| damping | selected | 0.00326459 | 0.0055333 | 0.0180144 | 0.552287 | 0.25115 | 0.667038 | 通过，最大 ΔNMSE=0 |
| damping | visible_truth | 0.00965536 | 0.0151432 | 0.0339139 | 0.552287 | 0.369962 | 2.1196 | — |
| damping | memory_truth | 0.00727097 | 0.0140412 | 0.0357093 | 0 | 0.156961 | 0.427421 | — |
| damping | both_truth | 0.0133712 | 0.0249837 | 0.0593256 | 0 | 0.231727 | 1.45265 | — |

隐态起点 RMSE 与能量列为各人群、各方案 16 个 train 样本的算术平均。系数方案名称对应评估代码中的冻结参数替换。


<a id="doc-d7906695b3c4-结果解读"></a>

#### 结果解读

三个合成人群中，单独把可见参数或记忆参数替换为真值后，0.5 秒、1 秒和 3 秒合并归一化 RMSE 均高于所选模型，说明原来拟合的两块系数在相互补偿。无新增项组同时换回两类系数后，3 秒误差降至 \(1.25\times10^{-10}\)。谐波组和调制耗散组还保留了原拟合的新增项幅值，因此双替换后仍有轨迹误差。校准将用完整 train 的可见动量积分联合估计两块系数和已选函数幅值。[独立结果审查](_sealed/machine_local_memory_diagnosis_result_review_20261114.json)


<a id="doc-d7906695b3c4-运行证据"></a>

#### 运行证据

实际操作计数如下：

| 操作 | 计数 |
|---|---:|
| checkpoint_decodes | 3 |
| matrix_products | 1152 |
| numpy_rhs | 233280 |
| qr | 1152 |
| rollouts | 120 |
| svd | 1152 |
| tangent_rhs | 17280 |
| torch_rhs | 195840 |
| triangular_solves | 1152 |

科学结果：`scientific_pass=True`。

## 回执与附件

- [machine_local_memory_diagnosis_20261114_evidence.json](_sealed/machine_local_memory_diagnosis_20261114_evidence.json)
- [machine_local_memory_diagnosis_result_review_20261114.json](_sealed/machine_local_memory_diagnosis_result_review_20261114.json)
- [machine_local_memory_diagnosis_review_20261114.json](_sealed/machine_local_memory_diagnosis_review_20261114.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_memory_diagnosis_v1_zh.md](_sealed/machine_local_memory_diagnosis_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_memory_diagnosis_20261114_zh.md](_sealed/machine_local_memory_diagnosis_20261114_zh.md)

</details>
