# TASK3 逐机记忆动量校准结果

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/03_kiki_baseline_memory.DESCRIPTION.md)

登记文件与配置

[拟合方案](../../../../configs/proposals/kiki_machine_local_memory_fit_v1.json) · [test 方案](../../../../configs/proposals/kiki_machine_local_memory_test_v1.json)

## 执行协议

<a id="doc-ab2ac0174d03"></a>

<a id="doc-ab2ac0174d03-task3-逐机记忆动量校准协议"></a>

### TASK3 逐机记忆动量校准协议

沿用冻结的 TASK3 train 346、val 107、test 76 名单，使用十机相对角度和转速、100 ms 历史、预测起点的机械功率零阶保持。复用 R20/R21 的 A06 锚点、train 尺度、历史重建、数值检查和 TASK3 评分。依据已通过的固定函数项合成校准门登记本轮真实比较。


<a id="doc-ab2ac0174d03-训练与选择"></a>

#### 训练与选择

四个候选从相同的 A06 可见参数、逐机端口起点和时间常数出发。A 是原始正弦恢复力和线性阻尼；其余三个分别增加二阶谐波、角度调制阻尼或两项。A 与 B 的无新增项候选使用同一个确定性拟合。新增幅值从零开始；四个支持各自校准，无需通过先拟合两项的模型生成初值。

用全部 train 的预测起点及后续可用 10 ms 状态建立动量积分约束。在每个事件的 50 ms 节点和真实最后一帧建方程。只有一个非零积分时刻的事件，其自由初始记忆力已经张成全部平衡力空间，共享参数的投影信息为零。记录这些事件的 ID，保留在 train、尺度与正式评分。每个候选最多 150 次不同的 train 目标计算及 100 次接受迭代。使用 QR 消除事件的初始记忆力，用有界最小二乘联合估计边刚度、可见阻尼、端口幅值和新增项幅值，外层优化十个时间常数。参数和方程跨事件共享；预测时按冻结模型从 100 ms 历史重建隐状态。

在初次求解、每 10 次接受迭代及最后一次接受迭代，用完整 val 计算沿用的事件等权 1 秒和 3 秒归一化平方误差。每个候选选完整有限 val 损失的最早最小点。若没有有效点，报告失败并保留用于评分的最后有限状态或初始锚点。B 在距最佳 val 损失不超过 max(1e-8, 最佳值的 1%) 的候选中优先选择非零新增项最少的方程，随后取较小损失，固定顺序打破平局。若所有候选均无有效 val 分数，B 采用 A 的方程并报告未能选择。B 选中无新增项时，两个结果对应同一方程、同一组系数。


<a id="doc-ab2ac0174d03-评价与预算"></a>

#### 评价与预算

关闭拟合和选择，冻结 A、B 的系数、函数项和尺度，在完整 train 和 val 上正式评分，再另行登记一次完整 test。所有名单均用 CUDA float64 从起点预测 8 秒，RK4 步长为 2.5/1.25 ms，每批 16，数值失败按既有规则逐例隔离一次。遵循 TASK3 的真实可用时长、固定尺度及五点稳定性规则，保留不合格和未决数。拟合失败或 RMSE 超门也完成三份名单的轨迹和稳定性报告。test 仅作为此前已公开的固定比较基准。部署决策为 KEEP_BASELINE。

算子上限以登记 JSON 为准。每次 fit 独立使用一个已批准的完整 train 读取阶段，静态与 train/val 归档按既有访问记录读取，四个候选共用内存中的资料。每种干扰量分解与求解最多 207600 次，共享线性求解与谱检查各最多 600 次。完整 val 监测最多 48 轮、每轮 7 批，使用 5 ms 步长。正式评分含双步长与逐例隔离预算。fit 只解码一次 A06；test 只解码两个冻结检查点，读取 76 个 test 事件各一次。

## 报告与结果

<a id="doc-703c5b791810"></a>

<a id="doc-703c5b791810-task3-逐机记忆动量校准结果"></a>

### TASK3 逐机记忆动量校准结果

三个固定函数项的[合成真值对照](../memory_calibration/README.md#doc-4d938b5af3d2)均通过恢复门槛。完整 TASK3 上，动量校准得到的预测远弱于 R20/R21，保留现有基线。A 和 B 的冻结参数及检查点哈希相同，两个完整评估的结果一致。


<a id="doc-703c5b791810-为什么合成通过task3-仍失败"></a>

#### 为什么合成通过，TASK3 仍失败

合成替换诊断保持函数项与 100 ms 初始化规则。分别单独换回真实记忆参数或真实刚度和阻尼时，三个 train 的 3 秒轨迹误差都升高，说明原拟合系数相互补偿。改用动量积分后，三个合成 val 的记忆核误差分别从 7.12%、12.01%、55.23% 降到 0.043%、0.205%、0.382%。

TASK3 的四个候选均在第 17 次接受迭代后遇到初始记忆力分解秩亏。每个候选的三次 val 检查中，只有第 0 次获得完整有效分数。该检查点已完成固定时间常数的线性系数求解，十个端口幅值全落到约 0.00001，45 条边刚度中 25 条、9 个阻尼系数中 6 个触及下限。A 的 train 动量均方误差由 6.987 降至最后一次有效求解的 0.307，但第 10、17 次接受迭代的 107 个 val 事件都缺少有效的监测分数。

冻结方程在 val 的历史拟合归一化 RMSE 中位数仍为 0.0191，预测起点隐态储能中位数约为 4.19×10¹¹。端口幅值很小时，大隐态可在短历史内补足部分可见受力，随后依其时间常数演化。自由初始记忆力与因果历史估计之间的失配是下一轮 train/val 可验证的假设。此次保存的秩亏错误属于训练时消去初始记忆力的矩阵。正式双步长评估的稳定性未决均来自轨迹在判定窗口内未收敛，数值失败为零。


<a id="doc-703c5b791810-与-r20r21-的同口径比较"></a>

#### 与 R20/R21 的同口径比较

| 模型 | split | 8 秒事件平均归一化 RMSE | 可评分样本 | 可判定稳定性准确率 |
|---|---|---:|---:|---:|
| R20 | val | 0.24373 | 95 | 106/106（100%） |
| R21 | val | 0.29089 | 95 | 103/103（100%） |
| 动量校准 A、B | val | 45.08277 | 95 | 12/85（14.12%） |
| R20 | test | 3.10252 | 67 | 58/73（79.45%） |
| R21 | test | 2.79558 | 67 | 55/72（76.39%） |
| 动量校准 A、B | test | 38.75363 | 67 | 9/57（15.79%） |

动量校准在 val 判出 12/12 个真实失稳事件，73/95 个真实稳定事件被预测为失步。test 分别为 9/9 和 48/67。完整名单准确率在两份名单上均为 null。另以真实可用时长评分的失稳样本，事件平均归一化 RMSE 为 train 1.6002（35/35）、val 1.1006（12/12）、test 0.9023（9/9）。[R20/R21 报告](../task3_r20_r21/README.md#doc-f2670f1dcdb3)

fit 运行：`kiki_machine_local_memory_fit_v1-fit-20261114T130732576418Z-d49040be-6b1c9970`。test 运行：`kiki_machine_local_memory_test_v1-test-20261114T131915206009Z-b80b7ea9-5b43bc56`。

固定名单为 train=346、val=107、test=76，输入采用 100 ms 历史和预测起点机械功率零阶保持。test 只用于冻结报告。


<a id="doc-703c5b791810-val-候选选择拟合阶段"></a>

#### val 候选选择（拟合阶段）

按预登记规则选择 无新增函数项，val 损失容差为 0.435668。

| 候选函数项 | train 目标调用 | 完成迭代 | 选中迭代 | 选中 val NMSE | 末次 val NMSE | val 失败数 | 收敛 | 停止原因 | 失败原因 | 零信息数 | 零信息 ID |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|
| neither | 34 | 17 | 0 | 43.5668 | null | 2 | False | numerical_failure | rank deficient nuisance momentum block | 3 | ieee39_expansion/expansion_0106, ieee39_postfault/pilot_0207, ieee39_postfault/pilot_0254 |
| harmonic | 33 | 17 | 0 | 43.5668 | null | 2 | False | numerical_failure | rank deficient nuisance momentum block | 3 | ieee39_expansion/expansion_0106, ieee39_postfault/pilot_0207, ieee39_postfault/pilot_0254 |
| damping | 34 | 17 | 0 | 43.5668 | null | 2 | False | numerical_failure | rank deficient nuisance momentum block | 3 | ieee39_expansion/expansion_0106, ieee39_postfault/pilot_0207, ieee39_postfault/pilot_0254 |
| both | 33 | 17 | 0 | 43.5668 | null | 2 | False | numerical_failure | rank deficient nuisance momentum block | 3 | ieee39_expansion/expansion_0106, ieee39_postfault/pilot_0207, ieee39_postfault/pilot_0254 |


<a id="doc-703c5b791810-轨迹误差"></a>

#### 轨迹误差

报告事件均值、合并评分点、误差最高 10% 事件均值与最差事件的归一化 RMSE。每项依次列出合格数、该时域可评样本数和完整名单数。预测使用 RTX 4060 Ti 的 CUDA float64，2.5 ms 主步长与 1.25 ms 精化步长。fit 与正式 train、val 评分共用 402.7 秒。


<a id="doc-703c5b791810-train"></a>

##### train

| 模型 | 时域 | 合格/可评/名单 | 事件均值 | pooled | CVaR10 | 最差 | 共同频率 RMSE (rad/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1 s | 315/315/346 | 1.10827 | 1.35973 | 2.88006 | 6.34192 | 1.29773 |
| A | 2 s | 311/311/346 | 4.10298 | 5.16478 | 10.7911 | 17.1923 | 0.987972 |
| A | 5 s | 311/311/346 | 23.1197 | 28.2304 | 56.4599 | 84.4589 | 0.780166 |
| A | 8 s | 311/311/346 | 54.5226 | 65.5096 | 128.504 | 195.019 | 0.624545 |
| B | 1 s | 315/315/346 | 1.10827 | 1.35973 | 2.88006 | 6.34192 | 1.29773 |
| B | 2 s | 311/311/346 | 4.10298 | 5.16478 | 10.7911 | 17.1923 | 0.987972 |
| B | 5 s | 311/311/346 | 23.1197 | 28.2304 | 56.4599 | 84.4589 | 0.780166 |
| B | 8 s | 311/311/346 | 54.5226 | 65.5096 | 128.504 | 195.019 | 0.624545 |

该名单中的失稳前缀子集

| 模型 | 子集数/合格数 | 事件均值 | pooled | CVaR10 | 最差 | 共同频率 RMSE (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| A | 35/35 | 1.6002 | 2.81708 | 4.06693 | 5.56187 | 3.2321 |
| B | 35/35 | 1.6002 | 2.81708 | 4.06693 | 5.56187 | 3.2321 |


<a id="doc-703c5b791810-val"></a>

##### val

| 模型 | 时域 | 合格/可评/名单 | 事件均值 | pooled | CVaR10 | 最差 | 共同频率 RMSE (rad/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1 s | 96/96/107 | 0.918985 | 1.04198 | 2.06245 | 2.70271 | 1.31949 |
| A | 2 s | 95/95/107 | 3.02153 | 3.93893 | 8.46748 | 10.675 | 0.993993 |
| A | 5 s | 95/95/107 | 18.1658 | 23.073 | 46.5853 | 55.0401 | 0.785138 |
| A | 8 s | 95/95/107 | 45.0828 | 55.5 | 108.738 | 125.743 | 0.627821 |
| B | 1 s | 96/96/107 | 0.918985 | 1.04198 | 2.06245 | 2.70271 | 1.31949 |
| B | 2 s | 95/95/107 | 3.02153 | 3.93893 | 8.46748 | 10.675 | 0.993993 |
| B | 5 s | 95/95/107 | 18.1658 | 23.073 | 46.5853 | 55.0401 | 0.785138 |
| B | 8 s | 95/95/107 | 45.0828 | 55.5 | 108.738 | 125.743 | 0.627821 |

该名单中的失稳前缀子集

| 模型 | 子集数/合格数 | 事件均值 | pooled | CVaR10 | 最差 | 共同频率 RMSE (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| A | 12/12 | 1.10062 | 1.78523 | 2.38014 | 2.39056 | 2.95128 |
| B | 12/12 | 1.10062 | 1.78523 | 2.38014 | 2.39056 | 2.95128 |


<a id="doc-703c5b791810-test"></a>

##### test

| 模型 | 时域 | 合格/可评/名单 | 事件均值 | pooled | CVaR10 | 最差 | 共同频率 RMSE (rad/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1 s | 67/67/76 | 1.62023 | 1.90437 | 3.2917 | 3.68258 | 0.833115 |
| A | 2 s | 67/67/76 | 4.18852 | 5.35228 | 11.8781 | 13.3004 | 0.650277 |
| A | 5 s | 67/67/76 | 17.7239 | 24.1619 | 58.6441 | 65.2544 | 0.558959 |
| A | 8 s | 67/67/76 | 38.7536 | 52.8334 | 127.695 | 141.736 | 0.449221 |
| B | 1 s | 67/67/76 | 1.62023 | 1.90437 | 3.2917 | 3.68258 | 0.833115 |
| B | 2 s | 67/67/76 | 4.18852 | 5.35228 | 11.8781 | 13.3004 | 0.650277 |
| B | 5 s | 67/67/76 | 17.7239 | 24.1619 | 58.6441 | 65.2544 | 0.558959 |
| B | 8 s | 67/67/76 | 38.7536 | 52.8334 | 127.695 | 141.736 | 0.449221 |

该名单中的失稳前缀子集

| 模型 | 子集数/合格数 | 事件均值 | pooled | CVaR10 | 最差 | 共同频率 RMSE (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| A | 9/9 | 0.902257 | 1.3887 | 1.9955 | 1.9955 | 1.79351 |
| B | 9/9 | 0.902257 | 1.3887 | 1.9955 | 1.9955 | 1.79351 |


<a id="doc-703c5b791810-轨迹派生稳定性"></a>

#### 轨迹派生稳定性

失稳为正类。混淆计数顺序为 TP/TN/FP/FN。确认检出率按全部真实失稳计，条件召回率按可判定的真实失稳计。若有未决、未覆盖或数值失败，完整名单准确率按协议保持 null。

| 模型 | split | 完整名单准确率 | 可判定准确率 |
|---|---|---:|---:|
| A、B | train | null | 36/285（12.63%） |
| A、B | val | null | 12/85（14.12%） |
| A、B | test | null | 9/57（15.79%） |

| 模型 | split | 名单 | 已判定 | 覆盖率 | 未决 | 未覆盖 | 数值失败 | 真失稳数 | 确认检出率 | 条件召回率 | 未确认失稳比例 | 可判定漏判率 | 可判定稳定误报率 | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | train | 346 | 285 | 0.8236994219653179 | 61 | 0 | 0 | 35 | 35/35 (1.0) | 35/35 (1.0) | 0.0 | 0.0 | 0.996 | 35/1/249/0 |
| B | train | 346 | 285 | 0.8236994219653179 | 61 | 0 | 0 | 35 | 35/35 (1.0) | 35/35 (1.0) | 0.0 | 0.0 | 0.996 | 35/1/249/0 |
| A | val | 107 | 85 | 0.794392523364486 | 22 | 0 | 0 | 12 | 12/12 (1.0) | 12/12 (1.0) | 0.0 | 0.0 | 1.0 | 12/0/73/0 |
| B | val | 107 | 85 | 0.794392523364486 | 22 | 0 | 0 | 12 | 12/12 (1.0) | 12/12 (1.0) | 0.0 | 0.0 | 1.0 | 12/0/73/0 |
| A | test | 76 | 57 | 0.75 | 19 | 0 | 0 | 9 | 9/9 (1.0) | 9/9 (1.0) | 0.0 | 0.0 | 1.0 | 9/0/48/0 |
| B | test | 76 | 57 | 0.75 | 19 | 0 | 0 | 9 | 9/9 (1.0) | 9/9 (1.0) | 0.0 | 0.0 | 1.0 | 9/0/48/0 |


<a id="doc-703c5b791810-ab-定义与运行证据"></a>

#### A/B 定义与运行证据

A 使用基础方程。B 的函数项由完整 val 选定。由于所选 support 也是 neither，A/B 使用同一模型，表中两者指标相同。fit 阶段完整 train/val 轨迹、test 阶段冻结 test 轨迹分别来自独立运行。test 冻结清单与 fit 最终 freeze.json 按原始字节核对。test=76 是既有固定比较名单，不是独立确认集。

| 阶段 | run ID | commit | manifest SHA-256 | operation counts |
|---|---|---|---|---|
| fit | `kiki_machine_local_memory_fit_v1-fit-20261114T130732576418Z-d49040be-6b1c9970` | `d49040be26f7431eac7175f7d4547f61377c359e` | `b12cdc6ff7d7de4deb0eeea25f691a9b67fa3d8c8dbfdba3b14d71f41eb1bf4e` | `{"checkpoint_loads": 1, "fits": 4, "matrix_products": 6720, "momentum_design_spectrum": 134, "momentum_linear_solves": 134, "momentum_nuisance_solves": 45962, "momentum_nuisance_svd": 45966, "momentum_objectives": 138, "momentum_qr": 45966, "qr": 7576, "rollouts": 776, "svd": 7576, "tangent_rhs": 94720, "torch_rhs": 2419200, "triangular_solves": 6720, "val_checks": 12}` |
| test | `kiki_machine_local_memory_test_v1-test-20261114T131915206009Z-b80b7ea9-5b43bc56` | `b80b7ea9911a2bd8a7c67683d1d2743447261405` | `79be9adcd10ddece61698d66b070368e52cdcbd8d8f1154d21a1b7d0c74a75ec` | `{"checkpoint_loads": 2, "matrix_products": 912, "qr": 912, "rollouts": 100, "svd": 912, "tangent_rhs": 14400, "torch_rhs": 403200, "triangular_solves": 912}` |

复核 `docs/reports/08_machine_local_topology/task3_baseline_memory/_sealed/kiki_machine_local_memory_fit_review_20261114.json`：APPROVE_EXACT_COMMON_ANCHOR_TASK3_MOMENTUM_FIT_AND_COMPLETE_TRAIN_VAL_EVALUATION。reviewer=independent_scientific_design_agent。approved=True。
复核 `docs/reports/08_machine_local_topology/task3_baseline_memory/_sealed/kiki_machine_local_memory_test_review_20261114.json`：未填写 decision 字段。reviewer=Astra independent scientific review。approved=True。


[独立结果复核](_sealed/kiki_machine_local_memory_result_review_20261114.json) · [证据收据](_sealed/kiki_machine_local_memory_20261114_evidence.json)

## 回执与附件

- [kiki_machine_local_memory_20261114_evidence.json](_sealed/kiki_machine_local_memory_20261114_evidence.json)
- [kiki_machine_local_memory_fit_review_20261114.json](_sealed/kiki_machine_local_memory_fit_review_20261114.json)
- [kiki_machine_local_memory_result_review_20261114.json](_sealed/kiki_machine_local_memory_result_review_20261114.json)
- [kiki_machine_local_memory_test_review_20261114.json](_sealed/kiki_machine_local_memory_test_review_20261114.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_machine_local_memory_v1_zh.md](_sealed/kiki_machine_local_memory_v1_zh.md)
- [docs/reports/08_machine_local_topology/kiki_machine_local_memory_20261114_zh.md](_sealed/kiki_machine_local_memory_20261114_zh.md)

</details>
