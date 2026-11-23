# kiki / TASK3 train-val-test 对照规范

[返回主题页](../README.md)

## 执行协议

<a id="doc-6c6486ac9f76"></a>

<a id="doc-6c6486ac9f76-kiki--task3-train-val-test-对照规范"></a>

### kiki / TASK3 train-val-test 对照规范

生效日期：2026-11-12。后续 KIKI 对比实验使用 TASK3 的固定名单：`train=346` 拟合，`val=107` 供预先声明的模型或阈值选择，`test=76` 作冻结后的最终评价。所有名单按来源限定 ID 冻结在[机器可读引用](../../../../configs/proposals/kiki_task3_split_reference_v1.json)。新实验不在 train 内再次拆分。历史协议、模型与结果保持原来的样本范围。


<a id="doc-6c6486ac9f76-数据人群"></a>

#### 数据人群

`train` 与 `val` 取自 `kiki/private_data/ieee39_postfault` 和 `kiki/private_data/ieee39_expansion` 的原始 `splits.json`；其中源字段 `validation` 在本文统一称为 `val`。按 TASK3 的 100 ms 长度条件保留合格事件。`test` 使用 TASK3 Campaign 2 最终冻结名单。源文件路径、完整 SHA-256、来源限定 ID 与长度资格名单见[机器可读引用](../../../../configs/proposals/kiki_task3_split_reference_v1.json)。

| 角色 | 当前冻结名单数 | 来源 | 用途 |
| --- | ---: | ---: | --- |
| train | 346 | 原始 source train 中合格的事件 | 拟合一个模型；归一化尺度只由 train 估计 |
| val | 107 | 原始 source val 中合格的事件 | 按预先声明的规则选择模型与阈值 |
| test | 76 | TASK3 Campaign 2 最终冻结名单 | 模型、预处理、阈值、代码和名单冻结后评价 |

原始 KIKI 官方拆分分配数为 train 369、val 117、test 105；索引按 100 ms 条件筛得 346、107、99。这些数字用于描述来源，不改变当前主测试名单。当前 `test=76` 来自 TASK3 Campaign 2 的独立冻结名单：86 个分配事件中 76 个满足同一历史长度条件，包含 expansion 43 个和额外 holdout 33 个。不能将这 76 个说成官方 105 个事件筛选后的结果。名单成员和 SHA-256 见机器可读引用。

旧 200 ms 冻结迁移名单 75 个事件是另一份历史比较名单，不与当前 `test=76` 合并。正式运行检查实际观测时间和数组完整性并报告名单覆盖。不能依据预测误差或稳定性结果删样本，也不把数值失败记作零误差；保留分配数、可用数、模型拒绝和数值失败各自的计数。完整来源差异见[元数据核对](#doc-0e9b59c24a5a)。


<a id="doc-6c6486ac9f76-与-task3-一致的输入和报告"></a>

#### 与 TASK3 一致的输入和报告

- 采用 `measured_history_100ms_zoh_v1`：十台发电机、0–100 ms 的 11 个真实观测点，以第 11 点为预测起点；模型使用 q/v 历史，功率为 `Pm[10]−Pm_equilibrium` 并向未来零阶保持。采用同一惯性中心坐标与单位转换。
- 不读取预测起点之后的功率、Pe、电压、故障身份、标签或未来汇总作为特征。每机隐藏状态的初始估计必须遵循相同因果输入预算。
- 对照表按 TASK3 定义报告 1/2/5/8 s 的事件平均、pooled、CVaR10、最差误差及共同频率 RMSE，保留真实可用区间。delta/omega 的 TASK3 固定比较尺度分别为 `0.47675314588683515`、`1.0033667249602924`；它们仅定义比较指标，不替代本模型只由 train 估计的训练尺度。
- 失步规则采用角范围超过 π、最大相对速度超过 0.02 pu、连续五点；末 1 秒角范围小于 π 且速度小于 0.005 pu 才记为窗口内安定。模型未决与覆盖数同时报告。若模型只预测相对运动，共同频率必须标明其输出约定，不能把没有预测的通道写成已学会。
- 三秒训练损失或项目内部诊断指标可另外保留，不混称为 TASK3 的比较指标。引用 TASK3 已发表数值时，必须匹配样本 ID、输入历史、预测时域、尺度、失败处理与汇总方式。


<a id="doc-6c6486ac9f76-新轮次的轨迹稳定性报告"></a>

##### 新轮次的轨迹稳定性报告

真实 TASK3 数据保留冻结的官方分类标签和计数，不按轨迹规则重标真值。预测标签按上文冻结的 TASK3 失步阈值、连续点数和观察窗口从预测轨迹派生，并报告预测标签与官方标签的一致性。合成已知真值数据则分别从真值轨迹和预测轨迹按同一冻结规则派生标签。阈值、连续点数和观察窗口在运行前固定，不依据 `val` 或 `test` 结果重新选择。失步条件未触发且末 1 秒安定条件也未满足时记为未决，不能直接记为安定。

失稳为正类。对 `train`、`val`、`test` 分别报告稳定性准确率、TP/TN/FP/FN、失稳召回率 `TP/(TP+FN)`、失稳漏判率 `FN/(TP+FN)` 和稳定误报率 `FP/(TN+FP)`。混淆矩阵指标按真值和预测均可判定的样本计算。只要存在未决、未覆盖或数值失败样本，总体准确率记为 `null`，并同时报告可判定数、可判定样本准确率与覆盖率。任何指标的分母为零时记为 `null`，不写成 100%。同时报告名单分配数、未覆盖数和数值失败数；未决、未覆盖或数值失败样本不计作安定。`test` 结果仅用于冻结报告，不用于阈值选择、调参或模型修改。该要求适用于新轮次，不改变历史协议、样本范围或既有结果。


<a id="doc-6c6486ac9f76-执行与冻结"></a>

#### 执行与冻结

用户已授权沿该标准自主推进。每个新实验登记具体拟合、val 选择和 test 评价预算。val 的选择规则先行冻结；test 在模型参数、预处理、阈值、源码和测试名单全部冻结后评价，test 结果不返回选择流程。分阶段授权分别记录拟合与 test 读取次数、冻结清单和 SHA-256，不复用已消费的测试授权。

迭代训练在拟合前登记最大更新数、完整 val 检查频率、停止规则和检查点选择规则。保留 train 损失与 val 曲线，汇报实际更新数、选中步数及停止原因。配对比较采用预先声明的共同更新节奏和停止规则。

每个新获授权的训练轮次都要形成完整的 train/val/test 评价记录。模型与评价合同冻结后，
第一次评估即覆盖完整冻结名单：`train=346`、`val=107`、`test=76`，不得先用两例或其他子集试跑，
除非用户明确要求子集试跑。无论 RMSE 是否达到晋升门，都要为三个名单生成完整轨迹和稳定性结果；
是否晋升与是否完成评估是两项独立判断。test 结果只用于该轮报告，不用于调参或修改模型。
不叠加额外噪声的快速初评也须覆盖三个完整名单，之后完成冻结评价合同要求的全部计算。
缺失、非有限或其他无效结果标为明确失败并报告覆盖情况，
不以零或臆造数值代替。

新真实实验登记 `data_policy=kiki_official_train_test_v1`、`training_split=train`、`val_use=official_val_for_preregistered_selection`、`test_population=task3_campaign2_final_test`、`internal_resplit=false`，并绑定 `configs/proposals/kiki_task3_split_reference_v1.json` 的 SHA-256。运行器在读取数据前核对来源限定名单、样本数、阶段批准与预算。历史实验的注册内容与样本范围保持不变。

TASK3 `test=76` 已有历史暴露。新结果用于同一固定名单上的比较评价，不作为未见数据的独立确认。用户已授权按该名单开展后续对比；需要独立确认时，另行登记独立样本。

真实实验的前置条件仍是对应算法修正通过新冻结合成验证；本规范的生效本身不触发真实数据读取。

## 历史记录

<a id="doc-0e9b59c24a5a"></a>

<a id="doc-0e9b59c24a5a-kiki-与-task3-数据划分核对"></a>

### KIKI 与 TASK3 数据划分核对

日期：2026-11-12。范围：只读数据拆分清单、索引元数据、源码、配置和已有报告；没有读取 HDF5、轨迹数组、NPZ 或缓存载荷，也没有执行训练、预测或评分。用户已授权未来按冻结名单执行测试；本轮只核对元数据，没有执行测试。

机器可读名单、绝对来源路径和完整 SHA-256 见 [kiki_task3_split_reference_v1.json](../../../../configs/proposals/kiki_task3_split_reference_v1.json)。


<a id="doc-0e9b59c24a5a-划分结论"></a>

#### 划分结论

用户要求今后的 KIKI 实验按 `train` 拟合、`test` 评价。TASK3 文档里的正式流程比二分更细：官方 `train` 拟合、官方 `val`（TASK3 源字段 `validation`，原文称 `DEV`）做有限选择；模型、阈值、代码和样本名单冻结后，再做一次 test。若本项目采用严格的 `train`/`test` 两分法，就不使用 `val` 做模型选择，所有选择须在接触 test 前固定。

官方划分从两份 KIKI `splits.json` 读取，名单原样保留，不重排分组或抽取内部子集。按 TASK3 100 ms 合格条件 `trajectory_end_s > 0.1 + 1e-9`，从拆分清单及 `dataset_index.csv` 元数据统计得到。下文统一称模型选择角色为 `val`：

| 官方角色 | 分配数 | 100 ms 可用数 | 100 ms 可用名单指纹¹ | TASK3 中的用途 |
|---|---:|---:|---|---|
| `train` | 369 | 346 | `eada35b23066e8abbcb36a62def9873610ecd6666b8f0f48fd8446fda16a85e6` | 拟合；所有特征尺度只用此角色估计 |
| `val`（源字段 `validation`） | 117 | 107 | `ef81998a0485bc71be5a50b9c20a12f0d2334a1b0c8f1757ebafa3d7671d95b4` | TASK3 原文称为 `DEV`，用于有限检查点/阈值选择；不是最终确认 |
| `test` | 105 | 99 | `475b68c18256ef89eb900e70248a9c9f545240556f705e7050b509aa02ac2b7f` | KIKI 官方测试角色；当前登记将 KIKI test 标记为已暴露，不能称为新鲜独立确认 |

¹ 指纹为按源顺序拼接两个来源、每个 ID 加来源目录名前缀、以换行连接后计算的 SHA-256；完整名单保留在源 `splits.json`。资格数只依据索引中的 `trajectory_end_s` 元数据计算，没有检查轨迹内容。

| 官方数据源 | `train` | `val`（源字段 `validation`） | `test` | `splits.json` SHA-256 | `dataset_index.csv` SHA-256 |
|---|---:|---:|---:|---|---|
| `E:/03-nn_Transfer_Func/kiki/private_data/ieee39_postfault` | 227（217 可用） | 74（68 可用） | 60（56 可用） | `02990d0ddb22e4153824a28f1e99fc0d22a0afb70a4d197e6a294e6da253dba8` | `3cb759749328986790fc6933c34aba416be10ef807135b4654f1ae970d84de9b` |
| `E:/03-nn_Transfer_Func/kiki/private_data/ieee39_expansion` | 142（129 可用） | 43（39 可用） | 45（43 可用） | `f1d12c0f60462c85991a41464f56fe247977cefaf5e458a22a4ecc0d7e61c10b` | `6e4c0f3b5329f2472201869aa1034d2e2097726c400064a4cdf1bcc54ac3d783` |

逐来源的合格名单指纹（ID 按原 `splits.json` 次序，以换行连接）：

| 数据源 | 角色 | 合格名单 SHA-256 |
|---|---|---|
| `ieee39_postfault` | `train` | `7bb5593acca1f9172037c2a2b9cd74f31b199b442023fba11b0e0c16c74a25e2` |
| `ieee39_postfault` | `val` | `b74da2d52128a07edb39edf56ecdd399515335f2d463e8a38fe96633e14564ec` |
| `ieee39_postfault` | `test` | `89b6d70d30f646640f2a5a812c2ce0dc8916b397106c7fda0cb3590d21bef9c4` |
| `ieee39_expansion` | `train` | `eaace9984a1a43633540eea80d3e554aa325b49661dca98b7e7ce023b2e11cec` |
| `ieee39_expansion` | `val` | `734940b7aea6a39c061ee5987106d237b44193f2d011318767f370b1be94bfc3` |
| `ieee39_expansion` | `test` | `ec3fe337d401f0bde49b95c9b08f5c2df6f372a48c8dc3fd09fc70f9d8bb9553` |


<a id="doc-0e9b59c24a5a-task3-的最终-test-与已消费复评名单"></a>

#### TASK3 的最终 test 与已消费复评名单

TASK3 Campaign 2 最终 test 是一份单独登记的 expansion 加 holdout supplement 名单，不是上表 KIKI 官方 `test` 105 例。它分配 86 例，10 例不满足 100 ms 数据长度条件，76 例进入最终评价：43 个 `expansion_*` 和 33 个 `holdout_*`。精确 eligible 名单在 `E:/03-nn_Transfer_Func/task3_baselines/runs/campaign2_final/holdout_access.json`，文件 SHA-256 为 `28f2eec7ffb713cc14408c214190c8779d225ca73b7d6f7cdae5b233af0618d3`，按排序 ID 换行计算的名单 SHA-256 为 `2d62deb1c5b97f896d5b30e5994c119a9b587eeea4f9d8cf702a751e149dfa34`。名单范围为：

- `expansion_0192–0205`、`0207–0217`、`0219`、`0221`、`0224–0239`；
- `holdout_0000–0005`、`0007`、`0009`、`0011`、`0013`、`0015–0019`、`0021`、`0023`、`0025`、`0027–0041`。

这 76 个 `test` 事件已于 2026-09-27 经用户授权读取并存入 TASK3 durable cache。TASK3 的 test 曝光说明位于 `E:/03-nn_Transfer_Func/task3_baselines/paper/campaign2_final/test_exposure_notice.md`，指出该名单不能再作为新鲜确认集，且不产生新的训练授权。对应 freeze manifest SHA-256 为 `180fa58db64cb78848ba09d3423430d8ffb4f2ac7ada1f47e009a574a11a09fa`。

kron-reduce 另有已消费的一次 200 ms 冻结模型复评：其 `test-coverage.json` 记录 75 例，为 43 个 expansion ID 加 32 个 holdout ID，文件 SHA-256 为 `565bc140696cba87b19d5a5400252f22822fae6834b5385c9cecc3536132de75`，名单指纹为 `fb36146f5e7bd61145a6c5d46ffbdd5bd023c6a9dc8abf614815af7ade265d14`。它与 TASK3 76 例最终名单、官方 KIKI `test` 105 例是三份不同名单；不得合并或互称。该复评的运行 ID 是 `kiki_task3_frozen_200ms_20261109-20261109T024831303717Z-aa86374c`，相关一次性授权已在 [research policy](../../../../configs/research_policy.json) 中记为 `consumed`。

官方 KIKI registry 的 `test_status` 是 `prior_exposure_detected_not_fresh`，`prospective_test_allowed` 为 `false`。因此，未来按官方 `train`/`test` 报告结果时必须披露已有暴露；若要声称独立确认，需要新注册的独立 KIKI 测试来源。TASK3 76 例仍可作为单独标注的历史参照，不能替代官方 test 或冒充未见数据。


<a id="doc-0e9b59c24a5a-task3-输入时钟与评价口径"></a>

#### TASK3 输入时钟与评价口径

TASK3 100 ms 输入是记录起点 `t=0` 至 `t=0.10 s` 的 11 点、十机 `delta/omega` 历史；预测起点是第 11 点，机械功率输入为 `Pm[10] - Pm_equilibrium`，从起点起按零阶保持。角度和速度投影到惯性中心坐标，速度由绝对 pu 转为同步偏差 `rad/s`。训练/预测不能使用起点之后的 `Pm`、`Pe`、电压、故障信息或未来标签。TASK3 文档、`configs/dataset.json` 和 `src/task3/protocol.py` 对应此合同。

TASK3 200 ms 冻结迁移合同是另一口径：21 点历史，功率取 `Pm[20] - Pm_equilibrium`，预测从 0.20 s 起算。该合同保存在 kron-reduce 的 `task3_200ms_origin_power_frozen_transfer_v1`，只说明已消费历史评估；不能代替新的 100 ms train/test 实验合同。

TASK3 100 ms 评分源码固定使用 `delta`/`omega` 尺度 `0.47675314588683515`/`1.0033667249602924`，报告 1/2/5/8 s 事件平均、pooled、CVaR10、最差值和共同频率 RMSE；另报稳定完整轨迹与失稳事件真实可用前缀。未来失步规则是角范围大于 `pi` 且最大相对速度大于 `0.02 pu` 连续 5 点；末 1 秒角范围小于 `pi`、速度小于 `0.005 pu` 才记为 `window_settled`。数值未决保留在覆盖与分类结果中。定义见 TASK3 `src/task3/protocol.py` 的 `metric_row`、`aggregate`、`classify`，阈值与完整流程见 `TASK3.md`。


<a id="doc-0e9b59c24a5a-本仓库现有治理与兼容边界"></a>

#### 本仓库现有治理与兼容边界

此次用户指令授权更新将来的默认划分约定，覆盖根目录 [AGENTS.md](../../../../AGENTS.md) 中“在 source `train` 内建立 `train/val`”的 prospective 规则。历史 run、已消费 authorization、协议、receipt 以及带源码哈希的冻结 experiment/workflow 配置继续保持原样；新口径应使用新版本的数据政策、输入/实验合同、workflow 和证据字段。

当前主要实现边界：

- [project_harness/policy.py](../../../../project_harness/policy.py) 先拒绝标记 `reads_test_performance` 的普通新 run，并要求十机、非空间聚类及已登记输入合同。
- [kiki_history.py](../../../../src/kron_reduce/data/kiki_history.py) 的 `allocate_roles` 会把原 source `train` 按 fault-line 再分为内部 `train/val`；`KikiHistorySamples.load` 仅允许两个角色，并在访问审计里固定 `test_ids_read=[]`。这不是官方 `train/test` reader。
- 使用此 reader 的历史 workflow 包括 `kiki_history_learning`、`kiki_joint_refinement`、`kiki_memory_order`、`kiki_paired_training`、`kiki_short_history`、`kiki_tilted_training`、`kiki_train_mechanism`、`kiki_train_refinement`。这些运行已有冻结记录，不应通过修改旧调用点把它们改写成新实验。
- 一次性 TASK3 reader `kiki_task3_frozen_eval.py`、其 approval、缓存名单及 receipt 属于已消费的回顾性特例，不是通用 test 开关。

后续需同步的 prospective 接入面是根目录 AGENTS、`configs/research_policy.json`、`project_harness/policy.py` 和 `tests/test_research_policy.py`；新增官方拆分只读 reader/白名单审计、`tests/test_kiki_history.py`、`tests/test_kiki_short_history_reader.py` 及相关 workflow tests；新的 experiment/workflow/population contracts、数据访问账本、审批/冻结名单和报告说明。建议把 `val` 与最终 `test` 分开登记：用户要求只用 `train`/`test` 时，`val` 不参与任何模型或阈值选择；`test` 在名单、预处理、模型和评分全部冻结后访问，运行结果明确标为已暴露的回顾性证据。


<a id="doc-0e9b59c24a5a-来源版本与-sha-256"></a>

#### 来源版本与 SHA-256

| 来源 | SHA-256 |
|---|---|
| TASK3 `TASK3.md` | `2f72bac535f232d22f16c945f67334019d57d0b8f3b1981c64e0d391e9c45f8d` |
| TASK3 `configs/dataset.json` | `70f47d533e07aef66328ec24fae265ee834480daa917a6669dbdb7666f809013` |
| TASK3 `src/task3/protocol.py` | `c09d9d8a142377b8d7ff414fa477b41d5b28b1e29cc633b8a9378b9d94cc9d59` |
| TASK3 `src/task3/data/kiki_dataset.py` | `da7dbe8d950a1ec3b8440c77acbc0433b2ce9e971aa950e92a05b27a0807d89c` |
| TASK3 `docs/campaign2_state.md` | `367bded3e98a1547b0a85587ce8784b46948dcb294a49d1225b98d887ec7a3bd` |
| TASK3 `runs/campaign2_final/holdout_access.json` | `28f2eec7ffb713cc14408c214190c8779d225ca73b7d6f7cdae5b233af0618d3` |
| TASK3 `runs/campaign2_final/freeze_manifest.json` | `180fa58db64cb78848ba09d3423430d8ffb4f2ac7ada1f47e009a574a11a09fa` |
| TASK3 test exposure notice | `0f6cedcd8412a84d6fa80048378124ce7cd1993dbdbfcdb0e3137a1db8a2c4d7` |
| kron-reduce `data/kiki_ieee39_v1/manifest.json` | `688d564f381f5ba472633cdcc3258f84baf7a3861616c5230432dba9c5e9ecb3` |
| kron-reduce `configs/research_policy.json` | `09b8f43458d68601067898fa5205c53586d22957717e98f0141960e43bfb56ab` |
| kron-reduce `configs/input_contracts.json` | `6c31fb878d3bed1d43bf70acf84995b5ec5d6d1f2ae64a1809febe0c00b3f83c` |
| kron-reduce `src/kron_reduce/data/kiki_history.py` | `c9bed1a04f6c924976a3799cfb9cc5e558954c61c8d97bd0e76eb726e2ab715a` |
| kron-reduce `project_harness/policy.py` | `ff1e45110707e6241a5d280e1812d299cb58e2a7d58257e426072f401dfcb150` |

数据源 split 与 index 文件的 SHA-256 见上方划分表。

<details>
<summary>原件与版本</summary>

- [docs/protocols/kiki_official_train_test_v1_zh.md](_sealed/kiki_official_train_test_v1_zh.md)
- [docs/research/kiki_task3_split_inventory_20261112_zh.md](_sealed/kiki_task3_split_inventory_20261112_zh.md)

</details>
