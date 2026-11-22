# Joint observer 确认结果（2026-11-12）

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_joint_observer_v1.json)

## 研究方案

<a id="doc-d1b07a56bdcb"></a>

<a id="doc-d1b07a56bdcb-含噪历史的联合初态估计"></a>

### 含噪历史的联合初态估计

日期：2026-11-12。核心设计与科学审核：Astra Ultra。

[S2 结果](../observer/README.md#doc-73d37091a634)已经验证模型历史响应能准确恢复 clean 状态，但 1% 历史噪声下 local/nonlocal 的 val 3 s NRMSE 为 0.052754/0.031516，超过 0.02。下一项修正只解除“首帧可见观测等于精确初态”的约束，检验该处理对噪声误差的贡献。

原方法在十一帧同一个加权历史损失中固定首帧 q/v，仅求十个隐状态；首帧残差恒为零。新方法在质量 COI 子空间内联合求 18 个可见初态坐标与十个隐状态，完整使用十一帧。固定动力学、100 ms 窗、历史尺度、无正则最小二乘、预测起点可见重置和未来门槛全部保持不变。该修正不预设误差一定降低：增加未知量可能放大短窗弱方向的噪声，历史拟合残差的必然下降也不能作为预测成功证据。

先在 S2 已保存的 local/nonlocal 两组含噪事件上作配对比较，只使用 train/val 的主 NMSE 比值判断机制支持。再以 `2026111221/2026111222` 两个新种子生成 local/nonlocal，各执行 clean 和 1% 历史噪声病例。所有参数均为固定真值，按原 32/16/16 角色和原精度门独立确认；旧 test 不参与配对机制选择。等率负例要求联合观测矩阵秩 27 并拒绝状态输出。

两旧组在 train/val 的四项主 NMSE 比值全部不超过 0.95，才标记“配对证据支持该机制”。新样本精度门与该标记分别报告：只有技术门和四个新样本病例的冻结精度门决定本次方法确认是否通过。若确认通过而配对标记未通过，可以继续登记参数学习，但不声称已识别旧失败的主要原因。确认失败则继续处理噪声模型或历史短窗敏感性，不追加本次拟合或改变已用门槛。

具体公式、来源、病例、预算及证据合同见 [执行协议](#doc-dfa4811cb776)。本阶段零参数训练、零梯度调用、零真实数据读取。用户已授权按门禁持续修正；通过本次固定参数验证后，仍需登记完整因果梯度检查、未知动力学训练和学得模型独立确认，才能继续真实数据阶段。未来数据完全复用 TASK3 的 train 346 / val 107 / test 76 名单规则；本阶段保持基线且不直接开放 T0。

## 执行协议

<a id="doc-dfa4811cb776"></a>

<a id="doc-dfa4811cb776-含噪历史联合初态估计协议-v1"></a>

### 含噪历史联合初态估计协议 v1

日期：2026-11-12。设计与科学审核：Astra Ultra。阶段：`machine_local_joint_observer_v1/fixed`。


<a id="doc-dfa4811cb776-1-唯一方法变更"></a>

#### 1. 唯一方法变更

沿用 [S2 协议](../observer/README.md#doc-6f68c96c4128)的十节点线性被动模型、真值参数、COI 约束、100 ms 十一帧输入、已知常值功率、历史尺度、预测起点可见重置、RK4 与预测指标。模型仍只有十个隐状态。

旧方法最小化十一帧加权历史残差，同时约束历史起点 q/v 等于首帧观测。新方法只解除这 18 个独立可见 COI 坐标的约束，与十个隐状态联合求解。没有新增 ridge、先验、噪声白化、观测帧、函数模块或动力学训练。首帧现在参与残差而非作为精确初始条件。


<a id="doc-dfa4811cb776-2-精确求解与因果链"></a>

#### 2. 精确求解与因果链

令 `a=m/‖m‖₂`、`w=a+e₁`，以确定 Householder 矩阵 `R=I−2wwᵀ/(wᵀw)` 的后九列构造 V。固定正质量下 `VᵀV=I₉`、`mᵀV=0`；构造不调用 QR/SVD。定义

\[
B=\operatorname{diag}(V,V,I_{10})\in\mathbb R^{30\times28},\qquad
x_h=B\xi_h,\quad\xi_h=(a_q,a_v,z_h).
\]

28 是待估计的完整初态坐标数，不是隐状态阶数。零全状态、事件已知功率构成基准 `b_e,j`；B 的 28 个初态列、零输入构成响应 `D_j`。每档子步各批量传播一次历史，输出 `j=0,…,10`。用 clean train 历史 RMS 尺度 `S_h` 构造

\[
O=\operatorname{stack}_{j=0}^{10}(S_h^{-1}HD_j)\in\mathbb R^{220\times28},\quad
r_e=\operatorname{stack}_{j=0}^{10}[S_h^{-1}(y_{e,j}-Hb_{e,j})].
\]

H 选择前 20 个可见坐标。每个病例、每档子步只做一次薄 SVD `O=UΣV_oᵀ`，数值秩阈值为 `64 eps64 max(shape(O)) σmax`。仅当秩为 28、`σmax/σmin≤1e8` 且全部有限，输出 `ξ̂=V_oΣ⁻¹Uᵀr`；否则拒绝状态和未来。没有截断补零或额外最小二乘库调用。

以 `x̂_e,j=b_e,j+D_j ξ̂_e,h` 重构历史，保留其预测起点 z，仅把起点 q/v 重置为最后一帧观测，再预测 8 s。保存 28 维 `initial_coordinates`、十维 `hidden_start`、包含 V 的 B、设计矩阵、右端项、SVD 因子、完整历史和未来，避免混淆联合坐标与隐状态。

首帧可直接区分 18 个可见坐标，隐状态在首帧没有可见响应；因此联合观测秩等于 18 加上已知可见初态时的隐状态观测秩。正常病例要求 28；原 τ 全为 0.3 s 的等率负例要求 27 并拒绝输出。该变更不创造新的可观测隐状态方向。


<a id="doc-dfa4811cb776-3-六病例与来源"></a>

#### 3. 六病例与来源

| 病例 | 数据与模型 | 用途 |
|---|---|---|
| `old_local_noise01` | S2 已保存 local 含噪输入、真值 L10 | train/val 配对机制比较 |
| `old_nonlocal_noise01` | S2 已保存 nonlocal 含噪输入、真值 G10 | train/val 配对机制比较 |
| `local_clean` | 新 local clean、真值 L10 | 独立 clean 确认 |
| `local_noise01` | 同一新 local，1% 历史噪声、真值 L10 | 独立 noisy 确认 |
| `nonlocal_clean` | 新 nonlocal clean、真值 G10 | 独立 clean 确认 |
| `nonlocal_noise01` | 同一新 nonlocal，1% 历史噪声、真值 G10 | 独立 noisy 确认 |

新种子 local=`2026111221`、nonlocal=`2026111222`，噪声种子各加 100。沿用 S2 的解析参数、初态/功率抽样次序、40 维 expm 真值生成、两端点核对、COI 噪声投影、尺度与固定 64 事件角色：train 32、val 16、test 16。两个新种子独立于已报告结果，规则在调用前冻结。评分前保存生成输入、角色和哈希。

新来源范围仅解码 S2 的两个输入 NPZ，各一次；旧含噪病例 JSON 只提供已审计指标，不再解码旧预测数组或恢复模型。来源运行是 `machine_local_observer_v1-fixed-20261112T095639821073Z-9d055fc1-374b3b37`。

| 来源文件 | SHA256 |
|---|---|
| `local_inputs.npz` | `5ce15448f879173fd4948e4a2408df9f24db43dc3453760b6426c6eb6929e70d` |
| `nonlocal_inputs.npz` | `aea21145e06000faf0317d2fdc614b1fc2e5ac04e74839919db5249c4aabe136` |
| `local_noise01.json` | `406a6728a0b87719e388f00e69217855cc2f4df05b739f414238166e5ce3a178` |
| `nonlocal_noise01.json` | `87ce2bd085758f5f454d42e4c09b6b43c99e9c2ed2311813383425802e9bc9f0` |

旧 test 结果只作描述，不参与机制标记、选择或阈值调整。旧 GF/Gz/B 补查已完成，不在此阶段重复。


<a id="doc-dfa4811cb776-4-固定指标与分流"></a>

#### 4. 固定指标与分流

每个正例用 `0.0025/0.00125 s` 各完成历史与未来，输出间隔 0.01 s。主 NRMSE 仍为 `sqrt((NRMSE_1s²+NRMSE_3s²)/2)`；时间、通道、事件权重和 train 尺度与 S2 相同。报告 train/val/test 的 1/3/8 s q/v 及合并指标、隐状态两个起点 RMSE。另报告联合估计历史起点 q/v 的 RMSE，以及原始首帧观测相对真值的同口径 RMSE，用于判断解除约束的实际效果；这些描述值不增加分解或传播。

| 门 | 规则 |
|---|---|
| 技术门 | 六正例两档均满列秩 28、条件数 ≤`1e8`；全部历史和未来有限且绝对值 ≤`1e12`；质量 COI ≤`1e-10`；0–8 s 可见步长精化差按目标尺度归一后最大值 ≤`1e-5` |
| 新 clean 预测 | local/nonlocal 每角色主及 3 s NRMSE 均 ≤0.001 |
| 新 clean 隐状态 | 每角色历史起点及预测起点隐状态 RMSE 均 ≤`1e-4` |
| 新 noisy 预测 | local/nonlocal 每角色主及 3 s NRMSE 均 ≤0.02 |
| 等率负例 | 联合秩 27，拒绝初态与未来 |

8 s 精度、含噪隐状态误差和旧病例预测精度为描述项；技术门仍覆盖旧病例。新 clean 的可见初态恢复误差亦报告，但不添加事后门槛。

`paired_mechanism_supported` 定义为：两个旧病例在 train 和 val 的四项主 NMSE 比值 `新/旧` 全部 ≤0.95。主 NMSE 是主 NRMSE 的平方，不能用 NRMSE 比值替代。这是配对效果的预定判据，不是显著性检验，也不证明首帧约束是唯一根因。历史拟合残差下降是解除约束后的必然性质，不用作机制判据。

`scientific_pass` 仅由上述技术、负例和四个新病例的精度门共同决定。通过则允许登记 observer 梯度检查和未知动力学训练；若配对机制标记未通过，保持这一事实，不宣称已识别旧失败主因。未通过则继续噪声模型或短窗敏感性修正。不得放宽当前阈值、增加 ridge 搜索或重跑当前范围。任何结果都不直接确认学得动力学、不晋升基线、不开放 T0。


<a id="doc-dfa4811cb776-5-单次预算与保存审核"></a>

#### 5. 单次预算与保存审核

一次不可变正式运行，禁止重试。全局上限：旧 NPZ 解码 2、恢复 0、rollout 25、SVD 13、expm 6、显式矩阵操作 30000。参数拟合、优化更新、梯度、独立 linear solve/least_squares、逐步 NumPy/Torch RHS 和真实数据读取均为零。状态推断的代价计入 SVD 和矩阵操作，不记作语料参数训练。

25 次 rollout 为六病例×两档×历史/未来，加一个等率负例历史；13 次 SVD 为对应十二正例加一个负例。新输入生成两组，各三次 expm。事件批量大小因 28 列响应增加，矩阵调用次数不乘事件数；实际耗时和内存单独记录。逐范围上限为 sources 解码 2、其他计数为零；truth 两范围各 expm 3/矩阵 900；六病例各 rollout 4/SVD 2/矩阵 4000；rank_control 为 rollout 1/SVD 1/矩阵 100。完整通过可求解门时的静态矩阵账为六病例各 3800、负例 37、两组真值各 814，合计 **24465**；Householder 系数构造不另计传播矩阵操作，状态求解与历史重构各档登记 3+11 次。准确源码与该静态表在执行前绑定，总上限不随结果增加。

沿用范围开始、已发出额度、成功返回、失败保留并全额消费的日志规则。权限、预算、身份、非有限数或日志错误立即停止。保存所有输入、角色、响应、SVD 因子、联合坐标、因果重构、各档预测、指标、门与配对标记、预算、来源及环境。另登记一次保存数组算术审核，每个输出 NPZ 最多解码一次，不调用传播、分解、求解或训练。

用户已授权持续修正及按门禁推进；本阶段仍使用新的准确合同和执行范围，旧范围保持关闭。后续真实数据完全复用 TASK3 train 346 / val 107 / test 76 名单规则，本协议不读取真实数据。

## 报告与结果

<a id="doc-5a351c33b6b1"></a>

<a id="doc-5a351c33b6b1-joint-observer-确认结果2026-11-12"></a>

### Joint observer 确认结果（2026-11-12）

本次按冻结的 [S2 协议](#doc-dfa4811cb776)评估已知固定动力学下的联合初态估计。结果为 `process PASS`、数值门通过、clean恢复门通过、含噪预测门未通过；科学门未通过，结论为 `KEEP_BASELINE`，`STOP_T0`。


<a id="doc-5a351c33b6b1-结果摘要"></a>

#### 结果摘要

- 四项既有病例 train/val 配对主 NMSE 比值均低于 0.95，支持联合初态估计对机制的贡献；此结果不改变新含噪病例未达预测门的结论。
- 六个正例的数值门通过；秩 27 负例被正确拒绝。nonlocal 条件数约 `6.64e4`，高于 local 的约 `2.36e3`，但仍低于协议上限 `1e8`。条件数描述观测算子的数值敏感性，不能单独代表可辨识性，也不保证对噪声稳健。
- 新 local/nonlocal clean 的主 NRMSE、3 s NRMSE 与隐状态恢复误差接近数值精度；新 1% 噪声病例仍未通过预先冻结的含噪预测门。
- 新 `test` 只用于执行预冻结确认门；不用于选择旧机制、调整阈值或后续超参数搜索。


<a id="doc-5a351c33b6b1-trainval-配对机制比较"></a>

##### train/val 配对机制比较

| 病例 | 角色 | 原主 NMSE | 联合主 NMSE | 比值 |
|---|---|---:|---:|---:|
| local | train | 0.00292742 | 0.00221885 | 0.757954 |
| local | val | 0.00213161 | 0.00144564 | 0.678191 |
| nonlocal | train | 0.000825049 | 0.000488161 | 0.591676 |
| nonlocal | val | 0.000835922 | 0.000364173 | 0.435655 |


<a id="doc-5a351c33b6b1-六病例指标"></a>

##### 六病例指标

NRMSE 为无量纲归一化均方根误差；hidden start/origin RMSE 为隐状态估计误差，按状态尺度计算。所有数值保留六位有效数字。

| 病例 | 数据角色 | 主 NRMSE | 3 s NRMSE | 8 s NRMSE | hidden start RMSE | hidden origin RMSE |
|---|---|---:|---:|---:|---:|---:|
| old local noisy | train | 0.0471047 | 0.0542211 | 0.0436711 | 0.138956 | 0.0824158 |
| old local noisy | val | 0.0380216 | 0.0432263 | 0.0344751 | 0.114374 | 0.0672018 |
| old local noisy | test | 0.0388501 | 0.0444171 | 0.0356727 | 0.126349 | 0.0718242 |
| old nonlocal noisy | train | 0.0220944 | 0.0226953 | 0.0165618 | 5.30575 | 0.719726 |
| old nonlocal noisy | val | 0.0190833 | 0.0193149 | 0.0139276 | 4.85218 | 0.658121 |
| old nonlocal noisy | test | 0.0227286 | 0.0238170 | 0.0173778 | 3.47103 | 0.471160 |
| local clean | train | 1.12788e-9 | 1.34735e-9 | 1.10921e-9 | 4.27151e-9 | 2.49715e-9 |
| local clean | val | 7.75118e-10 | 9.27217e-10 | 7.64127e-10 | 2.98158e-9 | 1.73687e-9 |
| local clean | test | 1.29020e-9 | 1.54242e-9 | 1.27053e-9 | 4.91591e-9 | 2.85841e-9 |
| local 1% noise | train | 0.0501005 | 0.0583487 | 0.0473159 | 0.151961 | 0.0911959 |
| local 1% noise | val | 0.0421248 | 0.0488559 | 0.0395723 | 0.124217 | 0.0735667 |
| local 1% noise | test | 0.0723848 | 0.0851560 | 0.0694195 | 0.237802 | 0.139691 |
| nonlocal clean | train | 3.01546e-11 | 2.90894e-11 | 2.02250e-11 | 3.26973e-8 | 4.71321e-9 |
| nonlocal clean | val | 2.26001e-11 | 2.18481e-11 | 1.51599e-11 | 2.73975e-8 | 3.84151e-9 |
| nonlocal clean | test | 2.38374e-11 | 2.30641e-11 | 1.59902e-11 | 2.85894e-8 | 4.18412e-9 |
| nonlocal 1% noise | train | 0.0211935 | 0.0220488 | 0.0160774 | 4.80450 | 0.651807 |
| nonlocal 1% noise | val | 0.0206160 | 0.0209830 | 0.0150452 | 4.33203 | 0.587745 |
| nonlocal 1% noise | test | 0.0223301 | 0.0234740 | 0.0171013 | 4.67436 | 0.634063 |


<a id="doc-5a351c33b6b1-执行与结论边界"></a>

#### 执行与结论边界

运行目录：`results/machine_local_joint_observer_v1-fixed-20261112T101522545706Z-6228cbca-99b92ecc`。源码提交：`6228cbca76194d5a65d0e637d41a1c0a9076438a`。预算返回：24,465 矩阵乘积、25 次 rollout、13 次 SVD、6 次 expm、2 次 NPZ 解码、0 次 checkpoint 加载；真实数据读取为 0。

独立算术审核通过：11 个 NPZ 各解码一次，核对 28 维初态基、状态解、两档历史与未来、各项指标、配对比值和预算；没有调用模型、传播、分解或求解。哈希与运行身份见[证据回执](_sealed/machine_local_joint_observer_20261112_evidence.json)。

本结果支持首帧约束对旧样本误差有贡献，尚未解决含噪预测，也不确认学得动力学或真实数据表现。维持 `KEEP_BASELINE`，T0 继续关闭。下一项已知先验与噪声的信息对照，将区分短窗口的统计限制和状态重置的额外误差。

## 复核与历史版本

<a id="doc-3586cbef42f7"></a>

<details>
<summary>联合初态估计结果审核</summary>

<a id="doc-3586cbef42f7-联合初态估计结果审核"></a>

### 联合初态估计结果审核

审核人：Astra Ultra。日期：2026-11-12。

接受本次结果证据。解除噪声首帧的精确约束得到预定配对证据支持，但独立新样本的含噪预测仍未通过；继续 `NOISE_MODEL_REPAIR_REQUIRED`。基线不晋升，T0 保持关闭。


<a id="doc-3586cbef42f7-主要结果"></a>

#### 主要结果

两组旧含噪事件的主 NMSE 新/旧比值如下。四项均低于预注册的 0.95，因此 `paired_mechanism_supported=true`。原始首帧 q/v 的 RMSE 也均高于联合估计的可见初态误差。这支持首帧精确约束确实贡献了旧方法的误差，不能据此解释全部剩余误差。

| 旧样本配对 | train 主 NMSE 比 | val 主 NMSE 比 |
|---|---:|---:|
| local | 0.757954 | 0.678191 |
| nonlocal | 0.591676 | 0.435655 |

新种子确认中的两个 clean 病例通过预测、隐状态恢复和数值门。新含噪病例的 **3 s 可见状态归一化 RMSE** 如下，冻结上限为 0.02；主 NRMSE 也未通过。

| 新样本，固定真值动力学 | train | val | test |
|---|---:|---:|---:|
| local，1% 历史噪声 | 0.058349 | 0.048856 | 0.085156 |
| nonlocal，1% 历史噪声 | 0.022049 | 0.020983 | 0.023474 |

旧配对标记仅使用 train/val。新样本的 train/val/test 均属于事前冻结的确认门，不能在运行后把新 test 改成门外描述项。配对改善不能覆盖新样本确认失败。

新 local/nonlocal 联合响应条件数分别为 2361.06/66393.65；val 历史起点隐状态 RMSE 分别为 0.1242/4.3320。联合初态解改善了首帧估计，却未消除含噪隐状态误差。数值满秩和条件数门通过仍不等于具有足够的抗噪预测信息。等率负例按要求得到秩 27 并拒绝状态和未来。


<a id="doc-3586cbef42f7-下一项最小诊断"></a>

#### 下一项最小诊断

使用已冻结的两组新输入，构造已知生成动力学、Gaussian 初态先验和已知噪声协方差下的准确后验对照。共享一次后验，分别输出完整后验均值预测，以及同一后验配合原可见末点重置的预测。前者检验给定信息下的最小期望平方误差，后者量化重置约束的额外代价；不新增种子、不搜索正则、不执行 Adam。

预期主/3 s **NMSE** 与 `0.02²=0.0004` 比较，NRMSE 仅由开方展示。对独立于 train 尺度的 val/test 事件，这给出已知生成分布条件下的期望平方风险参照；它不是有限 test 必然失败的证明。train 的尺度由其样本生成，相关风险解释保留为描述。该对照使用部署时未知的真值信息，不能直接晋升为预测算法；下一可用候选仍须把先验和噪声模型限定在 train 信息内，再独立确认。


<a id="doc-3586cbef42f7-证据"></a>

#### 证据

正式运行 `machine_local_joint_observer_v1-fixed-20261112T101522545706Z-6228cbca-99b92ecc`，源码 `6228cbca76194d5a65d0e637d41a1c0a9076438a`。十个范围全部返回，实际消耗为 2 次旧 NPZ 解码、25 次传播、13 次 SVD、6 次 expm、24,465 次登记矩阵操作，零参数拟合、零梯度和真实数据读取。

一次保存算术审计 PASS，11 个输出 NPZ 各解码一次，核对联合基、保存 SVD、28 维坐标与十维隐状态、因果重构、可见重置、两档有限性和 COI、全部角色指标、配对判据、新样本门与预算。未重新传播、分解或求解。本审核只读 JSON、源码和字节哈希，未解码 NPZ 或执行科学算子。

| 证据 | SHA256 |
|---|---|
| 机器计划 | `06a30710590b6be5e6218156d1c1454e18577ef1367bbde8e4d408bac2f19b17` |
| 协议 | `7873562f1c3b3596a76f3ab082a345c3e9c887b615e1478ae60821af272b9f6f` |
| 运行产物清单 | `fbcfae6e2d7e6cc64efb00726433021979c9f4ee9e07c7c2f87d4c0e6d500858` |
| `summary.json` | `3313b19ef02295945c63232f4a277c1a04dfbe407f2fa713f04081f74c6dabab` |
| 保存算术审计 | `6ea3a3173b2c2ddec628606a5ddd2d94a3ec6d9e301a024e1d757eb92869b6d6` |
| 审计脚本 | `52435babe5c433cdb7c4c4a50c6957f35ea979528fe726da76437180a972e014` |

后续见 [Gaussian 后验对照计划](../gaussian_oracle/README.md#doc-d60df53b26e4) 与 [执行协议](../gaussian_oracle/README.md#doc-aa5c32db9aee)。

</details>

## 回执与附件

- [machine_local_joint_observer_20261112_evidence.json](_sealed/machine_local_joint_observer_20261112_evidence.json)
- [machine_local_joint_observer_result_review_20261112.json](_sealed/machine_local_joint_observer_result_review_20261112.json)
- [machine_local_joint_observer_review_20261112.json](_sealed/machine_local_joint_observer_review_20261112.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_joint_observer_v1_zh.md](_sealed/machine_local_joint_observer_v1_zh.md)
- [docs/reports/08_machine_local_topology/_sealed/machine_local_joint_observer_20261112_zh.md](_sealed/original/machine_local_joint_observer_20261112_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_joint_observer_20261112_zh.md](_sealed/machine_local_joint_observer_20261112_zh.md)
- [docs/research/machine_local_joint_observer_plan_20261112_zh.md](_sealed/machine_local_joint_observer_plan_20261112_zh.md)
- [docs/research/machine_local_joint_observer_result_review_20261112_zh.md](_sealed/machine_local_joint_observer_result_review_20261112_zh.md)

</details>
