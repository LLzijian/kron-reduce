# Gaussian information oracle 诊断结果（2026-11-12）

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_gaussian_oracle_v1.json)

## 研究方案

<a id="doc-d60df53b26e4"></a>

<a id="doc-d60df53b26e4-100-ms-历史的-gaussian-后验信息对照"></a>

### 100 ms 历史的 Gaussian 后验信息对照

日期：2026-11-12。核心设计与科学审核：Astra Ultra。

[联合初态估计](../joint_observer/README.md#doc-3586cbef42f7)支持解除首帧精确约束的作用，但新样本的含噪预测仍未达到 0.02。下一项诊断直接计算已知生成动力学、初态先验与噪声条件下的后验均值及期望平方误差，判断目前的误差与信息允许的水平相差多少。

仅使用上一阶段两个新语料的冻结输入，各解码一次，不新增种子或搜索正则。初态先验取已知 Gaussian 生成分布，观测噪声使用实际 COI 投影后的协方差，在 18 维独立可见坐标中处理。用精确矩阵指数构建十一帧历史和未来响应，求 28 维初态后验。它是使用真值信息的对照，不是可部署算法。

同一后验产生两个输出：完整后验均值预测；保持其隐状态不变、把预测起点 q/v 重置为最后观测后的预测。第一项给出已知分布下的最小期望平方风险；第二项量化这一具体重置操作的额外条件风险。它不代表所有固定可见末点预测器的最优风险，因为其他隐状态选择可能部分补偿重置误差。

解析主/3 s NMSE 与 `0.02²=0.0004` 比较。对独立于 train 尺度的 val/test 事件，这个风险是已知生成条件下的总体期望平方误差参照，不是有限样本必须失败的判定。train 的尺度由自身样本生成，仅作描述。各角色实际 NRMSE 单独报告，不能把有限样本 MSE 差强制等同于后验风险分解。

技术门通过即形成有效诊断。若完整后验均值的预期风险仍不满足目标，先审查信息、噪声和观测合同；若满足，则准备仅由 train 信息确定的噪声/先验候选，并根据重置风险处理预测起点。两条路线都保留用户授权的持续修正工作，不盲目追加 Adam、不用 test 反搜正则、不改变已完成阶段的门槛。

执行细节见 [协议](#doc-aa5c32db9aee)。本次零参数拟合、零梯度、零真实数据读取，保持基线且不开启 T0。可用候选仍需独立新样本确认和未知动力学学习；后续真实数据完全复用 TASK3 train 346 / val 107 / test 76 名单规则。

## 执行协议

<a id="doc-aa5c32db9aee"></a>

<a id="doc-aa5c32db9aee-gaussian-后验信息对照协议-v1"></a>

### Gaussian 后验信息对照协议 v1

日期：2026-11-12。设计与科学审核：Astra Ultra。执行阶段：`machine_local_gaussian_oracle_v1/fixed`。研究目的见 [计划](#doc-d60df53b26e4)。


<a id="doc-aa5c32db9aee-1-数据与解释范围"></a>

#### 1. 数据与解释范围

本阶段只诊断已知线性生成分布下 100 ms 历史的抗噪预测信息。各读取上一阶段 `local_inputs.npz`、`nonlocal_inputs.npz` 一次，使用其已冻结的 1% 历史噪声、train 尺度、64 个事件和 32/16/16 角色。不新增种子，不选择超参数，不训练动力学或先验，不读取真实数据。

来源运行是 `machine_local_joint_observer_v1-fixed-20261112T101522545706Z-6228cbca-99b92ecc`。

| 来源 | SHA256 |
|---|---|
| `local_inputs.npz` | `fe56c9ac5730e76cd880a7d8a396a950f03f56cec5ce145b53000e56180324d9` |
| `nonlocal_inputs.npz` | `72307579c131d2702b9c7878a09ca303b3e1225930d3a9132d9efe046b28752f` |

动力学、初态分布和噪声分布使用生成真值，因而本方法是 oracle 对照，不能作为可部署算法或学得模型成绩。val/test 与 train 样本独立，条件于 train 生成的历史尺度和未来指标尺度时，下述 Gaussian 先验及噪声模型对它们成立。train 的尺度由自身样本计算，不能把同一公式无条件称作其准确条件风险，故 train 只作描述。当前数据均为已经暴露的诊断样本，不声称新的独立确认。


<a id="doc-aa5c32db9aee-2-已知先验与噪声"></a>

#### 2. 已知先验与噪声

沿用联合初态的正交质量 COI 基 V 及 `B=diag(V,V,I₁₀)`；`ξ=(Vᵀq_h,Vᵀv_h,z_h)` 为 28 维初态坐标。令 `P=I−m1ᵀ/(1ᵀm)`、`Q=Pᵀ`、`L=diag(Vᵀ,Vᵀ)`。已知输入功率与初态独立，条件于给定功率后先验仍为零均值，协方差为

\[
\Sigma_0=\operatorname{diag}(0.04V^\top QPV,\;0.01V^\top QPV,\;0.0225I_{10}).
\]

该协方差来自原 `q_h=0.2Qη_q`、`v_h=0.1Qη_v`、`z_h=0.15η_z` 抽样规则，不能用当前隐状态真值或未来目标估计。

每帧观测投到独立的 18 维可见坐标 `Ly_j`，避免对奇异的 20 维 COI 噪声矩阵作伪逆。历史通道尺度记为 `s_q,s_v`，已知每帧噪声协方差为

\[
R=10^{-4}\operatorname{diag}\left(
V^\top Q\operatorname{diag}(s_q^2)PV,\;
V^\top Q\operatorname{diag}(s_v^2)PV\right).
\]

11 帧噪声独立。这里使用的是原生成噪声在 COI 投影后的完整协方差，不以对角历史 RMS 权重代替白化。没有 ridge 路径或噪声幅度搜索。


<a id="doc-aa5c32db9aee-3-精确响应与一次后验"></a>

#### 3. 精确响应与一次后验

每族从原 40 维含常值输入的真值生成矩阵 A 计算一次 `exp(0.01A)`。批量包括 64 个零物理状态、给定功率的基准初态，以及 B 的 28 列零输入响应；共同递推 810 次，得到 `b_e,j` 与 `D_j`，`j=0,…,810`。不重新生成随机真值。

历史观测为 `r_e,j=L(y_e,j−Hb_e,j)`、`O_j=LHD_j`，`j=0,…,10`，H 选择前 20 个物理坐标。后验为

\[
K=\Sigma_0^{-1}+\sum_{j=0}^{10}O_j^\top R^{-1}O_j,\qquad
g_e=\sum_{j=0}^{10}O_j^\top R^{-1}r_{e,j},
\]
\[
\mu_e=K^{-1}g_e,\qquad\Sigma_{\rm post}=K^{-1}.
\]

每族分别对 prior、noise、precision 三个正定矩阵做 Cholesky，并调用四次 `cho_solve`：prior 逆、合并全部帧的 `R⁻¹O`、合并全部事件/帧的 `R⁻¹r`、`K⁻¹[I,g]`。后一个合并右端项一次给出协方差和全部事件均值；不对每事件重新分解。

源真实初态 ξ 只用于核对 `b+Dξ` 与已保存的 811 帧前 30 个物理状态，不参与后验求解、尺度或参数选择。常值功率保留为已知输入。保存矩阵、因子、每个合并右端项及解，以便后续纯算术核验。


<a id="doc-aa5c32db9aee-4-同一后验的两个预测与风险"></a>

#### 4. 同一后验的两个预测与风险

完整后验均值历史/未来为 `x̄_e,j=b_e,j+D_j μ_e`。其未来第 k 帧使用 `j=10+k`，是给定全部历史和功率时的条件均值。第二个预测保留 `x̄_e,10` 的 z，但把其 q/v 换成最后一帧观测。令

\[
\delta_e=L(y_{e,10}-H\bar x_{e,10}),\qquad
d_{e,k}=H D_k^{\rm visible}\,\delta_e.
\]

这里 `D_k^visible` 表示前 18 个可见初态响应列，代码对应 `D_k[:, :18]`；k 从预测起点重新计时。由于动力学时不变，已有 `D_0,…,D_800` 直接给出重置扰动响应，无需额外传播。重置版未来等于完整后验均值未来加该物理响应。

记 `F_k=HD_{10+k}`、未来指标尺度 `S=diag(target_scale)`。完整条件均值在第 k 帧的合并 NMSE 风险为

\[
v_k=\frac1{20}\operatorname{tr}(S^{-2}F_k\Sigma_{\rm post}F_k^\top).
\]

未来目标是 clean 状态，没有未来过程或观测噪声，因此不能再把 R 加入预测风险。主 NMSE 风险采用前 100 帧权重 `1/150`、后 200 帧 `1/600`；3 s NMSE 各帧权重为 `1/300`。均不包含预测起点。NRMSE 由风险开方展示，但目标比较始终使用 **NMSE ≤0.0004**。

给定当前历史，重置版相应期望风险为

\[
R_{{\rm reset},e}=R_{\rm mean}+\sum_{k=1}^{300}w_k\|S^{-1}d_{e,k}\|^2/20.
\]

额外项非负；它描述保持后验 z 不变、仅重置可见状态的代价，不是所有固定可见末点方法的最优风险。另选 z 可能部分补偿重置误差，本阶段不增加这类候选。有限样本实际 MSE 差还含随机交叉项，不能要求它等于上述风险增量或必为正。

分别报告两输出、全部角色的 1/3/8 s q/v 与合并实际 NRMSE、物理 RMSE、隐状态误差，及解析主/3 s NMSE、按角色汇总的重置条件风险和增量。解析风险不通过有限样本实际误差反校准，不依据 test 搜索先验或正则。


<a id="doc-aa5c32db9aee-5-验收与继续路线"></a>

#### 5. 验收与继续路线

技术门通过形成有效诊断，不要求信息目标必然可达。技术门包括：来源与参数身份一致；真值响应重构最大绝对差 ≤`1e-10`；三 Cholesky 成功且因子重构相对 Frobenius 误差 ≤`1e-12`；四求解的后向残差 `‖AX−B‖F/(‖A‖F‖X‖F+‖B‖F)` ≤`1e-12`（分母下限 `1e-300`）；prior 逆与后验逆的恒等式残差最大绝对值均 ≤`1e-8`；后验协方差对称误差 `‖Σpost−Σpostᵀ‖F/max(1,‖Σpost‖F)` ≤`1e-12`；全部量有限，状态绝对值 ≤`1e12`，完整历史与两输出质量 COI ≤`1e-10`。因子重构的相对误差分母为 `max(1,‖A‖F)`。风险方差项允许 `1e-12` 的浮点负误差，保留原值，不以裁剪掩盖失败。保存风险分解供算术复核。

`information_target_feasible` 要求两族完整条件均值的解析主和 3 s NMSE 均 ≤0.0004。它是给定生成分布、噪声、100 ms 窗和目标尺度下的期望平方风险判据，不是某个有限 test 集不可能达标的证明，也不外推到真实电网分布。

`scientific_pass` 表示两族技术门均通过。`reset_val_target_feasible` 要求两族 val 平均条件重置风险的主和 3 s NMSE 均 ≤0.0004，train/test 不参与该分流标记。

按顺序分流：技术门失败为 `ORACLE_INVALID`；技术门通过而信息目标未满足为 `INFORMATION_CONTRACT_REVIEW_REQUIRED`；信息目标满足但 val 重置目标未满足为 `RESET_INTERFACE_REPAIR_PRIORITY`；两目标均满足为 `REGISTER_TRAIN_ONLY_STATISTICAL_OBSERVER`。后两条只决定准备哪种由 train 信息确定的候选，不接受 oracle 为部署算法。重置版的失败不单独证明所有固定可见末点方法不可达。本阶段不改变旧结果和门槛、不晋升基线、不开放 T0。


<a id="doc-aa5c32db9aee-6-执行边界"></a>

#### 6. 执行边界

一次独立诊断范围，禁止重试。全局固定原语上限为旧 NPZ 解码 2、expm 2、批量轨迹传播 2、Cholesky 6、`cho_solve` 8；显式矩阵操作上限 **12000**，完整运行静态预期 **9805**。新随机数据生成、checkpoint 恢复、参数拟合、优化更新、梯度、SVD、least_squares、逐步 RHS 和真实数据读取均为零。

三个范围为 `sources`（只解码 2）、`local`、`nonlocal`；每族上限 expm 1、传播 1、Cholesky 3、`cho_solve` 4、矩阵 6000。矩阵静态账为 local **4902**、nonlocal **4903**，后者多一次冻结 DCT 端口构造。每族后验部分为十一帧正规方程 22 次，加三项因子重构、三项 prior/后验方程残差和两项噪声方程残差，共 30 次。其余传播、重构、方差与 COI 操作在准确源码中逐组登记；批量不乘事件数，不把库内部运算冒充逐算子硬件计数。保存时间、内存和运行库版本。

沿用已发出分组额度、正常返回闭合、失败保留并全额消费的规则。保存输入哈希、全部响应、后验方程/因子/解、两个预测、方差与偏差风险、实际指标和解释标签、预算与环境。另登记一次保存数组算术复核，每个输出 NPZ 最多解码一次，不调用模型、传播、分解或求解。用户已授权持续修正，但各阶段使用新的冻结范围；本次仍不触发真实训练。

## 报告与结果

<a id="doc-6bb34fcdab7d"></a>

<a id="doc-6bb34fcdab7d-gaussian-information-oracle-诊断结果2026-11-12"></a>

### Gaussian information oracle 诊断结果（2026-11-12）

本阶段用已知生成动力学参数、输入功率、初态先验和噪声协方差，评估 100 ms 历史对未来状态的条件信息。运行按冻结协议完成，技术门通过；两族完整后验的解析信息目标未同时满足，分流为 `INFORMATION_CONTRACT_REVIEW_REQUIRED`。这是一项生成分布下的 oracle 诊断，不是可部署估计器。


<a id="doc-6bb34fcdab7d-条件与风险定义"></a>

#### 条件与风险定义

输入为 local 与 nonlocal 两种已知线性生成动力学，64 个事件按 train/val/test = 32/16/16。历史有 11 帧、间隔 10 ms，噪声标准差为历史尺度的 1%；使用已知零均值 28 维高斯初态先验，协方差按 q/v/z 抽样尺度 `0.2/0.1/0.15` 构造，并使用 COI 投影后的完整观测噪声协方差。没有训练参数或搜索超参数。

表中期望主/3 s 风险是 NMSE，括号内为其平方根；它们是生成分布下的解析条件平方风险，不是有限样本预测误差。`full` 是完整后验均值，`reset` 是同一后验隐状态下将 q/v 换为历史末帧观测的对照。val/test 的实际 3 s NRMSE 单独列出。

| 族 | 角色 | full 期望主 NMSE (√) | full 期望 3 s NMSE (√) | reset 期望主 NMSE (√) | reset 期望 3 s NMSE (√) |
|---|---|---:|---:|---:|---:|
| local | val | 0.000490888 (0.0221560) | 0.000634753 (0.0251943) | 0.000574521 (0.0239692) | 0.000699206 (0.0264425) |
| local | test | 0.000490888 (0.0221560) | 0.000634753 (0.0251943) | 0.000590683 (0.0243040) | 0.000713021 (0.0267025) |
| nonlocal | val | 0.000329781 (0.0181599) | 0.000390167 (0.0197526) | 0.000445256 (0.0211011) | 0.000481728 (0.0219483) |
| nonlocal | test | 0.000329781 (0.0181599) | 0.000390167 (0.0197526) | 0.000426039 (0.0206407) | 0.000465363 (0.0215723) |

解析信息目标要求两族 full 的主与 3 s NMSE 均不超过 `0.0004`。nonlocal 达到该条件；local 的期望 3 s NMSE 平方根为 `0.0251943`，高于对应 `0.02` 目标。val reset 风险也未同时达到门槛：nonlocal reset 期望 3 s NMSE 平方根为 `0.0219483`。

| 族 | 角色 | full 实际 3 s NRMSE | reset 实际 3 s NRMSE |
|---|---|---:|---:|
| local | val | 0.0257293 | 0.0271125 |
| local | test | 0.0257319 | 0.0272144 |
| nonlocal | val | 0.0165220 | 0.0195783 |
| nonlocal | test | 0.0209551 | 0.0228208 |

实际 val/test 误差是有限样本结果，不能替代解析风险；test 仅按冻结设计描述，不据此选择先验、正则或其他参数。上述信息目标针对已知生成分布、100 ms 历史与 1% 噪声，不能外推为真实电网或任意有限 test 集的结论。


<a id="doc-6bb34fcdab7d-执行记录与边界"></a>

#### 执行记录与边界

运行目录：`results/machine_local_gaussian_oracle_v1-fixed-20261112T105208760887Z-4ef00ac3-3d2be473`。源码提交：`4ef00ac34070a431a8dd00eb12373d7e59de68d3`；CI [运行 43203727718](https://github.com/LLzijian/kron-reduce/actions/runs/43203727718) 成功。过程状态 `PASS`，技术门通过，`scientific_pass=true` 按协议表示技术门有效；信息目标标记为 false。预算返回：2 次 NPZ 解码、9,805 次矩阵乘积、2 次 expm、2 次 rollout、6 次 Cholesky、8 次线性求解；真实数据读取为 0，运行约 11.88 s。

保存数组的独立算术审核通过：4 个 NPZ 各解码一次，后验方程、两个输出、风险分解、逐项指标与预算记录一致；没有执行新的模型调用、传播、分解或求解。证据哈希见[回执](_sealed/machine_local_gaussian_oracle_20261112_evidence.json)。当前分流为 `INFORMATION_CONTRACT_REVIEW_REQUIRED`；维持 `KEEP_BASELINE`，`STOP_T0`。oracle 结果不确认学得动力学，也不构成部署算法。

## 复核与历史版本

<a id="doc-30f0945ff899"></a>

<details>
<summary>Gaussian 后验信息对照结果审核</summary>

<a id="doc-30f0945ff899-gaussian-后验信息对照结果审核"></a>

### Gaussian 后验信息对照结果审核

日期：2026-11-12。科学审核：Astra Ultra。结论：接受本次诊断证据；已知生成分布下，local 的 100 ms 抗噪预测期望风险超过原目标。保留原失败结果、基线与 T0 门禁。


<a id="doc-30f0945ff899-结果与含义"></a>

#### 结果与含义

本次固定已知动力学、初态先验和噪声分布，对同一历史比较完整后验均值，以及保留同一后验隐状态但强制重置末点可见状态的预测。主 NMSE 是 1 s 与 3 s NMSE 的均值，原目标为主和 3 s NMSE 均不超过 0.0004，即对应 NRMSE 不超过 0.02。

| 生成模型 | 完整后验主 NMSE | 完整后验 3 s NMSE | val 重置主条件 NMSE | val 重置 3 s 条件 NMSE |
|---|---:|---:|---:|---:|
| local | 0.0004908880915 | 0.0006347525990 | 0.0005745214107 | 0.0006992062514 |
| nonlocal | 0.0003297812388 | 0.0003901667053 | 0.0004452560673 | 0.0004817279012 |

完整条件均值在给定生成分布和观测合同下最小化期望平方误差。local 的两个期望指标均超门，nonlocal 均低于门槛。因此，当前 local 条件下不能通过更长训练保证达到原期望目标。该结论条件于已知生成分布、1% 历史噪声、train 尺度和 100 ms/11 帧输入；不表示任意有限 test 样本必然失败，也不外推到真实电网。

两族 val 的同后验重置条件风险均超门。其增加项分别为主 NMSE 0.00008363331917 和 0.00011547482855，说明当前重置接口增加了预测风险。这只评价“保持后验 z 不变、重置 q/v”的方案，不能推出所有固定可见末点方法都不可行。有限样本实际误差差还包含交叉项，不等于上述条件风险增量。

这些样本已用于前序诊断，本次不是新的独立确认。train 参与形成噪声与目标尺度，其风险仅作描述；上述条件风险解释适用于独立于 train 的 val/test。oracle 使用生成先验、噪声和动力学，不能作为可部署算法或学得模型成绩。


<a id="doc-30f0945ff899-证据接受"></a>

#### 证据接受

运行 `machine_local_gaussian_oracle_v1-fixed-20261112T105208760887Z-4ef00ac3-3d2be473` 使用源码提交 `4ef00ac34070a431a8dd00eb12373d7e59de68d3`。两族技术门通过：来源轨迹重构最大误差不超过 2.131e-14，逆恒等式误差不超过 2.203e-13，四方程后向残差不超过 3.062e-17。三项执行范围全部正常返回，矩阵操作为 9805，分族为 4902/4903；解码 2、expm 2、批量传播 2、Cholesky 6、求解 8，无拟合、梯度或真实数据读取。

独立保存算术审计通过，四个输出 NPZ 各解码一次。审计从保存数组重组先验、噪声协方差、观测方程、后验方程、因子和求解残差，复核均值、重置偏差、方差风险、全部角色的实际指标、分流与预算。审计没有再次计算 expm 或递推轨迹；来源重构和源码审核提供该部分的一致性证据。本审核只读 JSON、源码与字节哈希，未解码 NPZ 或调用科学算子。

| 绑定 | SHA256 |
|---|---|
| 机器计划 | `b7c0a0861ef7926154eefedb2a55b1cd7063b41dcff5a2da36687385e45f7177` |
| 协议 | `e5d805f8a7e1ab7e5e17b0ac136b9164c6b6976ef3cbd51b8096e18a96d21fa8` |
| 运行 artifact manifest | `2045dd15754e4866cb2af41b06a307971ed5704b1f66ccf6de0eed3ad77f02f6` |
| 运行 summary | `d7394c79e1c0c91f9a461375f4e31b1e933bf9682d0ce8b9368bf442e5d58120` |
| 保存算术 audit | `94d3506c4f11b980ed35ca626d5c37868d3ef4abef6eece9cd012f81bc4e19c7` |
| audit 脚本 | `06d7da6cf49ff7ada17e732160c7308e8f1ec13f73fa0e03d51867d990daf1b1` |

`approved=true` 表示接受已完成诊断证据；`scientific_pass=true` 表示技术诊断有效。`information_target_feasible=false`、`reset_val_target_feasible=false`，继续路线为 `INFORMATION_CONTRACT_REVIEW_REQUIRED`。基线保持，T0 不开放。


<a id="doc-30f0945ff899-下一步"></a>

#### 下一步

按[统计初始化计划](../statistical_initializer/README.md#doc-b45fdc06a514)准备唯一 train-only 候选，先隔离统计估计与初始化泛化，再独立处理未知动力学。TASK3 的 100 ms/11 帧输入、train/val/test 名单规则和旧失败门槛保持不变；新候选的机制检验不能覆盖原绝对指标失败。

</details>

<a id="doc-8c9819bcb54a"></a>

<details>
<summary>Gaussian 后验信息对照准确实现审核</summary>

<a id="doc-8c9819bcb54a-gaussian-后验信息对照准确实现审核"></a>

### Gaussian 后验信息对照准确实现审核

审核人：Astra Ultra。日期：2026-11-12。结论：**接受已登记的单次诊断执行**。保持基线，T0 关闭。

机器计划 [machine_local_gaussian_oracle_v1.json](../../../../configs/proposals/machine_local_gaussian_oracle_v1.json) 的 SHA256 为 `b7c0a0861ef7926154eefedb2a55b1cd7063b41dcff5a2da36687385e45f7177`，36 个实现绑定全部匹配。[协议](#doc-aa5c32db9aee) SHA256 为 `e5d805f8a7e1ab7e5e17b0ac136b9164c6b6976ef3cbd51b8096e18a96d21fa8`。完整绑定与执行范围见 [机器审核记录](_sealed/machine_local_gaussian_oracle_review_20261112.json)。

已核对已知初态协方差和 COI 投影后的完整噪声协方差、18 维观测与 28 维初态坐标、四次合并求解的数组方向、精确响应的时间索引，以及共享后验的完整均值/可见重置两个输出。未来目标为 clean 状态，风险没有额外添加未来测量噪声。主/3 s 解析 NMSE 与 0.0004 比较，技术有效性与风险是否满足目标分别记录。

静态预算为两个源 NPZ 各解码一次、2 次 expm、2 次批量传播、6 次 Cholesky、8 次 `cho_solve`、9805 次登记矩阵操作；矩阵上限为 12000。没有参数训练、正则搜索、新种子、梯度或真实数据读取。来源重构、因子重构、求解后向残差、协方差、有限性、COI 与单次执行约束符合协议。协调端报告 27 项工程测试及 Ruff 通过，本审核未重跑。

该对照使用真值动力学、先验和噪声，结果只说明指定生成分布和观测合同下的诊断。对独立 val/test，完整后验均值给出条件期望平方风险；train 因自身参与尺度生成，只作描述。有限样本误差不等于解析期望风险；同一后验的可见重置代价也不能证明所有固定可见末点方法的最优风险。任何诊断结果都不能直接确认学得模型或晋升为部署算法。

本审核只读取代码、JSON、测试源码和字节哈希，未解码 NPZ 或执行科学算子。用户已授权按门禁继续修正；本次批准只对应新的独立诊断范围，不恢复已消费授权。

执行前绑定更新：协调者已将 tests/test_gaussian_oracle_workflow.py 统一为 LF，并确认 Git 内容无语义变化。此次仅更新该测试文件的字节哈希及机器计划哈希，其余 35 项绑定、科学设计、预算与诊断批准结论不变。

</details>

## 回执与附件

- [machine_local_gaussian_oracle_20261112_evidence.json](_sealed/machine_local_gaussian_oracle_20261112_evidence.json)
- [machine_local_gaussian_oracle_result_review_20261112.json](_sealed/machine_local_gaussian_oracle_result_review_20261112.json)
- [machine_local_gaussian_oracle_review_20261112.json](_sealed/machine_local_gaussian_oracle_review_20261112.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_gaussian_oracle_v1_zh.md](_sealed/machine_local_gaussian_oracle_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_gaussian_oracle_20261112_zh.md](_sealed/machine_local_gaussian_oracle_20261112_zh.md)
- [docs/research/machine_local_gaussian_oracle_plan_20261112_zh.md](_sealed/machine_local_gaussian_oracle_plan_20261112_zh.md)
- [docs/research/machine_local_gaussian_oracle_result_review_20261112_zh.md](_sealed/machine_local_gaussian_oracle_result_review_20261112_zh.md)
- [docs/research/machine_local_gaussian_oracle_review_20261112_zh.md](_sealed/machine_local_gaussian_oracle_review_20261112_zh.md)

</details>
