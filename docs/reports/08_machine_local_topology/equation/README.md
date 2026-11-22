# 逐机方程与函数项恢复：合成结果

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_equation_v1.json)

## 研究方案

<a id="doc-052b12bfad13"></a>

<a id="doc-052b12bfad13-逐机方程联合学习与有限物理函数对照计划"></a>

### 逐机方程联合学习与有限物理函数对照计划

日期：2026-11-14。状态：准备稿；不授权训练、数据读取或模型晋升。


<a id="doc-052b12bfad13-研究问题"></a>

#### 研究问题

当前逐机模型通过局部互易端口学习记忆状态，但势能刚度和线性阻尼仍固定。下一步先检验这些已存在物理项的系数能否与端口、时间常数共同适配，再用有限物理函数对照判断谐波势能和状态调制耗散是否得到数据支持。评价重点是闭环轨迹、物理函数恢复及激励支持，不以分类精度替代动力学证据。


<a id="doc-052b12bfad13-三层候选"></a>

#### 三层候选

| 层 | 可训练内容 | 作用 |
|---|---|---|
| A：原方程系数 | 逐机端口幅值和时间常数，加 45 条边刚度系数和 9 个 COI 阻尼系数，共 74 个可训练量 | 检查原方程内部系数适配的作用 |
| 密集 B | A 同时增加谐波势能族和调制阻尼族两个标量，从与 A 相同的系数锚点联合拟合 | 形成两类函数的联合提案 |
| 三个支持集复拟合 | 从密集 B 提案分别拟合仅保留谐波、仅保留调制阻尼、两项均删除的简化支持集 | 判断每个函数族是否有独立支持，并控制不必要的自由度 |

阻尼坐标固定采用 A06 的原始未加权定义 `N=coi_basis(ones)`，由全一向量的欧氏正交补经 SVD 得到，不使用质量加权基或 Helmert 基。候选实现必须沿用同一坐标顺序与系数定义，以保证锚点和函数比较有共同含义。A 与 B 共享其余全部初始系数；B 的两个新标量从零开始。


<a id="doc-052b12bfad13-合成验证"></a>

#### 合成验证

建立三类已知真值语料：不含新增项的 null、含谐波势能项、含状态调制阻尼项。每类均完整划分 `train=16`、`val=8`、`test=8`，按事件 ID 固定角色。每类运行六个方程：共享物理参数的真值控制、A、密集 B，以及三个支持集复拟合。真值控制不进入候选初始化或优化器；候选不跨语料混用真值或拟合结果。R20/R21 只作为历史参考，不增加为本轮新拟合方程。

先在合成语料上完成方程与因果历史求解器验证。未来目标从 0.5 s 和 1 s 两个时长开始，训练梯度贯穿完整的因果 GN 历史求解、可见状态重置和自由未来 rollout。拟合使用 CPU；正式长时推理使用 GPU CUDA graph。比较端口与逐边物理系数、两类函数幅值、训练更新表现、轨迹误差及其随时间的变化；同时检查激励视图是否覆盖被估计函数所在的状态与边角范围。null 语料用于检查新增项是否在缺少真值支持时仍被选择。保留全部候选和角色结果，具体更新规则、采样方案、预算、数值失败处理与接受门槛由独立协议冻结。


<a id="doc-052b12bfad13-真实数据比较边界"></a>

#### 真实数据比较边界

真实比较沿用当前冻结的 TASK3 `train=346`、`val=107`、`test=76` 来源限定名单及既有输入、评分和稳定性合同。A06 提供边刚度和阻尼锚点；真实逐机端口幅值不读取 A06 的旧 g/τ，而按 (g_i=\lVert C_{A06}\rVert_F/\lVert P\rVert_F) 确定统一初值，时间常数统一取 `geomspace(0.06, 1.6, 10)`。A 与 B 的其余初始系数完全一致。按协议完成配对训练后，完整评价覆盖三个 split。`test` 仅作报告，不能选择模型、函数族、阈值或训练规则。当前真实 TASK3 结果继续作为固定的历史暴露比较基准，R20/R21 也只作历史比较参照；不改写为独立确认，也不因本计划改变现有基线或部署默认值。

本计划只界定研究方向。执行前仍需单独完成候选实现审核、协议冻结、登记预算及相应科学授权。


<a id="doc-052b12bfad13-合成结果后的诊断顺序"></a>

#### 合成结果后的诊断顺序

第一轮确认三组均能选中正确的函数项，记忆核恢复成为后续验证重点。下一项准备固定已知正确的函数项，在完整合成 train 上分别将端口与时间常数、刚度与阻尼替换为对应真值。比较 0.5 秒／1 秒与 3 秒轨迹误差，检验参数块之间的补偿以及短时目标对长时误差的敏感度。各对照沿用三步 GN。若进一步比较训练时域，两组共享起点、更新预算和 val 选择规则。诊断及训练须分别登记预算，冻结后的完整合成 test 用于报告。[首轮结果](#doc-afec4d4ead3a) · [独立结果审查](_sealed/machine_local_equation_result_review_20261114.json)

## 执行协议

<a id="doc-f8672d277241"></a>

<a id="doc-f8672d277241-逐机方程联合学习与函数项选择的合成协议"></a>

### 逐机方程联合学习与函数项选择的合成协议

2026-11-14。以三个完整的已知真值人群检验共享系数与函数项。真实 TASK3 比较另行登记，沿用冻结的 train 346、val 107、test 76。


<a id="doc-f8672d277241-方程与共同起点"></a>

#### 方程与共同起点

每台机保留一个隐态。A 学习 45 个边刚度、9 个阻尼系数、10 个局部端口幅值和 10 个时间常数，共 74 个量。阻尼为 \(D=N\operatorname{diag}(d)N^\top\)，\(N=\mathrm{coi\_basis}(\mathbf 1)\) 为 A06 使用的欧氏平衡基。\(k,d\) 用锚点乘指数表示，\(g\) 用指数表示，\(\tau\) 保留原有 \((0.02,5)\) 秒的对数插值参数化。逐边刚度直接学习，方程中无额外全局刚度乘数。

A 与 B 共享逐项相同的系数初值。以 \(e=B^\top q\) 和 \(\eta_*=B^\top q_*\) 表示相对角及平衡角，B 增加两个从精确零开始的非负幅值。二阶谐波的边力为 \(a k_{\rm anchor}[\sin(2(e+\eta_*))-\sin(2\eta_*)]\)，对应移位势能 \(a k_{\rm anchor}[\cos(2\eta_*)-\cos(2(e+\eta_*))-2e\sin(2\eta_*)]/2\)。调制耗散为 \(D_b(q)=\frac{b}{45}B\operatorname{diag}(\sin^2 e)B^\top\)，训练坐标为 \(b/8\)。这些项保留平衡点处零恢复力、半正定耗散和互易端口功率抵消。函数项选择保留 45 条候选边，曲率检查限定在平衡点的质量 COI 子空间。


<a id="doc-f8672d277241-人群与历史合同"></a>

#### 人群与历史合同

固定种子 20261114，以一次排列把 32 例分为 train 16、val 8、test 8。三个人群共用初值和输入生成方案。真值分别无新增项、仅二阶谐波 \(a=0.6\)、仅调制耗散 \(b=8\)。共同锚点为 \(k=0.12,d=0.8,g=0.4,\tau=\mathrm{geomspace}(0.06,1.6,10)\)，全部 74 个真值系数由生成器固定变化。每份名单四分之一为预先指定的高速激励。独立 NumPy 方程以 1.25/0.625 ms 双步长生成真值，每 10 ms 出帧。真值标签按 TASK3 判据从轨迹派生，保留全部预设事件。

各事件使用 11 帧功角和转速、起报前已知的平衡状态及常值平衡功率输入。三个 Gauss–Newton 更新使用当前方程的解析切线。闭环历史传播后，从第 11 帧可见状态和传播得到的隐态起报。train 每次更新重新求隐态，val 和 test 同样从本事件历史求隐态。未来真值给共享参数提供训练监督。历史尺度和目标尺度分别取完整 train 的历史 11 帧和预测后 100 帧的逐通道 RMS，下限为 \(10^{-5}\)。


<a id="doc-f8672d277241-拟合选择与冻结"></a>

#### 拟合、选择与冻结

每次更新使用完整 16 例 train 和一个 Adam 步，CPU float64 单线程，学习率 0.01，梯度范数上限为 10。以 5 ms 步长计算 0.5 秒／1 秒等权、事件等权、按固定 train 尺度归一化的可见状态 MSE。A 与完整 B 各最多 240 步，每步将 B 的两个训练坐标固定收缩 \(10^{-6}\) 后投影到 \([0,2]\)。每 20 步用完整 val 计算同一未加罚目标，选最早严格最小值。连续三次 val 检查相对改善不足 \(10^{-4}\) 时停止，最早第 80 步。无有效 val 候选时保留最后完整更新。数值失败记录并回滚本次更新。检查点保存参数、优化器状态和步数，预处理随人群单独保存。

从 B 的 val 最佳参数出发，用 fresh Adam 分别重拟合三种删项方案，各最多 80 步。被删除项精确为零。完整 B 与三种删项方案按完整 val 选择。距四者最低目标不超过 \(\max(10^{-8},0.01J_{\rm best})\) 的方案中，先选非零函数项最少者，再按 val 目标及固定顺序「均无、谐波、耗散、两项」决定。各删项重拟合单独记录预算。冻结人群全部候选后，正式评价完整 train、val、test。


<a id="doc-f8672d277241-评分与推进门禁"></a>

#### 评分与推进门禁

正式预测使用 Torch 2.13.0 CUDA 13.0，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，历史估计与未来积分均在 GPU float64 上执行。未来段使用现有 CUDA graph。拟合与 NumPy 真值生成使用 CPU。进入运行前验证 CUDA 和 CPU 的三次历史迭代、切线及未来预测一致性。

每个人群评价六个方程，包括真参数 GN、A、B 完整支持和三种删项方案。每份名单两档 2.5/1.25 ms 推到 8 秒。各档出现数值批失败时逐例隔离一次。逐事件记录 1/2/3/5/8 秒误差、历史残差、隐态误差与储能、历史设计条件数及 8 秒稳定性。稳定性报告完整准确率、可判定准确率和覆盖率、TP/TN/FP/FN、失稳召回、漏判和稳定误报率。未决、未覆盖和数值失败各自记录。零分母记 `null`。

- 真值两步长的可见归一化最大差不超过 \(10^{-6}\)，候选不超过 \(10^{-5}\)，COI 误差不超过 \(10^{-10}\)，分类一致且触发时刻至多相差两帧。GN 要求秩十、条件数不超过 \(10^8\)。
- 真参数对照在 train、val 的合并 3 秒归一化 RMSE 不超过 0.001，每例历史归一化最大误差不超过 \(10^{-7}\)，历史起点与预测起点隐态 RMSE 不超过 \(10^{-4}\)。两名单各包含安定与失稳事件，每例真值可判定且预测起点前无失步。真值与预测的标签及触发时刻符合两帧要求。
- 选中 B 在 train 和 val 的合并 3 秒与 8 秒归一化 RMSE 分别不超过 0.01 和 0.03。在完整 val 的每 100 ms 真值网格上，恢复力和可见耗散力的相对 RMS 误差分别不超过 3% 与 5%。记忆核在固定 32 点对数时延网格 0.01 至 3 秒上，相对 Frobenius RMS 误差不超过 5%。初始锚点至少一项函数误差超过该门槛。无新增项人群的 A 通过同一轨迹与函数门。
- B 的真值缺项对应幅值精确为零，真值存在项对应幅值相对误差不超过 10%。先按 val 预测选择，再检查真值支持。记录实际非零和声明支持。
- 六个冻结方程在 val 诊断网格的能量平衡相对误差不超过 \(10^{-10}\)，最小总耗散不低于 \(-10^{-12}\)，平衡曲率大于 \(10^{-8}\)。工程测试覆盖势能、PSD、解析切线和六组参数的完整历史链梯度。参数梯度工程检查使用 \(2\times10^{-5}\) 中心差分和 2% 相对容差，并保留绝对容差。

三个合成人群通过上述门槛后登记真实比较。test 全名单结果用于报告。合成结果支持在声明激励上的函数恢复判断，真实系数辨识和故障位置外推由后续独立证据检验。


<a id="doc-f8672d277241-调用预算"></a>

#### 调用预算

一次运行，真实数据、缓存和旧检查点读取为零。三个人群最多 15 次拟合、2160 次更新和反传、108 次完整 val 检查。真值生成最多 233,280 批 NumPy RHS。训练和监测最多 2268 批前向，每批 5 次传播、1120 次 Torch RHS、240 次切线 RHS，QR、SVD、投影乘法及三角求解各按事件计 3 次。两档正式评价含一次逐例隔离的上限为 6,300 次传播、25,401,600 次 Torch RHS、907,200 次切线 RHS 和每种分解 6,912 次。21 次函数诊断各计一次 Torch RHS，并记录 val 网格的曲率谱调用。

## 报告与结果

<a id="doc-afec4d4ead3a"></a>

<a id="doc-afec4d4ead3a-逐机方程与函数项恢复合成结果"></a>

### 逐机方程与函数项恢复：合成结果

2026-11-14。

完整 val 为三组合成真值都选对了函数项。所选方程的记忆核相对误差均超过 5%，调制耗散组的长时轨迹也未达到门槛。本轮保留默认基线。


<a id="doc-afec4d4ead3a-模型与范围"></a>

#### 模型与范围

A 学习 74 个共享系数，包括 45 个边刚度、9 个原有平衡基阻尼系数、10 个局部端口幅值和 10 个记忆时间常数。B 从同一起点增加二阶谐波恢复力 `a` 和状态调制耗散 `b/8`，两个幅值均从精确零开始。谐波项来自势能，调制耗散矩阵为半正定。

合成范围为 `null`、`harmonic`、`damping` 三个人群，每个人群按固定名单划分 train 16、val 8、test 8。每个人群评价六个冻结方程：真参数 `known`、A、完整 B、仅谐波、仅耗散、无新增项；删项方案从完整 B 的 val 最优点重拟合。拟合在 CPU float64 单线程进行，正式历史预测与轨迹积分在 CUDA float64 上进行。每个结果均包含 2.5 ms 与 1.25 ms 预测步长。


<a id="doc-afec4d4ead3a-轨迹误差与稳定性"></a>

#### 轨迹误差与稳定性

NRMSE 是协议定义的合并可见状态归一化 RMSE。稳定性为轨迹派生标签；混淆计数顺序为 TP/TN/FP/FN。表中同时列出完整名单准确率、可判定准确率、失稳召回率、漏判率、稳定误报率，以及覆盖数、未决、未覆盖、数值失败和未稳定预测计数。

| 人群 | 方程 | split | 例数 | 1 s NRMSE | 3 s NRMSE | 8 s NRMSE | 完整准确率 | 可判定准确率 | 失稳召回 | 漏判率 | 稳定误报 | TP/TN/FP/FN | 覆盖数/覆盖率 | 未决 | 未覆盖 | 数值失败 | 未稳定预测 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| null | known | train | 16 | 5.1479e-11 | 1.2531e-10 | 1.0939e-10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| null | known | val | 8 | 6.7924e-11 | 1.6705e-10 | 2.0815e-10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | known | test | 8 | 7.1495e-11 | 1.7405e-10 | 1.8025e-10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | a | train | 16 | 0.00048381 | 0.0018025 | 0.0027354 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| null | a | val | 8 | 0.0020525 | 0.0062334 | 0.023848 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | a | test | 8 | 0.0029384 | 0.010201 | 0.037074 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | both | train | 16 | 0.00055394 | 0.0024475 | 0.0046455 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| null | both | val | 8 | 0.0023524 | 0.0066626 | 0.026287 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | both | test | 8 | 0.0031167 | 0.0084347 | 0.011578 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | harmonic | train | 16 | 0.00016806 | 0.00058467 | 0.00095903 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| null | harmonic | val | 8 | 0.0007451 | 0.0021562 | 0.0091047 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | harmonic | test | 8 | 0.0010637 | 0.0037478 | 0.012322 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | damping | train | 16 | 0.00026667 | 0.0010522 | 0.0017057 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| null | damping | val | 8 | 0.0011432 | 0.0033855 | 0.0091137 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | damping | test | 8 | 0.0016467 | 0.0068766 | 0.024937 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | neither | train | 16 | 0.00016014 | 0.00057436 | 0.00098895 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| null | neither | val | 8 | 0.00067493 | 0.0019599 | 0.0083294 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| null | neither | test | 8 | 0.00089478 | 0.0031391 | 0.0099383 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | known | train | 16 | 1.2245e-10 | 2.619e-10 | 2.1339e-10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | known | val | 8 | 1.2002e-10 | 1.9589e-10 | 3.3445e-10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | known | test | 8 | 2.6186e-10 | 5.7513e-10 | 1.1772e-09 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | a | train | 16 | 0.027834 | 0.098519 | 0.13232 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | a | val | 8 | 0.053936 | 0.17824 | 0.94058 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | a | test | 8 | 0.02905 | 0.11817 | 0.19442 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | both | train | 16 | 0.0036005 | 0.011977 | 0.017754 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | both | val | 8 | 0.0089883 | 0.024742 | 0.067553 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | both | test | 8 | 0.012744 | 0.073147 | 0.16678 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | harmonic | train | 16 | 0.00035393 | 0.0013766 | 0.0016296 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | harmonic | val | 8 | 0.00098752 | 0.002925 | 0.0061246 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | harmonic | test | 8 | 0.0019767 | 0.011277 | 0.05666 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | damping | train | 16 | 0.0094898 | 0.032153 | 0.047919 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | damping | val | 8 | 0.037766 | 0.127 | 0.96401 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | damping | test | 8 | 0.028334 | 0.20713 | 0.76396 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | neither | train | 16 | 0.0094971 | 0.032144 | 0.047895 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | neither | val | 8 | 0.037074 | 0.12469 | 0.96316 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| harmonic | neither | test | 8 | 0.028163 | 0.20591 | 0.75956 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | known | train | 16 | 1.7097e-09 | 3.908e-09 | 3.5626e-09 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| damping | known | val | 8 | 2.9485e-09 | 5.2459e-09 | 4.9471e-09 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | known | test | 8 | 4.1293e-09 | 8.1509e-09 | 7.5359e-09 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | a | train | 16 | 0.27529 | 0.5771 | 0.9527 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| damping | a | val | 8 | 0.48126 | 0.95571 | 1.6748 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | a | test | 8 | 0.3764 | 0.98413 | 2.0929 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | both | train | 16 | 0.027252 | 0.050902 | 0.070795 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| damping | both | val | 8 | 0.29714 | 0.51353 | 0.43968 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | both | test | 8 | 0.1243 | 0.20862 | 0.21352 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | harmonic | train | 16 | 0.045615 | 0.08192 | 0.11652 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| damping | harmonic | val | 8 | 0.57639 | 0.93365 | 1.1192 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | harmonic | test | 8 | 0.23122 | 0.28688 | 0.23205 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | damping | train | 16 | 0.0055333 | 0.018014 | 0.028598 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| damping | damping | val | 8 | 0.049556 | 0.098219 | 0.085976 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | damping | test | 8 | 0.044826 | 0.11734 | 0.19065 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | neither | train | 16 | 0.045611 | 0.081932 | 0.11659 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 4/12/0/0 | 16/16 (100.0%) | 0 | 0 | 0 | 0 |
| damping | neither | val | 8 | 0.57611 | 0.93278 | 1.1171 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| damping | neither | test | 8 | 0.23098 | 0.28655 | 0.23189 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 2/6/0/0 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |


<a id="doc-afec4d4ead3a-函数恢复与物理门禁"></a>

#### 函数恢复与物理门禁

| 人群 | 选择的 B 支持 | 真值幅值 (a, b/8) | 选择幅值 (a, b/8) | 恢复力误差 | 耗散力误差 | 记忆核误差 | 刚度参数误差 | 阻尼矩阵误差 | 端口误差 | 时间常数误差 | 能量误差 | 最小耗散 | 最小曲率 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| null | neither | 0, 0 | 0, 0 | 0.002611 | 0.0037916 | 0.071183 | 0.0081251 | 0.0045995 | 0.066431 | 0.0070458 | 6.5161e-16 | 0.0065948 | 1.0349 |
| harmonic | harmonic | 0.6, 0 | 0.58322, 0 | 0.011306 | 0.0062682 | 0.1201 | 0.029783 | 0.0091586 | 0.10496 | 0.032823 | 5.5417e-16 | 0.012103 | 2.3485 |
| damping | damping | 0, 1 | 0, 0.93259 | 0.082946 | 0.027466 | 0.55229 | 0.22044 | 0.097554 | 0.39088 | 0.45687 | 6.0056e-16 | 0.0069201 | 1.0342 |

预先登记门禁逐项如下：

门槛按协议执行：真参数 train/val 合并 3 s NRMSE ≤0.001，历史最大归一化误差 ≤1×10⁻⁷，起点和预测起点隐态 RMSE ≤1×10⁻⁴；预测稳定性类别一致且失步触发相差不超过两帧。所选 B 的 train/val 合并 3 s、8 s NRMSE 分别 ≤0.01、≤0.03；val 恢复力、可见耗散力、记忆核相对误差分别 ≤3%、≤5%、≤5%。支持项幅值误差 ≤10%，缺失真值项保持精确零；能量误差 ≤1×10⁻¹⁰、最小耗散 ≥−1×10⁻¹²、平衡曲率 >1×10⁻⁸。无新增项人群的 A 另按相同轨迹与函数门检查。

| 人群 | 初始函数差异 | 真参数 train/val 对照 | 选择训练有效 | train/val 轨迹 | 函数恢复 | 支持与幅值 | 物理检查 | null 人群 A 恢复 | 科学门禁总结果 |
|---|---|---|---|---|---|---|---|---|---|
| null | 通过 | 通过 | 通过 | 通过 | 未通过 | 通过 | 通过 | 未通过 | 未通过 |
| harmonic | 通过 | 通过 | 通过 | 通过 | 未通过 | 通过 | 通过 | 通过 | 未通过 |
| damping | 通过 | 通过 | 通过 | 未通过 | 未通过 | 通过 | 通过 | 通过 | 未通过 |


<a id="doc-afec4d4ead3a-拟合轨迹与运行记录"></a>

#### 拟合轨迹与运行记录

`training.jsonl` 保存逐更新 train 目标；`selection.json` 保存每 20 步完整 val 检查和早停状态。B 初始拟合上限 240 步，删项重拟合上限 80 步。以下列出实际更新数、所选更新数及所选 val 目标，并链接两条记录。

| 人群 | 拟合器 | 更新数 | 选中更新 | 选中 val NMSE | train 曲线 | val 曲线 |
|---|---|---:|---:|---:|---|---|
| null | a | 240 | 240 | 2.5502e-06 | [null/a/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [null/a/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| null | both | 240 | 240 | 3.423e-06 | [null/both/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [null/both/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| null | harmonic | 80 | 80 | 3.3666e-07 | [null/harmonic/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [null/harmonic/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| null | damping | 80 | 80 | 8.1779e-07 | [null/damping/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [null/damping/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| null | neither | 80 | 80 | 2.759e-07 | [null/neither/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [null/neither/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| harmonic | a | 100 | 40 | 0.0016626 | [harmonic/a/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [harmonic/a/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| harmonic | both | 240 | 240 | 5.4581e-05 | [harmonic/both/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [harmonic/both/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| harmonic | harmonic | 80 | 80 | 5.9244e-07 | [harmonic/harmonic/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [harmonic/harmonic/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| harmonic | damping | 80 | 80 | 0.00080807 | [harmonic/damping/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [harmonic/damping/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| harmonic | neither | 80 | 80 | 0.00078052 | [harmonic/neither/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [harmonic/neither/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| damping | a | 80 | 20 | 0.16117 | [damping/a/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [damping/a/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| damping | both | 240 | 240 | 0.064616 | [damping/both/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [damping/both/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| damping | harmonic | 80 | 60 | 0.2508 | [damping/harmonic/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [damping/harmonic/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| damping | damping | 80 | 80 | 0.001688 | [damping/damping/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [damping/damping/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |
| damping | neither | 80 | 60 | 0.2506 | [damping/neither/training.jsonl](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) | [damping/neither/selection.json](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md) |

执行在 CPU 完成拟合和 NumPy 真值积分，在 GPU 完成历史预测与正式轨迹评分。实耗计数和登记上限如下；运行时长、解释器及库版本来自保存的运行摘要。

| 操作 | 实耗 | 登记上限 |
|---|---:|---:|
| curvature_spectra | 13608 | 13608 |
| fits | 15 | 15 |
| function_checks | 21 | 21 |
| gradients | 1860 | 2160 |
| matrix_products | 94968 | 113184 |
| numpy_rhs | 233280 | 233280 |
| optimizer_updates | 1860 | 2160 |
| qr | 94968 | 113184 |
| rollouts | 10305 | 17640 |
| svd | 94968 | 113184 |
| tangent_rhs | 546480 | 1451520 |
| torch_rhs | 4364661 | 27941781 |
| triangular_solves | 94968 | 113184 |
| val_checks | 93 | 108 |

运行时长：1868.95 秒。运行环境：Python 3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]；NumPy 2.5.2；SciPy 1.18.0；Torch 2.13.0+cu130；CUDA NVIDIA GeForce RTX 4060 Ti。真实数据读取数为 0。

运行目录：[`machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa`](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md)。来源提交：`2cf72ad4c6e99732375ebeee3c8f022c90ca4b90`。执行协议：[machine_local_equation_v1_zh.md](#doc-f8672d277241)。


<a id="doc-afec4d4ead3a-证据范围"></a>

#### 证据范围

test 在模型冻结后评分。保存结果核对涵盖逐事件标量汇总、函数项选择、调用预算及文件哈希。真实 TASK3 训练的前置条件为通过合成恢复门。

- [独立结果审查](_sealed/machine_local_equation_result_review_20261114.json)
- [保存结果核对](../../../../results/backup/08_machine_local_topology/02_s1_synthetic_learning_saved_checks.DESCRIPTION.md)

## 回执与附件

- [machine_local_equation_20261114_evidence.json](_sealed/machine_local_equation_20261114_evidence.json)
- [machine_local_equation_result_review_20261114.json](_sealed/machine_local_equation_result_review_20261114.json)
- [machine_local_equation_review_20261114.json](_sealed/machine_local_equation_review_20261114.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_equation_v1_zh.md](_sealed/machine_local_equation_v1_zh.md)
- [docs/reports/08_machine_local_topology/_sealed/machine_local_equation_20261114_zh.md](_sealed/original/machine_local_equation_20261114_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_equation_20261114_zh.md](_sealed/machine_local_equation_20261114_zh.md)
- [docs/research/machine_local_joint_equation_plan_20261114_zh.md](_sealed/machine_local_joint_equation_plan_20261114_zh.md)

</details>
