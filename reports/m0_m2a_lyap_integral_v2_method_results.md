# 基于积分弱形式与端口哈密顿约束的故障后降阶动力学：方法与单 seed 确认实验报告

**项目：** `kron-reduce`  
**数据：** canonical IEEE39 post-clear dataset  
**报告日期：** 2026-07-24  
**证据状态：** 方法、五折单-seed（324）完整运行、独立指标复算、积分收敛审计和证书审计均已完成；本轮判定为 **REJECT**

## English abstract

Accurate causal reintegration of post-clear power-system dynamics requires a correct forcing convention before additional model complexity is introduced. We revised the M0, M2A and Lyapunov-constrained models by separating equilibrium mechanical power, clearing-time mechanical power and the observed post-clear power schedule. All swing equations use the centred input `Delta Pm = Pm(t) - Pm_equilibrium` in mass-weighted centre-of-inertia coordinates. M0 and static M2A were reformulated as single weak-form constrained regression problems over 1-, 5- and 20-step windows. A causal first-order governor, dynamic edge-memory closure and augmented port-Hamiltonian residual were evaluated in a registered five-fold, single-seed confirmation run. The deterministic correction passed its engineering gate: five-fold validation mean RMSE decreased by 96.9% relative to the old M0 reference and the maximum COI drift was $1.33\times10^{-15}$. However, static M2A was rejected back to M0 in every fold and dynamic M2A was never adopted. The certified port-Hamiltonian candidate improved the H=100 test pooled omega $R^2$ from 0.5276 (M0) to 0.5998, but remained far below the target. The final predicted-trajectory classifier produced 0 true positives and 51 false negatives (FSR=1.0). An interpretation-only audit further showed that the same rule also labels every truncated true unstable trajectory as stable. Consequently, the round is rejected: numerical integration is convergent and the Lyapunov certificates pass, but neither reintegration accuracy nor instability detection meets the required objective.

## 摘要

本工作研究 canonical IEEE39 数据集上的故障清除后因果降阶动力学重积分。旧版结果显示，M0、M2A 和能量正则模型在长视界闭环积分中均存在明显误差，其中 M0 在主 ZOH 条件下的 H=100 mean RMSE 为 9.695。代码审计表明，首要问题不是 SINDy 求解轮数不足，而是机械功率口径不一致：电气耦合项以平衡点为中心，而机械功率仍以绝对 COI 功率进入方程。为此，本轮统一使用

\[
\Delta P_m(t)=P_m(t)-P_{m,\mathrm{eq}},
\]

并明确区分平衡点功率 `Pm_equilibrium`、清除时刻功率 `Pm_clear` 与条件观测日程 `Pm(t)`。M0 和静态 M2A 被改写为积分弱形式的一次约束求解；多视界闭环训练只保留为验证集晋级的候选后处理。进一步实现了因果一阶 governor、动态边记忆闭包和具有显式供能恒等式的端口哈密顿 LYAP 模型。五折、单 seed（324）完整运行表明：确定性修复门通过，五折 validation mean RMSE 相对旧 M0 下降 96.9%，最大 COI 漂移为 $1.33\times10^{-15}$；但静态 M2A 五折全部回退 M0，动态 M2A 五折均未采用。通过全部证书的端口哈密顿候选将 H=100 test pooled omega $R^2$ 从 M0 的 0.5276 提高到 0.5998，仍未达到目标。最终预测轨迹的 TP=0、FN=51，FSR=1.0；解释性审计还发现，相同规则对截断的真实失稳轨迹同样得到 FSR=1.0。因此本轮判定为 **REJECT**：方程和积分器修复有效，但当前残差库、长视界精度和分类视界均不足。

## 1. 问题定义与设计原则

### 1.1 任务与因果输入

预测从故障清除的精确时刻 $t=0$ 开始。主模型只接收清除状态 $(\delta_0,\omega_0)$、平衡点、惯量与图先验，以及预注册的机械功率策略。输出为未来转子角与转速轨迹。电气功率 `Pe`、电压、稳定标签、故障线路、事件时间、有效长度及未来统计均不作为主模型输入或选择变量。

结构保持电力系统模型强调网络耦合、能量与动力学坐标之间的一致性，这也是本工作保留 incidence 结构和内部力平衡的物理依据 [[1]](https://doi.org/10.1109/TPAS.1981.316883)。所有模型均在质量加权的惯性中心（centre of inertia, COI）坐标中计算，并使用

\[
m_i=M_i/\omega_s
\]

作为摆动方程中的有效质量。

### 1.2 证据分层

本报告区分三类证据：

1. **旧版完整运行证据：** 已封存的五折、三随机种子测试结果，用于描述原始问题。
2. **新版工程验收证据：** 本轮 F1 训练/验证诊断，用于验证确定性方程修复。
3. **确认性证据：** 新版五折、单 seed（324）完整运行，以及由此生成的 fault-line cluster bootstrap、governor 跨折门、LYAP 证书门、最终轨迹稳定性结果和独立复算证据。

F1 结果只用于确定性工程验收；最终接受判定完全依据五折冻结选择后的 test 结果和独立复算。

### 1.3 确认实验的冻结谱系

| 项目 | 冻结值 |
|---|---|
| 工作分支 | `codex/integral-v2-round1` |
| 方法实现提交 | `88c9314a07e786c8fc6c8b9b0af23f9c6ae64a59` |
| 单-seed 配置提交 | `474cfd117a630adfc4b1b96e7dfcedcab6402cfc` |
| 注册实验 | `paper_v2_integral_single_seed` |
| 配置文件 | `configs/experiments/paper_integral_v2_single_seed.toml` |
| folds / seed | `F1`–`F5` / `324` |
| 不可变完整运行 | `paper_v2_integral_single_seed-full_run-20260724T081902Z-474cfd11` |

本轮只允许上述 run 进入最终确认结论。早期 smoke run 只证明流水线可执行；按原三-seed配置启动后被用户要求停止的未完成目录继续保留，但不得用于性能、证书或接受判定。运行时不会覆盖任何历史结果目录。

## 2. 方法

### 2.1 机械功率语义与 COI 处理

数据适配层提供以下三个静态量或日程：

- `Pm_equilibrium`：平衡点 COI 机械功率；
- `Pm_clear`：故障清除时刻的机械功率；
- `Pm0`：暂时保留的兼容别名，语义等同于 `Pm_clear`；
- `exogenous["Pm"]`：COI 平衡后的机械功率日程。

角度与转速通过质量加权投影

\[
\Pi_{\mathrm{COI}}x=x-\frac{m^\top x}{\mathbf 1^\top m}\mathbf 1
\]

消除共同模态；功率通过

\[
\Pi_P p=p-\frac{\mathbf 1^\top p}{\mathbf 1^\top m}m
\]

消除净加速功率。两种投影均设计为幂等操作，因此无需修改 canonical 文件即可兼容已中心化和历史数据。COI 商空间的正交基由质量向量 $m$ 的零空间构造，不再使用仅适合质量归一化坐标的 $\sqrt m$。

### 2.2 统一闭环积分器

NumPy 推理与 Torch 候选微调共用相同的四阶 Runge–Kutta 定义。每个观测间隔被细分为不大于 0.01 s 的内部步长。ZOH 模式在所有 RK 阶段严格使用 `Pm_clear`；条件 observed 模式在 RK 阶段对相邻机械功率样本线性插值。每个阶段投影 COI 加速度，并在接受子步后重新中心化角度和速度。数值审计比较 `dt`、`dt/2` 与 `dt/4` 的轨迹或能量平衡误差。

### 2.3 M0：一次积分弱形式辨识

修正后的 M0 为

\[
m\dot\omega=\Delta P_m-D\omega-
B\left[K\odot\left(\sin(B^\top\delta)-\sin(B^\top\delta_{\mathrm{eq}})\right)\right].
\]

为避免直接对测量轨迹求导，在真实训练轨迹的窗口 ([t_a,t_b]) 上积分：

\[
m(\omega_b-\omega_a)=
\int_a^b\Delta P_m\,dt-
D\int_a^b\omega\,dt-
BK\int_a^b\Delta\sin\eta\,dt.
\]

窗口长度固定为 1、5 和 20 步。所有窗口被堆叠为一个线性方程组；它们不是不同 epoch，也不产生递归预测状态。设计矩阵只用训练集列尺度归一化，随后以非负 ridge/约束最小二乘一次求解 $D,K\ge0$。ridge 候选为 $10^{-8},10^{-6},10^{-4},10^{-2}$，只根据验证集闭环误差选择。积分弱形式与弱 SINDy 的共同动机是避免点导数放大高频噪声，同时保留可解释的稀疏回归问题 [[2]](https://doi.org/10.1137/20M1343166)。

### 2.4 M2A：一步图 SINDy 与验证门控微调

M2A 冻结选定的 M0，在真实训练轨迹上拟合力空间残差。静态库包括：

\[
\text{node}:\quad \omega|\omega|,\;\Delta P_m\omega,
\]

\[
\text{edge}:\quad \Delta\sin\eta,\;\Delta\omega_e,\;
\Delta\omega_e\Delta\sin\eta.
\]

节点力经过零总力投影，边力通过 incidence 矩阵映射，以保持内部力平衡。模型先以 ridge 和 sequential thresholded ridge regression（STRidge）获得 `one_step_solution`。这一过程延续了 SINDYc 将外部输入纳入稀疏动力学辨识的思想 [[3]](https://doi.org/10.1016/j.ifacol.2016.10.249)，但主求解目标采用积分弱形式而非点导数。

只有当一次解相对修正后 M0 的验证综合误差至少下降 2% 时，M2A 才被晋级；否则残差被回退为零。可选 `rollout_refined` 从一次解复制非零参数，在固定稀疏支撑下优化

\[
L=L_{\mathrm{rollout}}+\lambda\|\theta-\theta_{\mathrm{one-step}}\|_2^2.
\]

候选视界为 5、20、40 和 100 步，并使用 AdamW、梯度裁剪、学习率衰减和验证早停。该优化不是对同一线性方程重复求解，而是针对部署时的闭环分布偏移求解不同目标。

### 2.5 动态边记忆闭包

降阶模型省略的电气和控制状态可能产生非 Markov 闭包；一般降阶闭包需要记忆项的观点已有系统讨论 [[4]](https://doi.org/10.1137/18M1177263)。本工作实现

\[
\tau_e\dot z_e=-z_e+a_e\Delta\sin\eta_e+b_e\Delta\omega_e,
\]

\[
r_e=c_ez_e+d_e\Delta\omega_e+
e_e\Delta\omega_e\Delta\sin\eta_e.
\]

记忆初态由清除状态的准稳态表达式确定，不读取真实未来状态。时间常数从预注册网格选择，线性可辨识组合通过积分弱形式求解。若动态候选未在验证集改善至少 2%，部署自动回退到静态 M2A。

### 2.6 因果 governor 消融

主结果继续采用 `Pm_clear` 的零阶保持；observed `Pm(t)` 只作为条件重积分上界。调速器消融采用

\[
\tau_i\dot{\Delta P}_{m,i}=-\Delta P_{m,i}-g_i\omega_i,
\qquad \tau_i,g_i>0.
\]

令 \(\alpha_i=1/\tau_i\)、\(\beta_i=g_i/\tau_i\)，在训练集上使用积分约束最小二乘一次估计 \(\alpha_i,\beta_i\ge0\)。部署时仅使用 `Pm_clear`、`Pm_equilibrium` 和预测速度。该模型是 IEEE PES 调速器模型层级的低阶可辨识近似，不等同于详细汽轮机—调速器模型 [[5]](https://doi.org/10.17023/w0mn-xh56)。只有 H=100 验证 mean RMSE 改善至少 2%、至少四个折同向且失败率不增加时，才将其标记为有效消融。

### 2.7 LYAP：端口哈密顿动态记忆

旧版 `M2A-energy` 只叠加了与 M0 同形的阻尼和耦合项，表达能力有限。本轮将其替换为增广端口哈密顿模型：

\[
m\dot\omega=\Delta P_m-B\nabla_\eta V-D_n\omega
-B(D_e\odot B^\top\omega)+B(C\odot z),
\]

\[
\dot z=-\Lambda\odot z-C\odot B^\top\omega,
\]

其中 $D_n,D_e,\Lambda,\kappa>0$。储能函数为

\[
H=\frac12\omega^\top m\omega+
\sum_e\kappa_e\left[
\cos\eta_{e,\mathrm{eq}}-\cos\eta_e-
\sin\eta_{e,\mathrm{eq}}(\eta_e-\eta_{e,\mathrm{eq}})
\right]+\frac12z^\top z.
\]

交叉项在 \(\dot H\) 中结构抵消，得到

\[
\dot H=\omega^\top\Delta P_m-
\omega^\top D_n\omega-
(B^\top\omega)^\top D_e(B^\top\omega)-
z^\top\Lambda z.
\]

这种设计与端口哈密顿学习模型显式表示外部供能和耗散的思路一致 [[6]](https://doi.org/10.1103/PhysRevE.104.034312)。数值门依次检查：平衡点 $H=0$、梯度为零、正确 COI 商空间 Hessian 严格正定、解析与数值链式法则一致、供能不等式无违例、人工反阻尼违例能够被拒绝、离散能量误差随步长收敛、轨迹精度不劣于修正后 M0且无失败。只有全部通过时才使用 `M2A-LYAP`；否则依次降级为 `M2A-storage` 或 `M2A-energy-regularized`。

### 2.8 训练、选择与统计

所有归一化、ridge、稀疏阈值、早停和 governor 参数仅使用 train/validation。M0、静态 M2A、动态记忆的线性阶段和 governor 不设置 epoch，只记录求解器迭代次数。确认实验固定使用 seed 324；测试事件只能在五折的 selection marker 全部写入后读取。

轨迹误差同时报告 mean、median、pooled RMSE、pooled $R^2$、最坏 10% CVaR 和 failure rate。最终 M2A–M0 差异按 fault-line cluster bootstrap 计算 95% 置信区间；只有 RMSE 差值的置信区间上界小于零时才声称显著改善。稳定性标签只由预测轨迹事后推导，不参与动力学选择。

## 3. 结果

### 3.1 旧版完整运行揭示长视界闭环失配

旧版不可变运行 `paper_v1_full-full_run-20260722T221756Z-294ba849` 完成了 75 次模型拟合，另有 15 次 SINDYc 拟合在闭环积分中产生非有限值。H=100 主 ZOH 测试结果如下。

| 方法 | population | mean RMSE | median RMSE | failure rate |
|---|---:|---:|---:|---:|
| LSTM | 792 | 0.699 | 0.498 | 0 |
| PINN | 792 | 4.002 | 3.675 | 0 |
| M2A | 792 | 5.328 | 5.308 | 0 |
| M0 | 792 | 9.695 | 9.655 | 0 |
| M2A-energy | 792 | 9.695 | 9.655 | 0 |
| SINDYc | — | — | — | 15/15 fits failed |

M2A 虽优于原始 M0，但误差仍显著高于 LSTM。旧版能量模型与 M0 的结果几乎相同，说明其残差结构主要重复了 M0，而未形成有效的新闭包。旧版所有 15 个能量证书门均失败；投影 Hessian 的最小特征值接近零或为负，因此报告名称正确保持为 `M2A-energy-regularized`，而非 `M2A-LYAP`。

observed `Pm(t)` 并未系统性修复旧版模型：旧 M0 的 H=100 conditional mean RMSE 为 9.764，略高于 ZOH 的 9.695；旧 M2A 分别为 5.375 和 5.328。这表明在方程口径错误存在时，比较 Pm 策略不能可靠回答“实时 Pm 是否必要”。

### 3.2 修正后的 M0 通过 F1/H=100 工程门

新版 M0 在 F1 中使用 178 个训练事件和 34 个验证事件，以 1、5、20 步积分窗构造一次弱形式方程，并从四个 ridge 候选中选择 (10^{-8})。H=100 验证结果为：

| 指标 | 数值 |
|---|---:|
| mean RMSE | 0.36898 |
| median RMSE | 0.07685 |
| maximum case RMSE | 1.76025 |
| weak-form residual RMSE | 0.05995 |
| maximum COI drift | $4.44\times10^{-16}$ |

mean RMSE 低于预注册工程门 0.5，COI 漂移低于 $10^{-10}$。完整五折 validation 的 raw mean RMSE 为 0.3016，相对旧 M0 参考 9.695 降低 96.9%，最大 COI 漂移为 $1.33\times10^{-15}$，确定性修复门正式通过。

该结果直接支持两个判断。第一，机械功率必须先减去平衡点功率，再与以平衡点中心化的电气耦合项组合。第二，SINDy 主问题在给定库和弱形式后是一次线性/约束求解；增加 epoch 不能修复错误的方程口径。

### 3.3 软件与物理一致性验证

最终测试套件共 24 项，全部通过。新增测试覆盖：

- `Pm_equilibrium`/`Pm_clear` 语义和 \(\Delta P_m\) 中心化；
- COI 投影、质量零空间、共同角平移不变性和内部力平衡；
- 合成已知参数系统上的积分弱形式恢复；
- NumPy/Torch RK4 一致性、非均匀首步和步长收敛；
- 动态记忆与 save/load 可复现；
- governor 不读取未来机械功率；
- 端口哈密顿供能恒等式和人工反阻尼违例拒绝；
- 原有禁止字段、数据 split 和 test 隔离规则。

`project_harness audit` 与 `data-check` 均通过；canonical 数据检查确认 315 个事件和数据构建标识保持不变。未覆盖或修改历史结果目录，也未修改用户已有的 `reports/full_run_report.md`。

### 3.4 五折单-seed完整重积分结果

主 ZOH 条件下，H=100 的 test 结果如下。population=264，因为只有这些稳定事件具有至少 100 个真实有效步。

| 方法 | mean RMSE | median RMSE | pooled RMSE | omega $R^2$ | delta $R^2$ | worst-10% CVaR | failure rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.3480 | 0.2971 | 0.4569 | 0.5276 | 0.8886 | 0.9149 | 0 |
| M2A | 0.3480 | 0.2971 | 0.4569 | 0.5276 | 0.8886 | 0.9149 | 0 |
| M2A-dynamic | 0.3480 | 0.2971 | 0.4569 | 0.5276 | 0.8886 | 0.9149 | 0 |
| M0-governor | 0.3360 | 0.2813 | 0.4442 | 0.5515 | 0.9037 | 0.9082 | 0 |
| M2A-LYAP | **0.3196** | **0.2678** | **0.4213** | **0.5998** | 0.8979 | **0.8532** | 0 |
| PINN | 0.3693 | 0.2886 | 0.4737 | 0.4965 | 0.8600 | 0.9652 | 0 |
| LSTM | 0.6694 | 0.4974 | 0.8533 | -0.6627 | 0.6822 | 1.7099 | 0 |

端口哈密顿候选相对 M0 将 mean RMSE 降低 8.17%，omega $R^2$ 绝对提高 0.0722，但仍远低于目标。FINAL 在五折均由 validation 选择该候选，其视界退化为：

| Horizon | population | mean RMSE | pooled omega $R^2$ | failure rate |
|---:|---:|---:|---:|---:|
| 20 | 280 | 0.1940 | 0.7666 | 0 |
| 40 | 264 | 0.2330 | 0.6232 | 0 |
| 100 | 264 | 0.3196 | 0.5998 | 0 |

### 3.5 静态与动态 M2A 未形成有效残差

静态 M2A 的一次弱形式候选在五折 validation 上相对冻结 M0 的改善分别为 0.95%、0.79%、0%、0.47% 和 0%，全部低于预注册 2% 晋级门。因而五折的最终静态残差支撑均为空，部署结果严格回退为 M0。动态边记忆五折的相对改善均为 0，亦全部回退。相应 test 上 M2A、M2A-dynamic 与 M0 完全相同；fault-line cluster bootstrap 得到配对效应和 95% CI 均为 0。这不是“显著改善失败但略有收益”，而是当前残差没有进入最终模型。

governor 在五折 validation 中有四折同向，但平均改善仅 1.49%，未达到门槛；因此它是负消融，不替代主 ZOH。

### 3.6 LYAP证书与积分收敛

五折端口哈密顿候选均通过全部证书门并可正式命名为 `M2A-LYAP`。各折的 COI 商空间 Hessian 最小特征值均为 0.13665，最大解析供能恒等式误差不超过 $2.13\times10^{-14}$，人工反阻尼违例均被拒绝，且无 rollout failure。

独立 RK4 审计在每折各选一个 validation 事件，对主 ZOH 与 observed-Pm 两种模式比较内部步长 0.01、0.005 和 0.0025 s，共 10 行全部收敛。粗/中与中/细误差比为 6.81–6.91，证明本轮长视界误差不是内部步长未收敛造成的。

### 3.7 最终轨迹稳定性结果与不可辨识审计

FINAL 在 315 个互不重复的 test 事件上得到：TP=0、FN=51、FP=0、TN=264，故 FSR=1.0、FUR=0、balanced accuracy=0.5。独立复算逐案例预测标签后得到完全相同的混淆矩阵。

更关键的是，post-selection解释性审计把同一轨迹阈值规则直接应用于真实 test 轨迹，也得到 TP=0、FN=51。全部 51 个失稳事件只保留 2–37 个有效步，在阈值越界之前轨迹已截断。因此，“按每个事件最后一个可用点分类”对当前数据天然不可辨识。后续若继续使用最终预测轨迹判稳，应从清除状态在统一、因果的固定视界上外推，而不是把真实 `valid_length` 作为分类终点；`valid_length` 仍不得进入模型输入。

### 3.8 接受判定

| 门 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| H=100 pooled omega $R^2$ | 0.59985 | $>0.9$ | FAIL |
| 最终预测轨迹 FSR | 1.0 | $<0.2$ | FAIL |
| rollout failure rate | 0 | $=0$ | PASS |

因此本轮严格判定为 **REJECT**。确定性 M0 修复和 LYAP 证书被保留为有效工程成果，但当前 M2A残差设计、长视界精度和分类视界不能被接受。

## 4. 讨论

### 4.1 为什么一次求解优先于重复训练

经典 SINDy 在候选库和真实轨迹确定后，本质上是稀疏回归问题。多轮训练只有在目标发生变化时才有意义，例如从真实状态上的一步/积分残差转向预测状态上的闭环 rollout 误差。若一次弱形式求解已经使同一目标达到最优，继续对同一目标迭代不能增加信息，反而可能因数值容差、噪声或验证选择产生不必要的漂移。本工作因而将一次解设为主模型，将多视界优化限制为固定支撑、带信赖项、验证改善至少 2% 的候选。

### 4.2 为什么仅训练下一时刻仍不充分

一步 teacher forcing 能提供稳定初始化，并避免早期预测误差污染特征库；但部署时模型输入来自自身预测，局部误差会改变后续非线性特征和功率平衡。一步最优与闭环长视界最优在模型失配、噪声或离散积分不一致时不是同一问题。因此，本方案没有取消闭环验证，而是取消了“无条件、固定轮数的闭环训练”。闭环结果决定模型是否晋级，但不改变 test 隔离规则。

### 4.3 实时 Pm 的解释边界

机械功率中心化修复后，才能公平比较 ZOH、observed 和 governor。observed `Pm(t)` 使用未来已知输入，只能解释为条件重积分上界，不是因果预测。ZOH 是最保守的主策略；governor 则用一个可辨识低阶 ODE 把训练期 observed 信息压缩为部署期可用的因果状态方程。若 governor 未通过跨折门，结论应是“当前低阶 governor 不足以替代 ZOH”，而不是任意提高 governor 阶数或使用未来 Pm。

### 4.4 动态记忆与 LYAP 的关系

动态记忆用于处理降阶后未解析状态导致的非 Markov 残差；LYAP 结构用于约束供能与耗散。两者解决的问题不同：记忆可以提高表达能力，却不自动提供稳定性证书；端口哈密顿结构提供可审计的供能关系，却不保证轨迹精度一定优于 M0。因此，LYAP 命名门同时要求数学证书和精度不退化，避免用“结构正确”替代“预测有效”。

## 5. 局限性与下一轮候选假设

1. 本轮只使用一个随机种子；五折覆盖了数据划分差异，但不能估计训练随机性的跨 seed 方差。
2. H=100 omega $R^2$ 只覆盖 264 个具有完整 100 步真实轨迹的稳定事件，不能代表 51 个截断失稳事件的长期重积分精度。
3. 当前分类终点跟随真实 `valid_length`，而全部失稳事件在阈值越界前终止，因此连真实轨迹都无法通过现有规则恢复标签。FSR=1.0 既反映模型失败，也反映评估视界定义失败。
4. 静态残差库在五折都没有越过 2% validation 门；动态记忆也没有改善。仅扩大 epoch 无法解决表达能力和闭环可辨识性不足。
5. 一阶 governor 是辨识性优先的低阶消融，不代表详细汽轮机、阀门、限幅和死区动力学。
6. 学得的 $K$ 是当前降阶库中的有效参数，不应解释为真实 Kron 网络参数。

若用户批准下一轮，建议保持严格的 leakage-free 五折口径，并按每折冻结本轮 validation 选定的 M0 checkpoint，而不是用一个跨折全数据 checkpoint。随后只学习投影到 COI 零总力空间的 M2A 残差，并同时解决两个问题：

- **残差表达：** 在冻结 M0 上增加可解释的角度依赖刚度修正、状态依赖阻尼和低阶因果记忆；先用积分残差筛选支撑，再用固定支撑的多视界闭环目标校正幅值。
- **分类视界：** 对所有事件从清除状态统一因果外推到固定 H=100；真实可用段只用于轨迹误差，预测的完整固定视界用于事后稳定性判断。不得把标签、`valid_length` 或未来状态作为动力学输入。

该方案目前只是由本轮失败证据导出的候选假设，尚未创建分支或执行。按既定流程，本报告提交用户评估后，只有获得批准才启动下一轮。

## 6. 结论

本轮恢复了机械功率、平衡点、电气耦合和数值积分之间的一致性，并建立了零失败、数值收敛的 M0 基线。端口哈密顿残差通过五折全部证书，并把 H=100 omega $R^2$ 从 0.5276 提高到 0.5998；但静态和动态 M2A 残差均未被采用，最终精度仍低于目标，分类 FSR 为 1.0。独立审计确认这些结果不是汇总或积分步长错误，同时证明当前截断终点分类规则本身无法识别失稳。故本轮结论为 **REJECT**。后续应冻结每折 M0，只学习更有表达力且守恒投影的残差，并将稳定性评估改为统一固定视界的预测轨迹；在用户批准前不执行该方案。

## 参考文献

1. Bergen, A. R. & Hill, D. J. A structure preserving model for power system stability analysis. *IEEE Trans. Power Appar. Syst.* **PAS-100**, 25–35 (1981). [DOI](https://doi.org/10.1109/TPAS.1981.316883)
2. Messenger, D. A. & Bortz, D. M. Weak SINDy: Galerkin-based data-driven model selection. *Multiscale Model. Simul.* (2021). [DOI](https://doi.org/10.1137/20M1343166)
3. Brunton, S. L., Proctor, J. L. & Kutz, J. N. Sparse identification of nonlinear dynamics with control (SINDYc). *IFAC-PapersOnLine* **49**, 710–715 (2016). [DOI](https://doi.org/10.1016/j.ifacol.2016.10.249)
4. Pan, S. & Duraisamy, K. Data-driven discovery of closure models. *SIAM J. Appl. Dyn. Syst.* **17**, 2381–2413 (2018). [DOI](https://doi.org/10.1137/18M1177263)
5. IEEE PES Task Force on Turbine-Governor Modeling. *Dynamic Models for Turbine-Governors in Power System Studies* (2013). [DOI](https://doi.org/10.17023/w0mn-xh56)
6. Desai, S. A. *et al.* Port-Hamiltonian neural networks for learning explicit time-dependent dynamical systems. *Phys. Rev. E* **104**, 034312 (2021). [DOI](https://doi.org/10.1103/PhysRevE.104.034312)

## 附录 A：术语表

| 规范术语 | 本报告定义 | 禁止混用 |
|---|---|---|
| `Pm_equilibrium` | 平衡点 COI 机械功率 | 不得写成清除时 Pm0 |
| `Pm_clear` | 故障清除时机械功率 | 不得解释为平衡点功率 |
| observed `Pm(t)` | 条件重积分的已观测输入 | 不得称为因果预测输入 |
| 积分弱形式 | 对真实轨迹窗口积分后一次回归 | 不得称为多视界 epoch 训练 |
| `one_step_solution` | 一次弱形式/STRidge 主解 | 不等同于仅用单个离散下一步 |
| `rollout_refined` | 固定稀疏支撑的闭环候选 | 未过 2% 门不得替代主解 |
| `M2A-LYAP` | 全部数值证书及精度门通过的名称 | 不得用于仅有能量正则的模型 |

## 附录 B：主张—证据对应

| 主张 | 证据 | 状态 |
|---|---|---|
| 原始 M0 存在严重长视界误差 | 旧五折 H=100 ZOH mean RMSE 9.695 | 已支持 |
| Pm 中心化与统一积分显著改善 M0 | 五折 validation mean RMSE 下降 96.9%，COI drift $1.33\times10^{-15}$ | 已支持 |
| 当前静态 M2A 优于冻结 M0 | 五折均低于 2% 晋级门，test 与 M0 完全相同 | 不支持 |
| governor 可替代 ZOH | 平均 validation 改善 1.49%，未过门 | 不支持 |
| 当前动态记忆优于静态 M2A | 五折改善均为 0 | 不支持 |
| 模型可命名为 `M2A-LYAP` | 五折全部证书通过且精度不劣于 M0 | 已支持命名，但精度目标失败 |
| 本轮满足 omega $R^2$ 与 FSR 目标 | 0.59985 与 1.0 | 不支持；本轮 REJECT |
