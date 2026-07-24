# 基于积分弱形式与端口哈密顿约束的故障后降阶动力学：方法与阶段性结果报告

**项目：** `kron-reduce`  
**数据：** canonical IEEE39 post-clear dataset  
**报告日期：** 2026-07-24  
**证据状态：** 方法实现完成；旧版五折基线已封存；新版完成 F1 工程验收，尚未执行受门控的五折完整实验

## English abstract

Accurate causal reintegration of post-clear power-system dynamics requires a correct forcing convention before additional model complexity is introduced. We revised the M0, M2A and Lyapunov-constrained models by separating equilibrium mechanical power, clearing-time mechanical power and the observed post-clear power schedule. All swing equations now use the centered input, `Delta Pm = Pm(t) - Pm_equilibrium`, in mass-weighted centre-of-inertia coordinates. M0 and static M2A were reformulated as single weak-form constrained regression problems over 1-, 5- and 20-step windows, whereas multi-horizon rollout optimization was retained only as a validation-gated, support-preserving candidate. A causal first-order governor and a dynamic edge-memory closure were implemented as ablations. The former energy residual was replaced by an augmented port-Hamiltonian system with an explicit storage function and numerical certificate gates. On the F1 validation fold at a 100-step horizon, the corrected weak-form M0 achieved a mean trajectory RMSE of 0.3690 and a median RMSE of 0.0768 across 34 validation cases, with a maximum centre-of-inertia drift of $4.44\times10^{-16}$. This satisfies the predeclared F1 engineering gate of mean RMSE no greater than 0.5. These results establish the deterministic correction but do not yet establish five-fold superiority of M2A, governor dynamics or the Lyapunov-constrained model; those claims remain conditional on the registered full-run gates.

## 摘要

本工作研究 canonical IEEE39 数据集上的故障清除后因果降阶动力学重积分。旧版结果显示，M0、M2A 和能量正则模型在长视界闭环积分中均存在明显误差，其中 M0 在主 ZOH 条件下的 H=100 mean RMSE 为 9.695。代码审计表明，首要问题不是 SINDy 求解轮数不足，而是机械功率口径不一致：电气耦合项以平衡点为中心，而机械功率仍以绝对 COI 功率进入方程。为此，本轮统一使用

\[
\Delta P_m(t)=P_m(t)-P_{m,\mathrm{eq}},
\]

并明确区分平衡点功率 `Pm_equilibrium`、清除时刻功率 `Pm_clear` 与条件观测日程 `Pm(t)`。M0 和静态 M2A 被改写为积分弱形式的一次约束求解；多视界闭环训练只保留为验证集晋级的候选后处理。进一步实现了因果一阶 governor、动态边记忆闭包和具有显式供能恒等式的端口哈密顿 LYAP 模型。新版 M0 在 F1、H=100 的 34 个验证事件上获得 mean RMSE 0.3690、median RMSE 0.0768，最大 COI 漂移为 $4.44\times10^{-16}$，通过预设的 F1 工程门。该结果支持“先修正方程口径，再讨论模型复杂度”的结论；M2A、governor 和 `M2A-LYAP` 的五折性能与证书结论仍需独立完整运行确认。

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
3. **待完成的确认性证据：** 新版五折完整运行、fault-line cluster bootstrap、governor 跨折门和 LYAP 证书门。

因此，F1 结果不能替代五折测试结论，也不用于宣称新版 M2A 或 LYAP 已显著优于全部基线。

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

所有归一化、ridge、稀疏阈值、早停和 governor 参数仅使用 train/validation。M0、静态 M2A、动态记忆的线性阶段和 governor 不设置 epoch，只记录求解器迭代次数。测试事件只能在所有折和随机种子的 selection marker 写入后读取。

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

mean RMSE 低于预注册工程门 0.5，COI 漂移低于 (10^{-10})。若仅将该数值与旧版完整运行的 M0 mean RMSE 9.695 作尺度参照，降幅约为 96.2%。但两者对应不同证据层级：前者是 F1 validation 工程验收，后者是五折 test 聚合，因此该百分比不能作为正式的配对性能结论。正式结论仍需在新版五折完整运行中形成。

该结果直接支持两个判断。第一，机械功率必须先减去平衡点功率，再与以平衡点中心化的电气耦合项组合。第二，SINDy 主问题在给定库和弱形式后是一次线性/约束求解；增加 epoch 不能修复错误的方程口径。

### 3.3 软件与物理一致性验证

最终测试套件共 21 项，全部通过。新增测试覆盖：

- `Pm_equilibrium`/`Pm_clear` 语义和 \(\Delta P_m\) 中心化；
- COI 投影、质量零空间、共同角平移不变性和内部力平衡；
- 合成已知参数系统上的积分弱形式恢复；
- NumPy/Torch RK4 一致性、非均匀首步和步长收敛；
- 动态记忆与 save/load 可复现；
- governor 不读取未来机械功率；
- 端口哈密顿供能恒等式和人工反阻尼违例拒绝；
- 原有禁止字段、数据 split 和 test 隔离规则。

`project_harness audit` 与 `data-check` 均通过；canonical 数据检查确认 315 个事件和数据构建标识保持不变。未覆盖或修改历史结果目录，也未修改用户已有的 `reports/full_run_report.md`。

### 3.4 M2A、governor 与 LYAP 的当前证据状态

| 模块 | 实现状态 | 当前可支持结论 | 尚缺证据 |
|---|---|---|---|
| 静态 M2A | 完成 | 一次 STRidge 与固定支撑微调已实现 | 新版五折是否优于修正 M0 |
| 动态 M2A | 完成 | 因果准稳态初值、验证回退可用 | 记忆项是否在多数折稳定改善 |
| M0-governor | 完成 | 训练与部署输入满足因果约束 | H=100 改善≥2%、4/5 folds 同向 |
| 端口哈密顿 LYAP | 完成 | 结构供能恒等式与人工违例测试通过 | 数据域全证书、精度不退化和零失败 |
| `M2A-LYAP` 名称 | 门控完成 | 只有全部证书通过才会自动采用 | 尚无新版五折证书结果 |

因此，本报告不声称 M2A、governor 或 LYAP 已经超过新版 M0，也不提前使用 `M2A-LYAP` 作为确认性结果名称。

## 4. 讨论

### 4.1 为什么一次求解优先于重复训练

经典 SINDy 在候选库和真实轨迹确定后，本质上是稀疏回归问题。多轮训练只有在目标发生变化时才有意义，例如从真实状态上的一步/积分残差转向预测状态上的闭环 rollout 误差。若一次弱形式求解已经使同一目标达到最优，继续对同一目标迭代不能增加信息，反而可能因数值容差、噪声或验证选择产生不必要的漂移。本工作因而将一次解设为主模型，将多视界优化限制为固定支撑、带信赖项、验证改善至少 2% 的候选。

### 4.2 为什么仅训练下一时刻仍不充分

一步 teacher forcing 能提供稳定初始化，并避免早期预测误差污染特征库；但部署时模型输入来自自身预测，局部误差会改变后续非线性特征和功率平衡。一步最优与闭环长视界最优在模型失配、噪声或离散积分不一致时不是同一问题。因此，本方案没有取消闭环验证，而是取消了“无条件、固定轮数的闭环训练”。闭环结果决定模型是否晋级，但不改变 test 隔离规则。

### 4.3 实时 Pm 的解释边界

机械功率中心化修复后，才能公平比较 ZOH、observed 和 governor。observed `Pm(t)` 使用未来已知输入，只能解释为条件重积分上界，不是因果预测。ZOH 是最保守的主策略；governor 则用一个可辨识低阶 ODE 把训练期 observed 信息压缩为部署期可用的因果状态方程。若 governor 未通过跨折门，结论应是“当前低阶 governor 不足以替代 ZOH”，而不是任意提高 governor 阶数或使用未来 Pm。

### 4.4 动态记忆与 LYAP 的关系

动态记忆用于处理降阶后未解析状态导致的非 Markov 残差；LYAP 结构用于约束供能与耗散。两者解决的问题不同：记忆可以提高表达能力，却不自动提供稳定性证书；端口哈密顿结构提供可审计的供能关系，却不保证轨迹精度一定优于 M0。因此，LYAP 命名门同时要求数学证书和精度不退化，避免用“结构正确”替代“预测有效”。

## 5. 局限性与后续确认实验

1. 新版目前只有 F1 validation 工程验收，尚未产生五折 test 结果和 fault-line clustered 置信区间。
2. F1 误差分布明显偏斜，mean RMSE 0.3690 高于 median 0.0768，且最大个案为 1.7603；后续必须报告 CVaR 与失败事件，而不能只报告中位数。
3. 一阶 governor 是辨识性优先的低阶消融，不代表详细汽轮机、阀门、限幅和死区动力学。
4. 动态边记忆采用确定性准稳态初值；本轮未引入概率隐初态、best-of-N 或真实未来状态初始化。
5. 学得的 (K) 是当前降阶库中的有效参数，不应被解释为真实 Kron 网络参数。
6. `M2A-LYAP` 的最终名称必须由新完整运行生成的证书文件决定，不能根据单元测试或解析推导提前授予。

下一步应使用已注册配置 `configs/experiments/paper_integral_v2.toml` 创建新的不可变 run ID，依次完成五折 train/validation 选择、写入 selection marker、打开 test、生成 cluster bootstrap 和自动命名结果。旧完整运行继续作为基线保留。

## 6. 结论

本轮改进的主要成果不是增加了训练轮数，而是恢复了机械功率、平衡点、电气耦合和数值积分之间的一致性。以 \(\Delta P_m=P_m-P_{m,\mathrm{eq}}\) 为输入、采用积分弱形式一次求解并统一 COI-RK4 后，M0 在 F1/H=100 上通过 mean RMSE 0.5 的工程门，且数值 COI 漂移接近机器精度。这一结果将修正后的 M0 建立为后续模型比较的新基线。M2A、动态记忆、governor 与端口哈密顿 LYAP 已具备可复现实现和严格晋级门，但其性能和证书结论仍应等待新版五折完整证据，而不应从 F1 或结构推导中外推。

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
| Pm 中心化与统一积分能显著改善 M0 | 新 F1/H=100 mean RMSE 0.3690，COI drift $4.44\times10^{-16}$ | 工程验收支持；待五折确认 |
| 静态 M2A 应以一次 SINDy 解为主 | 算法结构与单元/集成测试 | 方法已实现；性能待确认 |
| governor 可替代 ZOH | 尚无跨折晋级结果 | 未支持 |
| 动态记忆优于静态 M2A | 尚无新版五折比较 | 未支持 |
| 模型可命名为 `M2A-LYAP` | 尚无新版全证书文件 | 未支持，名称保持门控 |
