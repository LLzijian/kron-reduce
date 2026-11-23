# 原始 10 机 M0 主线与端口结构发现协议

## 研究问题与边界

本轮只回答：在不使用七节点聚类先验的条件下，原始 10 台发电机是否能够成为 M0 主坐标系，以及 M0-3EG 之上的预测闭合与物理结构发现能否在同一坐标系中成立。实验仅使用 F1、seed 324 的训练集和验证集；测试集保持锁定。因此，本轮能够决定后续建模主线，不能宣称五折或测试泛化。

此前已接受的 M0-3EG → M0-GMD → M0-GMD-PM 链使用 `clustered_7`。后续 `generator_10` 仅出现在表征比较和聚类诊断中，不能倒推为已接受下游模型的坐标系。本轮在运行产物中保存原配置、协议、选择标记和 manifest 的 SHA-256，以避免依赖回忆作历史判断。

## 数据合同

- 数据集：canonical IEEE39 post-clear v1，combined dynamics，F1。
- 表征：`generator_10`；10 个节点、45 条完整有效候选边。完整图是 Kron 后发电机相互作用的参数化支撑，不是 39 节点物理线路表。
- 训练和验证轨迹必须至少有 100 个真实采样点；较短轨迹不得进入弱方程或模型选择。
- 拟合使用 observed \(P_m(t)\)，其重积分只定义为条件灵敏度；因果部署模型只读取 \(P_{m,\mathrm{clear}}\)、\(P_{m,\mathrm{eq}}\) 与预测速度。
- `Pe`、电压、稳定标签、故障时间、未来统计和 test ID 不进入拟合、归一化、阈值或选择。

## M0-3EG

原始 10 机模型保持 swing 功率平衡，并引入归一化暂态电势状态 \(q\)：

\[
\dot\delta=\omega,
\qquad
m\dot\omega=\Delta P_m-D\omega-
B\{K\odot[q_{\rm tail}q_{\rm head}\sin(B^\top\delta)
-\sin(B^\top\delta_{\rm eq})]\},
\]

\[
\tau\dot q=1-q-\frac{\alpha_{\rm eff}}{K_{\rm ref}}
[i_d(q,\delta)-i_d(1,\delta_{\rm eq})].
\]

其中 \(\Delta P_m=P_m-P_{m,\mathrm{eq}}\)，\(D\ge0\)、\(K\ge0\) 由多窗口积分弱形式求解；窗口 1、5、20、40、80 步一次堆叠，并采用 equal-case/equal-horizon 权重。\(\alpha\)、\(\tau\) 和大扰动激活阈值只由验证闭环选择，且包含 \(\alpha=0\) 的严格 M0 回退。学习到的 \(D,K\) 是约化有效系数，不宣称等于设备阻尼或物理线路参数。

## M0-GMD 与 M0-GMD-PM

M0-GMD 冻结 M0-3EG，先对训练残差做 POD，再在图特征和稳定一阶记忆库上进行组稀疏积分求解。它发现的是低秩残差模态，预测表达力强，但模态支撑不能直接解释为边拓扑。

M0-GMD-PM 在零总功率基 \(U_p\) 上定义

\[
a=U_p^\top(P_m-P_{m,\mathrm{eq}}),\qquad
v=U_p^\top\Pi_P(\omega),\qquad
\dot a=Aa+Gv.
\]

\(A,G\) 由 train-only observed \(P_m\) 的多窗口积分方程求解，并对 \(A\) 施加 Hurwitz 谱边界。部署从 \(P_{m,\mathrm{clear}}\) 初始化，不读取 observed future \(P_m\)。流程以反事实未来日程扰动实测这一因果不变性，而不是仅检查字段声明。

## M0-PSD：更强物理发现候选

为避免 GMD 的 POD 模态解释瓶颈，M0-PSD 在冻结 M0-3EG 后直接发现有效边流 \(r\)：

\[
m\dot\omega=f_{\rm 3EG}-Br,
\]

\[
r_e=\sum_{f\in\mathcal C}a_{f,e}\phi_f(\eta_e)
+d_{0,e}\nu_e+d_{1,e}\nu_e|\Delta\sin\eta_e|
+\sum_j c_{j,e}z_{j,e},
\]

\[
\tau_j\dot z_{j,e}=\nu_e-z_{j,e},
\qquad \eta=B^\top\delta,\quad\nu=B^\top\omega.
\]

保守库 \(\mathcal C\) 为 \(\Delta\sin\eta\)、\(\Delta\sin\eta|\Delta\sin\eta|\) 与 \((\Delta\sin\eta)^3\)，均对应可显式积分的边势能；静态耗散与记忆系数非负。对记忆元件取储能 \(S_{j,e}=c_{j,e}\tau_jz_{j,e}^2/2\)，和机械功率合并后产生 \(-c_{j,e}z_{j,e}^2\)，故不会由该支路主动注能。

所有系数以一次非负 ridge 弱形式求解；组阈值、时间常数和零闭合回退只用 train/val 选择。晋级必须同时满足：

1. 验证闭环相对 M0-3EG 的内部选择门通过，且存在非零边支撑；
2. 两个数据源独立支持的边集合 Jaccard 不低于 0.5；
3. H=100 精度相对 GMD 的退化不超过 2%，失败率为零；
4. 保守势在训练/验证定义域非负、解析/数值链式法则一致、COI 商空间局部刚度为正、内部力平衡和被动记忆恒等式通过；
5. RK4 在 \(dt,dt/2,dt/4\) 下收敛。

M0-PSD 只是端口结构残差候选，不是全局 Lyapunov 证书；无论结果如何均不得命名为 `M2A-LYAP`。被选边只代表当前完整有效发电机图上的交互，不能宣称恢复物理输电线路。

## 评估与 harness 顺序

验证报告 H=20、40、100 的 mean、median、pooled SSE、worst-10% CVaR、failure rate、按数据源结果及 fault-line cluster bootstrap。模型选择标记必须先于度量评估写入。流程顺序为 data → contracts → model unit → clean source commit → diagnostic → adversarial audit → acceptance → explicitly approved merge gate。科学候选允许得到负结论，但数据隔离、物理审计、积分收敛和产物哈希不能失败。

## 方法学依据

- 弱/积分 SINDy 通过对动力学采用弱变换，降低数值微分对噪声的放大：Messenger 与 Bortz，*Multiscale Modeling & Simulation* (2021)，[DOI: 10.1137/20M1343166](https://epubs.siam.org/doi/10.1137/20M1343166)。
- 结构保持 SINDy 表明字典与约束可以强制物理可实现性，但其结论仍依赖候选库是否正确：Lee、Trask 与 Stinis，[arXiv:2109.05364](https://arxiv.org/abs/2109.05364)。
- 端口哈密顿辨识将被动性和耗散不等式直接放入数据驱动模型：Morandin、Nicodemus 与 Unger，*SIAM Journal on Scientific Computing* (2023)，[DOI: 10.1137/22M149329X](https://epubs.siam.org/doi/10.1137/22M149329X)。
- 暂态数据可以用于稀疏电网结构恢复，但该任务具有物理拓扑观测假设；这与本项目的 Kron 后完整有效图不同，故只能支持“可做结构辨识”，不能支持“PSD 边等于真实线路”：*Journal of Computational Methods in Sciences and Engineering*，[DOI: 10.3233/JCM-215005](https://journals.sagepub.com/doi/10.3233/JCM-215005)。
