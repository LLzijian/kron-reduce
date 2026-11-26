# 基于 100 ms 观测的电力系统故障后能量结构灰箱动态等值与算子选择

## 摘要

由故障清除后 100 ms 的转子观测推演后续 8 s 的运动，需要同时刻画首摆和维持长时自由推演的数值可靠性。本文提出能量结构降阶模型 ES-ROM，将惯量中心坐标下的移位正弦网络势能、半正定阻尼和逐机互易记忆端口写成显式常微分方程，满足供能上界 $\dot H\le v^\top p$。该方法结合软半径因果高斯–牛顿初始化、积分动量初筛、梯度平衡的骨架拟合和闭环剪枝，从五种候选算子中选出随逐机功角偏移减弱的阻尼项。在 IEEE 39 节点系统独立生成的 94 例 `test` 上，ES-ROM 确认 32 例失稳中的 29 例（90.62%），62 例稳定样本无误报，94 例预测均数值合格，全样本准确率为 96.81%，在参与对比的零误报轨迹模型中召回最高（Graph-GRU 为 26/32），相对硬半径基线提升 12.50 个百分点（按故障组重采样的 95% 区间为 [2.78, 27.27] 个百分点）。同一冻结方程的主导模态频率绝对误差中位数为 0.0318 Hz，在真值与预测均触发失步的 29 例中准确复现 27 例割集，起报机械能区分失稳的受试者工作特征曲线下面积（AUROC）为 0.9526。

**关键词**　动态等值、能量建模、方程发现、暂态稳定性、电力系统动力学

## 一、引言

故障后的转子轨迹可以揭示机组何时失去同步以及哪些机组参与分离 [1]。同步相量量测可用于估计发电机的动态状态和辨识低阶等值模型 [2–4]。同调分析和 Koopman 模态可以识别摆动特征相近的机组 [5, 6]。多事件辨识可用来拟合等值负荷，按电气距离分组可用来聚合外部发电机 [7, 8]。

循环网络和时空图模型已用于直接判定暂态稳定性 [9, 10]。图模型与 Koopman 模型也可预测暂态轨迹 [11]。摆动方程残差可用于训练神经网络预测轨迹，物理信息神经网络也已用于求解各元件的动态响应并通过网络方程耦合 [12, 13]。

暂态能量法将扰动后的能量与临界能量比较，保结构模型则在暂态分析中保留网络拓扑 [14, 15]。Kron 降阶可保留代数网络以及满足特定条件的动态电路的端口行为 [16, 17]。消去动态状态后，保留状态的方程会出现记忆项和被消去状态初值的影响 [18–20]。

稀疏辨识从数据中选择方程项，积分形式可避免逐点计算导数 [21–23]。自编码器可以学习状态坐标，时延嵌入与循环解码器可以处理部分观测下的动力学辨识 [24–26]。算子推断直接拟合降阶动力学，约束回归可为特定模型施加能量或稳定性条件 [27–29]。端口哈密顿方法描述功率守恒的互连与耗散，已有工作将其用于图网络建模、非线性投影降阶与数据辨识 [30–33]。

ES-ROM 在共享的显式常微分方程中选择恢复力、阻尼、耦合与记忆算子，用故障清除后 100 ms 的转子观测初始化，推演后续 8 s 的运动。本文的主要贡献如下。

1. 建立惯量中心（COI）坐标下的灰箱模型，证明五个物理算子的任意组合均满足供能与耗散恒等式。
2. 结合软半径因果高斯–牛顿初始化、弱动量初筛、梯度平衡的局部拟合与后向剪枝，改善初态截断和阻尼偏大造成的首摆预测误差。
3. 在真值已知的系统中检验算子选择与隐状态恢复能力，并用 200 ms 动量窗诊断缺失的恢复力项。
4. 在 IEEE 39 节点系统 21 个故障组的 94 例独立样本上比较冻结模型，检验主导模态、失步机群和能量特征的一致性。

第二节定义预测与评价口径。第三、四节介绍方法。第五至七节分别报告合成验证、IEEE 39 节点系统对比及物理一致性分析。第八节总结。

## 二、问题定义与因果预测协议

### A. 惯量中心坐标

考虑故障清除、网络恢复至故障后拓扑后的十台同步发电机。设 $M=\operatorname{diag}(m_1,\ldots,m_{10})$ 为惯量矩阵，$m=M\mathbf{1}$。对展开后的转子功角 $\delta$ 和参考平衡点 $\delta^\star$，定义

$$
P=I-\frac{m\mathbf{1}^\top}{\mathbf{1}^\top m},\qquad
q=P^\top(\delta-\delta^\star),\qquad
v=P^\top\dot\delta.
$$

两组相对坐标满足 $m^\top q=m^\top v=0$。$P^\top$ 投影状态，$P$ 投影广义力，在 COI 子空间内有 $v^\top P=v^\top$。省略的电气和控制动态由有效耦合、阻尼和隐状态表示。

### B. 观测与自由推演

以故障清除为 $t=0$，起报时刻为 $t_o=0.10\,\mathrm{s}$。预测输入包括起报前每隔 10 ms 采集的 11 帧 $(q,v)$、参考平衡点、惯量、图先验，以及起报时刻已知的机械功率偏差向量 $p$。历史重建和未来预测均使用同一个常值 $p$。用这 11 帧估计隐状态后，将可见状态对齐到观测末帧，再用同一冻结方程自由推演 8 s。

离线训练和评价使用未来轨迹及其实际可用长度。故障身份用于分组，稳定性标签用于给训练损失加权和评估预测。预测阶段的输入范围由上述观测、起报功率与先验确定，电功率量测与未来机械功率序列均在这一范围之外。

### C. 轨迹误差与失稳判据

对事件 $e$ 的 $n_e$ 个评分点，归一化均方根误差（NRMSE）定义为

$$
\epsilon_e=\left\{\frac{1}{20n_e}\sum_{j=1}^{n_e}
\left[\frac{\lVert\hat q_j-q_j\rVert_2^2}{s_q^2}
+\frac{\lVert\hat v_j-v_j\rVert_2^2}{s_v^2}\right]\right\}^{1/2},
\qquad s_q=0.476753\,\mathrm{rad},\quad
s_v=1.003367\,\mathrm{rad/s}.
\tag{1}
$$

在指定样本范围内对 $\epsilon_e$ 按事件等权取平均。稳定样本使用固定时长，失稳样本使用起报后实际观测到的区间，最长为 8 s。

在拼接的历史与预测轨迹上，当 $q+\delta^\star$ 的角度跨度大于 $\pi$，且最大 COI 相对转速幅值 $\max_i\lvert v_i\rvert/(2\pi\cdot60)$ 大于 0.02 p.u.，两项同时持续 5 个 10 ms 采样点时，判为失步。若始终未触发失步，且末 100 帧的角度跨度均小于 $\pi$、最大相对转速幅值均小于 0.005 p.u.，则判为稳定。数值合格且未落入上述两种情况的预测记为未安定，数值不合格另计。

以失稳为正类，确认召回率为 $\mathrm{TP}/n_{\rm unstable}$，稳定误报率为 $\mathrm{FP}/n_{\rm stable}$。全部事件都有有效判定时计算全样本准确率。

## 三、能量结构灰箱模型与候选算子库

### A. 共享机电骨架

十机完全图有 45 条候选边，记有向关联矩阵为 $B=[b_1,\ldots,b_{45}]$，边 $e$ 的相对功角偏移和参考角差分别为 $d_e=b_e^\top q$ 和 $r_e=b_e^\top\delta^\star$。定义移位谐波势能与基础网络势能

$$
\Phi_j(d;r)=\frac{\cos(jr)-\cos(j(r+d))}{j}-d\sin(jr)
\tag{2}
$$

$$
U_0(q)=\sum_{e=1}^{45} k_e^0\Phi_1(d_e;r_e),\qquad k_e^0>0.
\tag{3}
$$

参考位移使得 $q=0$ 时边恢复力为零。取 $\mathbb R^{10}$ 中分量和为零的子空间的一组正交归一基 $N$，令 $D_0=N\operatorname{diag}(\beta_1,\ldots,\beta_9)N^\top$，其中 $\beta_j\ge0$。十个互易记忆端口的增益为 $C=P\operatorname{diag}(g_1,\ldots,g_{10})$，正衰减矩阵为 $\Lambda=\operatorname{diag}(1/\tau_1,\ldots,1/\tau_{10})$。从 $\dot z=C^\top v-\Lambda z$ 消去 $z$ 后，记忆力可写为

$$
Cz(t)=Ce^{-\Lambda t}z(0)+\int_0^t Ce^{-\Lambda(t-s)}C^\top v(s)\,ds.
\tag{4}
$$

第一项是未观测初态的后续贡献，第二项累积起始时刻之后的转子运动。骨架有 74 个共享系数，包括 45 个边刚度、9 个阻尼系数、10 个端口增益和 10 个时间常数。

### B. 增广动力学与候选算子

记算子组合为 $\mathcal S\subseteq\mathfrak G=\{\mathrm{G1},\ldots,\mathrm{G5}\}$。令 $\chi=(c,\xi)$ 表示启用的整体与空间耦合状态，全部启用时有 $c\in\mathbb R$、$\xi\in\mathbb R^9$。统一储能为

$$
H(q,v,z,\chi)=\tfrac12 v^\top Mv+U(q,\chi)+\tfrac12 z^\top z
+\tfrac12\kappa_c c^2+\tfrac12\kappa_\xi\lVert\xi\rVert_2^2.
\tag{5}
$$

相应的动力学为

$$
\begin{aligned}
\dot q &= v,\\
M\dot v &= P[p-\nabla_q U-D(q)v]-C(q)z,\\
\dot z &= C(q)^\top v-\Lambda z,\\
\dot\chi &= -G_\chi\nabla_\chi H.
\end{aligned}
\tag{6}
$$

$G_\chi=\operatorname{diag}(\mu_c,\mu_\xi I_9)$ 取启用的块。关闭耦合块时一并移除对应状态和储能项。所有算子均开启时有 40 个状态坐标，去掉两条 COI 约束后有 38 个独立状态。最终 ES-ROM 的 $q,v,z$ 共 30 个坐标，对应 28 个独立状态。$z$ 和 $\chi$ 分别表示有效记忆与耦合，辨识目标是保留机组的输入与输出动力学。

所有候选算子共用式（5）和式（6）的结构。对机组 $i$，定义局部角度偏移量 $s_i(q)=\sum_{e\in\mathcal E_i}d_e^2/\lvert\mathcal E_i\rvert$，其中 $\mathcal E_i$ 为与机组 $i$ 相连的边集。

- **G1，二阶谐波恢复力。** 在每条边上用 $(1-2a\cos(2r))\Phi_1+a\cos(r)\Phi_2$ 代替 $\Phi_1$，其中 $a\in[0,0.6]$。该算子加入保守的二阶谐波，同时保持平衡点恢复力与边的线性刚度。
- **G2，逐机功角软化阻尼。** 阻尼随局部功角偏移增大而减小。

  $$
  D(q)=D_0^{1/2}\left[I-\rho\operatorname{diag}\!\left(\frac{s_i(q)}{1+s_i(q)}\right)\right]D_0^{1/2}.
  \tag{7}
  $$

  $0\le\rho\le0.85$ 保证 $D(q)\succeq(1-\rho)D_0\succeq0$。

- **G3 与 G4，整体与空间动态耦合。** G3 给各边增加共同的对数倍率 $c$。G4 增加分量和为零的空间模式 $\lvert B\rvert^\top N_\chi\xi$，其中 $\lvert B\rvert$ 按元素取绝对值，$N_\chi$ 为零和子空间的 Helmert 基。将两项合记为 $u$，令 $k_e=k_e^0\exp(fu_e)$，其中

  $$
  f=\left[1+\sum_e\left\{
  \left(\frac{(u_e)_+}{\log 2}\right)^8+
  \left(\frac{(-u_e)_+}{\log 4}\right)^8
  \right\}\right]^{-1/8}.
  \tag{8}
  $$

  同一个缩放因子保证每条边的 $k_e/k_e^0$ 处于 $[1/4,2]$。两块均沿同一储能函数的负梯度演化，其中 $\mu_c,\mu_\xi\in[0,0.5]$，$\kappa_c=1$，$\kappa_\xi\in[0.05,1]$。

- **G5，状态依赖的饱和记忆端口。** 令 $C(q)=P\operatorname{diag}\bigl(g_i/(1+\gamma_s s_i(q))\bigr)$，其中 $\gamma_s\in[0,8]$。记忆驱动与力反馈共用同一衰减因子。

**表 1　候选算子与闭环筛选**

**A. `train` 上的弱动量初筛**

| 算子 | 物理作用 | 失稳样本动量残差 RMS 变化 |
|---|---|---:|
| G1 | 二阶谐波恢复力 | $+1.51\%$ |
| G2 | 逐机功角软化阻尼 | $-37.54\%$ |
| G3 | 整体动态耦合，给各边刚度施加同一个时变倍率 | $-5.97\%$ |
| G4 | 空间动态耦合，按分量和为零的空间模式调节边刚度 | $+0.14\%$ |
| G5 | 随功角变化的饱和记忆端口 | $-8.70\%$ |

**B. `val` 上的闭环对比**

| 算子组合 | 确认失稳数 | 稳定误报数 |
|---|---:|---:|
| {G2, G3, G5} | 11/12 | 0/95 |
| {G2, G5} | 11/12 | 0/95 |
| {G2}（ES-ROM） | 12/12 | 2/95 |

带符号变化量比较单个算子在最佳筛选系数下的结果与关闭全部算子时的基准结果。2.5 ms 和 1.25 ms 步长的结果在所示精度内一致。按事件加权的 RMS 使用起报后 0–200 ms 与 200–400 ms 的完整窗口，分别覆盖 `train` 中 26/35 与 14/35 个失稳样本。三个闭环组合在 `val` 的未安定数均为零。

### C. 供能与耗散恒等式

**命题 1**　算子库中的任意组合保持 COI 约束，并满足

$$
\begin{aligned}
\dot H
&=v^\top p-v^\top D(q)v-z^\top\Lambda z
-(\nabla_\chi H)^\top G_\chi\nabla_\chi H\\
&\le v^\top p.
\end{aligned}
\tag{9}
$$

**证明**　由 $\mathbf{1}^\top P=0$ 与 $\mathbf{1}^\top C(q)=0$，式（6）右侧的广义力分量和为零。因此 $m^\top\dot v=0$，且 $\dot q=v$ 保持角度约束。对式（5）求导得到机械项 $v^\top M\dot v+v^\top\nabla_q U$、记忆项 $z^\top\dot z$ 和耦合项 $(\nabla_\chi H)^\top\dot\chi$。由 $v^\top P=v^\top$，势能与恢复力项相消。互易端口两项 $-v^\top C(q)z$ 与 $z^\top C(q)^\top v$ 在每一时刻相消。其余耗散项由半正定性给出式（9）。

全部 32 种算子组合都通过了容差为 $10^{-12}$ 的合成储能收支检查。固定输入下，倾斜储能 $H-p^\top q$ 单调不增。$\Phi_1$ 中的线性参考项可能使展开角度空间上的势能无下界，因此全局稳定性结论还需独立验证下界与不变性条件。

## 四、因果初始化与自动算子发现

### A. 基于短观测历史的嵌套辨识

算子组合 $\mathcal S$ 和共享系数 $\theta$ 同时决定观测历史重建与起报后推演。每次评价 $(\theta,\mathcal S)$ 时，先用每个事件的 100 ms 观测求解事件隐初态，再在 `train` 上优化跨事件目标。`val` 用于选择检查点和算子组合。

在自由拟合初始记忆力的诊断中，端口增益下降至接近 $10^{-5}$ 时，可以由很大的估计记忆状态补偿。`val` 的起报时刻隐储能中位数达到 $4.19\times10^{11}$。因此，每次更新共享参数时都使用短历史重新估计记忆状态，使训练时的初始化与实际预测一致。

记 $y_j=(q_j,v_j)$，$j=0,\ldots,10$。将历史起始时刻的隐耦合状态设为零，并用下式参数化初始记忆状态

$$
z_h(\eta)=\frac{\eta}{\sqrt{1+\lVert\eta\rVert_2^2/(2R)^2}}.
\tag{10}
$$

由初态 $(y_0,z_h(\eta),0)$ 积分式（6），得到历史预测 $\tilde y_j(\eta;\theta,\mathcal S)$。内层目标为

$$
\frac{1}{200}\sum_{j=1}^{10}
\left\lVert W_h\bigl[\tilde y_j(\eta)-y_j\bigr]\right\rVert_2^2
+\lambda_h\lVert\eta/R\rVert_2^2,\qquad
\lambda_h=10^{-6}.
\tag{11}
$$

$W_h$ 由冻结在 `train` 上的历史归一化尺度的倒数构成。将 $W_h(\tilde y_j-y_j)/\sqrt{200}$ 与 $\sqrt{\lambda_h}\eta/R$ 拼接为残差向量 $r_h$，记 $J_h=\partial r_h/\partial\eta$。从 $\eta=0$ 开始做五次高斯–牛顿迭代，每一步求解 $\Delta\eta=\arg\min_d\lVert r_h+J_hd\rVert_2^2$。$J_h$ 由 ODE 的解析切线和式（10）的导数得到。用拟合后的初态重新积分至 $t_o$，保留末端隐状态并与实测 $(q_{10},v_{10})$ 对齐。

硬半径求解器将历史起点记忆状态投影到球 $\lVert z_h\rVert_2\le R$。在一例漏判失稳的配对初始化诊断中，硬约束使起始记忆储能饱和于 $R^2/2=21.85$，数值采用模型的储能尺度。式（10）允许记忆范数平滑接近 $2R$，并保持对待估坐标的可微性。固定的轻先验施加在无约束坐标 $\eta/R$ 上，用来抑制弱可观测方向。这样增加的是传递到起报点的隐状态及其记忆力的调节空间，起报的可见动能仍由实测转速确定。

### B. 首摆损失与局部骨架拟合

`train` 含 311 例稳定和 35 例失稳。对共享初始方程的梯度诊断表明，加权轨迹平方误差、单边转速误差和弱动量项的梯度范数分别为 15.81、0.24 和 1.33，对阻尼对数倍率的偏导数分别为 $-4.95$、$+0.19$ 和 $+1.05$。在该起点，沿轨迹损失下降会增大阻尼，沿两个首摆损失下降则会减小阻尼。本文据此在拟合起点平衡梯度量级，并以自由推演表现选取检查点。

轨迹损失 $\mathcal L_{\rm tr}$ 对 1 s 和 3 s 窗内的有效观测误差取平均。稳定样本采用转折点为 1 的 Huber 损失，失稳样本采用平方误差。弱形式动量残差为

$$
r_{ab}=v(t_b)-v(t_a)-\int_{t_a}^{t_b}M^{-1}F(q,v,h;p)\,dt.
\tag{12}
$$

$F$ 是式（6）中驱动转子的广义力。以因果求解得到的起报隐状态为初值，令 $h=(z,\chi)$ 沿线性插值的实测 $(q,v)$ 积分。动量除以惯量得到转速单位的残差。对起报后 0–200 ms 与 200–400 ms 的完整窗口，将各分量归一化后平方平均，得到 $\mathcal L_{\rm mom}$。在自由推演前 500 ms 内，$\mathcal L_{\rm vel}$ 惩罚转速归一化后 $\operatorname{sgn}(v_i)(v_i-\hat v_i)$ 的正部。两项均给失稳样本赋予稳定样本 8 倍的权重。

令 $\mathcal L_{\rm fs}=\mathcal L_{\rm mom}+\tfrac12\mathcal L_{\rm vel}$，外层目标为

$$
\begin{aligned}
\mathcal L &= \mathcal L_{\rm tr}+\alpha\mathcal L_{\rm fs}
+\lVert\theta-\theta_{\rm a}\rVert_2^2+0.1\mathcal L_E,\\
\alpha &=1.05\left.\frac{\lVert\nabla\mathcal L_{\rm tr}\rVert_2}
{\lVert\nabla\mathcal L_{\rm fs}\rVert_2}\right|_{\rm start}.
\end{aligned}
\tag{13}
$$

锚点 $\theta_{\rm a}$ 表示在拟合坐标中。各边刚度和阻尼采用相对于参考骨架的独立对数倍率，对应的倍率范围为 $[0.5,2]$，且 $g_i\ge0.01$、$\tau_i\in[0.02,5]\,\mathrm{s}$。$\mathcal L_E$ 惩罚起报记忆储能相对于 $\tfrac12(2R)^2$ 的正超量平方。每次拟合开始时用完整 `train` 确定梯度定标系数，后续固定该系数。

### C. 初筛、分阶段拟合与剪枝

图 1 和算法 1 概括离线选择和因果预测。弱动量初筛固定共享骨架，分别评价单个算子相对于关闭全部五块的基准。通过系数网格搜索和局部中点细化，以加权积分残差排序。进入后续筛选的算子需要在两档积分步长下均降低失稳样本的动量残差均方根（RMS）。拟合骨架时采用 $\rho=2/3$ 的 G2 阻尼。第一阶段开放 45 个独立边刚度和 9 个阻尼系数，以适应不同扰动下的局部耦合。第二阶段再开放记忆增益与时间常数。两阶段各执行 12 次全 `train` 更新，每三次更新用完整 `val` 检查一次。

**图 1　能量结构方程的离线算子选择与因果预测。** 从短历史估计隐状态和起报后的自由推演共用同一拟合方程。[原图 PDF](figures/method.pdf)

**算法 1　基于 `val` 约束的算子选择**

1. 用两个 200 ms 动量窗筛选 G1–G5，保留能降低失稳样本动量残差 RMS 的算子。
2. 先拟合 54 个局部骨架系数，再开放全部 74 个骨架系数。优化式（13）时嵌套式（11）的历史求解，各阶段用 `val` 选择检查点。
3. 在相同的重拟合预算下比较初筛组合与逐次删去算子的组合。先检查数值、轨迹和首摆条件，再应用 2% 等价准则，冻结合格组合。

后向剪枝从同一个局部骨架出发，每个候选执行 6 次完整更新，在第 0、3、6 次更新时检查。两档步长均需数值合格。`val` 候选需确认全部 12 例失稳，95 例稳定样本的误报数至多为 2，未安定数为 0。`train` 至少确认 35 例失稳中的 33 例。`val` 稳定样本 8 s NRMSE 中位数至多为 0.21，正确判稳样本的 8 s NRMSE 均值至多为 0.24。对 `train` 预选的 33 个机群分离窗口，分离速度残差的带符号中位数距零至少缩小 20%，绝对残差的中位数也须减小。

对误报数相同的合格组合，若删去算子后，稳定样本 NRMSE 中位数、正确判稳样本 NRMSE 均值和前 500 ms 峰值欠估均在较大组合的 2% 等价范围内，则选更简的组合。上述选择均在正式 `test` 预测前完成。

### D. 双步长数值核验

采用 CUDA 双精度四阶 Runge–Kutta 积分，训练步长为 5 ms。正式 ODE 自由推演采用 2.5 ms 主步长和 1.25 ms 核验步长，均输出到相同的 10 ms 网格。两档结果的 COI NRMSE 和归一化公共转速差均须不超过 $10^{-3}$，分类一致，成对的触发时刻相差不超过 20 ms。触发失步时，核验至两档中较晚的触发时刻，未触发时核验至预测终点。各轨迹指标还按对应的评分区间核验，完整预测是否有限另行记录。

## 五、合成验证与可辨识性边界

十机合成实验在五个已知真值系统 S0–S4 上拟合 $\{\mathrm{G3},\mathrm{G4},\mathrm{G5}\}$ 的八种算子组合。每个真值的合成 `train`、`val`、`test` 分别为 24、12、12 例，以 3 s 轨迹平方误差选模。S2 选出了饱和记忆端口，估计的 $\gamma_s=0.464$，真值为 0.60。沿拟合与选模轨迹的记忆核相对误差为 1.55%。五种真值各自选中的方程均正确判定 12/12 例合成 `test` 样本，最大归一化储能收支残差为 $4.29\times10^{-16}$。

在另一项已知系数的合成实验中，历史求解也估计耦合隐坐标，正确选出固定耦合、整体耦合、空间耦合及混合耦合四种结构，对应的 `val` 8 s 等权平均 NRMSE 为 $4.40\times10^{-9}$ 至 $6.56\times10^{-6}$（表 2）。另一项两算子实验也在相应的真值系统中选出了二阶谐波与状态调制阻尼。

**表 2　已知真值下的算子选择与 100 ms 历史初始化检验**

**A. 用无噪声历史初始化的轨迹选模**

| 系统 | 真值机制 | 选中机制 | 合成 `val` 8 s NRMSE | 合成 `test` 8 s NRMSE |
|---|---|---|---:|---:|
| S0 | 固定耦合与恒定端口 | 固定耦合与恒定端口 | 0.002207 | 0.002047 |
| S1 | 耦合的共同缩放与空间再分配 | 共同缩放 | 0.002666 | 0.002675 |
| S2 | 饱和记忆端口 | 饱和记忆端口 | 0.000680 | 0.000668 |
| S3 | 共同缩放、空间再分配与饱和端口 | 共同缩放与饱和端口 | 0.001729 | 0.001386 |
| S4 | 含库外五次谐波的恢复力 | 固定耦合与恒定端口 | 0.002289 | 0.002138 |

**B. 方程与真值系数固定时先验的作用**

| 系统 | 无噪声历史初始隐状态最大相对误差 | 含噪历史初始隐状态最大相对误差 | 含噪合成 `val` 8 s NRMSE |
|---|---:|---:|---:|
| S0 | $0.6624\to 6.55\times10^{-6}$ | $1.681\to3.067$ | $0.005331\to0.005750$ |
| S1 | $0.7496\to 5.78\times10^{-3}$ | $1.654\to3.433$ | $0.005195\to0.009579$ |
| S2 | $0.7312\to 5.48\times10^{-6}$ | $1.628\to2.946$ | $0.001516\to0.001850$ |
| S3 | $0.8005\to 1.19\times10^{-4}$ | $1.734\to3.612$ | $0.001714\to0.005741$ |

各系统的合成 `train`、`val`、`test` 分别为 24、12、12 例。8 s NRMSE 为 2.5 ms 步长下各事件的等权均值。A 中，选出的组合在无噪声和含噪历史下均正确判定各自合成 `test` 的 12/12 个事件。S4 有一个候选通过长时预测判据。在另一项系数已知的对照中，对含五阶谐波的真值，四个库内候选在两档步长下的最大单事件 200 ms 动量 NRMSE 均超过合成 `val` 的 $10^{-4}$ 阈值，固定耦合候选为 $8.79\times10^{-4}$。

B 中每个单元格依次列出使用先验和取消先验时的数值。初始隐状态误差为 $\lVert\hat x-x\rVert_2/\max(\lVert x\rVert_2,0.1)$，取 12 个合成 `val` 事件中的最大值。投影到 COI 坐标前，观测噪声的标准差为合成 `train` 历史尺度的 $10^{-3}$。

S4 的真值含候选库之外的五阶谐波，用来检验候选库的覆盖范围。至少一个库内候选在 S4 上通过长时轨迹误差和失步时序要求。在已知系数的耦合实验中，以全部 `val` 样本及两档步长的最大值计，四个库内候选的 200 ms 归一化动量误差均超过 $10^{-4}$，其中固定耦合候选为 $8.79\times10^{-4}$。当 8 s 轨迹指标接受近似方程时，短窗动量残差仍能揭示缺失的恢复力项。

这些结果给出了短历史辨识的三项边界。

1. **可见轨迹与隐状态的恢复精度。** 整体与空间耦合同时启用时，已知系数实验在 `val` 上的 8 s NRMSE 为 $6.56\times10^{-6}$，空间隐状态的最大相对误差仍为 38.2%。
2. **激励覆盖与算子可辨识性。** 在未知系数的 S1 和 S3 上，按 3 s 轨迹误差选模会删去真值中的空间耦合块。该激励和目标指标提供的空间系数辨识信息有限。
3. **正则化与测量噪声。** 在系数固定的 S3 中，取消先验使无噪声 `val` 的 8 s 等权平均 NRMSE 降至 $1.11\times10^{-8}$。历史噪声的标准差为 `train` 历史尺度的 $10^{-3}$ 时，取消先验使同一指标由 0.001714 增至 0.005741。固定轻先验有助于抑制弱灵敏度方向上的噪声。

## 六、IEEE 39 节点系统算例

### A. 数据与对比口径

采用 ANDES 2.0.0 [34] 建立 IEEE 39 节点十机高阶仿真模型。`train` 含 346 例，其中 311 例稳定、35 例失稳。`val` 含 8 个故障组的 107 例，其中 95 例稳定、12 例失稳，用于模型与阈值选择。

`test` 在同一运行点独立生成，采用三相母线故障及恢复后的网络拓扑。94 例中 62 例稳定、32 例失稳，覆盖 21 个母线与线路故障组、7 个 `train` 和 `val` 未见母线及 26 个近临界故障条件。32 例失稳中，31 例在故障清除后 0.2–1.5 s 触发失步。生成前固定故障条件与入组名额，核对 `test` 与 `train`、`val` 的事件互不重叠。真值沿用高阶仿真标签，其中稳定标签由 15 s 的观察窗口确认。

所有模型和判别阈值均在 `test` 推理前冻结。对比采用同一观测时长、起报机械功率、COI 误差口径、轨迹判稳规则和 `test` 名单。样本中稳定与失稳的比例由故障样本设计确定。

### B. 算子选择与组件消融

在 `train` 的弱动量初筛中，G2 降低失稳样本 RMS 37.54%，同时降低稳定样本 RMS 7.33%。G5 的饱和记忆和 G3 的整体耦合也使失稳样本 RMS 下降，G1 的二阶谐波和 G4 的空间耦合则使其上升（表 1）。

与硬半径基线相比，54 系数局部骨架拟合使 `train` 中 33 个机群分离窗口的带符号速度残差中位数由 $+2.551$ 降至 $+1.544$ rad/s。相对于骨架锚点，最大单边刚度改变量为 1.22%。局部骨架阶段选择第 12 次更新，开放 74 个系数的阶段选择第 0 次更新（图 2）。记忆增益和时间常数沿用该阶段的起始值。

初筛保留的两个较大组合均未确认 `val` 中全部 12 例失稳。完整初筛组合在 `val` 正确判稳样本上的 8 s NRMSE 均值为 0.31277，单 G2 组合对应 0.21494。单 G2 组合符合全部选择条件，确认 `train` 的 33/35 例与 `val` 的 12/12 例失稳，在 `val` 稳定样本上的 8 s NRMSE 中位数为 0.19026，误报 2/95。剪枝阶段选取第 0 次更新，即沿用局部骨架。

**图 2　两阶段骨架拟合。** 各阶段的训练损失分别按阶段初值归一化。轨迹曲线是 `val` 全部 95 个稳定样本的 8 s NRMSE 中位数。机群分离诊断取 `train` 33 个窗口带符号残差中位数的绝对值。虚线为对应的接受阈值。[原图 PDF](figures/curriculum.pdf)

表 3 对比同一 `test` 上的三组冻结方程与初始化方式。硬 $R$ 基线与软 $2R$ 消融共用整体缩放的骨架、记忆参数和功角相关阻尼。软 $2R$ 消融以光滑映射代替硬投影，并把先验施加到映射的待估坐标。ES-ROM 将同一软初始化与逐边拟合的局部骨架结合。

在硬 $R$ 基线、软 $2R$ 消融和 ES-ROM 下，`test` 的确认失稳召回分别为 25/32、27/32 和 29/32，三组稳定误报均为 0/62。在 32 个失稳样本上，取起报后前 500 ms 可用窗口内实测与预测最大相对转速幅值之差的正部，再按事件取中位数，三组分别为 1.7609、0.9863 和 0.3210 rad/s。ES-ROM 比硬 $R$ 基线少欠估 81.8%，全样本准确率为 96.81%，$F_1=0.9508$。

**表 3　`test` 上 94 个事件（62 个稳定、32 个失稳）的轨迹比较**

| 模型 | TP/TN/FP/FN | 未安定（失稳/稳定） | 数值不合格（失稳/稳定） | 全样本准确率（%） | 稳定样本 8 s NRMSE 均值 | 稳定样本 8 s NRMSE 中位数 | 失稳样本可用区间 NRMSE 均值 | 失稳样本可用区间 NRMSE 中位数 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 硬 $R$ 基线 | 25/62/0/6 | 1/0 | 0/0 | -- | 0.24809 | 0.26335 | 0.86075 | 0.66309 |
| 软 $2R$ 消融 | 27/62/0/5 | 0/0 | 0/0 | 94.68 | 0.25930 | 0.25972 | 0.73896 | 0.51106 |
| ES-ROM | 29/62/0/3 | 0/0 | 0/0 | 96.81 | 0.25122 | 0.24765 | 0.73181 | 0.51692 |
| Graph-GRU（扩展版） | 26/62/0/6 | 0/0 | 0/0 | 93.62 | 0.24017 | 0.23494 | 0.45094 | 0.30562 |
| GRU Seq2Seq（扩展版） | 24/62/0/8 | 0/0 | 0/0 | 91.49 | 0.14865 | 0.13241 | 0.44647 | 0.27366 |
| 潜空间神经 ODE（扩展版） | 18/62/0/11 | 3/0 | 0/0 | -- | 0.19116 | 0.20503 | 0.43973 | 0.35641 |
| 小型 Transformer | 0/62/0/18 | 14/0 | 0/0 | -- | 0.15584 | 0.15514 | 1.32959 | 1.32945 |
| Koopman-SHRED（渐进式） | 16/62/0/12 | 4/0 | 0/0 | -- | 0.21546 | 0.21982 | 0.44468 | 0.33013 |
| 小型 LASS | 25/62/0/6 | 1/0 | 0/0 | -- | 0.21363 | 0.20903 | 0.50537 | 0.39193 |
| 历史条件 DeepONet | 21/62/0/4 | 7/0 | 0/0 | -- | 0.26290 | 0.25970 | 0.45613 | 0.35862 |
| STAN（转子状态适配） | 32/31/10/0 | 0/21 | 0/0 | -- | 5.94096 | 0.66809 | 0.58742 | 0.32183 |
| 历史条件 ActNet | 14/54/0/0 | 18/8 | 0/0 | -- | 0.46026 | 0.45881 | 0.57729 | 0.47138 |
| I-IKNO | 32/0/62/0 | 0/0 | 0/0 | 34.04 | $1.096\times10^4$ | $6.658\times10^3$ | 1.38453 | 0.48493 |
| Hankel-DMD（修订版） | 32/0/62/0 | 0/0 | 0/0 | 34.04 | $3.058\times10^7$ | $3.479\times10^7$ | 1.16127 | 0.64145 |
| SINDy-SHRED（修订版） | 7/0/1/0 | 0/0 | 25/61 | -- | --（$0/62$） | --（$0/62$） | --（$22/32$） | --（$22/32$） |
| SINDy-SHRED（STLSQ） | 19/5/6/0 | 0/3 | 13/48 | -- | --（$8/62$） | --（$8/62$） | --（$30/32$） | --（$30/32$） |
| Hankel-DMD（初始版） | 32/0/62/0 | 0/0 | 0/0 | 34.04 | $2.497\times10^8$ | $2.004\times10^8$ | 1.61769 | 0.52486 |
| GRU Seq2Seq（初始版） | 1/62/0/29 | 2/0 | 0/0 | -- | 0.38675 | 0.37962 | 0.83000 | 0.76888 |
| 潜空间神经 ODE（初始版） | 0/62/0/32 | 0/0 | 0/0 | 65.96 | 0.36993 | 0.37987 | 1.17056 | 1.02563 |
| Koopman-SHRED（初始版） | 0/62/0/32 | 0/0 | 0/0 | 65.96 | 0.41716 | 0.41886 | 1.25069 | 1.13058 |

失稳为正类。TP、TN、FP、FN 依次为正确确认失稳、正确判稳、稳定误报和失稳漏判。确认召回率为 TP/32，稳定误报率为 FP/62。未安定与数值不合格均按真实失稳/稳定的顺序计数。事件全部可判定时，全样本准确率为 $(\mathrm{TP}+\mathrm{TN})/94$。若只报告已判定样本的准确率，分母为 $\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}$。

对事件 $i$，在起报后的 $n_i$ 个机组与时点组合上，分别用 $0.476753$ rad 和 $1.003367$ rad/s 归一化 COI 角度与转速误差。记两类归一化误差的平方和的一半为 $S_i$，则单事件 NRMSE 为 $\sqrt{S_i/n_i}$。表中均值和中位数均按事件等权统计。按评分点汇总的 NRMSE 为 $\sqrt{\sum_i S_i/\sum_i n_i}$。稳定样本使用完整 8 s，失稳样本使用起报后实际可用的观测点。轨迹覆盖不完整时，括号内给出合格事件数与总事件数。

**表 4　`test` 上相对于硬 $R$ 基线的配对差值**

| 指标 | 软 $2R$ 消融 | ES-ROM |
|---|---:|---:|
| 确认召回率（百分点） | $+6.25$ | $+12.50$ |
| 95% 区间 | $[0.00,16.13]$ | $[2.78,27.27]$ |
| 转速峰值欠估中位数（rad/s） | $-0.7746$ | $-1.4399$ |
| 95% 区间 | $[-1.2074,-0.1685]$ | $[-1.9591,-0.6686]$ |
| 稳定样本 8 s NRMSE 均值 | $+0.0112$ | $+0.0031$ |
| 95% 区间 | $[0.0017,0.0222]$ | $[-0.0062,0.0143]$ |

从 21 个母线与线路故障组中有放回进行 10,000 次配对重采样，全部重复均有效。对 32 个真实失稳事件，分别在起报后前 500 ms 的共同观测窗口内取真值和预测的 COI 相对转速幅值峰值，再取真值峰值减预测峰值的正部。2.5 ms 与 1.25 ms 步长的统计量在所示精度内一致。

以 21 个故障组为单位进行 10,000 次有放回配对重采样，各模型在每次重复中使用相同的组索引。表 4 给出双侧 95% 百分位区间。相对于硬 $R$ 基线，软 $2R$ 消融的召回率提高 6.25 个百分点，区间包含零，稳定样本 8 s NRMSE 均值增加 0.0112。ES-ROM 的召回率提高 12.50 个百分点，区间为 [2.78, 27.27] 个百分点，首摆峰值欠估中位数减少 1.44 rad/s，减少量区间为 [0.67, 1.96] rad/s。ES-ROM 的稳定样本 8 s NRMSE 均值差为 $+0.0031$，区间为 [$-0.0062$, 0.0143]，中位数由硬 $R$ 基线的 0.26335 降至 0.24765。按本次故障组抽样设计，配对区间支持 ES-ROM 的召回改善，观察到的稳定轨迹平均误差差异较小。

### C. 与轨迹预测模型对比

表 3 和图 3 对比三组物理模型以及 17 种基于文献方法实现的轨迹模型。对照覆盖循环网络、图模型、Transformer、潜空间神经 ODE、基于时延的稀疏辨识与模态方法，以及神经算子 [11, 26, 35–40]。STAN、LASS 和 ActNet 的适配使用同一对比任务的转子状态输入与可用监督 [41–43]。连续模型使用归档的积分核验，离散模型和按时间查询的模型使用 10 ms 输出网格。

两个 SINDy-SHRED 实现分别有 86/94 和 61/94 例的分类数值不合格。I-IKNO 与 Hankel-DMD 的稳定样本 8 s NRMSE 达到 $10^4$–$10^8$ 量级，均在 62 例稳定样本上报出失步。STAN 适配模型确认 32/32 例失稳，稳定误报 10/62，另有 21 例稳定样本未安定，稳定样本 8 s NRMSE 均值为 5.94。ES-ROM 在两档步长下的全部 94 条 8 s 轨迹均有限，逐例分类一致，未安定数为 0。

**图 3　`test` 的失稳确认召回率与稳定样本 8 s NRMSE。** 虚线连接稳定样本零误报条件下观测到的帕累托前沿。方形表示存在未安定判定，空心标记表示出现稳定误报。右图的误差轴采用对数尺度。SINDy-SHRED 标注给出没有完整稳定样本 NRMSE 时的确认召回与数值不合格数。[原图 PDF](figures/trajectory-comparison.pdf)

在 `test` 的零稳定误报轨迹模型中，ES-ROM 的确认召回最高。该条件下的循环模型对照 Graph-GRU 确认 26/32 例失稳，全样本准确率为 93.62%。两者共同确认 24 例失稳，ES-ROM 单独确认 5 例，Graph-GRU 单独确认 2 例，双方均未确认 1 例。

GRU Seq2Seq 和小型 Transformer 的稳定样本 8 s NRMSE 均值分别为 0.14865 和 0.15584，对应确认召回 24/32 和 0/32。潜空间神经 ODE 的失稳样本可用区间 NRMSE 均值最低，为 0.43973，确认召回为 18/32。ES-ROM 的稳定样本 8 s NRMSE 均值为 0.25122，确认召回为 29/32。整体轨迹误差与能否正确预测首摆失稳是两项不同的性能指标。

### D. 与直接暂态稳定分类器对比

表 5 对比 ES-ROM 的轨迹判定与直接 GRU、Extra Trees 及 Koopman 风险评分 [9, 44, 45]。经 `val` 定阈值的 GRU 在 `test` 确认 29/32 例失稳，误报 0/62，与 ES-ROM 一致。该 GRU 在 `val` 误报 5/95。对同一 GRU 使用固定阈值 0.5 时，`test` 确认 27/32 例，Extra Trees 在阈值 0.5 时确认 25/32 例。另一组固定阈值 GRU 确认 30/32 例，稳定误报为零。

经 `val` 定阈值的 Extra Trees 在 `test` 确认 31/32 例失稳并误报 13/62。以冻结的潜空间神经 ODE 为代理方程的 Koopman 风险评分确认 20/32 例，误报 8/62。ES-ROM 从预测轨迹本身作出稳定性判定，这条轨迹也提供振荡模态和机群分离信息。

**表 5　`test` 上的直接分类、风险评分与轨迹判定**

| 方法 | 阈值 | TP/32 | FP/62 | AUROC |
|---|---:|---:|---:|---:|
| GRU（初始版） | 0.5 | 30/32 | 0/62 | 0.9637 |
| GRU（修订版） | 0.5 | 27/32 | 0/62 | 0.9506 |
| GRU（修订版） | $0.00215034^{*}$ | 29/32 | 0/62 | 0.9506 |
| Extra Trees | 0.5 | 25/32 | 0/62 | 0.9819 |
| Extra Trees | $0.024^{*}$ | 31/32 | 13/62 | 0.9819 |
| 基于 PCDK 思路的风险评分 | $0.59861255^{*}$ | 20/32 | 8/62 | 0.8417 |
| ES-ROM | 轨迹判据 | 29/32 | 0/62 | -- |

星号标记在 `val` 上选定并冻结的阈值。两种 Extra Trees 拟合在阈值 0.5 时的 `test` 结果相同。每行均覆盖 94 个事件。AUROC 根据连续评分计算，以失稳为正类。ES-ROM 使用预测轨迹上的失步与安定条件作出判定。

## 七、故障后物理一致性与方法适用范围

### A. 主导机电振荡模态

在参考平衡点和零投影扰动输入处线性化冻结方程。扣除两条 COI 约束后，雅可比矩阵为 28 维。解析求导与自动微分的最大绝对差小于 $1.5\times10^{-14}$。从参考仿真的转速记录中，用分块 Hankel 矩阵的移位辨识估计经验模态 [46]。起报后的主窗和复核窗分别为 0.5–8 s 和 1.5–8 s，截断秩为 6、10 和 14。

按频率、阻尼和模态保证准则（MAC）配对。MAC 是两个复振型归一化内积的平方幅值。强匹配要求模型极点非增长、相对频率误差至多 10%、阻尼比绝对误差至多 0.05、$\mathrm{MAC}\ge0.80$。

`test` 的 62 例稳定样本中，53 例可提取符合要求的参考模态。在三组模型共同匹配的 52 个主导模态上，ES-ROM 的绝对频率误差中位数从硬 $R$ 基线的 0.0506 Hz 降至 0.0318 Hz，相对误差中位数从 7.62% 降至 4.79%，MAC 中位数为 0.9452。以全部稳定样本为分母，强匹配从 41/62 增至 52/62。`val` 的 63 个共同匹配模态上，硬 $R$ 基线与 ES-ROM 的绝对频率误差中位数分别为 0.05023 和 0.03292 Hz，其中 62 个模态的频率误差减小。$q=0$ 时功角软化项消失，因此该处的谱变化与共享骨架标定有关（表 6）。

### B. 有效耦合与惯量加权慢同调

相对于拟合前的刚度锚点，绝对变化最大的十条边中，有五条连向大惯量的外部等值机 G10，刚度降低 1.17%–1.22%。其余五条连接新英格兰区域内部机组，刚度增加 1.17%–1.22%（图 4）。这些变化在有效耦合图中体现出区间联络与区内联络的差异。

比较普通刚度拉普拉斯矩阵 $L(k^0)$ 的第二特征向量与惯量加权广义特征问题 $L(k^0)u=\lambda Mu$ 的第二特征向量所给出的两组机群 [5]。参考分群来自 `val` 的 95 个稳定样本在起报后末 2 s 的 COI 相对转速。纳入惯量后，与参考分群一致的机组对由 21/45 增至 29/45，调整兰德指数由 $-0.1111$ 增至 0.2593（图 5）。调整兰德指数衡量扣除随机一致性后两种分群的吻合程度。G10 所在组由七台机组缩小到 $\{\mathrm{G1},\mathrm{G2},\mathrm{G10}\}$，观测尾段中 G10 单独成组。$\lambda_3/\lambda_2\approx1.15$ 表明刚度–惯量问题的两个低频模态较接近，完整线性化还含阻尼和记忆效应。

**图 4　相对刚度锚点调整幅度最大的十条有效耦合边。** 红虚线表示外部等值机 G10 周围软化的连边，蓝实线表示新英格兰区域内部刚化的连边。[原图 PDF](figures/stiffness.pdf)

**图 5　普通与惯量加权刚度谱的慢同调分群。** 两种谱分群与起报后 6–8 s 的转速参考分群并列对比。[原图 PDF](figures/coherency.pdf)

### C. 失步割集

在参考轨迹和预测轨迹各自首次持续触发失步的时刻，将机组按功角排序，在相邻功角的最大间隙处划分两群。两群之间的机组对构成割集，功角较大的一群为领先机群。

在 `test` 的 32 个真实失稳样本中，ES-ROM 精确匹配割集 27 例，硬 $R$ 基线为 23 例。真值与 ES-ROM 预测均触发失步的样本有 29 例，割集和领先机群各有 27 例精确匹配，割集的 Jaccard 相似度均值为 0.9635。该相似度是两个割集中相同的跨群机组对数占其并集的比例。以全部失稳样本为分母同时衡量失稳确认与空间结构恢复，以两条轨迹均触发的样本为分母则衡量触发条件下的空间一致性。

**表 6　故障后模态对应、失步割集与起报能量排序**

**A. 主导模态与失步割集**

| 数据集 | 模型 | $\lvert\Delta f\rvert$ 中位数（Hz） | 相对频率误差中位数 | MAC 中位数 | 强匹配数 | 精确匹配数 $^{a}$ | Jaccard $^{b}$ |
|---|---|---:|---:|---:|---:|---:|---:|
| `val` | 硬 $R$ 基线 | 0.05023 | 7.57% | 0.94727 | 53/95 | 10/12 (10/11) | 0.9144 |
| `val` | ES-ROM | 0.03292 | 4.95% | 0.94532 | 58/95 | 10/12 (10/12) | 0.8818 |
| `test` | 硬 $R$ 基线 | 0.05058 | 7.62% | 0.94709 | 41/62 | 23/32 (23/25) | 0.9576 |
| `test` | 软 $2R$ 消融 | 0.05058 | 7.62% | 0.94709 | 41/62 | 26/32 (26/27) | 0.9804 |
| `test` | ES-ROM | 0.03177 | 4.79% | 0.94516 | 52/62 | 27/32 (27/29) | 0.9635 |

**B. `val` 与 `test` 的储能 AUROC**

| 模型 | `val` 机械能 | `val` 记忆储能 | `val` 总储能 | `test` 机械能 | `test` 记忆储能 | `test` 总储能 |
|---|---:|---:|---:|---:|---:|---:|
| 硬 $R$ 基线 | 0.9325 | 0.5877 | 0.9333 | 0.9526 | 0.5282 | 0.8926 |
| 软 $2R$ 消融 | 0.9325 | 0.6140 | 0.7175 | 0.9526 | 0.5983 | 0.7157 |
| ES-ROM | 0.9316 | 0.5623 | 0.6702 | 0.9526 | 0.5766 | 0.6956 |

主导模态指标的中位数基于共同匹配事件计算，`val` 为 63 例，`test` 为 52 例。强匹配要求相对频率误差 $\leq10\%$、阻尼比绝对误差 $\leq0.05$、MAC $\geq0.80$，且模型模态不增长。强匹配数以各划分的全部稳定样本为分母。

$^{a}$ 在 `val` 上计领先机群精确匹配，在 `test` 上计割集精确匹配，其次数与领先机群精确匹配相同。分别以全部失稳样本和真值、预测均触发失步的样本为分母，后一项加括号。$^{b}$ 跨群机组对的 Jaccard 相似度，在真值和预测均触发的事件上按事件平均。

机械能为动能与移位耦合势能之和。硬 $R$ 基线和软 $2R$ 消融的机械能逐例相同。ES-ROM 使用自身势能，`test` 机械能 AUROC 在所示精度内与两组对照相同。总储能还包括估计的记忆储能。

### D. 起报机械暂态能与记忆储能

在起报时刻 $t_o$，机械暂态能为 $E_{\rm mech}=\tfrac12 v^\top Mv+U(q)$。它使用实测可见状态与冻结骨架的移位势能。总储能 $H$ 的另一部分是估计所得的记忆储能 $E_z=\tfrac12\lVert z(t_o)\rVert_2^2$。

硬 $R$ 基线和软 $2R$ 消融共用骨架和可见起报状态，因此逐例机械能相同。三组模型在 `test` 的机械能 AUROC 均为 0.9526（按表中精度）。只改变初始化方式时，`test` 的总储能 AUROC 由 0.8926 变为 0.7157。ES-ROM 的总储能和记忆储能 AUROC 分别为 0.6956 和 0.5766。图 6 显示放宽记忆状态范围对储能分布的影响。机械项对失稳风险保持较高的排序区分度，配对消融支持将 $E_z$ 解释为与有效记忆力相关的储能。机械能 AUROC 描述的是本次故障分布下的排序性能。

**图 6　起报储能与失稳区分度。** 子图（a）–（c）分别给出 `train`、`val` 和 `test` 各能量分量的 AUROC。子图（d）和（e）按模型和真值标签给出 `val` 与 `test` 的记忆储能中位数及四分位区间。AUROC 以失稳为正类。[原图 PDF](figures/energy.pdf)

### E. 方法适用范围

辨识方程描述故障清除并恢复拓扑后的运动。估计临界切除时间还需要故障期间的动力学和拓扑切换规则。在 100 ms 观测与恒定机械功率的预测协议下，`test` 有 3 例真实失稳被判为稳定。

当机端电压深度跌落且功角大幅摆动时，采用固定等值内电势的移位正弦势能可能高估电磁恢复力，其保守形式省略了传输电导引起的非保守效应。从 $t_o=0.10\,\mathrm{s}$ 起保持机械功率恒定会省略后续调速器作用。模型也省略了临界摆动中较晚触发的励磁限制。各算子组合由互易端口和半正定阻尼保证 $\dot H\le v^\top p$，适用于有阻尼的故障后恢复与首摆分离。描述主动控制回路引起的低频负阻尼振荡，需要表示该回路注入的能量。

`test` 覆盖十机系统同一运行点的未见故障条件。跨运行点迁移、规模更大的电网以及高比例逆变器电源的系统需要独立验证。

## 八、结论

ES-ROM 选择出一个共享的显式故障后方程，包含 74 个骨架系数和一个逐机阻尼软化系数。用 100 ms 的转子观测初始化，在 `test` 上确认 90.62% 的失稳样本，稳定误报为零，全样本准确率为 96.81%。同一冻结方程给出的主导模态、失步割集和储能有助于解释预测结果。

结合故障期间与故障后动力学的切换能量模型可支持临界切除时间研究。动态机械功率状态可表示观测窗之后的调速器作用，考虑电压变化或非保守效应的网络项可刻画严重电压跌落，带闭环稳定性证书的有界有源端口可用于描述主动振荡。稀疏有效耦合图与逆变器控制算子可用于检验动量初筛和能量一致算子选择方法在规模更大、组成更多样的电网中的适用性。

## 参考文献

1. N. Hatziargyriou, J. Milanović, C. Rahmann, V. Ajjarapu, C. Cañizares, I. Erlich, D. Hill, I. Hiskens, I. Kamwa, B. Pal, P. Pourbeik, J. Sanchez-Gasca, A. Stanković, T. Van Cutsem, V. Vittal, and C. Vournas, “Definition and classification of power system stability—revisited & extended,” *IEEE Trans. Power Syst.*, 36(4):3271–3281, 2021. DOI: [10.1109/TPWRS.2020.3041774](https://doi.org/10.1109/TPWRS.2020.3041774).
2. A. G. Phadke and J. S. Thorp, *Synchronized Phasor Measurements and Their Applications*. Springer, 2008. DOI: [10.1007/978-0-387-76537-2](https://doi.org/10.1007/978-0-387-76537-2).
3. A. K. Singh and B. C. Pal, “Decentralized dynamic state estimation in power systems using unscented transformation,” *IEEE Trans. Power Syst.*, 29(2):794–804, 2014. DOI: [10.1109/TPWRS.2013.2281323](https://doi.org/10.1109/TPWRS.2013.2281323).
4. M. Shiroei, B. Mohammadi-Ivatloo, and M. Parniani, “Low-order dynamic equivalent estimation of power systems using data of phasor measurement units,” *Int. J. Electr. Power Energy Syst.*, 74:134–141, 2016. DOI: [10.1016/j.ijepes.2015.07.015](https://doi.org/10.1016/j.ijepes.2015.07.015).
5. P. V. Kokotović, B. Avramović, J. H. Chow, and J. R. Winkelman, “Coherency based decomposition and aggregation,” *Automatica*, 18(1):47–56, 1982. DOI: [10.1016/0005-1098(82)90025-5](https://doi.org/10.1016/0005-1098(82)90025-5).
6. Y. Susuki and I. Mezić, “Nonlinear Koopman modes and coherency identification of coupled swing dynamics,” *IEEE Trans. Power Syst.*, 26(4):1894–1904, 2011. DOI: [10.1109/TPWRS.2010.2103369](https://doi.org/10.1109/TPWRS.2010.2103369).
7. Z. Jiang, N. Tong, Y. Liu, Y. Xue, and A. G. Tarditi, “Enhanced dynamic equivalent identification method of large-scale power systems using multiple events,” *Electr. Power Syst. Res.*, 189:106569, 2020. DOI: [10.1016/j.epsr.2020.106569](https://doi.org/10.1016/j.epsr.2020.106569).
8. J. L. Jardim and A. M. Leite da Silva, “A methodology for computing robust dynamic equivalents of large power systems,” *Electr. Power Syst. Res.*, 143:513–521, 2017. DOI: [10.1016/j.epsr.2016.11.003](https://doi.org/10.1016/j.epsr.2016.11.003).
9. Q. Chen and H. Wang, “Time-adaptive transient stability assessment based on gated recurrent unit,” *International Journal of Electrical Power & Energy Systems*, 133:107156, 2021. DOI: [10.1016/j.ijepes.2021.107156](https://doi.org/10.1016/j.ijepes.2021.107156).
10. J. Liu, C. Yao, and L. Chen, “Time-adaptive transient stability assessment based on the gating spatiotemporal graph neural network and gated recurrent unit,” *Frontiers in Energy Research*, 10:885673, 2022. DOI: [10.3389/fenrg.2022.885673](https://doi.org/10.3389/fenrg.2022.885673).
11. S. P. Nandanoori, S. Guan, S. Kundu, S. Pal, K. Agarwal, Y. Wu, and S. Choudhury, “Graph neural network and Koopman models for learning networked dynamics: A comparative study on power grid transients prediction,” *IEEE Access*, 10, 2022. DOI: [10.1109/ACCESS.2022.3160710](https://doi.org/10.1109/ACCESS.2022.3160710).
12. G. S. Misyris, A. Venzke, and S. Chatzivasileiadis, “Physics-informed neural networks for power systems,” in *2020 IEEE Power & Energy Society General Meeting (PESGM)*, pp. 1–5. DOI: [10.1109/PESGM41954.2020.9282004](https://doi.org/10.1109/PESGM41954.2020.9282004).
13. J. Stiasny, B. Zhang, and S. Chatzivasileiadis, “PINNSim: A simulator for power system dynamics based on physics-informed neural networks,” *Electr. Power Syst. Res.*, 235:110796, 2024. DOI: [10.1016/j.epsr.2024.110796](https://doi.org/10.1016/j.epsr.2024.110796).
14. T. Athay, R. Podmore, and S. Virmani, “A practical method for the direct analysis of transient stability,” *IEEE Trans. Power App. Syst.*, PAS-98(2):573–584, 1979. DOI: [10.1109/TPAS.1979.319407](https://doi.org/10.1109/TPAS.1979.319407).
15. A. R. Bergen and D. J. Hill, “A structure preserving model for power system stability analysis,” *IEEE Trans. Power App. Syst.*, PAS-100(1):25–35, 1981. DOI: [10.1109/TPAS.1981.316883](https://doi.org/10.1109/TPAS.1981.316883).
16. F. Dörfler and F. Bullo, “Kron reduction of graphs with applications to electrical networks,” *IEEE Trans. Circuits Syst. I, Reg. Papers*, 60(1):150–163, 2013. DOI: [10.1109/TCSI.2012.2215780](https://doi.org/10.1109/TCSI.2012.2215780).
17. S. Y. Çalışkan and P. Tabuada, “Towards Kron reduction of generalized electrical networks,” *Automatica*, 50(10):2586–2590, 2014. DOI: [10.1016/j.automatica.2014.08.017](https://doi.org/10.1016/j.automatica.2014.08.017).
18. H. Mori, “Transport, collective motion, and Brownian motion,” *Prog. Theor. Phys.*, 33(3):423–455, 1965. DOI: [10.1143/PTP.33.423](https://doi.org/10.1143/PTP.33.423).
19. A. Gouasmi, E. J. Parish, and K. Duraisamy, “A priori estimation of memory effects in reduced-order models of nonlinear systems using the Mori–Zwanzig formalism,” *Proc. R. Soc. A*, 473(2205):20170385, 2017. DOI: [10.1098/rspa.2017.0385](https://doi.org/10.1098/rspa.2017.0385).
20. L. Pagnier, R. Delabays, and M. Tyloo, “Nontrivial Kron reduction for power grid dynamics modeling,” in *Proc. IEEE Kiel PowerTech*, 2025, pp. 1–6. DOI: [10.1109/PowerTech59965.2025.11180705](https://doi.org/10.1109/PowerTech59965.2025.11180705).
21. S. L. Brunton, J. L. Proctor, and J. N. Kutz, “Discovering governing equations from data by sparse identification of nonlinear dynamical systems,” *Proc. Natl. Acad. Sci. USA*, 113(15):3932–3937, 2016. DOI: [10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113).
22. H. Schaeffer and S. G. McCalla, “Sparse model selection via integral terms,” *Phys. Rev. E*, 96(2):023302, 2017. DOI: [10.1103/PhysRevE.96.023302](https://doi.org/10.1103/PhysRevE.96.023302).
23. D. A. Messenger and D. M. Bortz, “Weak SINDy: Galerkin-based data-driven model selection,” *Multiscale Model. Simul.*, 19(3):1474–1497, 2021. DOI: [10.1137/20M1343166](https://doi.org/10.1137/20M1343166).
24. K. Champion, B. Lusch, J. N. Kutz, and S. L. Brunton, “Data-driven discovery of coordinates and governing equations,” *Proc. Natl. Acad. Sci. USA*, 116(45):22445–22451, 2019. DOI: [10.1073/pnas.1906995116](https://doi.org/10.1073/pnas.1906995116).
25. J. Bakarji, K. Champion, J. N. Kutz, and S. L. Brunton, “Discovering governing equations from partial measurements with deep delay autoencoders,” *Proc. R. Soc. A*, 479(2276):20230422, 2023. DOI: [10.1098/rspa.2023.0422](https://doi.org/10.1098/rspa.2023.0422).
26. M. L. Gao, J. P. Williams, and J. N. Kutz, “Sparse identification of nonlinear dynamics and Koopman operators with shallow recurrent decoder networks,” *Proceedings of the National Academy of Sciences*, 123(16):e2508144123, 2026. DOI: [10.1073/pnas.2508144123](https://doi.org/10.1073/pnas.2508144123).
27. B. Peherstorfer and K. E. Willcox, “Data-driven operator inference for nonintrusive projection-based model reduction,” *Comput. Methods Appl. Mech. Eng.*, 306:196–215, 2016. DOI: [10.1016/j.cma.2016.03.025](https://doi.org/10.1016/j.cma.2016.03.025).
28. J.-C. Loiseau and S. L. Brunton, “Constrained sparse Galerkin regression,” *J. Fluid Mech.*, 838:42–67, 2018. DOI: [10.1017/jfm.2017.823](https://doi.org/10.1017/jfm.2017.823).
29. P. Goyal, I. Pontes Duff, and P. Benner, “Guaranteed stable quadratic models and their applications in SINDy and operator inference,” *Physica D*, 483:134893, 2025. DOI: [10.1016/j.physd.2025.134893](https://doi.org/10.1016/j.physd.2025.134893).
30. A. J. van der Schaft and B. M. Maschke, “Port-Hamiltonian systems on graphs,” *SIAM J. Control Optim.*, 51(2):906–937, 2013. DOI: [10.1137/110840091](https://doi.org/10.1137/110840091).
31. S. Chaturantabut, C. Beattie, and S. Gugercin, “Structure-preserving model reduction for nonlinear port-Hamiltonian systems,” *SIAM J. Sci. Comput.*, 38(5):B837–B865, 2016. DOI: [10.1137/15M1055085](https://doi.org/10.1137/15M1055085).
32. C. Neary and U. Topcu, “Compositional learning of dynamical system models using Port-Hamiltonian neural networks,” in *Proc. 5th Annu. Learn. Dyn. Control Conf.*, ser. Proc. Mach. Learn. Res., vol. 211, pp. 679–691, 2023. [https://proceedings.mlr.press/v211/neary23a.html](https://proceedings.mlr.press/v211/neary23a.html).
33. J. Rettberg, J. Kneifl, J. Herb, P. Buchfink, J. Fehr, and B. Haasdonk, “Data-driven identification of latent port-Hamiltonian systems,” *Computational Science and Engineering*, 2:4, 2025. DOI: [10.1007/s44207-025-00007-2](https://doi.org/10.1007/s44207-025-00007-2).
34. H. Cui, F. Li, and K. Tomsovic, “Hybrid symbolic-numeric framework for power system modeling and analysis,” *IEEE Trans. Power Syst.*, 36(2):1373–1384, 2021. DOI: [10.1109/TPWRS.2020.3017019](https://doi.org/10.1109/TPWRS.2020.3017019).
35. K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in *Proc. Conf. Empirical Methods in Natural Language Processing*, 2014, pp. 1724–1734. DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179).
36. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).
37. Y. Rubanova, R. T. Q. Chen, and D. K. Duvenaud, “Latent ODEs for irregularly-sampled time series,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019. [https://arxiv.org/abs/1907.03907](https://arxiv.org/abs/1907.03907).
38. H. Arbabi and I. Mezić, “Ergodic theory, dynamic mode decomposition, and computation of spectral properties of the Koopman operator,” *SIAM Journal on Applied Dynamical Systems*, 16(4):2096–2126, 2017. DOI: [10.1137/17M1125236](https://doi.org/10.1137/17M1125236).
39. L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, “Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators,” *Nature Machine Intelligence*, 3:218–229, 2021. DOI: [10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5).
40. X. Pu, W. Pei, H. Xiao, J. Xiong, X. Wang, G. Wang, and Y. Sun, “A dynamic trajectory computation method for power systems based on an improved invertible Koopman neural operator,” *Advanced Technology of Electrical Engineering and Energy*, 2026, early access. DOI: [10.12067/ATEEE2601079](https://doi.org/10.12067/ATEEE2601079).
41. S. Chung, Y. Zhang, M. Yue, and H. Cui, “Network interdependency-informed power system dynamic trajectory prediction utilizing black-box modeling of multiple inverter-based resources,” arXiv:2607.05843, 2026. [https://arxiv.org/abs/2607.05843](https://arxiv.org/abs/2607.05843).
42. H. Li, L. Mai, C. Xiao, E. Blasch, and Y. Weng, “Predicting power-system dynamic trajectories with foundation models,” arXiv:2604.14991, 2026. [https://arxiv.org/abs/2604.14991](https://arxiv.org/abs/2604.14991).
43. P. Ellinas, J. Vorwerk, and S. Chatzivasileiadis, “Tools to explain neural networks for power system dynamics,” arXiv:2608.08048, 2026. [https://arxiv.org/abs/2608.08048](https://arxiv.org/abs/2608.08048).
44. P. Geurts, D. Ernst, and L. Wehenkel, “Extremely randomized trees,” *Machine Learning*, 63(1):3–42, 2006. DOI: [10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1).
45. Z. Du, S. Shan, and Y. Liu, “Region of attraction estimation for power system transient stability via physics-constrained deep Koopman operator,” *IET Generation, Transmission & Distribution*, 20(1):e70428, 2026. DOI: [10.1049/gtd2.70428](https://doi.org/10.1049/gtd2.70428).
46. J.-N. Juang and R. S. Pappa, “An eigensystem realization algorithm for modal parameter identification and model reduction,” *Journal of Guidance, Control, and Dynamics*, 8(5):620–627, 1985. DOI: [10.2514/3.20031](https://doi.org/10.2514/3.20031).
