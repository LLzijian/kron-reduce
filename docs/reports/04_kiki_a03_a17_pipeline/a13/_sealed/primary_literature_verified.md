# 初态估计与短历史可辨识：原始文献核验

检索按 `nature-academic-search` 的 multi-source-search 流程完成，限于论文原文、作者预印本和官方出版社页面。问题范围按 [固定模型积分初态方案](integral_initializer_fixed_protocol_zh.md)：用有限 q/v 历史估计固定 pH 结构模型的隐藏起点状态。以下文献提供方法背景或理论边界，不替代该方案自己的合成真值检验和固定预算审计。

## 1. 积分估计可避免显式数值微分，但结论依赖模型和可辨识条件

**Itai Dattner、Chris A. J. Klaassen，2015。** “Optimal Rate of Direct Estimators in Systems of Ordinary Differential Equations Linear in Functions of the Parameters.” *Electronic Journal of Statistics* 9(2): 1939–1973. DOI: [10.1214/15-EJS1053](https://doi.org/10.1214/15-EJS1053)；作者预印本 [arXiv:1305.4126](https://arxiv.org/abs/1305.4126)。

核实内容：作者针对对未知参数线性的 ODE，提出积分形式的直接估计，避免估计状态导数；论文给出其模型类别中的参数可辨识条件及相应渐近率。原文还把初值 \(\xi\) 纳入估计式。这里可支持“积分残差方法能绕开显式数值微分，并可在特定假设下估计参数和初值”。

不能支持：它不证明本项目的隐藏变量 \(z_h\) 在部分可见、固定且可能失配的 pH 骨架下可恢复；更不证明降低本项目的积分最小二乘残差等于恢复真实物理隐藏状态。其假设、观测状态和估计对象与本方案不同。

## 2. pH 模型识别中，内部初态本身可以成为校准变量

**Michael Günther、Birgit Jacob、Claudia Totzeck，2023 年 arXiv 预印本。** “Structure-preserving identification of port-Hamiltonian systems — a sensitivity-based approach.” [arXiv:2301.02019](https://arxiv.org/abs/2301.02019)。

核实内容：作者描述了从时域输入输出数据校准 pH 系统的梯度方法，并在迭代中保持矩阵的斜对称和半正定结构；摘要明确说，只有输入输出观测时，内部状态初值也需要校准，并以合成数据展示方法可行性。

不能支持：这是数值校准方法及合成可行性结果，不是“输出误差下降即隐藏状态真实恢复”的一般定理，也没有验证本项目基于积分动量平衡的固定模型 LS 初态估计。请把它用作问题类比，不作本方法有效性的证据。

## 3. 连续时间局部弱可观测性有明确前提

**Robert Hermann、Arthur J. Krener，1977。** “Nonlinear Controllability and Observability.” *IEEE Transactions on Automatic Control* 22(5): 728–740. DOI: [10.1109/TAC.1977.1101601](https://doi.org/10.1109/TAC.1977.1101601)；作者提供的论文原文：[PDF](https://www.math.ucdavis.edu/~krener/1-25/10.IEEETAC77.pdf)。

核实内容：论文建立光滑连续时间非线性控制系统的可控性、可观测性框架；本方案引用的定理针对给定系统上的局部弱可观测性条件。

不能支持：连续时间的可观测性秩条件本身不等价于有限个离散样本下的稳定恢复，也不量化噪声、模型误差或短窗口的条件数。对本任务应把连续秩分析、实际离散矩阵秩、奇异值敏感度和独立真值误差分别报告。

## 4. 连续可观测不自动保留到离散采样

**S. Ammar、J.-C. Vivalda，2004。** “On the preservation of observability under sampling.” *Systems & Control Letters* 52(1): 7–15. DOI: [10.1016/j.sysconle.2003.08.008](https://doi.org/10.1016/j.sysconle.2003.08.008)；[官方出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S0167691103002925)。

核实内容：论文直接研究采样下可观测性的保持问题。出版社页面的引言指出，数字控制场景只在固定离散时刻观察连续系统；其结论针对采样间隔及系统条件，研究何时可保持可观测性。

不能支持：不能将其概括成“采样必然破坏可观测性”；同样也不能把连续可观测性或某个离散矩阵满秩，推广成当前短历史必然足以稳定恢复隐藏状态。具体定理条件须与本项目的观测映射、采样时钟和窗口匹配。

## 5. 连续系统可辨识仍可能遭遇系统混叠

**Zhexuan Zeng、Zuogong Yue、Alexandre Mauroy、Jorge Goncalves、Ye Yuan，2024（arXiv 预印本 2022）。** “A Sampling Theorem for Exact Identification of Continuous-time Nonlinear Dynamical Systems.” *IEEE Transactions on Automatic Control* 69(12): 8402–8417. DOI: [10.1109/TAC.2024.3409639](https://doi.org/10.1109/TAC.2024.3409639)；作者预印本：[arXiv:2204.14021](https://arxiv.org/abs/2204.14021)。

核实内容：作者研究从离散采样精确识别连续非线性动力系统的条件，给出 Koopman 框架下的采样周期/系统混叠判据；摘要明确指出，即使模型本身可辨识，低采样频率仍会使连续时间系统识别受挑战，并可能产生相同离散观测对应不同连续动力学的系统混叠。

不能支持：论文的系统类、频谱条件和识别目标须逐项核对；它不直接证明本项目特定隐藏状态不可辨，也不提供一个通用采样率阈值供当前历史使用。它支持的是谨慎边界：离散样本秩或拟合良好不能单独排除采样混叠。

## 对当前方案的适用边界

这些来源共同支持两点：积分形式可以减少显式求导带来的数值问题；结构化动力系统的初态及采样条件都应作为识别问题的一部分处理。它们没有替当前任务证明隐藏 \(z_h\) 已被物理恢复。方案中提出的可证伪检查仍需依赖已知合成真值、固定模型下的离散条件数和独立闭环历史重建；残差更小、离散秩满或连续秩条件成立，都不能单独升级为真实隐藏状态恢复证据。

检索范围说明：Ammar–Vivalda 官方页面可核对元数据与研究问题，但本次访问未取得出版社全文，因此这里只陈述页面可核实的范围，不转述其定理的详细假设。其余内容以作者 arXiv 页面/论文原文或 IEEE 出版社页面核验。
