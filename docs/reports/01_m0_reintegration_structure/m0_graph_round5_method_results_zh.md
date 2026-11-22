# Round 5 方法与结果报告：图结构 M0

## 核心结论

在 canonical IEEE39 故障清除后数据上，即使直接使用 observed
`Pm(t)`，并显式约束节点阻尼、边阻尼、局部刚度和线性化模态稳定，
M0-GRAPH 仍未稳定提升闭环重积分精度。因此 Round 5 在验证阶段判定为
**REJECT**；测试集未解锁，稳定性 FSR 未计算。

## 1. 方法

本轮只使用 seed 324 和五折 `combined_line_5fold_v1`。所有有效长度小于
100 的 train/validation 轨迹均在拟合、归一化和选择前排除。H=100
验证集共 177 条轨迹，覆盖 33 个 fault-line cluster。所有模型从精确
clearing state 起步，共用 observed `Pm(t)`、RK 阶段线性插值、COI 投影
和最大内部步长 0.01 s 的 RK4。

比较对象包括：

1. `M0`：重新拟合的积分弱形式 swing 模型；
2. `M0-DD`：在当前相同数据和积分器口径下，用二阶数值导数直接拟合的
   诊断基线；
3. `M0-GRAPH`：新增节点阻尼、边阻尼及平衡点锚定的 sine/cosine
   图势能力。

M0-GRAPH 使用

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s\Delta\sin\eta+K_c\Delta\cos\eta+D_eB^T\omega\}.
\]

其中 (D_n,D_e,K_s\ge0)，(K_c) 可带符号，并约束

\[
K_s\cos\eta_{eq}-K_c\sin\eta_{eq}\ge0.
\]

训练采用 1、5、20、40、100 个间隔的非重叠积分窗口，每个
“轨迹×窗口长度”分层等权。M0 和静态 M0-GRAPH 都没有 epoch；每个
ridge 只求解一次约束方程组。首次 run 暴露出 SLSQP 未收敛候选仍可被
物理门接纳的问题。修复版使用逐边二射线凸锥参数化，并硬性拒绝
`solver_success=false` 的候选；旧 run 保留但不作为最终证据。

## 2. 重积分结果

| 模型 | H=100 mean RMSE | median RMSE | pooled RMSE | worst-10% CVaR | pooled 频率 R² | failure rate |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.2846 | 0.1242 | 0.4326 | 0.9839 | 0.4366 | 0% |
| M0-DD | 0.2726 | 0.1195 | 0.4112 | 0.9214 | 0.4925 | 0% |
| M0-GRAPH | 0.5135 | 0.1432 | 1.1872 | 3.1893 | -3.2248 | 0% |
| FINAL | 0.2683 | 0.1408 | 0.3996 | 0.9028 | 0.5211 | 0% |

按折选择结果为：F1、F3、F5 选择 M0-DD，F2、F4 选择 M0-GRAPH。
FINAL 相对 M0 的 H=100 mean RMSE 只改善 5.72%，未达到 20% 门；五折
改善分别为 +4.16%、-1.64%、+11.64%、+7.89%、+4.07%。尽管
fault-line cluster bootstrap 的 FINAL−M0 配对差为 -0.0163，95% CI
为 [-0.0286, -0.00186]，但 pooled 频率 R² 仅 0.521，且 M0-GRAPH
只在 2/5 折入选，所以整体验证门失败。

![Round 5 验证汇总](../../../results/figures/m0-graph-round5-solver-fix/fig_round5_validation_summary.png)

## 3. 阻尼、刚度与模态审计

修复后五折求解器均成功收敛，迭代次数为 430--857；平衡点锚定、内部力
平衡、共同角平移不变性、局部刚度和线性化模态门均通过。但参数落在约束
边界附近：五折近零节点阻尼数分别为 7、6、6、4、6（每折共 7 个），
近零边阻尼数为 15、18、15、14、15（每折共 21 个）。最小 quotient
阻尼特征值在 F1、F2、F5 近似为零；最大模态实部虽均为负，但 F1 只有
-0.0037，距离中性稳定很近。

设计矩阵条件数在 (7.67\times10^3) 到 (9.28\times10^3) 之间，说明
70 参数的完整图模型辨识条件较差。更重要的是，弱形式残差不能预测闭环
精度：F5 的加权弱残差最小（2.59），但验证 mean RMSE 为 0.741；F2 的
弱残差更大（4.09），验证 mean RMSE 却只有 0.149。这说明“方程残差小、
局部模态稳定”并不等于“长程相位和幅值正确”。

![Round 5 参数审计](../../../results/figures/m0-graph-round5-solver-fix/fig_round5_parameter_audit.png)

轨迹图采用每折 M0-GRAPH−M0 H=100 RMSE 差的中位事件，不挑选最佳样本。
误差主要表现为初始瞬态之后的相位和幅值偏差，而不是 clearing state
错误。

![Round 5 典型轨迹](../../../results/figures/m0-graph-round5-solver-fix/fig_round5_trajectory_examples.png)

## 4. 解释与边界

本轮证据表明，物理可行性与预测充分性必须分开。M0-GRAPH 的局部曲率和
线性化模态是合格的，但完整图上的 sine/cosine/边阻尼共 70 个参数存在
明显冗余；高条件数和跨折边界解支持“辨识性不足”这一解释。该解释来自
诊断关联，并非与真实 Kron 参数比较后的因果证明。

M0-DD 在当前较干净的仿真数据上有小幅经验收益，但仍未达到精度门，且
不能解决物理参数辨识问题。因此它适合作为初始化或诊断基线，不应直接
替代主弱形式模型。

本轮已经使用 observed `Pm(t)`，所以当前失败不能归因于 ZOH。此时加入
Pm ODE 会把输入模型误差与尚未解决的状态动力学误差混合，不建议继续。

## 5. 决策

**Round 5：REJECT。** 因验证精度门失败，测试数据保持锁定；没有 test
重积分结果，也没有稳定性 FSR。这是预注册流程的正确执行，不是漏算。

建议下一轮只做一个受限改动：把 70 参数完整图降为低维 quotient 模态
阻尼/刚度模型，在低维空间保证半正定，再只增加少量与线性化算子绑定的
非线性校正。保持相同长轨迹人口、observed `Pm(t)`、M0/M0-DD 基线和
验证门；暂不增加 Pm ODE，也不做多步神经优化。该方案需下一轮明确批准
后再实施。

## 6. 可复现信息

- 修复版源码提交：`ac3a3dc867ea96dee1ecfd91a55a5bce21745c3a`
- 修复版不可变 run：`paper_m0_graph_round5_solver_fix_validation-validation_run-20260729T104104Z-ac3a3dc8`
- 测试：33 passed
- 项目审计：PASS
- 测试集读取：false；test IDs：空
