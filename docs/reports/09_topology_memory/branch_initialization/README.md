# 按耦合强度倍率估计各支路的初值

[返回主题页](../README.md)

## 运行结果与归档

- [运行结果（归档）](../../../../results/backup/09_topology_memory/05_branch_initialization.DESCRIPTION.md#run-1)

登记文件与配置

[方案](../../../../configs/proposals/topology_memory_rom_branch_init_v1.json)

## 执行协议

<a id="doc-9cb0993dde26"></a>

<a id="doc-9cb0993dde26-按耦合强度倍率估计各支路的初值"></a>

### 按耦合强度倍率估计各支路的初值

首次合成对照中，支路耦合使用真实起报状态时，200 ms 的最大归一化动量误差约为 \(10^{-11}\)。用 100 ms 历史估计初值后，单独改变各支路耦合时误差升至 \(4.15\times10^{-4}\)，同时改变整体和各支路耦合时升至 \(4.02\times10^{-3}\)。本对照固定参数，检验改用耦合强度的倍率求解能否改善初值估计。


<a id="doc-9cb0993dde26-执行前提与范围"></a>

#### 执行前提与范围

待原运行关闭并核验真值、名单及模型来源摘要后再登记本诊断。沿用五种冻结真值队列及各自完整 train 24、val 12、test 12。只重算启用“各支路耦合强度分别变化”的支路单项与“整体变化和各支路分别变化并存”候选；固定耦合和仅整体变化候选沿用不可变证据。test 仅作报告，不参与选择。全程不读取真实数据。


<a id="doc-9cb0993dde26-成对初始化"></a>

#### 成对初始化

两种方法均使用同一模型系数、100 ms 历史、五次 Gauss–Newton、记忆先验权重 0、记忆半径 1 及原冻结拓扑半径 0.6。逐例初始状态、历史次序、输入、精度和停止规则配对一致。

- **原坐标**：直接以原支路坐标 \(\xi\) 执行五次更新。
- **倍率坐标** 使用 \(w=\xi/\sqrt{1+\|\xi\|^2/b^2}\)。支路对数倍率由固定矩阵乘以 \(w\) 得到。初值灵敏度的支路列按 \(\partial\xi/\partial w=a(I+\xi\xi^\top/b^2)\) 转换，其中 \(a=\sqrt{1+\|\xi\|^2/b^2}\)。每次更新先把 \(\|w\|\) 限于物理半径 0.6 映射得到的范围，再反解 \(\xi=w/\sqrt{1-\|w\|^2/b^2}\)，随后执行原联合半径约束。整体耦合的求解坐标沿用 \(c\)，联合投影可能同时缩小 \(c\) 和 \(\xi\)。

两种方法使用同一组干净历史，方程和起报规则相同。五组真值档案按原校验值读取。额外预算为 4,800 次逐事件支路灵敏度矩阵乘法。


<a id="doc-9cb0993dde26-评分与判定"></a>

#### 评分与判定

仅对含支路变化的两种候选重算五种真值的完整 train、val、test，使用原冻结的 2.5 ms、1.25 ms 双步长。继续核对 200 ms 动量误差、隐状态误差、8 秒轨迹、稳定性、覆盖与数值失败。使用真实起报状态的短时对照仅在 val 执行。验收使用原阈值，test 结果只报告。

科学结论仅回答：在已知系数、不训练共享参数时，改变支路状态的优化坐标能否改善因果初值恢复并通过原有 val 门。它不证明参数拟合能恢复正确结构，不构成独立数据确认，也不授权真实 TASK3 工作。


<a id="doc-9cb0993dde26-产物与边界"></a>

#### 产物与边界

保存新运行的逐例历史与预测路径、成组隐状态误差、短时动量误差、覆盖与数值失败，并绑定原运行和复用评分的校验值。原运行保留独立回执。

## 回执与附件

- [topology_memory_rom_branch_init_v1_20261117_evidence.json](_sealed/topology_memory_rom_branch_init_v1_20261117_evidence.json)
- [topology_memory_rom_branch_init_v1_result_review_20261117.json](_sealed/topology_memory_rom_branch_init_v1_result_review_20261117.json)
- [topology_memory_rom_branch_init_v1_review_20261117.json](_sealed/topology_memory_rom_branch_init_v1_review_20261117.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/topology_memory_rom_branch_init_v1_zh.md](_sealed/topology_memory_rom_branch_init_v1_zh.md)

</details>
