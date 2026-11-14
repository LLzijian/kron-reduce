# 逐机方程与函数项恢复：合成结果

2026-11-14。

完整 val 为三组合成真值都选对了函数项。所选方程的记忆核相对误差均超过 5%，调制耗散组的长时轨迹也未达到门槛。本轮保留默认基线。

## 模型与范围

A 学习 74 个共享系数，包括 45 个边刚度、9 个原有平衡基阻尼系数、10 个局部端口幅值和 10 个记忆时间常数。B 从同一起点增加二阶谐波恢复力 `a` 和状态调制耗散 `b/8`，两个幅值均从精确零开始。谐波项来自势能，调制耗散矩阵为半正定。

合成范围为 `null`、`harmonic`、`damping` 三个人群，每个人群按固定名单划分 train 16、val 8、test 8。每个人群评价六个冻结方程：真参数 `known`、A、完整 B、仅谐波、仅耗散、无新增项；删项方案从完整 B 的 val 最优点重拟合。拟合在 CPU float64 单线程进行，正式历史预测与轨迹积分在 CUDA float64 上进行。每个结果均包含 2.5 ms 与 1.25 ms 预测步长。

## 轨迹误差与稳定性

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

## 函数恢复与物理门禁

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

## 拟合轨迹与运行记录

`training.jsonl` 保存逐更新 train 目标；`selection.json` 保存每 20 步完整 val 检查和早停状态。B 初始拟合上限 240 步，删项重拟合上限 80 步。以下列出实际更新数、所选更新数及所选 val 目标，并链接两条记录。

| 人群 | 拟合器 | 更新数 | 选中更新 | 选中 val NMSE | train 曲线 | val 曲线 |
|---|---|---:|---:|---:|---|---|
| null | a | 240 | 240 | 2.5502e-06 | [null/a/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/a/training.jsonl) | [null/a/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/a/selection.json) |
| null | both | 240 | 240 | 3.423e-06 | [null/both/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/both/training.jsonl) | [null/both/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/both/selection.json) |
| null | harmonic | 80 | 80 | 3.3666e-07 | [null/harmonic/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/harmonic/training.jsonl) | [null/harmonic/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/harmonic/selection.json) |
| null | damping | 80 | 80 | 8.1779e-07 | [null/damping/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/damping/training.jsonl) | [null/damping/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/damping/selection.json) |
| null | neither | 80 | 80 | 2.759e-07 | [null/neither/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/neither/training.jsonl) | [null/neither/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/null/neither/selection.json) |
| harmonic | a | 100 | 40 | 0.0016626 | [harmonic/a/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/a/training.jsonl) | [harmonic/a/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/a/selection.json) |
| harmonic | both | 240 | 240 | 5.4581e-05 | [harmonic/both/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/both/training.jsonl) | [harmonic/both/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/both/selection.json) |
| harmonic | harmonic | 80 | 80 | 5.9244e-07 | [harmonic/harmonic/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/harmonic/training.jsonl) | [harmonic/harmonic/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/harmonic/selection.json) |
| harmonic | damping | 80 | 80 | 0.00080807 | [harmonic/damping/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/damping/training.jsonl) | [harmonic/damping/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/damping/selection.json) |
| harmonic | neither | 80 | 80 | 0.00078052 | [harmonic/neither/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/neither/training.jsonl) | [harmonic/neither/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/harmonic/neither/selection.json) |
| damping | a | 80 | 20 | 0.16117 | [damping/a/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/a/training.jsonl) | [damping/a/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/a/selection.json) |
| damping | both | 240 | 240 | 0.064616 | [damping/both/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/both/training.jsonl) | [damping/both/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/both/selection.json) |
| damping | harmonic | 80 | 60 | 0.2508 | [damping/harmonic/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/harmonic/training.jsonl) | [damping/harmonic/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/harmonic/selection.json) |
| damping | damping | 80 | 80 | 0.001688 | [damping/damping/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/damping/training.jsonl) | [damping/damping/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/damping/selection.json) |
| damping | neither | 80 | 60 | 0.2506 | [damping/neither/training.jsonl](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/neither/training.jsonl) | [damping/neither/selection.json](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa/damping/neither/selection.json) |

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

运行目录：[`machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa`](../../results/machine_local_equation_v1-synthetic-20261114T112046562581Z-2cf72ad4-b66d43aa)。来源提交：`2cf72ad4c6e99732375ebeee3c8f022c90ca4b90`。执行协议：[machine_local_equation_v1_zh.md](../protocols/machine_local_equation_v1_zh.md)。

## 证据范围

test 在模型冻结后评分。保存结果核对涵盖逐事件标量汇总、函数项选择、调用预算及文件哈希。真实 TASK3 训练的前置条件为通过合成恢复门。

- [独立结果审查](../research/machine_local_equation_result_review_20261114.json)
- [保存结果核对](../../reports/generated/machine_local_equation_20261114/saved-check.json)
