# 合成记忆核系数校准

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/08_machine_local_topology/08_observer_calibration_diagnosis.DESCRIPTION.md)

登记文件与配置

[方案](../../../../configs/proposals/machine_local_memory_calibration_v1.json)

## 执行协议

<a id="doc-c39b748a5355"></a>

<a id="doc-c39b748a5355-可见动量积分校准与逐机记忆核复核"></a>

### 可见动量积分校准与逐机记忆核复核

系数替换诊断显示，单独换回一块真值参数后，三类系统的 3 秒误差均增大。校准保留已选函数项，通过完整 train 的可见动量约束重新估计共享系数。保留项的幅值参与拟合，缺项保持精确零。

记 \(P=I-m\mathbf 1^\top/\sum m\)、\(C=P\operatorname{diag}(g)\)、\(\Delta q=q(t)-q(0)\)。质量 COI 条件下 \(C^\top v=\operatorname{diag}(g)v\)。动量积分的记忆项为

\[
P\operatorname{diag}(h)g^2+P\operatorname{diag}(E)w_0,\quad
h_i=\int_0^t e^{-(t-s)/\tau_i}\Delta q_i(s)\,ds,\quad E_i=\tau_i(1-e^{-t/\tau_i}),\quad w_{0,i}=g_i z_{0,i}.
\]

以 \(Pp_0t-M[v(t)-v(0)]\) 为目标。恢复力使用 10 ms 功角、速度的三次 Hermite 插值和每区间八点 Gauss 求积。线性耗散积分为 \(D\Delta q\)，调制耗散积分用 \(J(e)=e/2-\sin(2e)/4\) 的端点差。使用每个事件的实际可用记录，在每 50 ms 及最后可用帧建立方程。权重由每个事件的方程数及 train 速度尺度确定。

给定时间常数，每条事件的 \(w_0\) 以 QR 消去。共享参数 \(k,d,g^2\) 下限为 \(10^{-10}\)。保留的谐波系数或调制耗散坐标 \(b/8\) 在 \([0,2]\)。列尺度化法方程与有界最小二乘要求设计条件数低于 \(10^6\)，干扰块奇异值比大于 \(10^{-10}\)，并检查有界梯度残差。L-BFGS-B 使用解析梯度优化 \(\log\tau\)，\(\tau\) 在 \([0.02001,4.999]\) 秒内。各组从相应冻结模型的时间常数出发。以初始动量残差缩放目标，每组最多 100 个接受迭代及 150 次不同目标计算。检查初始时间常数上的线性最优解、每十次接受迭代及最后接受点。完整 val 的 1 秒／3 秒归一化 MSE 等权目标选最早最小值。CPU float64 拟合，val 检查与正式预测在 CUDA float64 上运行。

干扰量只用于 train 方程求解。推理时沿用三次 GN，从本事件 100 ms 历史估计隐态。train 的可见轨迹是唯一的逐事件拟合目标。完整 val 用于选择，test 用于冻结后报告。


<a id="doc-c39b748a5355-对照和推进条件"></a>

#### 对照和推进条件

三类真值各覆盖 train 16、val 8、test 8。校准对象包括三组原先由 val 选中的方程和无新增项组的 A。四组各保存前后模型，并在三类系统上各评一个真参数对照。十一组方程均以 2.5/1.25 ms 两档预测全部事件至 8 秒。沿用原数值资格、轨迹指标和 TASK3 稳定性判据，批次求解出现数值异常时允许逐例隔离一次。

四个校准方程分别检查 train 和 val 的 3 秒、8 秒合并归一化 RMSE 上限 0.01、0.03，val 恢复力、可见耗散力和记忆核误差上限 3%、5%、5%，存在项的幅值误差上限 10%，缺项精确为零。真参数对照、能量平衡、耗散和局部曲率门沿用前轮。四个模型达标后登记完整 TASK3。本轮验证固定结构的系数恢复，沿用既有函数项选择结果。


<a id="doc-c39b748a5355-调用预算"></a>

#### 调用预算

四次冻结模型解码、四次拟合、600 次动量目标和梯度。干扰量 QR、SVD 和三角求解各最多 9600 次，共享线性求解与条件谱各 600 次。真值生成上限 233280 次 NumPy RHS。val 监测上限 48 批。十一组双步长正式评价含逐例隔离上限 3850 次传播、15523200 次 Torch RHS、554400 次切线 RHS、每种分解 4224 次。监测和物理诊断分别计入总预算。真实数据读数和旧优化器解码为零。

## 报告与结果

<a id="doc-4d938b5af3d2"></a>

<a id="doc-4d938b5af3d2-合成记忆核系数校准"></a>

### 合成记忆核系数校准

运行：`machine_local_memory_calibration_v1-calibration-20261114T125033660720Z-9b1cb1bc-9e0e3c25`。完整合成 train/val/test 评估。


<a id="doc-4d938b5af3d2-轨迹与稳定性"></a>

#### 轨迹与稳定性

每个模型分别列出 3 s、8 s pooled NRMSE 与覆盖情况；稳定性计数顺序为 TP/TN/FP/FN。


<a id="doc-4d938b5af3d2-null"></a>

##### null

| 模型 | split | 例数/覆盖 | 3 s NRMSE | 8 s NRMSE | 稳定性准确率 | 可判定准确率 | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---|
| known | train | 16/16 | 1.2531e-10 | 1.0939e-10 | 1 | 1 | 4/12/0/0 |
| known | val | 8/8 | 1.6705e-10 | 2.0815e-10 | 1 | 1 | 2/6/0/0 |
| known | test | 8/8 | 1.7405e-10 | 1.8025e-10 | 1 | 1 | 2/6/0/0 |
| selected_before | train | 16/16 | 0.00057436 | 0.00098895 | 1 | 1 | 4/12/0/0 |
| selected_before | val | 8/8 | 0.0019599 | 0.0083294 | 1 | 1 | 2/6/0/0 |
| selected_before | test | 8/8 | 0.0031391 | 0.0099383 | 1 | 1 | 2/6/0/0 |
| selected_calibrated | train | 16/16 | 2.5894e-05 | 2.2767e-05 | 1 | 1 | 4/12/0/0 |
| selected_calibrated | val | 8/8 | 3.3439e-05 | 3.4625e-05 | 1 | 1 | 2/6/0/0 |
| selected_calibrated | test | 8/8 | 3.5012e-05 | 3.6278e-05 | 1 | 1 | 2/6/0/0 |
| a_before | train | 16/16 | 0.0018025 | 0.0027354 | 1 | 1 | 4/12/0/0 |
| a_before | val | 8/8 | 0.0062334 | 0.023848 | 1 | 1 | 2/6/0/0 |
| a_before | test | 8/8 | 0.010201 | 0.037074 | 1 | 1 | 2/6/0/0 |
| a_calibrated | train | 16/16 | 7.9727e-06 | 6.9735e-06 | 1 | 1 | 4/12/0/0 |
| a_calibrated | val | 8/8 | 9.6313e-06 | 8.7442e-06 | 1 | 1 | 2/6/0/0 |
| a_calibrated | test | 8/8 | 8.6127e-06 | 8.3096e-06 | 1 | 1 | 2/6/0/0 |


<a id="doc-4d938b5af3d2-harmonic"></a>

##### harmonic

| 模型 | split | 例数/覆盖 | 3 s NRMSE | 8 s NRMSE | 稳定性准确率 | 可判定准确率 | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---|
| known | train | 16/16 | 2.619e-10 | 2.1339e-10 | 1 | 1 | 4/12/0/0 |
| known | val | 8/8 | 1.9589e-10 | 3.3445e-10 | 1 | 1 | 2/6/0/0 |
| known | test | 8/8 | 5.7513e-10 | 1.1772e-09 | 1 | 1 | 2/6/0/0 |
| selected_before | train | 16/16 | 0.0013766 | 0.0016296 | 1 | 1 | 4/12/0/0 |
| selected_before | val | 8/8 | 0.002925 | 0.0061246 | 1 | 1 | 2/6/0/0 |
| selected_before | test | 8/8 | 0.011277 | 0.05666 | 1 | 1 | 2/6/0/0 |
| selected_calibrated | train | 16/16 | 9.6694e-05 | 7.3054e-05 | 1 | 1 | 4/12/0/0 |
| selected_calibrated | val | 8/8 | 0.00012313 | 0.00015971 | 1 | 1 | 2/6/0/0 |
| selected_calibrated | test | 8/8 | 0.00011345 | 0.00010102 | 1 | 1 | 2/6/0/0 |


<a id="doc-4d938b5af3d2-damping"></a>

##### damping

| 模型 | split | 例数/覆盖 | 3 s NRMSE | 8 s NRMSE | 稳定性准确率 | 可判定准确率 | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---|
| known | train | 16/16 | 3.908e-09 | 3.5626e-09 | 1 | 1 | 4/12/0/0 |
| known | val | 8/8 | 5.2459e-09 | 4.9471e-09 | 1 | 1 | 2/6/0/0 |
| known | test | 8/8 | 8.1509e-09 | 7.5359e-09 | 1 | 1 | 2/6/0/0 |
| selected_before | train | 16/16 | 0.018014 | 0.028598 | 1 | 1 | 4/12/0/0 |
| selected_before | val | 8/8 | 0.098219 | 0.085976 | 1 | 1 | 2/6/0/0 |
| selected_before | test | 8/8 | 0.11734 | 0.19065 | 1 | 1 | 2/6/0/0 |
| selected_calibrated | train | 16/16 | 0.0002419 | 0.00021576 | 1 | 1 | 4/12/0/0 |
| selected_calibrated | val | 8/8 | 0.00028851 | 0.00026456 | 1 | 1 | 2/6/0/0 |
| selected_calibrated | test | 8/8 | 0.00027546 | 0.00025088 | 1 | 1 | 2/6/0/0 |


<a id="doc-4d938b5af3d2-拟合与门槛"></a>

#### 拟合与门槛

| 人群/起点 | train 目标调用 | 选中 val NMSE | 选中迭代 | 轨迹门 | 函数门 | 支持门 | 拟合有效 |
|---|---:|---:|---:|---|---|---|---|
| null/selected | 109 | 6.4692e-10 | 100 | True | True | True | True |
| null/a | 108 | 5.4041e-11 | 100 | True | True | True | True |
| harmonic/selected | 112 | 9.2157e-09 | 100 | True | True | True | True |
| damping/selected | 113 | 4.9871e-08 | 100 | True | True | True | True |


<a id="doc-4d938b5af3d2-主选模型校准前后记忆核误差"></a>

#### 主选模型：校准前后记忆核误差

| 人群 | 拟合起点 | 校准前 | 校准后 | 校准门全部通过 |
|---|---|---:|---:|---|
| null | selected | 0.071183 | 0.00042884 | True |
| harmonic | selected | 0.1201 | 0.0020516 | True |
| damping | selected | 0.55229 | 0.0038248 | True |


<a id="doc-4d938b5af3d2-物理检查与运行计数"></a>

#### 物理检查与运行计数

| 人群 | 已知真值对照 | 物理检查 | 科学门结果 |
|---|---|---|---|
| null | True | True | True |
| harmonic | True | True | True |
| damping | True | True | True |

| 操作 | 实际计数 |
|---|---:|
| checkpoint_decodes | 4 |
| curvature_spectra | 7128 |
| fits | 4 |
| function_checks | 11 |
| matrix_products | 3168 |
| momentum_design_spectrum | 442 |
| momentum_linear_solves | 442 |
| momentum_nuisance_solves | 7072 |
| momentum_nuisance_svd | 7072 |
| momentum_objectives | 442 |
| momentum_qr | 7072 |
| numpy_rhs | 233280 |
| qr | 3168 |
| rollouts | 550 |
| svd | 3168 |
| tangent_rhs | 58080 |
| torch_rhs | 1450251 |
| triangular_solves | 3168 |
| val_checks | 44 |

整体科学门结果：`True`。失败门和缺失指标原样保留；本报告不据结果推断参数唯一性。

复核 `docs/reports/08_machine_local_topology/memory_calibration/_sealed/machine_local_memory_calibration_review_20261114.json`：`APPROVE_EXACT_FIXED_SUPPORT_SYNTHETIC_COEFFICIENT_RECOVERY`，approved=True。

## 回执与附件

- [machine_local_memory_calibration_20261114_evidence.json](_sealed/machine_local_memory_calibration_20261114_evidence.json)
- [machine_local_memory_calibration_result_review_20261114.json](_sealed/machine_local_memory_calibration_result_review_20261114.json)
- [machine_local_memory_calibration_review_20261114.json](_sealed/machine_local_memory_calibration_review_20261114.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_memory_calibration_v1_zh.md](_sealed/machine_local_memory_calibration_v1_zh.md)
- [docs/reports/08_machine_local_topology/machine_local_memory_calibration_20261114_zh.md](_sealed/machine_local_memory_calibration_20261114_zh.md)

</details>
