# 合成记忆核系数校准

运行：`machine_local_memory_calibration_v1-calibration-20261114T125033660720Z-9b1cb1bc-9e0e3c25`。完整合成 train/val/test 评估。

## 轨迹与稳定性

每个模型分别列出 3 s、8 s pooled NRMSE 与覆盖情况；稳定性计数顺序为 TP/TN/FP/FN。

### null

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

### harmonic

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

### damping

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

## 拟合与门槛

| 人群/起点 | train 目标调用 | 选中 val NMSE | 选中迭代 | 轨迹门 | 函数门 | 支持门 | 拟合有效 |
|---|---:|---:|---:|---|---|---|---|
| null/selected | 109 | 6.4692e-10 | 100 | True | True | True | True |
| null/a | 108 | 5.4041e-11 | 100 | True | True | True | True |
| harmonic/selected | 112 | 9.2157e-09 | 100 | True | True | True | True |
| damping/selected | 113 | 4.9871e-08 | 100 | True | True | True | True |

## 主选模型：校准前后记忆核误差

| 人群 | 拟合起点 | 校准前 | 校准后 | 校准门全部通过 |
|---|---|---:|---:|---|
| null | selected | 0.071183 | 0.00042884 | True |
| harmonic | selected | 0.1201 | 0.0020516 | True |
| damping | selected | 0.55229 | 0.0038248 | True |

## 物理检查与运行计数

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

复核 `docs/research/machine_local_memory_calibration_review_20261114.json`：`APPROVE_EXACT_FIXED_SUPPORT_SYNTHETIC_COEFFICIENT_RECOVERY`，approved=True。
