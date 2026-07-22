---
title: "3.11 Protocol B M2A / LYAP Accuracy"
date: "2026-07-04"
status: experiment
---

# 3.11 Protocol B M2A / LYAP Accuracy

> **引用限定（2026-07-19 审查添加）**：本报告 mean ACC≈0.90 来自 accuracy-first 阈值
> 校准，同批 mean FSR 为 0.077–0.094（非零漏报）。引用 ACC 数字时必须附带该 FSR 范围
> 与校准目标限定。独立对抗审计见 [[3.11_to_3.16.1-Adversarial_Audit_And_Algorithm_Design]]。
> 整条稳定性阈值校准路线后被项目判为历史路线：稳定性结论只能由物理 rollout 派生。

Run mode: `full`.
Result directory: `E:\03-nn_Transfer_Func\swing_equation\results\03_kron_graph_sindy\02_cluster\3.11-protocol_b_m2a_acc`.

## Protocol

- Dense dataset: `E:\03-nn_Transfer_Func\swing_equation\data\external_sim\andes\ieee39_near_critical_unstable_v1_dense` with counts {'total': 262, 'stable': 112, 'near_critical': 34, 'unstable': 116}.
- Splits: fixed Protocol B fault-line group splits from 3.10.
- Metric: binary `stable` / `unstable` Accuracy. `near_critical` is excluded from binary metrics.
- Goal: M2A and M2A-LYAP family mean Accuracy over splits/seeds above 0.90.
- Leakage guard: generator-written result summaries such as `case_results.csv` maximum angle/frequency fields are forbidden as model inputs.
- PASSH-R calibration uses train+val labels only, scored by fault-line mean Accuracy, then fault-line mean Balanced Accuracy, with long-horizon tie-breaking.

## 3.10 Baseline

| method | family | variant | window_len | mean_Accuracy | mean_BalancedAccuracy | mean_FSR | mean_FUR | n_runs |
|---|---|---|---|---|---|---|---|---|
| M0 | physics_equation_rollout | clustered | 1 | 0.6725 | 0.6601 | 0.005263 | 0.6746 | 5 |
| M2A | physics_equation_rollout | clustered | 1 | 0.7206 | 0.7116 | 0.03503 | 0.5418 | 25 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 0.754 | 0.7364 | 0.005263 | 0.522 | 25 |

## 3.11 Summary

| method | m0_anchor | calibrator_type | n_runs | mean_Accuracy | std_Accuracy | min_Accuracy | mean_BalancedAccuracy | mean_FSR | mean_FUR |
|---|---|---|---|---|---|---|---|---|---|
| M2A-PASSH-R | 3.10_all | line_balanced_train_val_threshold_grid | 25 | 0.9031 | 0.1036 | 0.6716 | 0.8987 | 0.07684 | 0.1259 |
| M2A-LYAP-PASSH-R | 3.10_all | line_balanced_train_val_threshold_grid | 25 | 0.9003 | 0.05016 | 0.8358 | 0.9001 | 0.0939 | 0.1058 |

## Per Split / Seed Grid

| split_id | method | m0_anchor | seed | status | Accuracy | BalancedAccuracy | FSR | FUR | val_Accuracy | calib_line_mean_Accuracy | calib_line_mean_BalancedAccuracy | rollout_horizon | residual_alpha | passivity_gate_pi | angle_thr_pi | freq_thr_pu | tail_thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | M2A-PASSH-R | 3.10_all | 324 | ok | 0.6716 | 0.6493 | 0.1842 | 0.5172 | 0.9814 | 0.9819 | 0.9156 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B1 | M2A-LYAP-PASSH-R | 3.10_all | 324 | ok | 0.8358 | 0.8348 | 0.1579 | 0.1724 | 0.9441 | 0.9469 | 0.881 | 50 | 0 | 0.95 | 1 | 0.05 | 5 |
| B1 | M2A-PASSH-R | 3.10_all | 325 | ok | 0.6866 | 0.6583 | 0.1316 | 0.5517 | 0.9689 | 0.9702 | 0.9045 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B1 | M2A-LYAP-PASSH-R | 3.10_all | 325 | ok | 0.8358 | 0.8348 | 0.1579 | 0.1724 | 0.9441 | 0.9469 | 0.881 | 50 | 0 | 0.95 | 1 | 0.05 | 5 |
| B1 | M2A-PASSH-R | 3.10_all | 326 | ok | 0.6716 | 0.6493 | 0.1842 | 0.5172 | 0.9814 | 0.9824 | 0.915 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B1 | M2A-LYAP-PASSH-R | 3.10_all | 326 | ok | 0.8358 | 0.8348 | 0.1579 | 0.1724 | 0.9441 | 0.9469 | 0.881 | 50 | 0 | 0.95 | 1 | 0.05 | 5 |
| B1 | M2A-PASSH-R | 3.10_all | 327 | ok | 0.8507 | 0.848 | 0.1316 | 0.1724 | 0.9814 | 0.9821 | 0.9124 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B1 | M2A-LYAP-PASSH-R | 3.10_all | 327 | ok | 0.8358 | 0.8348 | 0.1579 | 0.1724 | 0.9441 | 0.9469 | 0.881 | 50 | 0 | 0.95 | 1 | 0.05 | 5 |
| B1 | M2A-PASSH-R | 3.10_all | 328 | ok | 0.7015 | 0.6838 | 0.1842 | 0.4483 | 0.9814 | 0.9816 | 0.9148 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B1 | M2A-LYAP-PASSH-R | 3.10_all | 328 | ok | 0.8358 | 0.8348 | 0.1579 | 0.1724 | 0.9441 | 0.9469 | 0.881 | 50 | 0 | 0.95 | 1 | 0.05 | 5 |
| B2 | M2A-PASSH-R | 3.10_all | 324 | ok | 0.9028 | 0.9186 | 0.1628 | 0 | 0.9359 | 0.9447 | 0.8095 | 400 | 1 | 0.95 | 2 | 0.05 | 3 |
| B2 | M2A-LYAP-PASSH-R | 3.10_all | 324 | ok | 1 | 1 | 0 | 0 | 0.9038 | 0.9099 | 0.7869 | 400 | 1 | 0.95 | 3 | 0.05 | 2 |
| B2 | M2A-PASSH-R | 3.10_all | 325 | ok | 0.9722 | 0.9767 | 0.04651 | 0 | 0.9551 | 0.9615 | 0.8241 | 400 | 1 | 0.95 | 1 | 0.05 | 3 |
| B2 | M2A-LYAP-PASSH-R | 3.10_all | 325 | ok | 0.9722 | 0.9767 | 0.04651 | 0 | 0.9423 | 0.9496 | 0.8146 | 400 | 1 | 0.95 | 3 | 0.05 | 3 |
| B2 | M2A-PASSH-R | 3.10_all | 326 | ok | 1 | 1 | 0 | 0 | 0.9551 | 0.9615 | 0.8241 | 400 | 1 | 0.95 | 1 | 0.05 | 3 |
| B2 | M2A-LYAP-PASSH-R | 3.10_all | 326 | ok | 1 | 1 | 0 | 0 | 0.9103 | 0.9134 | 0.7928 | 400 | 1 | 0.95 | 3 | 0.05 | 2 |
| B2 | M2A-PASSH-R | 3.10_all | 327 | ok | 0.9306 | 0.9419 | 0.1163 | 0 | 0.9487 | 0.955 | 0.8152 | 400 | 1 | 0.95 | 1.25 | 0.05 | 3 |
| B2 | M2A-LYAP-PASSH-R | 3.10_all | 327 | ok | 0.9861 | 0.9884 | 0.02326 | 0 | 0.9231 | 0.9292 | 0.8008 | 400 | 1 | 0.95 | 3 | 0.05 | 2 |
| B2 | M2A-PASSH-R | 3.10_all | 328 | ok | 0.9167 | 0.9302 | 0.1395 | 0 | 0.9423 | 0.9494 | 0.8131 | 400 | 1 | 0.95 | 1.25 | 0.05 | 3 |
| B2 | M2A-LYAP-PASSH-R | 3.10_all | 328 | ok | 0.9722 | 0.9767 | 0.04651 | 0 | 0.9295 | 0.9357 | 0.8059 | 400 | 1 | 0.95 | 3 | 0.05 | 8 |
| B3 | M2A-PASSH-R | 3.10_all | 324 | ok | 0.9718 | 0.9714 | 0.05714 | 0 | 0.9618 | 0.9617 | 0.8199 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B3 | M2A-LYAP-PASSH-R | 3.10_all | 324 | ok | 0.8873 | 0.8857 | 0.2286 | 0 | 0.9108 | 0.9156 | 0.7919 | 100 | 1 | 0.95 | 1 | 0.05 | 5 |
| B3 | M2A-PASSH-R | 3.10_all | 325 | ok | 0.9577 | 0.9571 | 0.08571 | 0 | 0.9427 | 0.9454 | 0.8049 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B3 | M2A-LYAP-PASSH-R | 3.10_all | 325 | ok | 0.8592 | 0.8571 | 0.2857 | 0 | 0.9108 | 0.9136 | 0.773 | 50 | 0.5 | 0.95 | 1 | 0.05 | 5 |
| B3 | M2A-PASSH-R | 3.10_all | 326 | ok | 0.9577 | 0.9571 | 0.08571 | 0 | 0.9299 | 0.9342 | 0.7967 | 200 | 1 | 0.95 | 1 | 0.05 | 3 |
| B3 | M2A-LYAP-PASSH-R | 3.10_all | 326 | ok | 0.8873 | 0.8857 | 0.2286 | 0 | 0.9172 | 0.9235 | 0.7958 | 100 | 1 | 0.95 | 1 | 0.05 | 5 |
| B3 | M2A-PASSH-R | 3.10_all | 327 | ok | 0.9577 | 0.9571 | 0.08571 | 0 | 0.9427 | 0.9454 | 0.8049 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B3 | M2A-LYAP-PASSH-R | 3.10_all | 327 | ok | 0.8873 | 0.8857 | 0.2286 | 0 | 0.9108 | 0.9153 | 0.7863 | 100 | 0.5 | 0.95 | 1 | 0.05 | 5 |
| B3 | M2A-PASSH-R | 3.10_all | 328 | ok | 0.9718 | 0.9714 | 0.05714 | 0 | 0.9427 | 0.9454 | 0.8049 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B3 | M2A-LYAP-PASSH-R | 3.10_all | 328 | ok | 0.8732 | 0.8714 | 0.2571 | 0 | 0.9108 | 0.9136 | 0.773 | 50 | 0 | 0.95 | 1 | 0.05 | 5 |
| B4 | M2A-PASSH-R | 3.10_all | 324 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9342 | 0.9342 | 0.801 | 200 | 0 | 0.95 | 3 | 0.05 | 5 |
| B4 | M2A-LYAP-PASSH-R | 3.10_all | 324 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9342 | 0.9342 | 0.801 | 200 | 0 | 0.95 | 3 | 0.05 | 5 |
| B4 | M2A-PASSH-R | 3.10_all | 325 | ok | 0.8947 | 0.8823 | 0.02326 | 0.2121 | 0.9474 | 0.9418 | 0.8047 | 100 | 0.5 | 0.95 | 1 | 0.05 | 2 |
| B4 | M2A-LYAP-PASSH-R | 3.10_all | 325 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9342 | 0.9342 | 0.801 | 200 | 0 | 0.95 | 3 | 0.05 | 5 |
| B4 | M2A-PASSH-R | 3.10_all | 326 | ok | 0.8947 | 0.8858 | 0.04651 | 0.1818 | 0.9671 | 0.9651 | 0.8246 | 400 | 1 | 0.95 | 1.1 | 0.05 | 2 |
| B4 | M2A-LYAP-PASSH-R | 3.10_all | 326 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9342 | 0.9342 | 0.801 | 200 | 0 | 0.95 | 3 | 0.05 | 5 |
| B4 | M2A-PASSH-R | 3.10_all | 327 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9539 | 0.9478 | 0.8119 | 100 | 0.5 | 0.95 | 1 | 0.05 | 2 |
| B4 | M2A-LYAP-PASSH-R | 3.10_all | 327 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9342 | 0.9342 | 0.801 | 200 | 0 | 0.95 | 3 | 0.05 | 5 |
| B4 | M2A-PASSH-R | 3.10_all | 328 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9342 | 0.9342 | 0.801 | 200 | 0 | 0.95 | 3 | 0.05 | 5 |
| B4 | M2A-LYAP-PASSH-R | 3.10_all | 328 | ok | 0.9079 | 0.8975 | 0.02326 | 0.1818 | 0.9342 | 0.9342 | 0.801 | 200 | 0 | 0.95 | 3 | 0.05 | 5 |
| B5 | M2A-PASSH-R | 3.10_all | 324 | ok | 0.9859 | 0.9839 | 0.03226 | 0 | 0.9618 | 0.9641 | 0.8278 | 200 | 1 | 0.95 | 1.1 | 0.05 | 3 |
| B5 | M2A-LYAP-PASSH-R | 3.10_all | 324 | ok | 0.8873 | 0.8964 | 0.03226 | 0.175 | 0.9554 | 0.9615 | 0.8241 | 400 | 1 | 0.95 | 1 | 0.05 | 2 |
| B5 | M2A-PASSH-R | 3.10_all | 325 | ok | 0.9718 | 0.9677 | 0.06452 | 0 | 0.9618 | 0.9641 | 0.8278 | 200 | 1 | 0.95 | 1.1 | 0.05 | 3 |
| B5 | M2A-LYAP-PASSH-R | 3.10_all | 325 | ok | 0.8873 | 0.8964 | 0.03226 | 0.175 | 0.9554 | 0.9615 | 0.8241 | 400 | 1 | 0.95 | 1 | 0.05 | 2 |
| B5 | M2A-PASSH-R | 3.10_all | 326 | ok | 1 | 1 | 0 | 0 | 0.9554 | 0.9592 | 0.8231 | 200 | 1 | 0.95 | 1.25 | 0.05 | 2 |
| B5 | M2A-LYAP-PASSH-R | 3.10_all | 326 | ok | 0.9014 | 0.9125 | 0 | 0.175 | 0.949 | 0.9566 | 0.8193 | 400 | 1 | 0.95 | 1 | 0.05 | 2 |
| B5 | M2A-PASSH-R | 3.10_all | 327 | ok | 0.9859 | 0.9839 | 0.03226 | 0 | 0.9682 | 0.9703 | 0.8312 | 200 | 1 | 0.95 | 1 | 0.05 | 2 |
| B5 | M2A-LYAP-PASSH-R | 3.10_all | 327 | ok | 0.8732 | 0.8839 | 0.03226 | 0.2 | 0.949 | 0.9566 | 0.8193 | 400 | 1 | 0.95 | 1 | 0.05 | 2 |
| B5 | M2A-PASSH-R | 3.10_all | 328 | ok | 1 | 1 | 0 | 0 | 0.9554 | 0.9592 | 0.8231 | 200 | 1 | 0.95 | 1.5 | 0.05 | 2 |
| B5 | M2A-LYAP-PASSH-R | 3.10_all | 328 | ok | 0.9155 | 0.925 | 0 | 0.15 | 0.9554 | 0.9615 | 0.8241 | 400 | 1 | 0.95 | 1 | 0.05 | 3 |

## Audit

| severity | check | message |
|---|---|---|
| OK | audit | No blocking findings. |

## Output Files

- `run_config.json`
- `baseline_310_protocol_b_summary.csv`
- `protocol_b_full_grid_results.csv`
- `protocol_b_stability_summary.csv`
- `per_case_predictions.csv`
- `calibrator_grid_results.csv`
- `m0_anchor_ablation.csv`
- `audit_findings_table.csv`
- `invalidated_leakage_rows.csv` when old leakage rows are encountered
- `artifact_hashes.csv`
