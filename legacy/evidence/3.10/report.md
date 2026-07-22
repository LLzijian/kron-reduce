---
title: "3.10 Final Paper Comparison"
date: "2026-07-03"
status: final-candidate
---

# 3.10 Final Paper Comparison

Run mode: `full`.
Result directory: `E:\03-nn_Transfer_Func\swing_equation\results\03_kron_graph_sindy\02_cluster\3.10-final_paper_comparison_protocol_ab`.

This report is the replacement final comparison experiment for 3.10. It uses two clean protocols: Protocol A for blind external generalization and Protocol B for dense boundary adaptation with fault-line group splits.

All rollout and controlled-operator methods receive `P_m(t)` as an observed exogenous input. The controlled input is represented in the centered model as `dpm(t)=P_m(t)-P_m0`, the same convention used by the M0/M2A/M2A-LYAP equations.

## 1. Experiment Protocol

- Highorder stable dataset: `E:\03-nn_Transfer_Func\swing_equation\data\external_sim\andes\ieee39_highorder_dataset` with train/val/test = {'train': 70, 'val': 15, 'test': 15}.
- Protocol A external set: `E:\03-nn_Transfer_Func\swing_equation\data\external_sim\andes\ieee39_near_critical_unstable_v1` with counts {'total': 143, 'stable': 67, 'near_critical': 13, 'unstable': 63}; only stable/unstable enter TP/FN/FP/TN.
- Protocol B dense set: `E:\03-nn_Transfer_Func\swing_equation\data\external_sim\andes\ieee39_near_critical_unstable_v1_dense` with counts {'total': 262, 'stable': 112, 'near_critical': 34, 'unstable': 116}. Dense contains 143 v1 cases and has 119 additional boundary cases, so v1 is not reused as a held-out test in Protocol B.
- Highorder and v1 fault-line overlap: `[]`.

Protocol B fixed splits:

| Split | Train fault lines | Val fault line | Test fault lines |
|---|---|---|---|
| B1 | Line_2, Line_37, Line_42, Line_43, Line_44, Line_46 | Line_36 | Line_35, Line_41, Line_45 |
| B2 | Line_2, Line_37, Line_41, Line_43, Line_44, Line_45 | Line_35 | Line_36, Line_42, Line_46 |
| B3 | Line_2, Line_35, Line_36, Line_41, Line_42, Line_45 | Line_46 | Line_37, Line_43, Line_44 |
| B4 | Line_2, Line_36, Line_37, Line_41, Line_43, Line_45 | Line_44 | Line_35, Line_42, Line_46 |
| B5 | Line_2, Line_35, Line_41, Line_42, Line_44, Line_46 | Line_37 | Line_36, Line_43, Line_45 |

## 2. Method Capability And Input Fairness

| method | reference | family | input_form | required_input_window | pm_mode | rollout_capable | classifier_capable |
|---|---|---|---|---|---|---|---|
| M0 | This work baseline; report 3.5 | physics_equation_rollout | clustered 7-area only | 1 point | Pm(t) observed exogenous input | yes | rollout-derived |
| M2A | This work residual Graph-SINDy | physics_equation_rollout | clustered 7-area only | 1 point | Pm(t) observed exogenous input | yes | rollout-derived |
| M2A-LYAP | This work Lyapunov-constrained residual; report 3.8.4 | physics_equation_rollout | clustered 7-area only | 1 point | Pm(t) observed exogenous input | yes | rollout-derived |
| SINDYc | Brunton et al. 2016 | sparse_identification | clustered and unclustered | 1 point | Pm(t) observed exogenous input | yes | rollout-derived |
| DMD | Schmid 2010 | operator_forecast | clustered and unclustered | 20/40/100/200 | Pm(t) observed exogenous input | forecast | rollout-derived |
| HODMD | Higher-order DMD literature | operator_forecast | clustered and unclustered | 20/40/100/200 | Pm(t) observed exogenous input | forecast | rollout-derived |
| EDMD-Koopman | Williams et al. 2015 | operator_forecast | clustered and unclustered | 20/40/100/200 | Pm(t) observed exogenous input | forecast | rollout-derived |
| LSTM-rollout | Sequence forecasting baseline | blackbox_rollout | clustered and unclustered | 20/40/100/200 | Pm(t) observed exogenous input | forecast | rollout-derived |
| CNN / LSTM / CNN+LSTM / Transformer | Supervised TSA baselines | direct_classifier | clustered and unclustered | 20/40/100/200 | Pm(t) observed exogenous input | no | Protocol B only |

## 3. Protocol A Complete Stability Results

Protocol A trains on highorder stable train/val and tests stability on untouched v1 external cases. Supervised direct classifiers are marked not applicable here.

| method | family | variant | window_len | n_runs | FSR | FUR | Accuracy | BalancedAccuracy |
|---|---|---|---|---|---|---|---|---|
| DMD | operator_forecast | clustered | 100 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | clustered | 20 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | clustered | 200 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | clustered | 40 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 100 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 20 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 200 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 40 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | clustered | 100 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | clustered | 20 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | clustered | 200 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | clustered | 40 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | unclustered | 100 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | unclustered | 20 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | unclustered | 200 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | unclustered | 40 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 5 | 0.0413 ± 0.0749 | 1.0000 ± 0.0000 | 0.4646 ± 0.0363 | 0.4794 ± 0.0374 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 5 | 0.0190 ± 0.0254 | 0.9881 ± 0.0112 | 0.4815 ± 0.0104 | 0.4964 ± 0.0108 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 5 | 0.0571 ± 0.0456 | 0.9970 ± 0.0060 | 0.4585 ± 0.0215 | 0.4729 ± 0.0222 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 5 | 0.0349 ± 0.0431 | 0.9791 ± 0.0202 | 0.4785 ± 0.0255 | 0.4930 ± 0.0259 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 5 | 0.0032 ± 0.0063 | 1.0000 ± 0.0000 | 0.4831 ± 0.0031 | 0.4984 ± 0.0032 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 5 | 0.0000 ± 0.0000 | 0.9881 ± 0.0239 | 0.4908 ± 0.0123 | 0.5060 ± 0.0119 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| M0 | physics_equation_rollout | clustered | 1 | 1 | 0.2698 ± 0.0000 | 0.0000 ± 0.0000 | 0.8692 ± 0.0000 | 0.8651 ± 0.0000 |
| M2A | physics_equation_rollout | clustered | 1 | 5 | 0.0952 ± 0.0246 | 0.0149 ± 1.7347e-18 | 0.9462 ± 0.0119 | 0.9449 ± 0.0123 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 5 | 0.2857 ± 0.0000 | 0.0000 ± 0.0000 | 0.8615 ± 1.1102e-16 | 0.8571 ± 1.1102e-16 |
| SINDYc | sparse_identification | clustered | 1 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |
| SINDYc | sparse_identification | unclustered | 1 | 5 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4846 ± 5.5511e-17 | 0.5000 ± 0.0000 |

Protocol A per-seed/per-variant stability grid:

| method | family | variant | window_len | seed | status | tp | fn | fp | tn | FSR | FUR | Accuracy | BalancedAccuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | physics_equation_rollout | clustered | 1 | 0 | ok | 46 | 17 | 0 | 67 | 0.2698412698412698 | 0.0 | 0.8692307692307693 | 0.8650793650793651 |
| M2A | physics_equation_rollout | clustered | 1 | 324 | ok | 60 | 3 | 1 | 66 | 0.047619047619047616 | 0.014925373134328358 | 0.9692307692307692 | 0.968727789623312 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | 45 | 18 | 0 | 67 | 0.2857142857142857 | 0.0 | 0.8615384615384616 | 0.8571428571428572 |
| M2A | physics_equation_rollout | clustered | 1 | 325 | ok | 57 | 6 | 1 | 66 | 0.09523809523809523 | 0.014925373134328358 | 0.9461538461538461 | 0.9449182658137882 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | 45 | 18 | 0 | 67 | 0.2857142857142857 | 0.0 | 0.8615384615384616 | 0.8571428571428572 |
| M2A | physics_equation_rollout | clustered | 1 | 326 | ok | 56 | 7 | 1 | 66 | 0.1111111111111111 | 0.014925373134328358 | 0.9384615384615385 | 0.9369817578772802 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | 45 | 18 | 0 | 67 | 0.2857142857142857 | 0.0 | 0.8615384615384616 | 0.8571428571428572 |
| M2A | physics_equation_rollout | clustered | 1 | 327 | ok | 56 | 7 | 1 | 66 | 0.1111111111111111 | 0.014925373134328358 | 0.9384615384615385 | 0.9369817578772802 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | 45 | 18 | 0 | 67 | 0.2857142857142857 | 0.0 | 0.8615384615384616 | 0.8571428571428572 |
| M2A | physics_equation_rollout | clustered | 1 | 328 | ok | 56 | 7 | 1 | 66 | 0.1111111111111111 | 0.014925373134328358 | 0.9384615384615385 | 0.9369817578772802 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | 45 | 18 | 0 | 67 | 0.2857142857142857 | 0.0 | 0.8615384615384616 | 0.8571428571428572 |
| SINDYc | sparse_identification | clustered | 1 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 20 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 20 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | 63 | 0 | 66 | 1 | 0.0 | 0.9850746268656716 | 0.49230769230769234 | 0.5074626865671642 |
| DMD | operator_forecast | clustered | 40 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 40 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | 63 | 0 | 66 | 1 | 0.0 | 0.9850746268656716 | 0.49230769230769234 | 0.5074626865671642 |
| DMD | operator_forecast | clustered | 100 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 100 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | 62 | 1 | 67 | 0 | 0.015873015873015872 | 1.0 | 0.47692307692307695 | 0.49206349206349204 |
| DMD | operator_forecast | clustered | 200 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 200 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | 59 | 4 | 67 | 0 | 0.06349206349206349 | 1.0 | 0.45384615384615384 | 0.46825396825396826 |
| SINDYc | sparse_identification | clustered | 1 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 40 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 40 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | 57 | 6 | 66 | 1 | 0.09523809523809523 | 0.9850746268656716 | 0.4461538461538462 | 0.4598436389481166 |
| DMD | operator_forecast | clustered | 100 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 100 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 200 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 200 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | 62 | 1 | 67 | 0 | 0.015873015873015872 | 1.0 | 0.47692307692307695 | 0.49206349206349204 |
| SINDYc | sparse_identification | clustered | 1 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| SINDYc | sparse_identification | clustered | 1 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 20 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 20 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | 61 | 2 | 65 | 2 | 0.031746031746031744 | 0.9701492537313433 | 0.4846153846153846 | 0.4990523572613125 |
| DMD | operator_forecast | clustered | 40 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 40 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | 58 | 5 | 66 | 1 | 0.07936507936507936 | 0.9850746268656716 | 0.45384615384615384 | 0.4677801468846245 |
| DMD | operator_forecast | clustered | 100 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 100 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | 51 | 12 | 67 | 0 | 0.19047619047619047 | 1.0 | 0.3923076923076923 | 0.40476190476190477 |
| DMD | operator_forecast | clustered | 200 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 200 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | 55 | 8 | 67 | 0 | 0.12698412698412698 | 1.0 | 0.4230769230769231 | 0.4365079365079365 |
| SINDYc | sparse_identification | clustered | 1 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 20 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 20 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | 59 | 4 | 66 | 1 | 0.06349206349206349 | 0.9850746268656716 | 0.46153846153846156 | 0.47571665482113246 |
| DMD | operator_forecast | clustered | 40 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 40 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | 63 | 0 | 63 | 4 | 0.0 | 0.9402985074626866 | 0.5153846153846153 | 0.5298507462686567 |
| DMD | operator_forecast | clustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | clustered | 200 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | clustered | 200 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | 58 | 5 | 66 | 1 | 0.07936507936507936 | 0.9850746268656716 | 0.45384615384615384 | 0.4677801468846245 |
| SINDYc | sparse_identification | unclustered | 1 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 20 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 20 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | 63 | 0 | 63 | 4 | 0.0 | 0.9402985074626866 | 0.5153846153846153 | 0.5298507462686567 |
| DMD | operator_forecast | unclustered | 40 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 40 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 100 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 100 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 200 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 200 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| SINDYc | sparse_identification | unclustered | 1 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 40 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 40 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 100 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 100 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | 62 | 1 | 67 | 0 | 0.015873015873015872 | 1.0 | 0.47692307692307695 | 0.49206349206349204 |
| DMD | operator_forecast | unclustered | 200 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 200 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| SINDYc | sparse_identification | unclustered | 1 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| SINDYc | sparse_identification | unclustered | 1 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 20 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 20 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 40 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 40 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 100 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 100 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 200 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 200 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| SINDYc | sparse_identification | unclustered | 1 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 20 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 20 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 40 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 40 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| DMD | operator_forecast | unclustered | 200 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| HODMD | operator_forecast | unclustered | 200 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | 63 | 0 | 67 | 0 | 0.0 | 1.0 | 0.4846153846153846 | 0.5 |
| CNN | direct_classifier | N/A | N/A | - | not_applicable | - | - | - | - | - | - | - | - |
| LSTM | direct_classifier | N/A | N/A | - | not_applicable | - | - | - | - | - | - | - | - |
| CNN+LSTM | direct_classifier | N/A | N/A | - | not_applicable | - | - | - | - | - | - | - | - |
| Transformer | direct_classifier | N/A | N/A | - | not_applicable | - | - | - | - | - | - | - | - |

## 4. Protocol A Complete Trajectory Results

These rows use the highorder internal test split for trajectory rollout/forecast fidelity.

| method | family | variant | window_len | n_runs | rollout_r2 | mean_omega_r2 | rmse | failure_rate |
|---|---|---|---|---|---|---|---|---|
| LSTM-rollout | blackbox_rollout | clustered | 100 | 5 | -9.6858e+05 ± 5.7943e+05 | -9.8922e+05 ± 6.1197e+05 | 3.9802 ± 0.9642 | 0.0000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 5 | -9.2005e+05 ± 3.4126e+05 | -9.3937e+05 ± 3.4687e+05 | 3.9867 ± 0.4730 | 0.0000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 5 | -8.4750e+05 ± 5.9884e+05 | -8.7342e+05 ± 6.2882e+05 | 3.2447 ± 0.4518 | 0.0000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 5 | -1.4171e+06 ± 9.1016e+05 | -1.4624e+06 ± 9.5641e+05 | 3.8312 ± 0.6491 | 0.0000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 5 | -8.3989e+05 ± 3.8049e+05 | -8.5104e+05 ± 3.9367e+05 | 3.7333 ± 0.2730 | 0.0000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 5 | -7.2273e+05 ± 2.1071e+05 | -7.3663e+05 ± 2.1882e+05 | 4.0439 ± 0.5466 | 0.0000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 5 | -7.7038e+05 ± 3.3339e+05 | -7.8693e+05 ± 3.4553e+05 | 3.4352 ± 0.3103 | 0.0000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 5 | -1.2269e+06 ± 3.8917e+05 | -1.2550e+06 ± 4.1493e+05 | 4.2741 ± 0.4369 | 0.0000 ± 0.0000 |
| M0 | physics_equation_rollout | clustered | 1 | 1 | 0.8701 ± 0.0000 | 0.9061 ± 0.0000 | 0.0065 ± 0.0000 | 0.0000 ± 0.0000 |
| M2A | physics_equation_rollout | clustered | 1 | 5 | 0.9152 ± 0.0106 | 0.9613 ± 0.0024 | 0.0033 ± 4.6093e-04 | 0.0000 ± 0.0000 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 5 | 0.7181 ± 0.0360 | 0.7571 ± 0.0353 | 0.0081 ± 4.8921e-04 | 0.0000 ± 0.0000 |

Protocol A per-seed/per-variant trajectory grid:

| method | family | variant | window_len | seed | status | rollout_r2 | mean_omega_r2 | worst_case_r2 | rmse | failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | physics_equation_rollout | clustered | 1 | 0 | ok | 0.8700924614584331 | 0.9061042258981755 | 0.7060227417207119 | 0.00646567475541801 | 0.0 |
| M2A | physics_equation_rollout | clustered | 1 | 324 | ok | 0.9129312486807379 | 0.9609846744659146 | 0.7080734680366576 | 0.0032334616266056137 | 0.0 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | 0.7468502510616711 | 0.7851847589808666 | 0.34877960072111436 | 0.007665849315731937 | 0.0 |
| M2A | physics_equation_rollout | clustered | 1 | 325 | ok | 0.9208111651687173 | 0.9625824199466052 | 0.7089421713882458 | 0.003170421565934397 | 0.0 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | 0.7282248460136079 | 0.7668624300833933 | 0.324928906931125 | 0.007981983488148058 | 0.0 |
| M2A | physics_equation_rollout | clustered | 1 | 326 | ok | 0.8983955711263791 | 0.9644572013458801 | 0.6686233925525966 | 0.004155445718457295 | 0.0 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | 0.7350651420117751 | 0.7737808037376908 | 0.3209503189745283 | 0.00798058189149459 | 0.0 |
| M2A | physics_equation_rollout | clustered | 1 | 327 | ok | 0.9305086893556311 | 0.9570900754410434 | 0.741716594464752 | 0.00298970361663155 | 0.0 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | 0.7332796423396117 | 0.7722255805497308 | 0.3158019192318222 | 0.007806403286114568 | 0.0 |
| M2A | physics_equation_rollout | clustered | 1 | 328 | ok | 0.9133090984718167 | 0.9612960928092428 | 0.5735440867959769 | 0.0028355977173118293 | 0.0 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | 0.6472235369095507 | 0.6875787501957776 | 0.09659846835435248 | 0.009045427387828578 | 0.0 |
| SINDYc | sparse_identification | clustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | -1228473.485116779 | -1260363.3842146802 | -5472904.080258872 | 4.670776010925052 | 0.0 |
| DMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | -3093579.589613244 | -3228760.4576813183 | -34891407.321780905 | 4.297140060390926 | 0.0 |
| DMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | -2052853.3442545459 | -2137571.1561181587 | -19124852.093826197 | 5.6892812166882445 | 0.0 |
| DMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | -2033807.8257768494 | -2119471.8821233595 | -18964313.18007061 | 3.979155209920617 | 0.0 |
| SINDYc | sparse_identification | clustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | -891321.5162464535 | -911563.5837332739 | -4684751.030043072 | 3.751561936056814 | 0.0 |
| DMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | -643715.4580362894 | -658163.342376062 | -3640933.0977231343 | 3.276157082902282 | 0.0 |
| DMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | -995439.0877857686 | -1023343.0026109183 | -7350486.846016714 | 3.7198169305607793 | 0.0 |
| DMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | -432586.19446947495 | -436151.33841954794 | -1728124.3732533806 | 3.086037134985884 | 0.0 |
| SINDYc | sparse_identification | clustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | -1350737.6474359767 | -1369211.758555349 | -12867384.176588248 | 4.357667055033418 | 0.0 |
| DMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | -1282449.223105361 | -1292943.5504687536 | -7872157.093980693 | 4.440867784652563 | 0.0 |
| DMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | -749069.5016833381 | -713305.9241666656 | -2563164.97209783 | 4.281090739190974 | 0.0 |
| DMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | -683740.6917513456 | -697347.9163751983 | -4405084.170461381 | 3.5155453182458194 | 0.0 |
| SINDYc | sparse_identification | clustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | -720655.6374428311 | -738514.9359977408 | -3666031.354010058 | 3.8234775820631 | 0.0 |
| DMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | -574696.392060601 | -584464.822255159 | -6555294.721975753 | 2.8379161886392126 | 0.0 |
| DMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | -354503.5529110163 | -354826.1774766755 | -2250822.88227814 | 2.9674546969854574 | 0.0 |
| DMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | -511040.69641894766 | -524910.6764286335 | -3155578.2244543103 | 2.931494115080257 | 0.0 |
| SINDYc | sparse_identification | clustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | -409063.13925994677 | -417184.7743763044 | -1424423.3601069322 | 3.3299790275299967 | 0.0 |
| DMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | -1491251.7669324635 | -1547580.7524128642 | -11505433.316540863 | 4.303817608966841 | 0.0 |
| DMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | -691035.7850360785 | -717033.1477199096 | -6262675.940180575 | 3.243316664692728 | 0.0 |
| DMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | -576349.096580958 | -589209.1398967581 | -4808058.066032646 | 2.7112445719266502 | 0.0 |
| SINDYc | sparse_identification | unclustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | -558491.172828288 | -569350.5916314577 | -3141911.369526917 | 3.5207486231618126 | 0.0 |
| DMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | -897857.1005001017 | -916970.8369324337 | -8989981.057174677 | 3.9184732520306205 | 0.0 |
| DMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | -700419.6028968474 | -712521.7420389842 | -6070052.628365453 | 3.7982989373186293 | 0.0 |
| DMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | -623495.8113297352 | -627345.43794228 | -5646630.125742807 | 3.2414679250242235 | 0.0 |
| SINDYc | sparse_identification | unclustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | -495909.60920339293 | -501748.470939661 | -3664858.834847933 | 3.304104977995793 | 0.0 |
| DMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | -921096.3700749743 | -927961.6858665581 | -7856207.993657623 | 3.8030692818395293 | 0.0 |
| DMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | -357012.7090443627 | -362131.92929912783 | -1771115.926248859 | 3.449871147400065 | 0.0 |
| DMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | -430886.96629132173 | -437745.470280946 | -3026228.490723781 | 2.9438334220390776 | 0.0 |
| SINDYc | sparse_identification | unclustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | -1035939.9865592568 | -1065411.7973038899 | -10759730.241891192 | 4.196491285839425 | 0.0 |
| DMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | -1967877.5629311348 | -2048964.48590803 | -22791358.315503858 | 4.923478848550271 | 0.0 |
| DMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | -887427.174169043 | -876687.1391302834 | -8731738.696255766 | 3.93904501194677 | 0.0 |
| DMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | -813543.9032931923 | -840809.4242871243 | -9159702.607802626 | 3.813862805865902 | 0.0 |
| SINDYc | sparse_identification | unclustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | -615987.1239779904 | -622513.7826555929 | -3631711.60157549 | 4.470460098750388 | 0.0 |
| DMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | -1153897.1501640491 | -1168708.257428326 | -7642278.90945288 | 4.653013936773892 | 0.0 |
| DMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | -737572.680692145 | -742698.0618610848 | -7291653.453070548 | 3.3889725500458865 | 0.0 |
| DMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | -592741.5479941702 | -600109.8870707912 | -4626277.102655895 | 3.5104790469116103 | 0.0 |
| SINDYc | sparse_identification | unclustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| DMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | -907338.1888113997 | -924111.6556707259 | -5785161.826850795 | 4.727772669611573 | 0.0 |
| DMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | -1193827.0451723807 | -1212465.9728390358 | -10020247.709308792 | 4.072363306843892 | 0.0 |
| DMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | -1517023.451797659 | -1561161.5224879018 | -18933181.438414462 | 4.090115005461967 | 0.0 |
| DMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| HODMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | -1391250.0898278083 | -1428617.1630466804 | -16083970.74501091 | 3.6664255613252874 | 0.0 |

## 5. Protocol B Complete Stability Results

Protocol B trains/validates/tests on dense using fixed held-out fault-line group splits. All direct classifiers are evaluated only here.

| method | family | variant | window_len | n_runs | FSR | FUR | Accuracy | BalancedAccuracy |
|---|---|---|---|---|---|---|---|---|
| CNN | direct_classifier | clustered | 100 | 25 | 0.0584 ± 0.0520 | 0.0000 ± 0.0000 | 0.9676 ± 0.0298 | 0.9708 ± 0.0260 |
| CNN | direct_classifier | clustered | 20 | 25 | 0.0715 ± 0.0767 | 0.2635 ± 0.2771 | 0.8472 ± 0.1152 | 0.8325 ± 0.1368 |
| CNN | direct_classifier | clustered | 200 | 25 | 0.0510 ± 0.0506 | 0.0000 ± 0.0000 | 0.9719 ± 0.0290 | 0.9745 ± 0.0253 |
| CNN | direct_classifier | clustered | 40 | 25 | 0.0575 ± 0.0723 | 0.0183 ± 0.0441 | 0.9598 ± 0.0448 | 0.9621 ± 0.0421 |
| CNN | direct_classifier | unclustered | 100 | 25 | 0.0572 ± 0.0452 | 0.0200 ± 0.0980 | 0.9587 ± 0.0499 | 0.9614 ± 0.0484 |
| CNN | direct_classifier | unclustered | 20 | 25 | 0.0756 ± 0.0617 | 0.2952 ± 0.2857 | 0.8319 ± 0.1070 | 0.8146 ± 0.1308 |
| CNN | direct_classifier | unclustered | 200 | 25 | 0.0690 ± 0.0548 | 0.0000 ± 0.0000 | 0.9624 ± 0.0314 | 0.9655 ± 0.0274 |
| CNN | direct_classifier | unclustered | 40 | 25 | 0.0559 ± 0.0634 | 0.0668 ± 0.0917 | 0.9408 ± 0.0418 | 0.9386 ± 0.0454 |
| CNN+LSTM | direct_classifier | clustered | 100 | 25 | 0.0102 ± 0.0138 | 0.0308 ± 0.0697 | 0.9820 ± 0.0272 | 0.9795 ± 0.0333 |
| CNN+LSTM | direct_classifier | clustered | 20 | 25 | 0.0749 ± 0.0711 | 0.2938 ± 0.2911 | 0.8326 ± 0.1145 | 0.8157 ± 0.1388 |
| CNN+LSTM | direct_classifier | clustered | 200 | 25 | 0.0146 ± 0.0190 | 0.0348 ± 0.1191 | 0.9772 ± 0.0582 | 0.9753 ± 0.0652 |
| CNN+LSTM | direct_classifier | clustered | 40 | 25 | 0.0152 ± 0.0218 | 0.0171 ± 0.0306 | 0.9843 ± 0.0146 | 0.9839 ± 0.0159 |
| CNN+LSTM | direct_classifier | unclustered | 100 | 25 | 0.0083 ± 0.0135 | 0.0014 ± 0.0068 | 0.9955 ± 0.0065 | 0.9952 ± 0.0072 |
| CNN+LSTM | direct_classifier | unclustered | 20 | 25 | 0.0846 ± 0.0577 | 0.2992 ± 0.3011 | 0.8271 ± 0.1095 | 0.8081 ± 0.1344 |
| CNN+LSTM | direct_classifier | unclustered | 200 | 25 | 0.0067 ± 0.0176 | 0.0024 ± 0.0082 | 0.9955 ± 0.0099 | 0.9955 ± 0.0093 |
| CNN+LSTM | direct_classifier | unclustered | 40 | 25 | 0.0133 ± 0.0272 | 0.0377 ± 0.0533 | 0.9767 ± 0.0222 | 0.9745 ± 0.0254 |
| DMD | operator_forecast | clustered | 100 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | clustered | 20 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | clustered | 200 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | clustered | 40 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 100 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 20 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 200 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| DMD | operator_forecast | unclustered | 40 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 100 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 20 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 200 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | clustered | 40 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 100 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 20 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 200 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| EDMD-Koopman | operator_forecast | unclustered | 40 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| HODMD | operator_forecast | clustered | 100 | 25 | 0.0869 ± 0.1494 | 0.6800 ± 0.4665 | 0.6312 ± 0.1661 | 0.6166 ± 0.1745 |
| HODMD | operator_forecast | clustered | 20 | 25 | 0.0682 ± 0.1347 | 0.7011 ± 0.4377 | 0.6323 ± 0.1705 | 0.6153 ± 0.1759 |
| HODMD | operator_forecast | clustered | 200 | 25 | 0.0954 ± 0.1650 | 0.6800 ± 0.4665 | 0.6269 ± 0.1607 | 0.6123 ± 0.1696 |
| HODMD | operator_forecast | clustered | 40 | 25 | 0.0813 ± 0.1453 | 0.6850 ± 0.4598 | 0.6315 ± 0.1694 | 0.6168 ± 0.1779 |
| HODMD | operator_forecast | unclustered | 100 | 25 | 0.0000 ± 0.0000 | 0.9576 ± 0.1958 | 0.5504 ± 0.1092 | 0.5212 ± 0.0979 |
| HODMD | operator_forecast | unclustered | 20 | 25 | 0.0000 ± 0.0000 | 0.9636 ± 0.1781 | 0.5477 ± 0.1025 | 0.5182 ± 0.0891 |
| HODMD | operator_forecast | unclustered | 200 | 25 | 0.0000 ± 0.0000 | 0.9285 ± 0.2444 | 0.5630 ± 0.1297 | 0.5358 ± 0.1222 |
| HODMD | operator_forecast | unclustered | 40 | 25 | 0.0000 ± 0.0000 | 0.9600 ± 0.1960 | 0.5493 ± 0.1089 | 0.5200 ± 0.0980 |
| LSTM | direct_classifier | clustered | 100 | 25 | 0.0057 ± 0.0116 | 0.0066 ± 0.0231 | 0.9945 ± 0.0103 | 0.9939 ± 0.0122 |
| LSTM | direct_classifier | clustered | 20 | 25 | 0.0530 ± 0.0503 | 0.2724 ± 0.2895 | 0.8547 ± 0.1179 | 0.8373 ± 0.1422 |
| LSTM | direct_classifier | clustered | 200 | 25 | 0.0112 ± 0.0140 | 0.0751 ± 0.1743 | 0.9631 ± 0.0719 | 0.9568 ± 0.0868 |
| LSTM | direct_classifier | clustered | 40 | 25 | 0.0203 ± 0.0302 | 0.0306 ± 0.0462 | 0.9754 ± 0.0209 | 0.9746 ± 0.0231 |
| LSTM | direct_classifier | unclustered | 100 | 25 | 0.0022 ± 0.0076 | 0.0133 ± 0.0438 | 0.9931 ± 0.0190 | 0.9922 ± 0.0219 |
| LSTM | direct_classifier | unclustered | 20 | 25 | 0.0570 ± 0.0518 | 0.2580 ± 0.2672 | 0.8593 ± 0.0988 | 0.8425 ± 0.1207 |
| LSTM | direct_classifier | unclustered | 200 | 25 | 0.0022 ± 0.0076 | 0.0415 ± 0.1132 | 0.9801 ± 0.0492 | 0.9781 ± 0.0563 |
| LSTM | direct_classifier | unclustered | 40 | 25 | 0.0191 ± 0.0289 | 0.0535 ± 0.0790 | 0.9666 ± 0.0291 | 0.9637 ± 0.0358 |
| LSTM-rollout | blackbox_rollout | clustered | 100 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | clustered | 20 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | clustered | 200 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | clustered | 40 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 100 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 20 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 200 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| LSTM-rollout | blackbox_rollout | unclustered | 40 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| M0 | physics_equation_rollout | clustered | 1 | 5 | 0.0053 ± 0.0105 | 0.6746 ± 0.2011 | 0.6725 ± 0.1265 | 0.6601 ± 0.1018 |
| M2A | physics_equation_rollout | clustered | 1 | 25 | 0.0350 ± 0.0381 | 0.5418 ± 0.3340 | 0.7206 ± 0.1725 | 0.7116 ± 0.1590 |
| M2A-LYAP | physics_equation_rollout | clustered | 1 | 25 | 0.0053 ± 0.0105 | 0.5220 ± 0.1855 | 0.7540 ± 0.0814 | 0.7364 ± 0.0907 |
| SINDYc | sparse_identification | clustered | 1 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| SINDYc | sparse_identification | unclustered | 1 | 25 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5320 ± 0.0587 | 0.5000 ± 0.0000 |
| Transformer | direct_classifier | clustered | 100 | 25 | 0.0098 ± 0.0134 | 0.0012 ± 0.0059 | 0.9944 ± 0.0068 | 0.9945 ± 0.0069 |
| Transformer | direct_classifier | clustered | 20 | 25 | 0.0401 ± 0.0497 | 0.1460 ± 0.1566 | 0.9136 ± 0.0620 | 0.9070 ± 0.0732 |
| Transformer | direct_classifier | clustered | 200 | 25 | 0.0098 ± 0.0196 | 0.0000 ± 0.0000 | 0.9944 ± 0.0112 | 0.9951 ± 0.0098 |
| Transformer | direct_classifier | clustered | 40 | 25 | 0.0219 ± 0.0315 | 0.0000 ± 0.0000 | 0.9876 ± 0.0179 | 0.9890 ± 0.0157 |
| Transformer | direct_classifier | unclustered | 100 | 25 | 0.0111 ± 0.0139 | 0.0000 ± 0.0000 | 0.9944 ± 0.0069 | 0.9944 ± 0.0069 |
| Transformer | direct_classifier | unclustered | 20 | 25 | 0.0657 ± 0.0455 | 0.1461 ± 0.1858 | 0.9013 ± 0.0705 | 0.8941 ± 0.0831 |
| Transformer | direct_classifier | unclustered | 200 | 25 | 0.0148 ± 0.0272 | 0.0000 ± 0.0000 | 0.9916 ± 0.0154 | 0.9926 ± 0.0136 |
| Transformer | direct_classifier | unclustered | 40 | 25 | 0.0168 ± 0.0248 | 0.0014 ± 0.0068 | 0.9899 ± 0.0139 | 0.9909 ± 0.0124 |

Protocol B per-split/per-seed stability grid:

| split_id | method | family | variant | window_len | seed | status | tp | fn | fp | tn | FSR | FUR | Accuracy | BalancedAccuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | 37 | 1 | 22 | 7 | 0.02631578947368421 | 0.7586206896551724 | 0.6567164179104478 | 0.6075317604355717 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | 36 | 2 | 7 | 22 | 0.05263157894736842 | 0.2413793103448276 | 0.8656716417910447 | 0.852994555353902 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | 37 | 1 | 9 | 20 | 0.02631578947368421 | 0.3103448275862069 | 0.8507462686567164 | 0.8316696914700545 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | 34 | 4 | 7 | 22 | 0.10526315789473684 | 0.2413793103448276 | 0.835820895522388 | 0.8266787658802177 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | 37 | 1 | 8 | 21 | 0.02631578947368421 | 0.27586206896551724 | 0.8656716417910447 | 0.8489110707803993 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | 35 | 3 | 7 | 22 | 0.07894736842105263 | 0.2413793103448276 | 0.8507462686567164 | 0.8398366606170599 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | 37 | 1 | 13 | 16 | 0.02631578947368421 | 0.4482758620689655 | 0.7910447761194029 | 0.7627041742286751 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | 35 | 3 | 6 | 23 | 0.07894736842105263 | 0.20689655172413793 | 0.8656716417910447 | 0.8570780399274047 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | 37 | 1 | 12 | 17 | 0.02631578947368421 | 0.41379310344827586 | 0.8059701492537313 | 0.77994555353902 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | 35 | 3 | 6 | 23 | 0.07894736842105263 | 0.20689655172413793 | 0.8656716417910447 | 0.8570780399274047 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | 37 | 1 | 12 | 17 | 0.02631578947368421 | 0.41379310344827586 | 0.8059701492537313 | 0.77994555353902 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 20 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 20 | 324 | ok | 35 | 3 | 3 | 26 | 0.07894736842105263 | 0.10344827586206896 | 0.9104477611940298 | 0.9088021778584392 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 40 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 40 | 324 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 100 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 100 | 324 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 200 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 200 | 324 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 20 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 20 | 325 | ok | 30 | 8 | 0 | 29 | 0.21052631578947367 | 0.0 | 0.8805970149253731 | 0.8947368421052632 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 40 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 40 | 325 | ok | 26 | 12 | 0 | 29 | 0.3157894736842105 | 0.0 | 0.8208955223880597 | 0.8421052631578947 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 100 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 100 | 325 | ok | 26 | 12 | 0 | 29 | 0.3157894736842105 | 0.0 | 0.8208955223880597 | 0.8421052631578947 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 200 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 200 | 325 | ok | 26 | 12 | 0 | 29 | 0.3157894736842105 | 0.0 | 0.8208955223880597 | 0.8421052631578947 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 20 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 20 | 326 | ok | 26 | 12 | 0 | 29 | 0.3157894736842105 | 0.0 | 0.8208955223880597 | 0.8421052631578947 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 40 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 40 | 326 | ok | 25 | 13 | 0 | 29 | 0.34210526315789475 | 0.0 | 0.8059701492537313 | 0.8289473684210527 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 100 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 100 | 326 | ok | 25 | 13 | 0 | 29 | 0.34210526315789475 | 0.0 | 0.8059701492537313 | 0.8289473684210527 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 200 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 200 | 326 | ok | 25 | 13 | 0 | 29 | 0.34210526315789475 | 0.0 | 0.8059701492537313 | 0.8289473684210527 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | clustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | clustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 20 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 40 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 100 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 200 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 20 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 40 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 100 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 200 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 20 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 40 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 100 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 200 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | DMD | operator_forecast | unclustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | 38 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5671641791044776 | 0.5 |
| B1 | CNN | direct_classifier | clustered | 20 | 324 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | LSTM | direct_classifier | clustered | 20 | 324 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN+LSTM | direct_classifier | clustered | 20 | 324 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | Transformer | direct_classifier | clustered | 20 | 324 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN | direct_classifier | clustered | 40 | 324 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | clustered | 40 | 324 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN+LSTM | direct_classifier | clustered | 40 | 324 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | clustered | 40 | 324 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN | direct_classifier | clustered | 100 | 324 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | clustered | 100 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 100 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | clustered | 100 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 200 | 324 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | clustered | 200 | 324 | ok | 38 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9850746268656716 | 0.9827586206896552 |
| B1 | CNN+LSTM | direct_classifier | clustered | 200 | 324 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | clustered | 200 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 20 | 325 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | LSTM | direct_classifier | clustered | 20 | 325 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN+LSTM | direct_classifier | clustered | 20 | 325 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | Transformer | direct_classifier | clustered | 20 | 325 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN | direct_classifier | clustered | 40 | 325 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | clustered | 40 | 325 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN+LSTM | direct_classifier | clustered | 40 | 325 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | clustered | 40 | 325 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN | direct_classifier | clustered | 100 | 325 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 100 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 100 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | clustered | 100 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 200 | 325 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 200 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 200 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | clustered | 200 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 20 | 326 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | clustered | 20 | 326 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN+LSTM | direct_classifier | clustered | 20 | 326 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | Transformer | direct_classifier | clustered | 20 | 326 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN | direct_classifier | clustered | 40 | 326 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 40 | 326 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN+LSTM | direct_classifier | clustered | 40 | 326 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | clustered | 40 | 326 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN | direct_classifier | clustered | 100 | 326 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 100 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 100 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | clustered | 100 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 200 | 326 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 200 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 200 | 326 | ok | 38 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9850746268656716 | 0.9827586206896552 |
| B1 | Transformer | direct_classifier | clustered | 200 | 326 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN | direct_classifier | clustered | 20 | 327 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | LSTM | direct_classifier | clustered | 20 | 327 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN+LSTM | direct_classifier | clustered | 20 | 327 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | Transformer | direct_classifier | clustered | 20 | 327 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN | direct_classifier | clustered | 40 | 327 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | LSTM | direct_classifier | clustered | 40 | 327 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN+LSTM | direct_classifier | clustered | 40 | 327 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | clustered | 40 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 100 | 327 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 100 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 100 | 327 | ok | 38 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9850746268656716 | 0.9827586206896552 |
| B1 | Transformer | direct_classifier | clustered | 100 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 200 | 327 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 200 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 200 | 327 | ok | 37 | 1 | 1 | 28 | 0.02631578947368421 | 0.034482758620689655 | 0.9701492537313433 | 0.9696007259528131 |
| B1 | Transformer | direct_classifier | clustered | 200 | 327 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN | direct_classifier | clustered | 20 | 328 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | LSTM | direct_classifier | clustered | 20 | 328 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN+LSTM | direct_classifier | clustered | 20 | 328 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | Transformer | direct_classifier | clustered | 20 | 328 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN | direct_classifier | clustered | 40 | 328 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | clustered | 40 | 328 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN+LSTM | direct_classifier | clustered | 40 | 328 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | clustered | 40 | 328 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN | direct_classifier | clustered | 100 | 328 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 100 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | clustered | 100 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | clustered | 100 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | clustered | 200 | 328 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | clustered | 200 | 328 | ok | 37 | 1 | 3 | 26 | 0.02631578947368421 | 0.10344827586206896 | 0.9402985074626866 | 0.9351179673321235 |
| B1 | CNN+LSTM | direct_classifier | clustered | 200 | 328 | ok | 37 | 1 | 0 | 29 | 0.02631578947368421 | 0.0 | 0.9850746268656716 | 0.986842105263158 |
| B1 | Transformer | direct_classifier | clustered | 200 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 20 | 324 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 20 | 324 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 20 | 324 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | Transformer | direct_classifier | unclustered | 20 | 324 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN | direct_classifier | unclustered | 40 | 324 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 40 | 324 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 40 | 324 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | unclustered | 40 | 324 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN | direct_classifier | unclustered | 100 | 324 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | LSTM | direct_classifier | unclustered | 100 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 100 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 100 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 200 | 324 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 200 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 200 | 324 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 200 | 324 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN | direct_classifier | unclustered | 20 | 325 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | LSTM | direct_classifier | unclustered | 20 | 325 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 20 | 325 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | Transformer | direct_classifier | unclustered | 20 | 325 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN | direct_classifier | unclustered | 40 | 325 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 40 | 325 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 40 | 325 | ok | 37 | 1 | 0 | 29 | 0.02631578947368421 | 0.0 | 0.9850746268656716 | 0.986842105263158 |
| B1 | Transformer | direct_classifier | unclustered | 40 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 100 | 325 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | unclustered | 100 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 100 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 100 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 200 | 325 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | unclustered | 200 | 325 | ok | 38 | 0 | 2 | 27 | 0.0 | 0.06896551724137931 | 0.9701492537313433 | 0.9655172413793103 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 200 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 200 | 325 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 20 | 326 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 20 | 326 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 20 | 326 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | Transformer | direct_classifier | unclustered | 20 | 326 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN | direct_classifier | unclustered | 40 | 326 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 40 | 326 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 40 | 326 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | Transformer | direct_classifier | unclustered | 40 | 326 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN | direct_classifier | unclustered | 100 | 326 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | unclustered | 100 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 100 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 100 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 200 | 326 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | unclustered | 200 | 326 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 200 | 326 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | Transformer | direct_classifier | unclustered | 200 | 326 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | CNN | direct_classifier | unclustered | 20 | 327 | ok | 31 | 7 | 0 | 29 | 0.18421052631578946 | 0.0 | 0.8955223880597015 | 0.9078947368421053 |
| B1 | LSTM | direct_classifier | unclustered | 20 | 327 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 20 | 327 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | Transformer | direct_classifier | unclustered | 20 | 327 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN | direct_classifier | unclustered | 40 | 327 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 40 | 327 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 40 | 327 | ok | 37 | 1 | 0 | 29 | 0.02631578947368421 | 0.0 | 0.9850746268656716 | 0.986842105263158 |
| B1 | Transformer | direct_classifier | unclustered | 40 | 327 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN | direct_classifier | unclustered | 100 | 327 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | LSTM | direct_classifier | unclustered | 100 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 100 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 100 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 200 | 327 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 200 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 200 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 200 | 327 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 20 | 328 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 20 | 328 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 20 | 328 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | Transformer | direct_classifier | unclustered | 20 | 328 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | CNN | direct_classifier | unclustered | 40 | 328 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 40 | 328 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 40 | 328 | ok | 34 | 4 | 0 | 29 | 0.10526315789473684 | 0.0 | 0.9402985074626866 | 0.9473684210526316 |
| B1 | Transformer | direct_classifier | unclustered | 40 | 328 | ok | 36 | 2 | 0 | 29 | 0.05263157894736842 | 0.0 | 0.9701492537313433 | 0.9736842105263157 |
| B1 | CNN | direct_classifier | unclustered | 100 | 328 | ok | 35 | 3 | 0 | 29 | 0.07894736842105263 | 0.0 | 0.9552238805970149 | 0.9605263157894737 |
| B1 | LSTM | direct_classifier | unclustered | 100 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 100 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 100 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN | direct_classifier | unclustered | 200 | 328 | ok | 32 | 6 | 0 | 29 | 0.15789473684210525 | 0.0 | 0.9104477611940298 | 0.9210526315789473 |
| B1 | LSTM | direct_classifier | unclustered | 200 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | CNN+LSTM | direct_classifier | unclustered | 200 | 328 | ok | 38 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B1 | Transformer | direct_classifier | unclustered | 200 | 328 | ok | 37 | 1 | 0 | 29 | 0.02631578947368421 | 0.0 | 0.9850746268656716 | 0.986842105263158 |
| B2 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | 43 | 0 | 13 | 16 | 0.0 | 0.4482758620689655 | 0.8194444444444444 | 0.7758620689655172 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | 40 | 3 | 18 | 11 | 0.06976744186046512 | 0.6206896551724138 | 0.7083333333333334 | 0.6547714514835605 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | 43 | 0 | 11 | 18 | 0.0 | 0.3793103448275862 | 0.8472222222222222 | 0.8103448275862069 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | 40 | 3 | 19 | 10 | 0.06976744186046512 | 0.6551724137931034 | 0.6944444444444444 | 0.6375300721732158 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | 43 | 0 | 11 | 18 | 0.0 | 0.3793103448275862 | 0.8472222222222222 | 0.8103448275862069 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | 40 | 3 | 11 | 18 | 0.06976744186046512 | 0.3793103448275862 | 0.8055555555555556 | 0.7754611066559743 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | 43 | 0 | 15 | 14 | 0.0 | 0.5172413793103449 | 0.7916666666666666 | 0.7413793103448276 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | 39 | 4 | 20 | 9 | 0.09302325581395349 | 0.6896551724137931 | 0.6666666666666666 | 0.6086607858861267 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | 43 | 0 | 11 | 18 | 0.0 | 0.3793103448275862 | 0.8472222222222222 | 0.8103448275862069 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | 39 | 4 | 14 | 15 | 0.09302325581395349 | 0.4827586206896552 | 0.75 | 0.7121090617481957 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | 43 | 0 | 11 | 18 | 0.0 | 0.3793103448275862 | 0.8472222222222222 | 0.8103448275862069 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 20 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 20 | 328 | ok | 30 | 13 | 4 | 25 | 0.3023255813953488 | 0.13793103448275862 | 0.7638888888888888 | 0.7798716920609463 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 40 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 40 | 328 | ok | 29 | 14 | 0 | 29 | 0.32558139534883723 | 0.0 | 0.8055555555555556 | 0.8372093023255813 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 100 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 100 | 328 | ok | 30 | 13 | 0 | 29 | 0.3023255813953488 | 0.0 | 0.8194444444444444 | 0.8488372093023255 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | clustered | 200 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | clustered | 200 | 328 | ok | 30 | 13 | 0 | 29 | 0.3023255813953488 | 0.0 | 0.8194444444444444 | 0.8488372093023255 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 20 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 40 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 100 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | DMD | operator_forecast | unclustered | 200 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | 43 | 0 | 29 | 0 | 0.0 | 1.0 | 0.5972222222222222 | 0.5 |
| B2 | CNN | direct_classifier | clustered | 20 | 324 | ok | 43 | 0 | 21 | 8 | 0.0 | 0.7241379310344828 | 0.7083333333333334 | 0.6379310344827587 |
| B2 | LSTM | direct_classifier | clustered | 20 | 324 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | CNN+LSTM | direct_classifier | clustered | 20 | 324 | ok | 43 | 0 | 21 | 8 | 0.0 | 0.7241379310344828 | 0.7083333333333334 | 0.6379310344827587 |
| B2 | Transformer | direct_classifier | clustered | 20 | 324 | ok | 43 | 0 | 5 | 24 | 0.0 | 0.1724137931034483 | 0.9305555555555556 | 0.9137931034482758 |
| B2 | CNN | direct_classifier | clustered | 40 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | LSTM | direct_classifier | clustered | 40 | 324 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | CNN+LSTM | direct_classifier | clustered | 40 | 324 | ok | 43 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9861111111111112 | 0.9827586206896552 |
| B2 | Transformer | direct_classifier | clustered | 40 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 100 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | clustered | 100 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | clustered | 100 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 100 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | clustered | 200 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | LSTM | direct_classifier | clustered | 200 | 324 | ok | 42 | 1 | 16 | 13 | 0.023255813953488372 | 0.5517241379310345 | 0.7638888888888888 | 0.7125100240577386 |
| B2 | CNN+LSTM | direct_classifier | clustered | 200 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 200 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 20 | 325 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | LSTM | direct_classifier | clustered | 20 | 325 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | CNN+LSTM | direct_classifier | clustered | 20 | 325 | ok | 43 | 0 | 21 | 8 | 0.0 | 0.7241379310344828 | 0.7083333333333334 | 0.6379310344827587 |
| B2 | Transformer | direct_classifier | clustered | 20 | 325 | ok | 43 | 0 | 12 | 17 | 0.0 | 0.41379310344827586 | 0.8333333333333334 | 0.7931034482758621 |
| B2 | CNN | direct_classifier | clustered | 40 | 325 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | LSTM | direct_classifier | clustered | 40 | 325 | ok | 43 | 0 | 4 | 25 | 0.0 | 0.13793103448275862 | 0.9444444444444444 | 0.9310344827586207 |
| B2 | CNN+LSTM | direct_classifier | clustered | 40 | 325 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | Transformer | direct_classifier | clustered | 40 | 325 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 100 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | clustered | 100 | 325 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | CNN+LSTM | direct_classifier | clustered | 100 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 100 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | clustered | 200 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | clustered | 200 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN+LSTM | direct_classifier | clustered | 200 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 200 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | clustered | 20 | 326 | ok | 43 | 0 | 17 | 12 | 0.0 | 0.5862068965517241 | 0.7638888888888888 | 0.7068965517241379 |
| B2 | LSTM | direct_classifier | clustered | 20 | 326 | ok | 43 | 0 | 18 | 11 | 0.0 | 0.6206896551724138 | 0.75 | 0.6896551724137931 |
| B2 | CNN+LSTM | direct_classifier | clustered | 20 | 326 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | Transformer | direct_classifier | clustered | 20 | 326 | ok | 43 | 0 | 13 | 16 | 0.0 | 0.4482758620689655 | 0.8194444444444444 | 0.7758620689655172 |
| B2 | CNN | direct_classifier | clustered | 40 | 326 | ok | 43 | 0 | 5 | 24 | 0.0 | 0.1724137931034483 | 0.9305555555555556 | 0.9137931034482758 |
| B2 | LSTM | direct_classifier | clustered | 40 | 326 | ok | 43 | 0 | 4 | 25 | 0.0 | 0.13793103448275862 | 0.9444444444444444 | 0.9310344827586207 |
| B2 | CNN+LSTM | direct_classifier | clustered | 40 | 326 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | Transformer | direct_classifier | clustered | 40 | 326 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | clustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN+LSTM | direct_classifier | clustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | clustered | 200 | 326 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | LSTM | direct_classifier | clustered | 200 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN+LSTM | direct_classifier | clustered | 200 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 200 | 326 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 20 | 327 | ok | 43 | 0 | 14 | 15 | 0.0 | 0.4827586206896552 | 0.8055555555555556 | 0.7586206896551724 |
| B2 | LSTM | direct_classifier | clustered | 20 | 327 | ok | 43 | 0 | 17 | 12 | 0.0 | 0.5862068965517241 | 0.7638888888888888 | 0.7068965517241379 |
| B2 | CNN+LSTM | direct_classifier | clustered | 20 | 327 | ok | 43 | 0 | 20 | 9 | 0.0 | 0.6896551724137931 | 0.7222222222222222 | 0.6551724137931034 |
| B2 | Transformer | direct_classifier | clustered | 20 | 327 | ok | 43 | 0 | 10 | 19 | 0.0 | 0.3448275862068966 | 0.8611111111111112 | 0.8275862068965517 |
| B2 | CNN | direct_classifier | clustered | 40 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | LSTM | direct_classifier | clustered | 40 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | clustered | 40 | 327 | ok | 43 | 0 | 2 | 27 | 0.0 | 0.06896551724137931 | 0.9722222222222222 | 0.9655172413793103 |
| B2 | Transformer | direct_classifier | clustered | 40 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 100 | 327 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | clustered | 100 | 327 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN+LSTM | direct_classifier | clustered | 100 | 327 | ok | 43 | 0 | 9 | 20 | 0.0 | 0.3103448275862069 | 0.875 | 0.8448275862068966 |
| B2 | Transformer | direct_classifier | clustered | 100 | 327 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | clustered | 200 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | LSTM | direct_classifier | clustered | 200 | 327 | ok | 43 | 0 | 16 | 13 | 0.0 | 0.5517241379310345 | 0.7777777777777778 | 0.7241379310344828 |
| B2 | CNN+LSTM | direct_classifier | clustered | 200 | 327 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | Transformer | direct_classifier | clustered | 200 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 20 | 328 | ok | 43 | 0 | 21 | 8 | 0.0 | 0.7241379310344828 | 0.7083333333333334 | 0.6379310344827587 |
| B2 | LSTM | direct_classifier | clustered | 20 | 328 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | CNN+LSTM | direct_classifier | clustered | 20 | 328 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | Transformer | direct_classifier | clustered | 20 | 328 | ok | 43 | 0 | 12 | 17 | 0.0 | 0.41379310344827586 | 0.8333333333333334 | 0.7931034482758621 |
| B2 | CNN | direct_classifier | clustered | 40 | 328 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | LSTM | direct_classifier | clustered | 40 | 328 | ok | 43 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9861111111111112 | 0.9827586206896552 |
| B2 | CNN+LSTM | direct_classifier | clustered | 40 | 328 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | Transformer | direct_classifier | clustered | 40 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | clustered | 100 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | clustered | 100 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | clustered | 100 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 100 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | clustered | 200 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | LSTM | direct_classifier | clustered | 200 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN+LSTM | direct_classifier | clustered | 200 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | clustered | 200 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | unclustered | 20 | 324 | ok | 43 | 0 | 20 | 9 | 0.0 | 0.6896551724137931 | 0.7222222222222222 | 0.6551724137931034 |
| B2 | LSTM | direct_classifier | unclustered | 20 | 324 | ok | 43 | 0 | 20 | 9 | 0.0 | 0.6896551724137931 | 0.7222222222222222 | 0.6551724137931034 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 20 | 324 | ok | 43 | 0 | 20 | 9 | 0.0 | 0.6896551724137931 | 0.7222222222222222 | 0.6551724137931034 |
| B2 | Transformer | direct_classifier | unclustered | 20 | 324 | ok | 43 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9861111111111112 | 0.9827586206896552 |
| B2 | CNN | direct_classifier | unclustered | 40 | 324 | ok | 43 | 0 | 6 | 23 | 0.0 | 0.20689655172413793 | 0.9166666666666666 | 0.896551724137931 |
| B2 | LSTM | direct_classifier | unclustered | 40 | 324 | ok | 43 | 0 | 4 | 25 | 0.0 | 0.13793103448275862 | 0.9444444444444444 | 0.9310344827586207 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 40 | 324 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | Transformer | direct_classifier | unclustered | 40 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | unclustered | 100 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 100 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 100 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | Transformer | direct_classifier | unclustered | 100 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | unclustered | 200 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 200 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 200 | 324 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | unclustered | 200 | 324 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | unclustered | 20 | 325 | ok | 43 | 0 | 20 | 9 | 0.0 | 0.6896551724137931 | 0.7222222222222222 | 0.6551724137931034 |
| B2 | LSTM | direct_classifier | unclustered | 20 | 325 | ok | 43 | 0 | 18 | 11 | 0.0 | 0.6206896551724138 | 0.75 | 0.6896551724137931 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 20 | 325 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | Transformer | direct_classifier | unclustered | 20 | 325 | ok | 43 | 0 | 12 | 17 | 0.0 | 0.41379310344827586 | 0.8333333333333334 | 0.7931034482758621 |
| B2 | CNN | direct_classifier | unclustered | 40 | 325 | ok | 43 | 0 | 5 | 24 | 0.0 | 0.1724137931034483 | 0.9305555555555556 | 0.9137931034482758 |
| B2 | LSTM | direct_classifier | unclustered | 40 | 325 | ok | 43 | 0 | 5 | 24 | 0.0 | 0.1724137931034483 | 0.9305555555555556 | 0.9137931034482758 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 40 | 325 | ok | 43 | 0 | 4 | 25 | 0.0 | 0.13793103448275862 | 0.9444444444444444 | 0.9310344827586207 |
| B2 | Transformer | direct_classifier | unclustered | 40 | 325 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | unclustered | 100 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 100 | 325 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 100 | 325 | ok | 43 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9861111111111112 | 0.9827586206896552 |
| B2 | Transformer | direct_classifier | unclustered | 100 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | unclustered | 200 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 200 | 325 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 200 | 325 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | Transformer | direct_classifier | unclustered | 200 | 325 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | unclustered | 20 | 326 | ok | 43 | 0 | 19 | 10 | 0.0 | 0.6551724137931034 | 0.7361111111111112 | 0.6724137931034483 |
| B2 | LSTM | direct_classifier | unclustered | 20 | 326 | ok | 43 | 0 | 16 | 13 | 0.0 | 0.5517241379310345 | 0.7777777777777778 | 0.7241379310344828 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 20 | 326 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | Transformer | direct_classifier | unclustered | 20 | 326 | ok | 43 | 0 | 10 | 19 | 0.0 | 0.3448275862068966 | 0.8611111111111112 | 0.8275862068965517 |
| B2 | CNN | direct_classifier | unclustered | 40 | 326 | ok | 43 | 0 | 5 | 24 | 0.0 | 0.1724137931034483 | 0.9305555555555556 | 0.9137931034482758 |
| B2 | LSTM | direct_classifier | unclustered | 40 | 326 | ok | 43 | 0 | 8 | 21 | 0.0 | 0.27586206896551724 | 0.8888888888888888 | 0.8620689655172413 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 40 | 326 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | Transformer | direct_classifier | unclustered | 40 | 326 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | unclustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | unclustered | 100 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | unclustered | 200 | 326 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 200 | 326 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 200 | 326 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | Transformer | direct_classifier | unclustered | 200 | 326 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | unclustered | 20 | 327 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | LSTM | direct_classifier | unclustered | 20 | 327 | ok | 43 | 0 | 17 | 12 | 0.0 | 0.5862068965517241 | 0.7638888888888888 | 0.7068965517241379 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 20 | 327 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | Transformer | direct_classifier | unclustered | 20 | 327 | ok | 43 | 0 | 17 | 12 | 0.0 | 0.5862068965517241 | 0.7638888888888888 | 0.7068965517241379 |
| B2 | CNN | direct_classifier | unclustered | 40 | 327 | ok | 43 | 0 | 9 | 20 | 0.0 | 0.3103448275862069 | 0.875 | 0.8448275862068966 |
| B2 | LSTM | direct_classifier | unclustered | 40 | 327 | ok | 43 | 0 | 6 | 23 | 0.0 | 0.20689655172413793 | 0.9166666666666666 | 0.896551724137931 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 40 | 327 | ok | 43 | 0 | 4 | 25 | 0.0 | 0.13793103448275862 | 0.9444444444444444 | 0.9310344827586207 |
| B2 | Transformer | direct_classifier | unclustered | 40 | 327 | ok | 43 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9861111111111112 | 0.9827586206896552 |
| B2 | CNN | direct_classifier | unclustered | 100 | 327 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 100 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 100 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | Transformer | direct_classifier | unclustered | 100 | 327 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | unclustered | 200 | 327 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 200 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 200 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | Transformer | direct_classifier | unclustered | 200 | 327 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | unclustered | 20 | 328 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | LSTM | direct_classifier | unclustered | 20 | 328 | ok | 43 | 0 | 16 | 13 | 0.0 | 0.5517241379310345 | 0.7777777777777778 | 0.7241379310344828 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 20 | 328 | ok | 43 | 0 | 22 | 7 | 0.0 | 0.7586206896551724 | 0.6944444444444444 | 0.6206896551724138 |
| B2 | Transformer | direct_classifier | unclustered | 20 | 328 | ok | 43 | 0 | 4 | 25 | 0.0 | 0.13793103448275862 | 0.9444444444444444 | 0.9310344827586207 |
| B2 | CNN | direct_classifier | unclustered | 40 | 328 | ok | 43 | 0 | 8 | 21 | 0.0 | 0.27586206896551724 | 0.8888888888888888 | 0.8620689655172413 |
| B2 | LSTM | direct_classifier | unclustered | 40 | 328 | ok | 43 | 0 | 3 | 26 | 0.0 | 0.10344827586206896 | 0.9583333333333334 | 0.9482758620689655 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 40 | 328 | ok | 43 | 0 | 1 | 28 | 0.0 | 0.034482758620689655 | 0.9861111111111112 | 0.9827586206896552 |
| B2 | Transformer | direct_classifier | unclustered | 40 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN | direct_classifier | unclustered | 100 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 100 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 100 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | Transformer | direct_classifier | unclustered | 100 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | CNN | direct_classifier | unclustered | 200 | 328 | ok | 42 | 1 | 0 | 29 | 0.023255813953488372 | 0.0 | 0.9861111111111112 | 0.9883720930232558 |
| B2 | LSTM | direct_classifier | unclustered | 200 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | CNN+LSTM | direct_classifier | unclustered | 200 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B2 | Transformer | direct_classifier | unclustered | 200 | 328 | ok | 43 | 0 | 0 | 29 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | 35 | 0 | 33 | 3 | 0.0 | 0.9166666666666666 | 0.5352112676056338 | 0.5416666666666666 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | 35 | 0 | 4 | 32 | 0.0 | 0.1111111111111111 | 0.9436619718309859 | 0.9444444444444444 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | 35 | 0 | 20 | 16 | 0.0 | 0.5555555555555556 | 0.7183098591549296 | 0.7222222222222222 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | 34 | 1 | 6 | 30 | 0.02857142857142857 | 0.16666666666666666 | 0.9014084507042254 | 0.9023809523809524 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | 35 | 0 | 25 | 11 | 0.0 | 0.6944444444444444 | 0.647887323943662 | 0.6527777777777778 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | 35 | 0 | 4 | 32 | 0.0 | 0.1111111111111111 | 0.9436619718309859 | 0.9444444444444444 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | 35 | 0 | 20 | 16 | 0.0 | 0.5555555555555556 | 0.7183098591549296 | 0.7222222222222222 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | 34 | 1 | 3 | 33 | 0.02857142857142857 | 0.08333333333333333 | 0.9436619718309859 | 0.944047619047619 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | 35 | 0 | 21 | 15 | 0.0 | 0.5833333333333334 | 0.704225352112676 | 0.7083333333333334 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | 34 | 1 | 6 | 30 | 0.02857142857142857 | 0.16666666666666666 | 0.9014084507042254 | 0.9023809523809524 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | 35 | 0 | 21 | 15 | 0.0 | 0.5833333333333334 | 0.704225352112676 | 0.7083333333333334 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 20 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 20 | 324 | ok | 32 | 3 | 0 | 36 | 0.08571428571428572 | 0.0 | 0.9577464788732394 | 0.9571428571428571 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 40 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 40 | 324 | ok | 30 | 5 | 0 | 36 | 0.14285714285714285 | 0.0 | 0.9295774647887324 | 0.9285714285714286 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 100 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 100 | 324 | ok | 30 | 5 | 0 | 36 | 0.14285714285714285 | 0.0 | 0.9295774647887324 | 0.9285714285714286 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 200 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 200 | 324 | ok | 30 | 5 | 0 | 36 | 0.14285714285714285 | 0.0 | 0.9295774647887324 | 0.9285714285714286 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | clustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | clustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 20 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 40 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 100 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 200 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | DMD | operator_forecast | unclustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | 35 | 0 | 36 | 0 | 0.0 | 1.0 | 0.49295774647887325 | 0.5 |
| B3 | CNN | direct_classifier | clustered | 20 | 324 | ok | 33 | 2 | 0 | 36 | 0.05714285714285714 | 0.0 | 0.971830985915493 | 0.9714285714285714 |
| B3 | LSTM | direct_classifier | clustered | 20 | 324 | ok | 33 | 2 | 7 | 29 | 0.05714285714285714 | 0.19444444444444445 | 0.8732394366197183 | 0.8742063492063492 |
| B3 | CNN+LSTM | direct_classifier | clustered | 20 | 324 | ok | 33 | 2 | 8 | 28 | 0.05714285714285714 | 0.2222222222222222 | 0.8591549295774648 | 0.8603174603174604 |
| B3 | Transformer | direct_classifier | clustered | 20 | 324 | ok | 35 | 0 | 8 | 28 | 0.0 | 0.2222222222222222 | 0.8873239436619719 | 0.8888888888888888 |
| B3 | CNN | direct_classifier | clustered | 40 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 40 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 40 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 40 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 20 | 325 | ok | 33 | 2 | 0 | 36 | 0.05714285714285714 | 0.0 | 0.971830985915493 | 0.9714285714285714 |
| B3 | LSTM | direct_classifier | clustered | 20 | 325 | ok | 33 | 2 | 1 | 35 | 0.05714285714285714 | 0.027777777777777776 | 0.9577464788732394 | 0.9575396825396825 |
| B3 | CNN+LSTM | direct_classifier | clustered | 20 | 325 | ok | 33 | 2 | 6 | 30 | 0.05714285714285714 | 0.16666666666666666 | 0.8873239436619719 | 0.888095238095238 |
| B3 | Transformer | direct_classifier | clustered | 20 | 325 | ok | 35 | 0 | 8 | 28 | 0.0 | 0.2222222222222222 | 0.8873239436619719 | 0.8888888888888888 |
| B3 | CNN | direct_classifier | clustered | 40 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 40 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 40 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 40 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 20 | 326 | ok | 33 | 2 | 8 | 28 | 0.05714285714285714 | 0.2222222222222222 | 0.8591549295774648 | 0.8603174603174604 |
| B3 | LSTM | direct_classifier | clustered | 20 | 326 | ok | 33 | 2 | 3 | 33 | 0.05714285714285714 | 0.08333333333333333 | 0.9295774647887324 | 0.9297619047619048 |
| B3 | CNN+LSTM | direct_classifier | clustered | 20 | 326 | ok | 33 | 2 | 9 | 27 | 0.05714285714285714 | 0.25 | 0.8450704225352113 | 0.8464285714285714 |
| B3 | Transformer | direct_classifier | clustered | 20 | 326 | ok | 34 | 1 | 0 | 36 | 0.02857142857142857 | 0.0 | 0.9859154929577465 | 0.9857142857142858 |
| B3 | CNN | direct_classifier | clustered | 40 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 40 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 40 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 40 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 100 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 100 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 100 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 100 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 20 | 327 | ok | 35 | 0 | 9 | 27 | 0.0 | 0.25 | 0.8732394366197183 | 0.875 |
| B3 | LSTM | direct_classifier | clustered | 20 | 327 | ok | 34 | 1 | 7 | 29 | 0.02857142857142857 | 0.19444444444444445 | 0.8873239436619719 | 0.8884920634920634 |
| B3 | CNN+LSTM | direct_classifier | clustered | 20 | 327 | ok | 34 | 1 | 8 | 28 | 0.02857142857142857 | 0.2222222222222222 | 0.8732394366197183 | 0.8746031746031746 |
| B3 | Transformer | direct_classifier | clustered | 20 | 327 | ok | 35 | 0 | 10 | 26 | 0.0 | 0.2777777777777778 | 0.8591549295774648 | 0.8611111111111112 |
| B3 | CNN | direct_classifier | clustered | 40 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 40 | 327 | ok | 35 | 0 | 1 | 35 | 0.0 | 0.027777777777777776 | 0.9859154929577465 | 0.9861111111111112 |
| B3 | CNN+LSTM | direct_classifier | clustered | 40 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 40 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 200 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 200 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 200 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 200 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 20 | 328 | ok | 35 | 0 | 9 | 27 | 0.0 | 0.25 | 0.8732394366197183 | 0.875 |
| B3 | LSTM | direct_classifier | clustered | 20 | 328 | ok | 33 | 2 | 8 | 28 | 0.05714285714285714 | 0.2222222222222222 | 0.8591549295774648 | 0.8603174603174604 |
| B3 | CNN+LSTM | direct_classifier | clustered | 20 | 328 | ok | 32 | 3 | 8 | 28 | 0.08571428571428572 | 0.2222222222222222 | 0.8450704225352113 | 0.846031746031746 |
| B3 | Transformer | direct_classifier | clustered | 20 | 328 | ok | 35 | 0 | 7 | 29 | 0.0 | 0.19444444444444445 | 0.9014084507042254 | 0.9027777777777778 |
| B3 | CNN | direct_classifier | clustered | 40 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 40 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 40 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 40 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | clustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | clustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | clustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | clustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 20 | 324 | ok | 32 | 3 | 5 | 31 | 0.08571428571428572 | 0.1388888888888889 | 0.8873239436619719 | 0.8876984126984127 |
| B3 | LSTM | direct_classifier | unclustered | 20 | 324 | ok | 33 | 2 | 5 | 31 | 0.05714285714285714 | 0.1388888888888889 | 0.9014084507042254 | 0.901984126984127 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 20 | 324 | ok | 32 | 3 | 11 | 25 | 0.08571428571428572 | 0.3055555555555556 | 0.8028169014084507 | 0.8043650793650794 |
| B3 | Transformer | direct_classifier | unclustered | 20 | 324 | ok | 33 | 2 | 4 | 32 | 0.05714285714285714 | 0.1111111111111111 | 0.9154929577464789 | 0.9158730158730158 |
| B3 | CNN | direct_classifier | unclustered | 40 | 324 | ok | 33 | 2 | 0 | 36 | 0.05714285714285714 | 0.0 | 0.971830985915493 | 0.9714285714285714 |
| B3 | LSTM | direct_classifier | unclustered | 40 | 324 | ok | 35 | 0 | 2 | 34 | 0.0 | 0.05555555555555555 | 0.971830985915493 | 0.9722222222222222 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 40 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 40 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 100 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 200 | 324 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 20 | 325 | ok | 33 | 2 | 11 | 25 | 0.05714285714285714 | 0.3055555555555556 | 0.8169014084507042 | 0.8186507936507936 |
| B3 | LSTM | direct_classifier | unclustered | 20 | 325 | ok | 32 | 3 | 7 | 29 | 0.08571428571428572 | 0.19444444444444445 | 0.8591549295774648 | 0.859920634920635 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 20 | 325 | ok | 32 | 3 | 9 | 27 | 0.08571428571428572 | 0.25 | 0.8309859154929577 | 0.8321428571428571 |
| B3 | Transformer | direct_classifier | unclustered | 20 | 325 | ok | 33 | 2 | 6 | 30 | 0.05714285714285714 | 0.16666666666666666 | 0.8873239436619719 | 0.888095238095238 |
| B3 | CNN | direct_classifier | unclustered | 40 | 325 | ok | 34 | 1 | 4 | 32 | 0.02857142857142857 | 0.1111111111111111 | 0.9295774647887324 | 0.9301587301587302 |
| B3 | LSTM | direct_classifier | unclustered | 40 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 40 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 40 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 100 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 200 | 325 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 20 | 326 | ok | 33 | 2 | 11 | 25 | 0.05714285714285714 | 0.3055555555555556 | 0.8169014084507042 | 0.8186507936507936 |
| B3 | LSTM | direct_classifier | unclustered | 20 | 326 | ok | 33 | 2 | 6 | 30 | 0.05714285714285714 | 0.16666666666666666 | 0.8873239436619719 | 0.888095238095238 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 20 | 326 | ok | 32 | 3 | 7 | 29 | 0.08571428571428572 | 0.19444444444444445 | 0.8591549295774648 | 0.859920634920635 |
| B3 | Transformer | direct_classifier | unclustered | 20 | 326 | ok | 33 | 2 | 3 | 33 | 0.05714285714285714 | 0.08333333333333333 | 0.9295774647887324 | 0.9297619047619048 |
| B3 | CNN | direct_classifier | unclustered | 40 | 326 | ok | 35 | 0 | 1 | 35 | 0.0 | 0.027777777777777776 | 0.9859154929577465 | 0.9861111111111112 |
| B3 | LSTM | direct_classifier | unclustered | 40 | 326 | ok | 35 | 0 | 2 | 34 | 0.0 | 0.05555555555555555 | 0.971830985915493 | 0.9722222222222222 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 40 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 40 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 100 | 326 | ok | 35 | 0 | 18 | 18 | 0.0 | 0.5 | 0.7464788732394366 | 0.75 |
| B3 | LSTM | direct_classifier | unclustered | 100 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 100 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 100 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 200 | 326 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 20 | 327 | ok | 32 | 3 | 7 | 29 | 0.08571428571428572 | 0.19444444444444445 | 0.8591549295774648 | 0.859920634920635 |
| B3 | LSTM | direct_classifier | unclustered | 20 | 327 | ok | 33 | 2 | 6 | 30 | 0.05714285714285714 | 0.16666666666666666 | 0.8873239436619719 | 0.888095238095238 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 20 | 327 | ok | 33 | 2 | 7 | 29 | 0.05714285714285714 | 0.19444444444444445 | 0.8732394366197183 | 0.8742063492063492 |
| B3 | Transformer | direct_classifier | unclustered | 20 | 327 | ok | 33 | 2 | 10 | 26 | 0.05714285714285714 | 0.2777777777777778 | 0.8309859154929577 | 0.8325396825396825 |
| B3 | CNN | direct_classifier | unclustered | 40 | 327 | ok | 33 | 2 | 0 | 36 | 0.05714285714285714 | 0.0 | 0.971830985915493 | 0.9714285714285714 |
| B3 | LSTM | direct_classifier | unclustered | 40 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 40 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 40 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 100 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 200 | 327 | ok | 34 | 1 | 0 | 36 | 0.02857142857142857 | 0.0 | 0.9859154929577465 | 0.9857142857142858 |
| B3 | LSTM | direct_classifier | unclustered | 200 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 200 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 200 | 327 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 20 | 328 | ok | 33 | 2 | 8 | 28 | 0.05714285714285714 | 0.2222222222222222 | 0.8591549295774648 | 0.8603174603174604 |
| B3 | LSTM | direct_classifier | unclustered | 20 | 328 | ok | 33 | 2 | 0 | 36 | 0.05714285714285714 | 0.0 | 0.971830985915493 | 0.9714285714285714 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 20 | 328 | ok | 32 | 3 | 3 | 33 | 0.08571428571428572 | 0.08333333333333333 | 0.9154929577464789 | 0.9154761904761904 |
| B3 | Transformer | direct_classifier | unclustered | 20 | 328 | ok | 33 | 2 | 7 | 29 | 0.05714285714285714 | 0.19444444444444445 | 0.8732394366197183 | 0.8742063492063492 |
| B3 | CNN | direct_classifier | unclustered | 40 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 40 | 328 | ok | 35 | 0 | 1 | 35 | 0.0 | 0.027777777777777776 | 0.9859154929577465 | 0.9861111111111112 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 40 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 40 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 100 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN | direct_classifier | unclustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | LSTM | direct_classifier | unclustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | CNN+LSTM | direct_classifier | unclustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B3 | Transformer | direct_classifier | unclustered | 200 | 328 | ok | 35 | 0 | 0 | 36 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | 43 | 0 | 14 | 19 | 0.0 | 0.42424242424242425 | 0.8157894736842105 | 0.7878787878787878 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | 43 | 0 | 28 | 5 | 0.0 | 0.8484848484848485 | 0.631578947368421 | 0.5757575757575758 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | 43 | 0 | 28 | 5 | 0.0 | 0.8484848484848485 | 0.631578947368421 | 0.5757575757575758 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | 43 | 0 | 26 | 7 | 0.0 | 0.7878787878787878 | 0.6578947368421053 | 0.6060606060606061 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | 43 | 0 | 28 | 5 | 0.0 | 0.8484848484848485 | 0.631578947368421 | 0.5757575757575758 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | 43 | 0 | 26 | 7 | 0.0 | 0.7878787878787878 | 0.6578947368421053 | 0.6060606060606061 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | 43 | 0 | 27 | 6 | 0.0 | 0.8181818181818182 | 0.6447368421052632 | 0.5909090909090909 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | 43 | 0 | 24 | 9 | 0.0 | 0.7272727272727273 | 0.6842105263157895 | 0.6363636363636364 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | 43 | 0 | 27 | 6 | 0.0 | 0.8181818181818182 | 0.6447368421052632 | 0.5909090909090909 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | 43 | 0 | 26 | 7 | 0.0 | 0.7878787878787878 | 0.6578947368421053 | 0.6060606060606061 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | 43 | 0 | 28 | 5 | 0.0 | 0.8484848484848485 | 0.631578947368421 | 0.5757575757575758 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 20 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 20 | 324 | ok | 37 | 6 | 7 | 26 | 0.13953488372093023 | 0.21212121212121213 | 0.8289473684210527 | 0.8241719520789288 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 40 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 40 | 324 | ok | 35 | 8 | 0 | 33 | 0.18604651162790697 | 0.0 | 0.8947368421052632 | 0.9069767441860466 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 100 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 100 | 324 | ok | 30 | 13 | 0 | 33 | 0.3023255813953488 | 0.0 | 0.8289473684210527 | 0.8488372093023255 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 200 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 200 | 324 | ok | 30 | 13 | 0 | 33 | 0.3023255813953488 | 0.0 | 0.8289473684210527 | 0.8488372093023255 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 100 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 100 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 200 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 200 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 20 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 20 | 328 | ok | 42 | 1 | 0 | 33 | 0.023255813953488372 | 0.0 | 0.9868421052631579 | 0.9883720930232558 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 40 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 40 | 328 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 100 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 100 | 328 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | clustered | 200 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | clustered | 200 | 328 | ok | 32 | 11 | 0 | 33 | 0.2558139534883721 | 0.0 | 0.8552631578947368 | 0.872093023255814 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 20 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 40 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 100 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 200 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 100 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 327 | ok | 43 | 0 | 31 | 2 | 0.0 | 0.9393939393939394 | 0.5921052631578947 | 0.5303030303030303 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 200 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 327 | ok | 43 | 0 | 7 | 26 | 0.0 | 0.21212121212121213 | 0.9078947368421053 | 0.8939393939393939 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 20 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 328 | ok | 43 | 0 | 3 | 30 | 0.0 | 0.09090909090909091 | 0.9605263157894737 | 0.9545454545454546 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 40 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 100 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | DMD | operator_forecast | unclustered | 200 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | 43 | 0 | 33 | 0 | 0.0 | 1.0 | 0.5657894736842105 | 0.5 |
| B4 | CNN | direct_classifier | clustered | 20 | 324 | ok | 37 | 6 | 17 | 16 | 0.13953488372093023 | 0.5151515151515151 | 0.6973684210526315 | 0.6726568005637773 |
| B4 | LSTM | direct_classifier | clustered | 20 | 324 | ok | 37 | 6 | 18 | 15 | 0.13953488372093023 | 0.5454545454545454 | 0.6842105263157895 | 0.6575052854122622 |
| B4 | CNN+LSTM | direct_classifier | clustered | 20 | 324 | ok | 40 | 3 | 18 | 15 | 0.06976744186046512 | 0.5454545454545454 | 0.7236842105263158 | 0.6923890063424947 |
| B4 | Transformer | direct_classifier | clustered | 20 | 324 | ok | 40 | 3 | 8 | 25 | 0.06976744186046512 | 0.24242424242424243 | 0.8552631578947368 | 0.8439041578576463 |
| B4 | CNN | direct_classifier | clustered | 40 | 324 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | clustered | 40 | 324 | ok | 43 | 0 | 4 | 29 | 0.0 | 0.12121212121212122 | 0.9473684210526315 | 0.9393939393939394 |
| B4 | CNN+LSTM | direct_classifier | clustered | 40 | 324 | ok | 43 | 0 | 1 | 32 | 0.0 | 0.030303030303030304 | 0.9868421052631579 | 0.9848484848484849 |
| B4 | Transformer | direct_classifier | clustered | 40 | 324 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN | direct_classifier | clustered | 100 | 324 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | clustered | 100 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | clustered | 100 | 324 | ok | 43 | 0 | 1 | 32 | 0.0 | 0.030303030303030304 | 0.9868421052631579 | 0.9848484848484849 |
| B4 | Transformer | direct_classifier | clustered | 100 | 324 | ok | 43 | 0 | 1 | 32 | 0.0 | 0.030303030303030304 | 0.9868421052631579 | 0.9848484848484849 |
| B4 | CNN | direct_classifier | clustered | 200 | 324 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | clustered | 200 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | clustered | 200 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | clustered | 200 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | clustered | 20 | 325 | ok | 37 | 6 | 18 | 15 | 0.13953488372093023 | 0.5454545454545454 | 0.6842105263157895 | 0.6575052854122622 |
| B4 | LSTM | direct_classifier | clustered | 20 | 325 | ok | 39 | 4 | 18 | 15 | 0.09302325581395349 | 0.5454545454545454 | 0.7105263157894737 | 0.6807610993657505 |
| B4 | CNN+LSTM | direct_classifier | clustered | 20 | 325 | ok | 39 | 4 | 16 | 17 | 0.09302325581395349 | 0.48484848484848486 | 0.7368421052631579 | 0.7110641296687807 |
| B4 | Transformer | direct_classifier | clustered | 20 | 325 | ok | 39 | 4 | 4 | 29 | 0.09302325581395349 | 0.12121212121212122 | 0.8947368421052632 | 0.8928823114869626 |
| B4 | CNN | direct_classifier | clustered | 40 | 325 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | clustered | 40 | 325 | ok | 41 | 2 | 1 | 32 | 0.046511627906976744 | 0.030303030303030304 | 0.9605263157894737 | 0.9615926708949964 |
| B4 | CNN+LSTM | direct_classifier | clustered | 40 | 325 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | Transformer | direct_classifier | clustered | 40 | 325 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN | direct_classifier | clustered | 100 | 325 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | clustered | 100 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | clustered | 100 | 325 | ok | 43 | 0 | 1 | 32 | 0.0 | 0.030303030303030304 | 0.9868421052631579 | 0.9848484848484849 |
| B4 | Transformer | direct_classifier | clustered | 100 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | clustered | 200 | 325 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | clustered | 200 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | clustered | 200 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | clustered | 200 | 325 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | CNN | direct_classifier | clustered | 20 | 326 | ok | 37 | 6 | 16 | 17 | 0.13953488372093023 | 0.48484848484848486 | 0.7105263157894737 | 0.6878083157152924 |
| B4 | LSTM | direct_classifier | clustered | 20 | 326 | ok | 39 | 4 | 16 | 17 | 0.09302325581395349 | 0.48484848484848486 | 0.7368421052631579 | 0.7110641296687807 |
| B4 | CNN+LSTM | direct_classifier | clustered | 20 | 326 | ok | 37 | 6 | 16 | 17 | 0.13953488372093023 | 0.48484848484848486 | 0.7105263157894737 | 0.6878083157152924 |
| B4 | Transformer | direct_classifier | clustered | 20 | 326 | ok | 40 | 3 | 6 | 27 | 0.06976744186046512 | 0.18181818181818182 | 0.881578947368421 | 0.8742071881606766 |
| B4 | CNN | direct_classifier | clustered | 40 | 326 | ok | 37 | 6 | 3 | 30 | 0.13953488372093023 | 0.09090909090909091 | 0.881578947368421 | 0.8847780126849893 |
| B4 | LSTM | direct_classifier | clustered | 40 | 326 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN+LSTM | direct_classifier | clustered | 40 | 326 | ok | 42 | 1 | 1 | 32 | 0.023255813953488372 | 0.030303030303030304 | 0.9736842105263158 | 0.9732205778717407 |
| B4 | Transformer | direct_classifier | clustered | 40 | 326 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN | direct_classifier | clustered | 100 | 326 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | clustered | 100 | 326 | ok | 43 | 0 | 2 | 31 | 0.0 | 0.06060606060606061 | 0.9736842105263158 | 0.9696969696969697 |
| B4 | CNN+LSTM | direct_classifier | clustered | 100 | 326 | ok | 43 | 0 | 3 | 30 | 0.0 | 0.09090909090909091 | 0.9605263157894737 | 0.9545454545454546 |
| B4 | Transformer | direct_classifier | clustered | 100 | 326 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | clustered | 200 | 326 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | clustered | 200 | 326 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | clustered | 200 | 326 | ok | 43 | 0 | 2 | 31 | 0.0 | 0.06060606060606061 | 0.9736842105263158 | 0.9696969696969697 |
| B4 | Transformer | direct_classifier | clustered | 200 | 326 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | clustered | 20 | 327 | ok | 36 | 7 | 14 | 19 | 0.16279069767441862 | 0.42424242424242425 | 0.7236842105263158 | 0.7064834390415786 |
| B4 | LSTM | direct_classifier | clustered | 20 | 327 | ok | 40 | 3 | 16 | 17 | 0.06976744186046512 | 0.48484848484848486 | 0.75 | 0.722692036645525 |
| B4 | CNN+LSTM | direct_classifier | clustered | 20 | 327 | ok | 36 | 7 | 18 | 15 | 0.16279069767441862 | 0.5454545454545454 | 0.6710526315789473 | 0.645877378435518 |
| B4 | Transformer | direct_classifier | clustered | 20 | 327 | ok | 39 | 4 | 12 | 21 | 0.09302325581395349 | 0.36363636363636365 | 0.7894736842105263 | 0.7716701902748414 |
| B4 | CNN | direct_classifier | clustered | 40 | 327 | ok | 36 | 7 | 0 | 33 | 0.16279069767441862 | 0.0 | 0.9078947368421053 | 0.9186046511627908 |
| B4 | LSTM | direct_classifier | clustered | 40 | 327 | ok | 42 | 1 | 3 | 30 | 0.023255813953488372 | 0.09090909090909091 | 0.9473684210526315 | 0.9429175475687104 |
| B4 | CNN+LSTM | direct_classifier | clustered | 40 | 327 | ok | 42 | 1 | 0 | 33 | 0.023255813953488372 | 0.0 | 0.9868421052631579 | 0.9883720930232558 |
| B4 | Transformer | direct_classifier | clustered | 40 | 327 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | CNN | direct_classifier | clustered | 100 | 327 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | clustered | 100 | 327 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | clustered | 100 | 327 | ok | 43 | 0 | 5 | 28 | 0.0 | 0.15151515151515152 | 0.9342105263157895 | 0.9242424242424243 |
| B4 | Transformer | direct_classifier | clustered | 100 | 327 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | clustered | 200 | 327 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | clustered | 200 | 327 | ok | 43 | 0 | 17 | 16 | 0.0 | 0.5151515151515151 | 0.7763157894736842 | 0.7424242424242424 |
| B4 | CNN+LSTM | direct_classifier | clustered | 200 | 327 | ok | 40 | 3 | 20 | 13 | 0.06976744186046512 | 0.6060606060606061 | 0.6973684210526315 | 0.6620859760394644 |
| B4 | Transformer | direct_classifier | clustered | 200 | 327 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | clustered | 20 | 328 | ok | 37 | 6 | 18 | 15 | 0.13953488372093023 | 0.5454545454545454 | 0.6842105263157895 | 0.6575052854122622 |
| B4 | LSTM | direct_classifier | clustered | 20 | 328 | ok | 39 | 4 | 18 | 15 | 0.09302325581395349 | 0.5454545454545454 | 0.7105263157894737 | 0.6807610993657505 |
| B4 | CNN+LSTM | direct_classifier | clustered | 20 | 328 | ok | 36 | 7 | 18 | 15 | 0.16279069767441862 | 0.5454545454545454 | 0.6710526315789473 | 0.645877378435518 |
| B4 | Transformer | direct_classifier | clustered | 20 | 328 | ok | 40 | 3 | 1 | 32 | 0.06976744186046512 | 0.030303030303030304 | 0.9473684210526315 | 0.9499647639182522 |
| B4 | CNN | direct_classifier | clustered | 40 | 328 | ok | 37 | 6 | 3 | 30 | 0.13953488372093023 | 0.09090909090909091 | 0.881578947368421 | 0.8847780126849893 |
| B4 | LSTM | direct_classifier | clustered | 40 | 328 | ok | 42 | 1 | 1 | 32 | 0.023255813953488372 | 0.030303030303030304 | 0.9736842105263158 | 0.9732205778717407 |
| B4 | CNN+LSTM | direct_classifier | clustered | 40 | 328 | ok | 42 | 1 | 1 | 32 | 0.023255813953488372 | 0.030303030303030304 | 0.9736842105263158 | 0.9732205778717407 |
| B4 | Transformer | direct_classifier | clustered | 40 | 328 | ok | 42 | 1 | 0 | 33 | 0.023255813953488372 | 0.0 | 0.9868421052631579 | 0.9883720930232558 |
| B4 | CNN | direct_classifier | clustered | 100 | 328 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | clustered | 100 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | clustered | 100 | 328 | ok | 43 | 0 | 4 | 29 | 0.0 | 0.12121212121212122 | 0.9473684210526315 | 0.9393939393939394 |
| B4 | Transformer | direct_classifier | clustered | 100 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | clustered | 200 | 328 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | clustered | 200 | 328 | ok | 43 | 0 | 4 | 29 | 0.0 | 0.12121212121212122 | 0.9473684210526315 | 0.9393939393939394 |
| B4 | CNN+LSTM | direct_classifier | clustered | 200 | 328 | ok | 43 | 0 | 1 | 32 | 0.0 | 0.030303030303030304 | 0.9868421052631579 | 0.9848484848484849 |
| B4 | Transformer | direct_classifier | clustered | 200 | 328 | ok | 42 | 1 | 0 | 33 | 0.023255813953488372 | 0.0 | 0.9868421052631579 | 0.9883720930232558 |
| B4 | CNN | direct_classifier | unclustered | 20 | 324 | ok | 41 | 2 | 18 | 15 | 0.046511627906976744 | 0.5454545454545454 | 0.7368421052631579 | 0.7040169133192389 |
| B4 | LSTM | direct_classifier | unclustered | 20 | 324 | ok | 40 | 3 | 18 | 15 | 0.06976744186046512 | 0.5454545454545454 | 0.7236842105263158 | 0.6923890063424947 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 20 | 324 | ok | 37 | 6 | 18 | 15 | 0.13953488372093023 | 0.5454545454545454 | 0.6842105263157895 | 0.6575052854122622 |
| B4 | Transformer | direct_classifier | unclustered | 20 | 324 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | CNN | direct_classifier | unclustered | 40 | 324 | ok | 39 | 4 | 3 | 30 | 0.09302325581395349 | 0.09090909090909091 | 0.9078947368421053 | 0.9080338266384778 |
| B4 | LSTM | direct_classifier | unclustered | 40 | 324 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 40 | 324 | ok | 43 | 0 | 3 | 30 | 0.0 | 0.09090909090909091 | 0.9605263157894737 | 0.9545454545454546 |
| B4 | Transformer | direct_classifier | unclustered | 40 | 324 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | CNN | direct_classifier | unclustered | 100 | 324 | ok | 38 | 5 | 0 | 33 | 0.11627906976744186 | 0.0 | 0.9342105263157895 | 0.9418604651162791 |
| B4 | LSTM | direct_classifier | unclustered | 100 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 100 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 100 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 200 | 324 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | unclustered | 200 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 200 | 324 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 200 | 324 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | CNN | direct_classifier | unclustered | 20 | 325 | ok | 39 | 4 | 16 | 17 | 0.09302325581395349 | 0.48484848484848486 | 0.7368421052631579 | 0.7110641296687807 |
| B4 | LSTM | direct_classifier | unclustered | 20 | 325 | ok | 40 | 3 | 18 | 15 | 0.06976744186046512 | 0.5454545454545454 | 0.7236842105263158 | 0.6923890063424947 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 20 | 325 | ok | 41 | 2 | 18 | 15 | 0.046511627906976744 | 0.5454545454545454 | 0.7368421052631579 | 0.7040169133192389 |
| B4 | Transformer | direct_classifier | unclustered | 20 | 325 | ok | 41 | 2 | 16 | 17 | 0.046511627906976744 | 0.48484848484848486 | 0.7631578947368421 | 0.7343199436222692 |
| B4 | CNN | direct_classifier | unclustered | 40 | 325 | ok | 39 | 4 | 3 | 30 | 0.09302325581395349 | 0.09090909090909091 | 0.9078947368421053 | 0.9080338266384778 |
| B4 | LSTM | direct_classifier | unclustered | 40 | 325 | ok | 43 | 0 | 3 | 30 | 0.0 | 0.09090909090909091 | 0.9605263157894737 | 0.9545454545454546 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 40 | 325 | ok | 43 | 0 | 4 | 29 | 0.0 | 0.12121212121212122 | 0.9473684210526315 | 0.9393939393939394 |
| B4 | Transformer | direct_classifier | unclustered | 40 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 100 | 325 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | unclustered | 100 | 325 | ok | 43 | 0 | 7 | 26 | 0.0 | 0.21212121212121213 | 0.9078947368421053 | 0.8939393939393939 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 100 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 100 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 200 | 325 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | unclustered | 200 | 325 | ok | 43 | 0 | 16 | 17 | 0.0 | 0.48484848484848486 | 0.7894736842105263 | 0.7575757575757576 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 200 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 200 | 325 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 20 | 326 | ok | 39 | 4 | 18 | 15 | 0.09302325581395349 | 0.5454545454545454 | 0.7105263157894737 | 0.6807610993657505 |
| B4 | LSTM | direct_classifier | unclustered | 20 | 326 | ok | 43 | 0 | 19 | 14 | 0.0 | 0.5757575757575758 | 0.75 | 0.7121212121212122 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 20 | 326 | ok | 39 | 4 | 18 | 15 | 0.09302325581395349 | 0.5454545454545454 | 0.7105263157894737 | 0.6807610993657505 |
| B4 | Transformer | direct_classifier | unclustered | 20 | 326 | ok | 40 | 3 | 11 | 22 | 0.06976744186046512 | 0.3333333333333333 | 0.8157894736842105 | 0.7984496124031008 |
| B4 | CNN | direct_classifier | unclustered | 40 | 326 | ok | 37 | 6 | 3 | 30 | 0.13953488372093023 | 0.09090909090909091 | 0.881578947368421 | 0.8847780126849893 |
| B4 | LSTM | direct_classifier | unclustered | 40 | 326 | ok | 41 | 2 | 1 | 32 | 0.046511627906976744 | 0.030303030303030304 | 0.9605263157894737 | 0.9615926708949964 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 40 | 326 | ok | 43 | 0 | 3 | 30 | 0.0 | 0.09090909090909091 | 0.9605263157894737 | 0.9545454545454546 |
| B4 | Transformer | direct_classifier | unclustered | 40 | 326 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN | direct_classifier | unclustered | 100 | 326 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | unclustered | 100 | 326 | ok | 43 | 0 | 2 | 31 | 0.0 | 0.06060606060606061 | 0.9736842105263158 | 0.9696969696969697 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 100 | 326 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 100 | 326 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 200 | 326 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | unclustered | 200 | 326 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 200 | 326 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 200 | 326 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | CNN | direct_classifier | unclustered | 20 | 327 | ok | 37 | 6 | 17 | 16 | 0.13953488372093023 | 0.5151515151515151 | 0.6973684210526315 | 0.6726568005637773 |
| B4 | LSTM | direct_classifier | unclustered | 20 | 327 | ok | 43 | 0 | 18 | 15 | 0.0 | 0.5454545454545454 | 0.7631578947368421 | 0.7272727272727273 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 20 | 327 | ok | 39 | 4 | 18 | 15 | 0.09302325581395349 | 0.5454545454545454 | 0.7105263157894737 | 0.6807610993657505 |
| B4 | Transformer | direct_classifier | unclustered | 20 | 327 | ok | 40 | 3 | 16 | 17 | 0.06976744186046512 | 0.48484848484848486 | 0.75 | 0.722692036645525 |
| B4 | CNN | direct_classifier | unclustered | 40 | 327 | ok | 40 | 3 | 1 | 32 | 0.06976744186046512 | 0.030303030303030304 | 0.9473684210526315 | 0.9499647639182522 |
| B4 | LSTM | direct_classifier | unclustered | 40 | 327 | ok | 43 | 0 | 6 | 27 | 0.0 | 0.18181818181818182 | 0.9210526315789473 | 0.9090909090909092 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 40 | 327 | ok | 43 | 0 | 4 | 29 | 0.0 | 0.12121212121212122 | 0.9473684210526315 | 0.9393939393939394 |
| B4 | Transformer | direct_classifier | unclustered | 40 | 327 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN | direct_classifier | unclustered | 100 | 327 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | unclustered | 100 | 327 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 100 | 327 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 100 | 327 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 200 | 327 | ok | 37 | 6 | 0 | 33 | 0.13953488372093023 | 0.0 | 0.9210526315789473 | 0.9302325581395349 |
| B4 | LSTM | direct_classifier | unclustered | 200 | 327 | ok | 43 | 0 | 11 | 22 | 0.0 | 0.3333333333333333 | 0.8552631578947368 | 0.8333333333333333 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 200 | 327 | ok | 43 | 0 | 1 | 32 | 0.0 | 0.030303030303030304 | 0.9868421052631579 | 0.9848484848484849 |
| B4 | Transformer | direct_classifier | unclustered | 200 | 327 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 20 | 328 | ok | 37 | 6 | 18 | 15 | 0.13953488372093023 | 0.5454545454545454 | 0.6842105263157895 | 0.6575052854122622 |
| B4 | LSTM | direct_classifier | unclustered | 20 | 328 | ok | 39 | 4 | 18 | 15 | 0.09302325581395349 | 0.5454545454545454 | 0.7105263157894737 | 0.6807610993657505 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 20 | 328 | ok | 37 | 6 | 18 | 15 | 0.13953488372093023 | 0.5454545454545454 | 0.6842105263157895 | 0.6575052854122622 |
| B4 | Transformer | direct_classifier | unclustered | 20 | 328 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | CNN | direct_classifier | unclustered | 40 | 328 | ok | 40 | 3 | 3 | 30 | 0.06976744186046512 | 0.09090909090909091 | 0.9210526315789473 | 0.919661733615222 |
| B4 | LSTM | direct_classifier | unclustered | 40 | 328 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 40 | 328 | ok | 40 | 3 | 0 | 33 | 0.06976744186046512 | 0.0 | 0.9605263157894737 | 0.9651162790697674 |
| B4 | Transformer | direct_classifier | unclustered | 40 | 328 | ok | 41 | 2 | 0 | 33 | 0.046511627906976744 | 0.0 | 0.9736842105263158 | 0.9767441860465116 |
| B4 | CNN | direct_classifier | unclustered | 100 | 328 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | unclustered | 100 | 328 | ok | 43 | 0 | 2 | 31 | 0.0 | 0.06060606060606061 | 0.9736842105263158 | 0.9696969696969697 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 100 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | Transformer | direct_classifier | unclustered | 100 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN | direct_classifier | unclustered | 200 | 328 | ok | 39 | 4 | 0 | 33 | 0.09302325581395349 | 0.0 | 0.9473684210526315 | 0.9534883720930232 |
| B4 | LSTM | direct_classifier | unclustered | 200 | 328 | ok | 43 | 0 | 0 | 33 | 0.0 | 0.0 | 1.0 | 1.0 |
| B4 | CNN+LSTM | direct_classifier | unclustered | 200 | 328 | ok | 43 | 0 | 1 | 32 | 0.0 | 0.030303030303030304 | 0.9868421052631579 | 0.9848484848484849 |
| B4 | Transformer | direct_classifier | unclustered | 200 | 328 | ok | 42 | 1 | 0 | 33 | 0.023255813953488372 | 0.0 | 0.9868421052631579 | 0.9883720930232558 |
| B5 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | 31 | 0 | 33 | 7 | 0.0 | 0.825 | 0.5352112676056338 | 0.5875 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | 31 | 0 | 19 | 21 | 0.0 | 0.475 | 0.7323943661971831 | 0.7625 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | 31 | 0 | 17 | 23 | 0.0 | 0.425 | 0.7605633802816901 | 0.7875 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | 31 | 0 | 15 | 25 | 0.0 | 0.375 | 0.7887323943661971 | 0.8125 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | 31 | 0 | 19 | 21 | 0.0 | 0.475 | 0.7323943661971831 | 0.7625 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | 31 | 0 | 10 | 30 | 0.0 | 0.25 | 0.8591549295774648 | 0.875 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 20 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 20 | 325 | ok | 14 | 17 | 3 | 37 | 0.5483870967741935 | 0.075 | 0.7183098591549296 | 0.6883064516129033 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 40 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 40 | 325 | ok | 14 | 17 | 5 | 35 | 0.5483870967741935 | 0.125 | 0.6901408450704225 | 0.6633064516129032 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 100 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 100 | 325 | ok | 14 | 17 | 0 | 40 | 0.5483870967741935 | 0.0 | 0.7605633802816901 | 0.7258064516129032 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 200 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 200 | 325 | ok | 11 | 20 | 0 | 40 | 0.6451612903225806 | 0.0 | 0.7183098591549296 | 0.6774193548387097 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | clustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | clustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 20 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 40 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 100 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 200 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | DMD | operator_forecast | unclustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | 31 | 0 | 40 | 0 | 0.0 | 1.0 | 0.43661971830985913 | 0.5 |
| B5 | CNN | direct_classifier | clustered | 20 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 20 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 20 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 20 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 40 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 40 | 324 | ok | 31 | 0 | 1 | 39 | 0.0 | 0.025 | 0.9859154929577465 | 0.9875 |
| B5 | CNN+LSTM | direct_classifier | clustered | 40 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 40 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 100 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 100 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 100 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 100 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 200 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 200 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 200 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 200 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 20 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 20 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 20 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 20 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 100 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 100 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 100 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 100 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | clustered | 200 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 200 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 200 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 200 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 20 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 20 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 20 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 20 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | clustered | 200 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 200 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 200 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 200 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 20 | 327 | ok | 31 | 0 | 3 | 37 | 0.0 | 0.075 | 0.9577464788732394 | 0.9625 |
| B5 | LSTM | direct_classifier | clustered | 20 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 20 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 20 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 40 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 40 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 40 | 327 | ok | 31 | 0 | 1 | 39 | 0.0 | 0.025 | 0.9859154929577465 | 0.9875 |
| B5 | Transformer | direct_classifier | clustered | 40 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 100 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 100 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 100 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 100 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | clustered | 200 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 200 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 200 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 200 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 20 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 20 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | clustered | 20 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 20 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 40 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | clustered | 40 | 328 | ok | 31 | 0 | 1 | 39 | 0.0 | 0.025 | 0.9859154929577465 | 0.9875 |
| B5 | CNN+LSTM | direct_classifier | clustered | 40 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 40 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | clustered | 100 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 100 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 100 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | clustered | 100 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | clustered | 200 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | clustered | 200 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | clustered | 200 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | clustered | 200 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 20 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | unclustered | 20 | 324 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 20 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 20 | 324 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN | direct_classifier | unclustered | 40 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | unclustered | 40 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 40 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 40 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 100 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | unclustered | 100 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 100 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 100 | 324 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | unclustered | 200 | 324 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | LSTM | direct_classifier | unclustered | 200 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 200 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 200 | 324 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 20 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | unclustered | 20 | 325 | ok | 31 | 0 | 1 | 39 | 0.0 | 0.025 | 0.9859154929577465 | 0.9875 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 20 | 325 | ok | 28 | 3 | 0 | 40 | 0.0967741935483871 | 0.0 | 0.9577464788732394 | 0.9516129032258065 |
| B5 | Transformer | direct_classifier | unclustered | 20 | 325 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN | direct_classifier | unclustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | unclustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 40 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 100 | 325 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | LSTM | direct_classifier | unclustered | 100 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 100 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 100 | 325 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | unclustered | 200 | 325 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | LSTM | direct_classifier | unclustered | 200 | 325 | ok | 31 | 0 | 2 | 38 | 0.0 | 0.05 | 0.971830985915493 | 0.975 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 200 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 200 | 325 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 20 | 326 | ok | 31 | 0 | 1 | 39 | 0.0 | 0.025 | 0.9859154929577465 | 0.9875 |
| B5 | LSTM | direct_classifier | unclustered | 20 | 326 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 20 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 20 | 326 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN | direct_classifier | unclustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | unclustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 40 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | unclustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 100 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | unclustered | 200 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | unclustered | 200 | 326 | ok | 31 | 0 | 4 | 36 | 0.0 | 0.1 | 0.9436619718309859 | 0.95 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 200 | 326 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 200 | 326 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 20 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | LSTM | direct_classifier | unclustered | 20 | 327 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 20 | 327 | ok | 28 | 3 | 0 | 40 | 0.0967741935483871 | 0.0 | 0.9577464788732394 | 0.9516129032258065 |
| B5 | Transformer | direct_classifier | unclustered | 20 | 327 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN | direct_classifier | unclustered | 40 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | unclustered | 40 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 40 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 40 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 100 | 327 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | LSTM | direct_classifier | unclustered | 100 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 100 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 100 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | unclustered | 200 | 327 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | LSTM | direct_classifier | unclustered | 200 | 327 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 200 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 200 | 327 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 20 | 328 | ok | 28 | 3 | 0 | 40 | 0.0967741935483871 | 0.0 | 0.9577464788732394 | 0.9516129032258065 |
| B5 | LSTM | direct_classifier | unclustered | 20 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 20 | 328 | ok | 27 | 4 | 0 | 40 | 0.12903225806451613 | 0.0 | 0.9436619718309859 | 0.935483870967742 |
| B5 | Transformer | direct_classifier | unclustered | 20 | 328 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | CNN | direct_classifier | unclustered | 40 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | LSTM | direct_classifier | unclustered | 40 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 40 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | Transformer | direct_classifier | unclustered | 40 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN | direct_classifier | unclustered | 100 | 328 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | LSTM | direct_classifier | unclustered | 100 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 100 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 100 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | CNN | direct_classifier | unclustered | 200 | 328 | ok | 29 | 2 | 0 | 40 | 0.06451612903225806 | 0.0 | 0.971830985915493 | 0.967741935483871 |
| B5 | LSTM | direct_classifier | unclustered | 200 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |
| B5 | CNN+LSTM | direct_classifier | unclustered | 200 | 328 | ok | 30 | 1 | 0 | 40 | 0.03225806451612903 | 0.0 | 0.9859154929577465 | 0.9838709677419355 |
| B5 | Transformer | direct_classifier | unclustered | 200 | 328 | ok | 31 | 0 | 0 | 40 | 0.0 | 0.0 | 1.0 | 1.0 |

## 6. Protocol B Neural Window Sweep

This table lists every neural direct-classifier window and input form, aggregated over seeds and fault-line splits.

| method | family | variant | window_len | n_runs | FSR | FUR | Accuracy | BalancedAccuracy |
|---|---|---|---|---|---|---|---|---|
| CNN | direct_classifier | clustered | 100 | 25 | 0.0584 ± 0.0520 | 0.0000 ± 0.0000 | 0.9676 ± 0.0298 | 0.9708 ± 0.0260 |
| CNN | direct_classifier | clustered | 20 | 25 | 0.0715 ± 0.0767 | 0.2635 ± 0.2771 | 0.8472 ± 0.1152 | 0.8325 ± 0.1368 |
| CNN | direct_classifier | clustered | 200 | 25 | 0.0510 ± 0.0506 | 0.0000 ± 0.0000 | 0.9719 ± 0.0290 | 0.9745 ± 0.0253 |
| CNN | direct_classifier | clustered | 40 | 25 | 0.0575 ± 0.0723 | 0.0183 ± 0.0441 | 0.9598 ± 0.0448 | 0.9621 ± 0.0421 |
| CNN | direct_classifier | unclustered | 100 | 25 | 0.0572 ± 0.0452 | 0.0200 ± 0.0980 | 0.9587 ± 0.0499 | 0.9614 ± 0.0484 |
| CNN | direct_classifier | unclustered | 20 | 25 | 0.0756 ± 0.0617 | 0.2952 ± 0.2857 | 0.8319 ± 0.1070 | 0.8146 ± 0.1308 |
| CNN | direct_classifier | unclustered | 200 | 25 | 0.0690 ± 0.0548 | 0.0000 ± 0.0000 | 0.9624 ± 0.0314 | 0.9655 ± 0.0274 |
| CNN | direct_classifier | unclustered | 40 | 25 | 0.0559 ± 0.0634 | 0.0668 ± 0.0917 | 0.9408 ± 0.0418 | 0.9386 ± 0.0454 |
| CNN+LSTM | direct_classifier | clustered | 100 | 25 | 0.0102 ± 0.0138 | 0.0308 ± 0.0697 | 0.9820 ± 0.0272 | 0.9795 ± 0.0333 |
| CNN+LSTM | direct_classifier | clustered | 20 | 25 | 0.0749 ± 0.0711 | 0.2938 ± 0.2911 | 0.8326 ± 0.1145 | 0.8157 ± 0.1388 |
| CNN+LSTM | direct_classifier | clustered | 200 | 25 | 0.0146 ± 0.0190 | 0.0348 ± 0.1191 | 0.9772 ± 0.0582 | 0.9753 ± 0.0652 |
| CNN+LSTM | direct_classifier | clustered | 40 | 25 | 0.0152 ± 0.0218 | 0.0171 ± 0.0306 | 0.9843 ± 0.0146 | 0.9839 ± 0.0159 |
| CNN+LSTM | direct_classifier | unclustered | 100 | 25 | 0.0083 ± 0.0135 | 0.0014 ± 0.0068 | 0.9955 ± 0.0065 | 0.9952 ± 0.0072 |
| CNN+LSTM | direct_classifier | unclustered | 20 | 25 | 0.0846 ± 0.0577 | 0.2992 ± 0.3011 | 0.8271 ± 0.1095 | 0.8081 ± 0.1344 |
| CNN+LSTM | direct_classifier | unclustered | 200 | 25 | 0.0067 ± 0.0176 | 0.0024 ± 0.0082 | 0.9955 ± 0.0099 | 0.9955 ± 0.0093 |
| CNN+LSTM | direct_classifier | unclustered | 40 | 25 | 0.0133 ± 0.0272 | 0.0377 ± 0.0533 | 0.9767 ± 0.0222 | 0.9745 ± 0.0254 |
| LSTM | direct_classifier | clustered | 100 | 25 | 0.0057 ± 0.0116 | 0.0066 ± 0.0231 | 0.9945 ± 0.0103 | 0.9939 ± 0.0122 |
| LSTM | direct_classifier | clustered | 20 | 25 | 0.0530 ± 0.0503 | 0.2724 ± 0.2895 | 0.8547 ± 0.1179 | 0.8373 ± 0.1422 |
| LSTM | direct_classifier | clustered | 200 | 25 | 0.0112 ± 0.0140 | 0.0751 ± 0.1743 | 0.9631 ± 0.0719 | 0.9568 ± 0.0868 |
| LSTM | direct_classifier | clustered | 40 | 25 | 0.0203 ± 0.0302 | 0.0306 ± 0.0462 | 0.9754 ± 0.0209 | 0.9746 ± 0.0231 |
| LSTM | direct_classifier | unclustered | 100 | 25 | 0.0022 ± 0.0076 | 0.0133 ± 0.0438 | 0.9931 ± 0.0190 | 0.9922 ± 0.0219 |
| LSTM | direct_classifier | unclustered | 20 | 25 | 0.0570 ± 0.0518 | 0.2580 ± 0.2672 | 0.8593 ± 0.0988 | 0.8425 ± 0.1207 |
| LSTM | direct_classifier | unclustered | 200 | 25 | 0.0022 ± 0.0076 | 0.0415 ± 0.1132 | 0.9801 ± 0.0492 | 0.9781 ± 0.0563 |
| LSTM | direct_classifier | unclustered | 40 | 25 | 0.0191 ± 0.0289 | 0.0535 ± 0.0790 | 0.9666 ± 0.0291 | 0.9637 ± 0.0358 |
| Transformer | direct_classifier | clustered | 100 | 25 | 0.0098 ± 0.0134 | 0.0012 ± 0.0059 | 0.9944 ± 0.0068 | 0.9945 ± 0.0069 |
| Transformer | direct_classifier | clustered | 20 | 25 | 0.0401 ± 0.0497 | 0.1460 ± 0.1566 | 0.9136 ± 0.0620 | 0.9070 ± 0.0732 |
| Transformer | direct_classifier | clustered | 200 | 25 | 0.0098 ± 0.0196 | 0.0000 ± 0.0000 | 0.9944 ± 0.0112 | 0.9951 ± 0.0098 |
| Transformer | direct_classifier | clustered | 40 | 25 | 0.0219 ± 0.0315 | 0.0000 ± 0.0000 | 0.9876 ± 0.0179 | 0.9890 ± 0.0157 |
| Transformer | direct_classifier | unclustered | 100 | 25 | 0.0111 ± 0.0139 | 0.0000 ± 0.0000 | 0.9944 ± 0.0069 | 0.9944 ± 0.0069 |
| Transformer | direct_classifier | unclustered | 20 | 25 | 0.0657 ± 0.0455 | 0.1461 ± 0.1858 | 0.9013 ± 0.0705 | 0.8941 ± 0.0831 |
| Transformer | direct_classifier | unclustered | 200 | 25 | 0.0148 ± 0.0272 | 0.0000 ± 0.0000 | 0.9916 ± 0.0154 | 0.9926 ± 0.0136 |
| Transformer | direct_classifier | unclustered | 40 | 25 | 0.0168 ± 0.0248 | 0.0014 ± 0.0068 | 0.9899 ± 0.0139 | 0.9909 ± 0.0124 |

## 7. Protocol B Trajectory / Forecast Grid

Rollout-capable methods also produce held-out dense trajectory forecasts. Direct classifiers have no trajectory row.

| split_id | method | family | variant | window_len | seed | status | rollout_r2 | mean_omega_r2 | worst_case_r2 | rmse | failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | -1.132909713900822e+23 | -4.1872532246045566e+21 | -6.309501970074158e+23 | 123.09641683208355 | 0.0 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | -1.7260296299893257e+25 | -8.456320299974825e+23 | -1.1606765241935018e+26 | 946.878715120068 | 0.0 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | -5.883704710690621e+26 | -3.0131935516322632e+25 | -3.753755153896213e+27 | 5997.727194894869 | 0.0 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | -1.4044406753898719e+25 | -6.614706830217725e+23 | -1.0696055989173582e+26 | 839.9698954378997 | 0.0 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | -2.443484045873771e+26 | -1.3279039319099243e+25 | -2.279483923362666e+27 | 3367.4512556333816 | 0.0 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | -4.018132292600318e+25 | -1.9319093846919666e+24 | -2.756347187969122e+26 | 1441.1340465015908 | 0.0 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | -7.959539784732213e+26 | -4.163819121157612e+25 | -4.817403858711779e+27 | 7075.673004719669 | 0.0 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | -1.4222397519087074e+25 | -6.929140480737612e+23 | -7.149018413320654e+25 | 871.1398353475907 | 0.0 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | -3.795986289469813e+27 | -1.8200813957166123e+26 | -1.9532362084665256e+28 | 15976.034764269538 | 0.0 |
| B1 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | -1.8974749760003742e+25 | -8.692522329060325e+23 | -1.55259381307306e+26 | 975.3167751431181 | 0.0 |
| B1 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | -3.3452579641138444e+27 | -1.6569518807835295e+26 | -1.7990454560202347e+28 | 14822.844634491417 | 0.0 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 20 | 324 | ok | -4.802047998196006e+18 | -4.256059646155294e+17 | -5.1131385424985465e+19 | 0.6860168402620597 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | -3.1840149918101155e+22 | -2.8382061772784234e+22 | -2.9871207696922793e+23 | 77.5106352496572 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 40 | 324 | ok | -4.686554304755675e+18 | -4.203483303164225e+17 | -4.974356578218525e+19 | 0.5298344926301439 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | -2.565279616343894e+22 | -1.7973949705695974e+22 | -1.8287877309672454e+23 | 67.49425979908533 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 100 | 324 | ok | -4.34596448093785e+18 | -4.047616577423481e+17 | -4.5663245640262975e+19 | 0.4330771649931041 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | -2.4211812427551046e+22 | -2.165707862167157e+22 | -3.6494658453486886e+23 | 66.28658547548916 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 200 | 324 | ok | -3.798177984826969e+18 | -3.791927544475336e+17 | -3.913829183173612e+19 | 0.38535774223998626 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | -3.179533866546978e+22 | -2.6922248683817007e+22 | -3.425515103362969e+23 | 61.859469702536295 | 0.0 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 20 | 325 | ok | -1.0907149941728051e+18 | -3.190675453329348e+17 | -1.0189446213501716e+19 | 0.5110523424994801 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | -2.539874351431161e+22 | -1.998552136374682e+22 | -2.4811328471514533e+23 | 62.53484036128914 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 40 | 325 | ok | -1.0637467009715896e+18 | -3.1141601421489946e+17 | -9.887147998075111e+18 | 0.38522678572849234 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | -1.7417144385591252e+22 | -1.3663231983963618e+22 | -2.1256866651481715e+23 | 74.23141634940127 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 100 | 325 | ok | -9.839108328825398e+17 | -2.887959093667161e+17 | -9.008455442303411e+18 | 0.269465460517873 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | -1.8086745831013718e+22 | -1.3690678604375116e+22 | -1.692959426660507e+23 | 61.039833752778634 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 200 | 325 | ok | -8.554665040606476e+17 | -2.5235430167887306e+17 | -7.923281619709599e+18 | 0.20874501616183938 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | -9.864086592857624e+21 | -6.010192662669758e+21 | -5.067874021640713e+22 | 52.74774381698583 | 0.0 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 20 | 326 | ok | -7.264670154640265e+17 | -5.823316580808108e+16 | -7.648207900953091e+18 | 0.3713827460508974 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | -1.9352364097559077e+22 | -1.559290813680225e+22 | -1.4269327875063304e+23 | 74.99606210589663 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 40 | 326 | ok | -7.073300060742207e+17 | -5.750177595803037e+16 | -7.51749509220304e+18 | 0.27100832971861105 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | -1.78793644779136e+22 | -1.4944652136875611e+22 | -2.4632044366697577e+23 | 72.49261134786248 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 100 | 326 | ok | -6.514584095373046e+17 | -5.532192301290009e+16 | -7.124028933520572e+18 | 0.20001547116375398 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | -1.712323102400755e+22 | -1.4891597534971866e+22 | -2.793215259857787e+23 | 59.010259045887935 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 200 | 326 | ok | -5.634567668490049e+17 | -5.179535571000189e+16 | -6.464584538329511e+18 | 0.17137218782250518 | 0.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | -1.4103683659729297e+22 | -1.24458762185149e+22 | -1.124835303492383e+23 | 47.68721802178389 | 0.0 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | -3.9130763045627195e+22 | -2.357030427081321e+22 | -2.2028405605475342e+23 | 82.39617522499219 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | -2.1899328174553685e+22 | -1.5285304737228558e+22 | -1.7802632977666913e+23 | 61.006268588617324 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | -3.2675070690138746e+22 | -2.3824023613907323e+22 | -2.2040865019449285e+23 | 65.16710475624507 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | -1.3880801906970279e+22 | -9.005252572246412e+21 | -6.902846119762076e+22 | 39.69583623006891 | 0.0 |
| B1 | SINDYc | sparse_identification | clustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | -2.6388926767916827e+22 | -2.0363224575465996e+22 | -2.5833071996750853e+23 | 62.817550053565824 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | -2.766602170335799e+22 | -2.2601603222545776e+22 | -3.066808270719908e+23 | 61.47239858527323 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | -1.3771644821557498e+22 | -8.650103866534312e+21 | -9.757556516922204e+22 | 50.822805565849826 | 0.0 |
| B1 | DMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | -1.4246255461462369e+22 | -1.0710702798446883e+22 | -1.7926700398847348e+23 | 48.90197995201847 | 0.0 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | -3.656088808383273e+22 | -2.651751718083908e+22 | -2.9755150930173728e+23 | 72.96691726330283 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | -3.645941277955982e+22 | -2.21939182730863e+22 | -3.008734159596187e+23 | 71.79653959798186 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | -3.4620580426856954e+22 | -2.832244915263388e+22 | -3.815366650749999e+23 | 90.27531145413468 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | -2.890484899891479e+22 | -2.2171463377367956e+22 | -2.115402828188345e+23 | 70.44897578017682 | 0.0 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | -4.008309455029561e+22 | -3.373731505353113e+22 | -4.8521413542225784e+23 | 85.23705682287058 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | -3.1117737861008436e+22 | -2.6038475760367896e+22 | -2.215780294732667e+23 | 70.39947201096587 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | -2.6714557989058477e+22 | -2.2818502293657957e+22 | -2.1786432169779075e+23 | 71.98680786108879 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | -1.5122112886043836e+22 | -1.218154705931595e+22 | -1.4005299916264799e+23 | 52.88002537779098 | 0.0 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | -6.187896091326727e+22 | -5.339250103334604e+22 | -4.3752989204189834e+23 | 80.17301203054579 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | -5.668499978317053e+22 | -4.5574966959104015e+22 | -4.601703182303263e+23 | 80.60385875789135 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | -5.058693411667866e+22 | -4.453805236725558e+22 | -2.9749415474150436e+23 | 74.77791940315699 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | -3.3682120932362916e+22 | -2.7723576057563397e+22 | -2.4348177001750116e+23 | 63.09828311390629 | 0.0 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | -5.426794451376979e+22 | -4.39820277940621e+22 | -3.968628426757168e+23 | 83.84159750836812 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | -4.203627036065302e+22 | -3.6447805925109174e+22 | -4.8472688485679334e+23 | 66.89324862405863 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | -4.760987767722177e+22 | -3.845169068556187e+22 | -3.854197278902348e+23 | 67.87770411085236 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | -2.5268323420722673e+22 | -1.90850036605253e+22 | -1.922298609358931e+23 | 54.19116052141494 | 0.0 |
| B1 | SINDYc | sparse_identification | unclustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | DMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | -3.5604709344654788e+22 | -2.6343194188894165e+22 | -3.724978248489357e+23 | 71.10781535209044 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | -3.5934508324917244e+22 | -2.8104497870858197e+22 | -2.181507889530876e+23 | 68.21825908076616 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | -4.393470929111527e+22 | -3.685778654148496e+22 | -3.266794621721278e+23 | 71.42040186466899 | 0.0 |
| B1 | DMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | HODMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B1 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | -1.9421366601883553e+22 | -1.5592273376231352e+22 | -1.4226262490908318e+23 | 53.61760563200858 | 0.0 |
| B2 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | -1.636550635273257e+22 | -7.53696179806271e+20 | -1.670434941120249e+23 | 31.414204722249714 | 0.0 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | -1.337186525245661e+24 | -9.203272083795541e+22 | -1.4274980846348296e+25 | 260.81012211982977 | 0.0 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | -1.2194706095877551e+20 | -2.1661829133533647e+19 | -5.583325209106718e+20 | 52.09245246787866 | 0.28915662650602414 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | -6.462417188480647e+20 | -2.5687081842768433e+19 | -1.20285910227743e+22 | 5.833668568873483 | 0.0 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | -2.8989856904757887e+20 | -3.355689756337276e+19 | -2.8128857359643564e+21 | 4620.142096013595 | 0.0 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | -7.53782761266068e+20 | -2.4189010361822597e+19 | -4.0943549658623304e+21 | 6.0603617298030725 | 0.0 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | -4.628627290745377e+28 | -1.4451515233041075e+28 | -5.424857909479725e+29 | 22369.473127000496 | 0.3373493975903614 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | -1.094142717577675e+21 | -4.545248474408362e+19 | -9.005466530426408e+21 | 7.4106237320694115 | 0.0 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | -9.957477768955542e+26 | -4.677950475756974e+26 | -8.172868285110735e+28 | 880.9500288888938 | 0.0 |
| B2 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | -5.428766369881779e+20 | -4.674807366543649e+19 | -9.695235526892848e+21 | 5.686614981832779 | 0.0 |
| B2 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | -1.4612905913925515e+28 | -5.402798746018748e+27 | -4.0339966614847464e+29 | 11051.433087990745 | 0.32530120481927716 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | -5.8069407943870875e+22 | -4.630971945436513e+22 | -5.678932746659648e+23 | 84.21974778555823 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | -5.341731354973774e+22 | -4.3924344232913056e+22 | -5.418466825709836e+23 | 72.73852109723336 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | -5.443761025448088e+22 | -4.1242457375775245e+22 | -5.7416995025029975e+23 | 74.76624349076168 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | -2.735751692013232e+22 | -1.8810642426916487e+22 | -2.356795941031737e+23 | 57.0595451579912 | 0.0 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | -1.487439499158972e+22 | -9.399276499301553e+21 | -1.1178640707523206e+23 | 61.67426410470408 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | -1.5795117993335613e+22 | -1.2244862588645345e+22 | -1.222020149798869e+23 | 61.41433013850968 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | -1.3471796951478042e+22 | -7.691614998095214e+21 | -1.1441698859074395e+23 | 51.75220367694021 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | -5.885422907478591e+21 | -4.518899319094186e+21 | -3.6488207374519984e+22 | 48.53671059210151 | 0.0 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | -4.488798188432296e+22 | -3.97771659905792e+22 | -4.345126812263859e+23 | 76.06708701932548 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | -4.513555024995294e+22 | -4.001026704437937e+22 | -3.8505625833614245e+23 | 72.08727455433585 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | -3.1333955496450342e+22 | -2.787533621090158e+22 | -1.9428046125702886e+23 | 71.8539312419929 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | -1.9273173667502256e+22 | -1.6891071152262759e+22 | -1.8451494900684726e+23 | 65.6999097718491 | 0.0 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | -1.8016158393739904e+22 | -1.0969780523707052e+22 | -1.1629493820366147e+23 | 55.761415063471176 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | -2.0584128082032544e+22 | -1.8176501130359107e+22 | -1.550702035074632e+23 | 59.82522750422397 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | -2.545378567841186e+22 | -2.2487965941183963e+22 | -1.470972025267175e+23 | 68.0358584308901 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | -1.4493011433833087e+22 | -1.1825515573468006e+22 | -1.3266289474792125e+23 | 40.26053911059097 | 0.0 |
| B2 | SINDYc | sparse_identification | clustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 20 | 328 | ok | -1.029682229269269e+19 | -9.286459595773572e+17 | -1.2236842911484507e+20 | 0.44795327268246043 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | -1.8483378533861845e+22 | -1.4062990947238973e+22 | -1.2492512768411153e+23 | 51.901720246084714 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 40 | 328 | ok | -1.0085618473831184e+19 | -9.233601005754913e+17 | -1.198200640984288e+20 | 0.35206797489208136 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | -1.458171948778615e+22 | -1.103483981005453e+22 | -2.986193655910205e+23 | 48.86908064735177 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 100 | 328 | ok | -9.472751638434005e+18 | -9.081846545588403e+17 | -1.1243269911894709e+20 | 0.30321459041889626 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | -1.8801710441701013e+22 | -1.455807403796058e+22 | -9.157421214516678e+22 | 54.82402547823141 | 0.0 |
| B2 | DMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | clustered | 200 | 328 | ok | -8.513300458032624e+18 | -8.851464147857961e+17 | -1.0089281483866687e+20 | 0.28027884141802284 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | -1.3280701443129513e+22 | -1.1663360994684e+22 | -8.683370747796504e+22 | 51.686490098076504 | 0.0 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | -7.334942736343891e+22 | -4.471071163898272e+22 | -4.4897663740195446e+23 | 82.98521855938688 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | -5.6826785188548404e+22 | -3.668434568413645e+22 | -3.81004480832401e+23 | 75.53012970610847 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | -4.1971201124658e+22 | -2.247159053298797e+22 | -3.5629274196872586e+23 | 66.18506605488686 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | -3.395657854688708e+22 | -2.557122860376373e+22 | -2.6898063035114185e+23 | 57.978740108656424 | 0.0 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | -5.5418570705278306e+22 | -4.44672000777784e+22 | -4.797903360149295e+23 | 69.10680897809326 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | -5.412162538801664e+22 | -4.3245940242551216e+22 | -6.38960282718804e+23 | 68.36430464168889 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | -3.4521059193757165e+22 | -2.5693055461788484e+22 | -3.3733459666918256e+23 | 67.65725666327725 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | -1.8512383955288304e+22 | -1.3410869475130242e+22 | -1.3477593759069787e+23 | 47.54585080772087 | 0.0 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 326 | ok | -8.150598409985238e+26 | -7.948894846763568e+26 | -9.068346495849225e+27 | 8358.657980009622 | 0.8192771084337349 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | -3.584279999337373e+22 | -2.7960025876736044e+22 | -2.1734079995443626e+23 | 78.19112923520828 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 326 | ok | -4.256548564755142e+26 | -4.170004866312042e+26 | -6.310692795003788e+27 | 6906.720529804354 | 0.7590361445783133 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | -3.876172902996e+22 | -2.6524007520018444e+22 | -2.8440409672585998e+23 | 85.60388386722012 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 326 | ok | -6.116629378712846e+25 | -5.946151218513606e+25 | -1.1672554565016173e+27 | 8940.007669346747 | 0.6867469879518072 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | -3.3766904688246174e+22 | -2.6907366206886136e+22 | -1.6267840820129937e+23 | 78.37077892462848 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 326 | ok | -8.593107607748719e+25 | -8.049919079685768e+25 | -5.092585391866915e+27 | 4096.369764886308 | 0.2650602409638554 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | -1.7348584784165134e+22 | -1.2881087150179876e+22 | -1.4236045904170044e+23 | 60.04653331169419 | 0.0 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | -6.013991106676102e+22 | -4.73676696746372e+22 | -3.675084241162485e+23 | 82.40012500585698 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | -6.078184721476594e+22 | -4.43128117196114e+22 | -4.303350155568563e+23 | 80.73423389515366 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | -4.195934978203295e+22 | -2.865958421992059e+22 | -2.7156394797181325e+23 | 67.29387582075167 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | -4.120459768993933e+22 | -3.3520278969234183e+22 | -2.4545432596239338e+23 | 61.26861665118124 | 0.0 |
| B2 | SINDYc | sparse_identification | unclustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | DMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 20 | 328 | ok | -6.201767051468978e+25 | -6.012392886110952e+25 | -4.7039533255608515e+26 | 1835.075244437385 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | -5.457294811254785e+22 | -4.703989963901746e+22 | -3.954048086597837e+23 | 78.61211123098799 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 40 | 328 | ok | -4.866550935786574e+25 | -4.730008248999839e+25 | -4.037115604656526e+26 | 1220.1167545972567 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | -3.303757118709829e+22 | -2.5353593055484693e+22 | -3.1184389632322016e+23 | 62.84350302779955 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 100 | 328 | ok | -2.1071833839128936e+25 | -2.04655901727164e+25 | -1.7644951074950134e+26 | 571.6172117456035 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | -4.319344906809452e+22 | -3.4465502254358875e+22 | -2.1266277091229608e+23 | 60.80500478272999 | 0.0 |
| B2 | DMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | HODMD | operator_forecast | unclustered | 200 | 328 | ok | -5.448702700509933e+24 | -5.296432500711982e+24 | -4.36498602396096e+25 | 257.08168688276896 | 0.0 |
| B2 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B2 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | -3.0373534021493572e+22 | -2.5172168276497037e+22 | -2.1705775490648856e+23 | 54.90503545743749 | 0.0 |
| B3 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | -4.697271206106907e+22 | -1.9626457470597852e+21 | -2.5698127188599575e+23 | 79.83080562588032 | 0.0 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | -5.5317803366823986e+23 | -3.6562231202426902e+22 | -1.6915391369255186e+25 | 47.877141955007566 | 0.0 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | -1.7555480362393453e+21 | -6.754514814135934e+19 | -9.867907485276293e+21 | 12.533496177332298 | 0.0 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | -5.055728164904356e+23 | -3.7767875164550352e+22 | -8.945856211829257e+24 | 82.14400807039125 | 0.0 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | -1.3366609826615152e+27 | -6.772882722625429e+25 | -6.163630442711364e+27 | 10636.459331539354 | 0.0 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | -7.510057092789893e+23 | -5.59452792800699e+22 | -8.771970388973312e+24 | 106.51161518622628 | 0.0 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | -2.6553911372715765e+21 | -7.534113519467102e+19 | -1.4861733803898442e+22 | 15.290339189243882 | 0.0 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | -7.635048479367765e+23 | -5.093757828097062e+22 | -1.4214000026474632e+25 | 60.324156747945445 | 0.0 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | -1.7754493442241523e+21 | -6.95868618356828e+19 | -9.306127108661357e+21 | 13.022206277004804 | 0.0 |
| B3 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | -8.967573681558301e+23 | -6.70228658545929e+22 | -1.1144864926625244e+25 | 119.35068224263844 | 0.0 |
| B3 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | -1.837124781989661e+24 | -6.027114964887847e+22 | -7.865198876618309e+24 | 333.16376505250724 | 0.0 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 20 | 324 | ok | -5.2288604917013955e+19 | -9.525032506504393e+18 | -3.7954746649252875e+20 | 1.2830087882011405 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | -6.406856132630395e+22 | -5.856718760277248e+22 | -4.326320259010261e+23 | 70.81166440164395 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 40 | 324 | ok | -5.168590538467834e+19 | -9.347467783093008e+18 | -3.7593237894515314e+20 | 1.1499380728872217 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | -2.3371325866478387e+22 | -1.7554809097667457e+22 | -1.5448313851435846e+23 | 63.3425866589046 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 100 | 324 | ok | -4.98647608398354e+19 | -8.825114472533225e+18 | -3.650720126332065e+20 | 1.026172376658024 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | -3.4544815259139327e+22 | -2.9492729234558296e+22 | -2.5136507120737355e+23 | 59.31836293807919 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 200 | 324 | ok | -4.678391132153879e+19 | -8.013483835467977e+18 | -3.461159217075472e+20 | 0.8632899199187454 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | -1.451378861064325e+22 | -1.0612502742703213e+22 | -1.4918866969590717e+23 | 51.42681173126734 | 0.0 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | -2.084025975782316e+22 | -1.3650158304366745e+22 | -1.273663586493688e+23 | 46.93291822241439 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | -1.6965099449655163e+22 | -1.1280646759685428e+22 | -1.007680912524814e+23 | 46.70570636132532 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | -9.652501476592736e+21 | -6.20353730074478e+21 | -5.715448102093833e+22 | 43.958020514757834 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | -1.2056240175521607e+22 | -9.457648640084907e+21 | -8.978582643174564e+22 | 36.95276564546372 | 0.0 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | -2.989149762623244e+22 | -2.498144150822377e+22 | -2.0088606360230303e+23 | 61.84147247108436 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | -3.2956969661519834e+22 | -2.6540183098239322e+22 | -2.1149571741484706e+23 | 64.90688125151738 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | -2.5745242230533707e+22 | -2.1084204654202194e+22 | -1.7094942167940614e+23 | 61.469236253289324 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | -2.2861288118205995e+22 | -1.7108068416748199e+22 | -1.0420742115847833e+23 | 60.018462840095694 | 0.0 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | -2.3888527801455043e+22 | -1.9977965781004105e+22 | -1.33550471523191e+23 | 55.92874922273509 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | -1.326809104922571e+22 | -1.0390155114951188e+22 | -1.1752251593138462e+23 | 43.717839460151865 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | -2.1212559704577693e+22 | -1.737647875603918e+22 | -1.1493224379906218e+23 | 54.078718261071785 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 200 | 327 | ok | -1455973187611.7217 | -1107428466836.777 | -1455973187611.7217 | 41089.10231811928 | 0.9878048780487805 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | -1.3316339408985307e+22 | -1.0742804664815425e+22 | -6.2402453093182905e+22 | 40.79711843193408 | 0.0 |
| B3 | SINDYc | sparse_identification | clustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 20 | 328 | ok | -1.7320022213682832e+28 | -1.3966107949907856e+28 | -1.7320022213682832e+28 | 30272.153734566862 | 0.9878048780487805 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | -2.451038220580145e+22 | -1.7236358014104852e+22 | -1.4623474369390391e+23 | 47.12486385020474 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 40 | 328 | ok | -9.341565364130109e+27 | -7.53261644431217e+27 | -9.341565364130109e+27 | 22232.020098921326 | 0.9878048780487805 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | -2.9518731230345525e+22 | -2.129495590044413e+22 | -1.1696369026776224e+23 | 60.37423794609284 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 100 | 328 | ok | -1.4656280804045525e+27 | -1.1818046423418173e+27 | -1.4656280804045525e+27 | 8806.046745314503 | 0.9878048780487805 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | -2.053369569639508e+22 | -1.4300533524674904e+22 | -1.4972459548866366e+23 | 45.342572083958835 | 0.0 |
| B3 | DMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | clustered | 200 | 328 | ok | -4.518626580842704e+28 | -3.643628308806945e+28 | -8.542593920555062e+28 | 43484.218849778146 | 0.8902439024390244 |
| B3 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | -1.8050140832393767e+22 | -1.037101721376743e+22 | -9.321881879864319e+22 | 39.25425035170329 | 0.0 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | -3.7148552245251907e+22 | -2.457825160315985e+22 | -2.053433349601836e+23 | 70.2203124526641 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | -3.402586028601274e+22 | -2.3075158094941184e+22 | -1.5136813900660697e+23 | 57.243322905064716 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | -2.0583636963558693e+22 | -1.3033185852157768e+22 | -1.0462279368846227e+23 | 52.06257414596643 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | -1.9225826432775652e+22 | -1.3643047019316974e+22 | -1.2098245307431916e+23 | 41.05211027934089 | 0.0 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | -3.5551640609924303e+22 | -2.7318676684656857e+22 | -2.1025724693284808e+23 | 60.67920798227679 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | -3.912205905307066e+22 | -3.1363362860708852e+22 | -2.3475222466117405e+23 | 63.62042714429669 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | -2.8031389703692744e+22 | -2.259074768309741e+22 | -1.293724650724282e+23 | 57.877395048905285 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | -3.958645558883626e+22 | -3.162632577561795e+22 | -1.9779100638542537e+23 | 59.62248596158726 | 0.0 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | -4.679144043516194e+22 | -3.859914323861771e+22 | -2.2607663028146693e+23 | 76.11875376934286 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | -4.249356382470489e+22 | -3.4983633925668522e+22 | -2.4681834197716087e+23 | 67.54525702978613 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | -4.52448134400832e+22 | -3.837523148786209e+22 | -2.5162326662488275e+23 | 63.36283654501728 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | -3.0289563295150443e+22 | -2.3350980990263137e+22 | -1.720274776105606e+23 | 63.66558447864187 | 0.0 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 327 | ok | -3.24068927085397e+25 | -9.129552326214311e+24 | -3.212484043552637e+26 | 862.8603949859889 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | -3.870423313981578e+22 | -2.377653875526084e+22 | -1.3929451016917701e+23 | 65.88427078959428 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 327 | ok | -2.614877872922765e+25 | -7.917739889048497e+24 | -2.5770048468049264e+26 | 632.2238248963381 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | -4.3669657387120625e+22 | -3.1138550799261197e+22 | -2.5871703651268393e+23 | 63.603763329178214 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 327 | ok | -1.2899031299450729e+25 | -4.788504861489561e+24 | -1.2425562483990075e+26 | 416.2553003219165 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | -3.5082482451918877e+22 | -2.5046429033865595e+22 | -1.4646889541808317e+23 | 58.18976988735556 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 327 | ok | -3.232102273079244e+24 | -1.6251616040303713e+24 | -2.922673993372805e+25 | 222.63007847430117 | 0.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | -3.241422429544356e+22 | -2.4585309835686394e+22 | -1.2612633771790635e+23 | 62.416863393179995 | 0.0 |
| B3 | SINDYc | sparse_identification | unclustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | DMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | -3.778641168402524e+22 | -2.798873306799753e+22 | -1.7316114238174172e+23 | 72.07098770387245 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | -3.208497450454934e+22 | -2.3295855842507167e+22 | -1.6087541090239064e+23 | 64.90008332352214 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | -2.626520346120413e+22 | -1.955173003522343e+22 | -1.815343371222141e+23 | 54.9841234741212 | 0.0 |
| B3 | DMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | HODMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B3 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | -2.5911522056605517e+22 | -1.93886951027013e+22 | -1.6163591258896452e+23 | 51.72622825748775 | 0.0 |
| B4 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | -6.447300399488195e+22 | -2.54298049597357e+21 | -4.3859221437028546e+23 | 90.0228998050678 | 0.0 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | -2.890276796988178e+24 | -1.0987011874522046e+23 | -2.6629801490164143e+25 | 497.47539125900903 | 0.0 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | -4.437481736202747e+28 | -1.9118370797146716e+27 | -2.2612205203886443e+29 | 77391.64059101699 | 0.0 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | -3.62917819163298e+24 | -1.4186423092205412e+23 | -3.0378910327418716e+25 | 564.9524742001167 | 0.0 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | -5.443341011534463e+28 | -2.3577935169629002e+27 | -2.846193522896449e+29 | 85729.11113316346 | 0.0 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | -2.73619518886278e+22 | -1.260142757787939e+21 | -8.723758765358574e+23 | 16.44468848724505 | 0.0 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | -2.352622541659292e+28 | -9.947317470492204e+26 | -1.3583727251463278e+29 | 54538.36670827264 | 0.0 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | -1.3299718669821235e+23 | -5.329741728575812e+21 | -2.848202478817902e+24 | 41.508996891260146 | 0.0 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | -2.5786089130339493e+28 | -1.090414077953885e+27 | -1.3447001793018576e+29 | 58219.58421393658 | 0.0 |
| B4 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | -2.765906010206214e+24 | -1.1756633457375761e+23 | -3.1364712645108303e+25 | 461.39789790781737 | 0.0 |
| B4 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | -1.7521542812629394e+29 | -7.432165340585114e+27 | -8.840765331705975e+29 | 155056.0095571703 | 0.0 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 20 | 324 | ok | -8.416392498060136e+18 | -1.9623749500015014e+18 | -7.156837763547661e+19 | 0.8219867359933941 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | -7.549904583200154e+21 | -5.134953990353839e+21 | -5.669618929927746e+22 | 46.55614096440358 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 40 | 324 | ok | -8.193606718231352e+18 | -1.9116660183656425e+18 | -6.957240464653336e+19 | 0.6239591795938098 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | -7.727906107476028e+21 | -4.817631807831133e+21 | -5.4160543900349795e+22 | 41.70568206508845 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 100 | 324 | ok | -7.554997573384073e+18 | -1.7679699140390556e+18 | -6.403208262079454e+19 | 0.47292529214590784 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | -8.136682032330689e+21 | -6.272395873258476e+21 | -6.214941892035509e+22 | 37.12316942130083 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 200 | 324 | ok | -6.598227322652066e+18 | -1.5690127817042132e+18 | -5.5677467023660425e+19 | 0.4068807972103464 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | -6.73707130774648e+21 | -5.335346111926847e+21 | -7.396894445112522e+22 | 32.93660731335367 | 0.0 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | -4.4250534107793593e+21 | -3.0004000061573547e+21 | -2.7676341203287415e+22 | 32.304460609849215 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | -3.602276213953228e+21 | -2.395838355888012e+21 | -3.9493220529533664e+22 | 30.623463147557995 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | -4.3076854453267595e+21 | -3.116852494701877e+21 | -4.452980389422305e+22 | 35.1585340960067 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | -4.198422861909526e+21 | -2.2894428285857125e+21 | -4.761188445541855e+22 | 43.42247274299159 | 0.0 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | -9.212032065958326e+21 | -7.892426077197677e+21 | -9.54573239015506e+22 | 44.23887154103634 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | -1.0100740560602737e+22 | -9.116308893545838e+21 | -1.3282384238702454e+23 | 45.99710849190683 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | -4.0794334159491176e+21 | -3.61904781700809e+21 | -5.61007428620709e+22 | 32.673794246349864 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | -7.085041576830231e+21 | -6.304102837088156e+21 | -6.6189218251203106e+22 | 41.50532849576988 | 0.0 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 20 | 327 | ok | -3.7159455727172117e+21 | -3.6553924781532995e+21 | -4.807656517736334e+22 | 46.84710743284967 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | -7.72619893840868e+21 | -6.451009742288772e+21 | -4.2029088094862065e+22 | 34.15790334956573 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 40 | 327 | ok | -2.898235251038663e+21 | -2.8454969905725513e+21 | -3.604853062312359e+22 | 21.745682454867108 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | -9.451084042807808e+21 | -8.124364414457996e+21 | -5.451999573509152e+22 | 37.672060393342335 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 100 | 327 | ok | -1.6099147745131427e+21 | -1.5783087627361554e+21 | -2.073772750516228e+22 | 15.086959814236062 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | -5.425746692494721e+21 | -4.1430955659107867e+21 | -4.464636957009482e+22 | 29.15570671912983 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 200 | 327 | ok | -5.786233822907575e+20 | -5.624025196009475e+20 | -7.312640010079644e+21 | 4.648323552186359 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | -5.947837155935511e+21 | -4.4208214127823615e+21 | -3.3217523436396315e+22 | 29.025938384669487 | 0.0 |
| B4 | SINDYc | sparse_identification | clustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 20 | 328 | ok | -5.103190803935839e+18 | -2.7058806538200607e+18 | -4.7747835417208414e+19 | 0.5889175883095513 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | -6.743290180735386e+21 | -4.997366689385867e+21 | -7.438130475658607e+22 | 36.34362125197528 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 40 | 328 | ok | -4.881388663930483e+18 | -2.5521207694698547e+18 | -4.560609159127343e+19 | 0.4079675145778509 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | -4.928530584565522e+21 | -3.9261011060960075e+21 | -5.6748635180429e+22 | 34.612650708019814 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 100 | 328 | ok | -4.263573002187949e+18 | -2.134754521643106e+18 | -3.964483824377563e+19 | 0.3373617950137214 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | -2.810218684356188e+21 | -2.3283152474396717e+21 | -3.6283078413127816e+22 | 25.979407812336188 | 0.0 |
| B4 | DMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | clustered | 200 | 328 | ok | -3.3769953595480294e+18 | -1.5686989258361444e+18 | -3.110394860301242e+19 | 0.2932924795261274 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | -3.952584457281292e+21 | -3.227769871205488e+21 | -4.157486869288087e+22 | 25.531663570394603 | 0.0 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | -2.0128326534999765e+22 | -1.311278077472381e+22 | -1.3998660540348854e+23 | 47.481993695584364 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | -2.192122932440958e+22 | -1.4379322294710458e+22 | -1.3634082633408533e+23 | 47.88078985462648 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | -1.91151673708152e+22 | -1.2644505311886279e+22 | -1.1366197558529865e+23 | 44.341967234145734 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | -1.0904463311573812e+22 | -6.082773762708407e+21 | -8.750371143342667e+22 | 42.91673737809977 | 0.0 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | -1.1337568535087854e+22 | -8.633572434333005e+21 | -1.3094671036680966e+23 | 42.42833254938663 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | -9.967002087347039e+21 | -5.789166558135378e+21 | -1.3807532848276211e+23 | 38.66970177253354 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | -1.0268954529351867e+22 | -7.382831040024015e+21 | -8.227536367450458e+22 | 47.224204141288574 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | -1.2156350053882733e+22 | -5.931095112531542e+21 | -1.3830023855929648e+23 | 41.00619316559551 | 0.0 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | -1.8359240008212402e+22 | -1.2905433337780635e+22 | -1.2058883495999963e+23 | 54.000515159744786 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | -1.666554315856793e+22 | -9.483203056647367e+21 | -1.2403553531520839e+23 | 51.16477212130986 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | -1.2244289891700597e+22 | -9.606122999951366e+21 | -7.62977489571559e+22 | 53.58820423928357 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | -2.4434854609680154e+22 | -1.1386300517900993e+22 | -2.327945436264016e+23 | 49.78550302360851 | 0.0 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 327 | ok | -1.5861403677498173e+20 | -1.304431470819446e+20 | -2.0446069718172767e+21 | 6.586492236791442 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | -1.6612386920775568e+22 | -8.152912147300956e+21 | -1.2666740718858547e+23 | 53.02565374120928 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 327 | ok | -1.3999302619476597e+20 | -1.130334530431705e+20 | -1.8530184804114398e+21 | 3.6939007638302455 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | -1.5757649382798638e+22 | -7.120298475344516e+21 | -1.1842843315722743e+23 | 47.81314216724441 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 327 | ok | -1.1834432223108725e+20 | -9.463896134463747e+19 | -1.5896494420849963e+21 | 2.4273200454592887 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | -8.213057031946858e+21 | -5.547172102567783e+21 | -8.383747188189458e+22 | 42.891707760758145 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 327 | ok | -8.686183801076289e+19 | -6.77764380607135e+19 | -1.126521321724547e+21 | 1.422960197449694 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | -1.0197829906632354e+22 | -6.370164243501151e+21 | -7.008404229760876e+22 | 42.807374723452654 | 0.0 |
| B4 | SINDYc | sparse_identification | unclustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | DMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 20 | 328 | ok | -1.0271884414243085e+20 | -1.4434003860003367e+19 | -1.0993394822589276e+21 | 1.2257206954423783 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | -1.4725146026140255e+22 | -1.1332003695919651e+22 | -9.190541768824454e+22 | 46.96149837182859 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 40 | 328 | ok | -9.825515149672882e+19 | -1.3538393097983257e+19 | -1.0510275158928869e+21 | 0.9234292633712412 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | -1.7232554641598106e+22 | -1.2515649629097581e+22 | -1.520183594319e+23 | 47.56796567107141 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 100 | 328 | ok | -8.599419793248931e+19 | -1.1433659310051623e+19 | -9.167656520591477e+20 | 0.787479773150235 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | -2.824476272489741e+22 | -1.3445467670346392e+22 | -3.104006055737719e+23 | 53.68922941810782 | 0.0 |
| B4 | DMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | HODMD | operator_forecast | unclustered | 200 | 328 | ok | -6.868907662283907e+19 | -8.688349428080446e+18 | -7.32043635626611e+20 | 0.6953857211448793 | 0.0 |
| B4 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B4 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | -2.8388227793892177e+22 | -2.2404848942337524e+22 | -3.130197520000977e+23 | 43.24947063353194 | 0.0 |
| B5 | M0 | physics_equation_rollout | clustered | 1 | 0 | ok | -1.0162711860017087e+23 | -3.773824962593799e+21 | -5.5379929278282995e+23 | 95.83951390841055 | 0.0 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 324 | ok | -2.032791794589205e+23 | -1.560521430779979e+22 | -4.617194875529064e+24 | 56.79307865234069 | 0.0 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 324 | ok | -4.551967722224065e+25 | -4.528923719316248e+24 | -6.0690901146401726e+26 | 1427.007358822948 | 0.0 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 325 | ok | -4.490306041949563e+24 | -2.400591953643529e+23 | -1.0350593675505683e+26 | 320.0000974768219 | 0.0 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 325 | ok | -8.318009378338776e+26 | -4.168414252777545e+25 | -4.869815184848542e+27 | 6870.275926073904 | 0.0 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 326 | ok | -1.1477041977439742e+23 | -1.0931467187525482e+22 | -2.483355279293813e+24 | 69.67756325166235 | 0.0 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 326 | ok | -1.5289144038456652e+25 | -1.9321210759040259e+24 | -2.4918037817797603e+26 | 972.5589646643322 | 0.0 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 327 | ok | -2.4245750091418445e+23 | -1.729474074286017e+22 | -5.270092084353395e+24 | 57.09097254283273 | 0.0 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 327 | ok | -3.547029686124431e+26 | -2.3027673973810506e+25 | -2.48535557465427e+27 | 4758.584622176747 | 0.0 |
| B5 | M2A | physics_equation_rollout | clustered | 1 | 328 | ok | -1.6752789543390452e+23 | -9.916818521268544e+21 | -2.816111301602842e+24 | 60.18936735123982 | 0.0 |
| B5 | M2A-LYAP | physics_equation_rollout | clustered | 1 | 328 | ok | -2.386882656350828e+25 | -1.9682954327712258e+24 | -3.782882131870059e+26 | 1016.6135695321038 | 0.0 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 324 | ok | -3.1060289917585837e+22 | -2.5994834721855093e+22 | -2.8059541364402973e+23 | 68.7602921915964 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 324 | ok | -2.686243677692455e+22 | -1.9804333313824374e+22 | -2.031842746852586e+23 | 86.35883224535772 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 324 | ok | -2.4974614196569055e+22 | -1.9467458158622195e+22 | -2.0866559383093248e+23 | 65.95248310526357 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 324 | ok | -1.4374738178973534e+22 | -1.138304800426087e+22 | -9.895705996814818e+22 | 61.417592740856946 | 0.0 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 20 | 325 | ok | -1.5634234224541665e+19 | -8.485481598077354e+17 | -2.09875109942586e+20 | 0.7332918107453669 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 325 | ok | -1.7018538650931827e+22 | -1.2394791308910018e+22 | -1.1987009784767668e+23 | 49.067083038003126 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 40 | 325 | ok | -1.5214378651766624e+19 | -8.122111889960237e+17 | -2.0435605795396213e+20 | 0.7541248971068022 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 325 | ok | -1.317416289633815e+22 | -1.0541159043400599e+22 | -7.334352658783358e+22 | 47.65580992915558 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 100 | 325 | ok | -1.3997786865201312e+19 | -7.282098557945027e+17 | -1.8815948130214753e+20 | 0.5755126905653569 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 325 | ok | -1.1584536032161667e+22 | -9.341455912152574e+21 | -7.439586015684266e+22 | 46.16537019577619 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 200 | 325 | ok | -1.2020472394363806e+19 | -5.994209129489687e+17 | -1.6189056299137876e+20 | 0.41551696350414746 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 325 | ok | -1.1345331312009903e+22 | -9.118861096882945e+21 | -7.620047682629739e+22 | 51.52788600570245 | 0.0 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 20 | 326 | ok | -1.7292616482529554e+23 | -1.7057924647567652e+23 | -2.486388538673786e+24 | 280.4274660955301 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 326 | ok | -3.6356341762327763e+22 | -2.61625622605707e+22 | -1.786872351673076e+23 | 74.1341309494614 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 40 | 326 | ok | -1.304261578528927e+23 | -1.2864944492534861e+23 | -1.7748792613610837e+24 | 224.3588492737642 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 326 | ok | -3.828656670323194e+22 | -2.7552726249916603e+22 | -1.730339617768076e+23 | 81.21876514936481 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 100 | 326 | ok | -5.590140562529007e+22 | -5.51331470787109e+22 | -8.254547038246192e+23 | 128.36286622334978 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 326 | ok | -2.995405605648826e+22 | -2.026724121275335e+22 | -1.789033016356461e+23 | 63.78398855450311 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 200 | 326 | ok | -1.3704407250562347e+22 | -1.3511077480480624e+22 | -1.9552611319746077e+23 | 24.898810030763798 | 0.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 326 | ok | -2.306590652655878e+22 | -1.8180941186135507e+22 | -1.677173598706677e+23 | 57.54242557898993 | 0.0 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 327 | ok | -1.9302844103073942e+22 | -1.5251031143716045e+22 | -1.2965461440870992e+23 | 58.33481962013577 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 327 | ok | -1.3643050658455718e+22 | -1.0635062809661392e+22 | -9.306398033119829e+22 | 49.3779878539159 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 327 | ok | -1.4861149990732188e+22 | -1.130815288855083e+22 | -9.021050136675441e+22 | 49.73950615668224 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 327 | ok | -1.1821964651632477e+22 | -8.738926617265512e+21 | -6.024824813940483e+22 | 36.774781921058846 | 0.0 |
| B5 | SINDYc | sparse_identification | clustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 20 | 328 | ok | -1.9504888519040803e+22 | -1.3829376537823768e+22 | -1.848550041425539e+23 | 49.8010195929337 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 40 | 328 | ok | -1.3419029378671437e+22 | -8.981393693389778e+21 | -7.876910431503483e+22 | 46.4396195272906 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 100 | 328 | ok | -1.452807266837004e+22 | -1.0540267155865991e+22 | -1.3202316637954387e+23 | 46.065738996951964 | 0.0 |
| B5 | DMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | clustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | clustered | 200 | 328 | ok | -1.0475222487220026e+22 | -7.691654073306062e+21 | -6.92627978775947e+22 | 42.173383903661524 | 0.0 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 324 | ok | -3.1129152771661475e+22 | -2.227028945341897e+22 | -1.988860333546513e+23 | 66.70789860978965 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 324 | ok | -3.286443713122921e+22 | -2.3228271834770337e+22 | -2.4090667728375e+23 | 75.68263395280164 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 324 | ok | -2.236056833437687e+22 | -1.5812630606948516e+22 | -1.2440732919548252e+23 | 50.946911941686814 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 324 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 324 | ok | -1.867484028567611e+22 | -1.22821422377113e+22 | -1.3466982191470491e+23 | 58.42186240451976 | 0.0 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 325 | ok | -271289944502.71588 | -288300871119.43726 | -356625718420.29236 | 40803.743994652395 | 0.9529411764705882 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 325 | ok | -2.1614514060223712e+22 | -1.4959685890212652e+22 | -1.3949821524329557e+23 | 56.44203079458008 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 325 | ok | -169615705827.26987 | -181511636706.40594 | -169615705827.26987 | 29425.564518011368 | 0.9882352941176471 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 325 | ok | -2.537434022236331e+22 | -1.8557230882862332e+22 | -1.440277734123983e+23 | 58.05749183205419 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 325 | ok | -15421096845.329504 | -16118158823.292719 | -160271635360.94037 | 22846.25343735943 | 0.5294117647058824 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 325 | ok | -3.4775409497608836e+22 | -2.7303849703403726e+22 | -2.0893781647771076e+23 | 74.4192713982482 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 325 | ok | -4.4834166863562965e+27 | -4.3461340719116216e+27 | -6.97860052011619e+28 | 9097.041340607575 | 0.2588235294117647 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 325 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 325 | ok | -2.074513060313299e+22 | -1.358609041582742e+22 | -1.1658111456034118e+23 | 50.2319225853539 | 0.0 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 326 | ok | -4.9059667541758755e+22 | -3.753001443823955e+22 | -4.070427819470141e+23 | 68.60037358515878 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 326 | ok | -4.9666489237707e+22 | -3.5345435547737105e+22 | -3.174785113443847e+23 | 71.30875226478396 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 326 | ok | -5.036508581860622e+22 | -3.912790349934824e+22 | -3.476122229100028e+23 | 75.5222534102936 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 326 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 326 | ok | -3.8474569042116825e+22 | -3.1702171947842367e+22 | -2.7866467043441328e+23 | 63.25173312642611 | 0.0 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 327 | ok | -2.4256497029271127e+22 | -1.904296693682528e+22 | -1.7574345725338457e+23 | 59.3658498598249 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 327 | ok | -2.2390780608476265e+22 | -1.7620698004385794e+22 | -1.8076654802348725e+23 | 65.39681084018423 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 327 | ok | -1.9332380703312056e+22 | -1.5052523146998392e+22 | -1.6920286121764283e+23 | 51.33819755870747 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 327 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 327 | ok | -1.4760075452742304e+22 | -1.1676318120289476e+22 | -1.0939673258979329e+23 | 48.00468888364769 | 0.0 |
| B5 | SINDYc | sparse_identification | unclustered | 1 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | DMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 20 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 20 | 328 | ok | -3.7037165301008475e+22 | -2.819767770210461e+22 | -2.4931849773102086e+23 | 68.96537825098791 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 40 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 40 | 328 | ok | -2.6659925778268326e+22 | -1.8749913939825809e+22 | -1.7774185934412232e+23 | 57.956833211396436 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 100 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 100 | 328 | ok | -2.5125921555856247e+22 | -1.6472912166381725e+22 | -1.3800062292600767e+23 | 58.71217330858474 | 0.0 |
| B5 | DMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | HODMD | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | EDMD-Koopman | operator_forecast | unclustered | 200 | 328 | failed | nan | nan | nan | nan | 1.0 |
| B5 | LSTM-rollout | blackbox_rollout | unclustered | 200 | 328 | ok | -2.2687091284376044e+22 | -1.6275083159848624e+22 | -1.1730742276903629e+23 | 55.73168811586489 | 0.0 |

## 8. Adversarial Audit

| severity | check | message |
|---|---|---|
| OK | all | No audit findings. |

## 9. Paper-Ready Conclusions And Limits

- Protocol A is the clean blind external generalization claim: no dense/v1 boundary cases are used in training.
- Protocol B is an adaptation and early-warning claim: all methods see the same dense train/val fault lines and are tested only on held-out fault lines.
- M0/M2A/M2A-LYAP are 7-area clustered physics methods; comparison methods are reported in both clustered and unclustered forms.
- The direct neural classifiers are not a fair Protocol A comparison because the highorder training set has no unstable labels; they are therefore Protocol B methods.

## Output Files

- `protocol_a_full_grid_results.csv`
- `protocol_a_stability_summary.csv`
- `protocol_a_trajectory_summary.csv`
- `protocol_b_full_grid_results.csv`
- `protocol_b_stability_summary.csv`
- `protocol_b_window_sweep_summary.csv`
- `per_case_predictions.csv`
- `method_capability_table.csv`
- `audit_findings.json` and `artifact_hashes.csv`
