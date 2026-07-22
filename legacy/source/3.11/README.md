# 3.11 Protocol B M2A / LYAP Accuracy

This task is an isolated follow-up to 3.10.  It targets Protocol B accuracy for
the paper methods, especially M2A and M2A-LYAP, without modifying the 3.10
experiment harness.

Entry point:

```powershell
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --quick-smoke --resume
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --run-all --resume
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --run-all --resume --skip-cls
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --run-all --resume --skip-passh
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --run-all --passh-only --reuse-310-models
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --run-all --passh-robust-only --reuse-310-models
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --report-only
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_311_protocol_b_m2a_acc/run_311_protocol_b_m2a_acc.py --report-only --quick-report
```

Output directory:

```text
results/03_kron_graph_sindy/02_cluster/3.11-protocol_b_m2a_acc/
```

Quick-smoke output directory:

```text
results/03_kron_graph_sindy/02_cluster/3.11-protocol_b_m2a_acc_quick_smoke/
```

Report:

```text
reports/hist_3.11-Protocol_B_M2A_LYAP_Accuracy.md
```

Protocol:

- Reuses the fixed 3.10 Protocol B dense fault-line group splits.
- Uses the same binary metric convention as 3.10: only `stable` and `unstable`
  enter Accuracy, while `near_critical` is counted as non-binary.
- Fits M0 anchors with `all`, `stable_near`, and `stable_only` train cases.
- Evaluates calibrated physics rollouts:
  - `M2A-BCal`
  - `M2A-LYAP-BCal`
  - `M2A-PASSH`
  - `M2A-LYAP-PASSH`
  - `M2A-PASSH-R`
  - `M2A-LYAP-PASSH-R`
  - `M2A-CLS`
  - `M2A-LYAP-CLS`

The BCal variants keep the learned residual parameters and select only a
low-dimensional rollout threshold grid on the validation fault line.  The CLS
variants add a short classification-loss fine-tuning stage before the same
validation-only threshold calibration.  The PASSH variants add a label-free
passivity projection in the small-signal stability region and choose the
finite prediction horizon plus residual gain on the validation fault line.  The
PASSH-R variants keep the same rollout model but select horizon, residual gain,
and rollout thresholds from a fixed grid using train+val labels only, with
fault-line mean Accuracy as the primary objective, fault-line mean Balanced
Accuracy as the secondary objective, and long-horizon tie-breaking for late
instability detection.  The
`--reuse-310-models` mode evaluates this new rollout structure using the
already trained 3.10 Protocol B parameter files and records that source in
`run_config.json`.
Generator-written
result summaries such as `case_results.csv` maximum angle/frequency fields are
forbidden as model inputs because they leak post-hoc simulation outcomes.  Full
runs execute CLS and PASSH by default; use `--skip-cls` or `--skip-passh` for
ablations.  Quick smoke runs skip CLS unless `--run-cls` is passed.

Main outputs:

```text
protocol_b_full_grid_results.csv
protocol_b_stability_summary.csv
per_case_predictions.csv
calibrator_grid_results.csv
m0_anchor_ablation.csv
baseline_310_protocol_b_summary.csv
audit_findings_table.csv
invalidated_leakage_rows.csv
artifact_hashes.csv
run_config.json
```
