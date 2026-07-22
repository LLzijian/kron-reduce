# 3.10 Final Paper Comparison

This directory contains the final Protocol A/B comparison harness for the
Kron / Graph-SINDy paper experiment.

Entry point:

```powershell
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_310_final_paper_comparison/run_final_310_comparison.py --quick-smoke --resume
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_310_final_paper_comparison/run_final_310_comparison.py --run-all --resume
C:/ProgramData/miniconda3/python.exe discovery/3.2-kron/hist_310_final_paper_comparison/run_final_310_comparison.py --report-only --resume
```

Output directory:

```text
results/03_kron_graph_sindy/02_cluster/3.10-final_paper_comparison_protocol_ab/
```

Quick-smoke output directory:

```text
results/03_kron_graph_sindy/02_cluster/3.10-final_paper_comparison_protocol_ab_quick_smoke/
```

Report:

```text
reports/hist_3.10-Final_Paper_Comparison.md
```

Quick-smoke report:

```text
reports/hist_3.10-Final_Paper_Comparison_quick_smoke.md
```

Code layout:

```text
run_final_310_comparison.py       # CLI only
support/
  common.py                       # paths, metrics, CSV/JSON, hash helpers
  data_protocol.py                # Protocol A/B datasets and fault-line splits
  audit.py                        # adversarial audit checks
  reporting.py                    # report and capability tables
  pipeline.py                     # orchestration
our_methods/
  m0.py                           # M0 refit and rollout
  m2a.py                          # M2A training wrapper
  m2a_lyap.py                     # M2A-LYAP training wrapper
  evaluation.py                   # paper-method per-case evaluation
comparison_methods/
  sindyc.py                       # controlled sparse polynomial baseline
  operator_methods.py             # DMD/HODMD/EDMD controlled operators
  neural_forecast.py              # LSTM rollout with exogenous Pm(t)
  neural_classifiers.py           # Protocol B direct classifiers
  evaluation.py                   # comparison-method grid evaluation
```

Protocols:

- Protocol A: highorder stable train/val/test for training and internal
  trajectory testing, untouched v1 external set for blind stability testing.
- Protocol B: dense boundary set with fixed fault-line group splits B1-B5.

All controlled rollout methods use observed `P_m(t)` as the exogenous input.
M0, M2A, and M2A-LYAP are fixed to the 7-area clustered representation.
Comparison methods are evaluated in both clustered and unclustered forms.
Rollout-derived stability rows report `rollout_failure_rate`; rows with one or
more rollout exceptions are marked `degraded` so numerical divergence is visible
instead of silently appearing as an ordinary successful stability prediction.
`--report-only` regenerates tables/reports from existing CSVs without rewriting
the original `run_config.json`.
