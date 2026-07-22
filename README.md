# kron-reduce

Clean, paper-oriented implementation of reduced IEEE39 post-clear dynamics.
The project rebuilds the defensible scientific core of historical work 3.10
and 3.11 on the audited `ieee39_postclear_canonical_v1` dataset. Historical
scores are provenance only: all paper models must be trained from scratch on
the canonical line-disjoint splits.

## Scientific scope

The primary task is closed-loop prediction of `delta(t), omega(t)` from the
exact clearing state. The default post-clear mechanical-power policy is a
zero-order hold at `Pm(tc)`. A run with the observed synchronous `Pm(t)` is a
separately reported conditional-rollout sensitivity, not a causal forecast.
Stability scores are derived from predicted trajectories and are exploratory.

Main methods are M0, M2A, energy-structured M2A, SINDYc, a shared
physics-informed neural ODE, and an autoregressive LSTM. Operator forecasts and
direct prefix classifiers are supplementary because they use a different
information budget.

## Commands

```powershell
C:/ProgramData/miniconda3/python.exe -m pytest -q
C:/ProgramData/miniconda3/python.exe -m project_harness list
C:/ProgramData/miniconda3/python.exe -m project_harness audit
C:/ProgramData/miniconda3/python.exe -m project_harness data-check
C:/ProgramData/miniconda3/python.exe -m project_harness git-check
C:/ProgramData/miniconda3/python.exe -m project_harness run paper_smoke --stage smoke
C:/ProgramData/miniconda3/python.exe -m project_harness run paper_support_audit --stage support_audit
C:/ProgramData/miniconda3/python.exe -m project_harness loop status
```

Full paper runs remain locked until the source baseline is committed, the data
support audit passes, and `full_run_approved` is explicitly approved.
After approval, the registered `paper_v1_full` runner freezes every fold/seed
selection record before opening test trajectories. Summarization, automated
adversarial checks, LaTeX table generation, and manuscript assembly are separate
one-stage commands in the project manifest; publication remains a human gate.
