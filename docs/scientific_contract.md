# Scientific contract

Effective 2026-10-23: new real-data experiments use the registered kiki sources
and retain ten generator nodes; spatial clustering is paused. Synthetic truth
checks and explicitly requested historical replay retain their separate roles.
Effective 2026-11-12, new KIKI comparisons use the frozen TASK3-compatible
cohorts: `train=346` for fitting, `val=107` for predeclared selection, and
`test=76` for final evaluation. The test list is TASK3 Campaign 2's 43 expansion
events plus 33 extra holdout events. It is not the official source `test` list
screened from 105 allocated / 99 length-qualified events; those counts remain
provenance facts only. Do not create another internal train/val split. The
[benchmark standard](protocols/kiki_official_train_test_v1_zh.md) defines the
source-qualified lists, 100 ms input and TASK3 scoring. The fixed test list has
prior exposure and supports comparative reporting; independent-confirmation
claims require a separately registered population. The user has authorized this
prospective change; exact run budgets and synthetic prerequisites still apply.
The active rules are machine-checked by `configs/research_policy.json`.

The historical `kiki_a04_development_20261023` exception permitted one
registered pre-certification development evaluation inside original source
train, following its frozen internal train/val, causal inputs, candidates and
budget. The empirical input support audit is fitted on train only, is labelled
uncertified, and does not filter the full comparison or risk denominator.
That consumed authorization does not open new reads, deployment or promotion.
See `docs/protocols/kiki_a04_development_authorization_20261023_zh.md`.

- Prediction begins at exact clearing (`t=0`).
- A history-initialized task declares a different forecast origin and real
  observation count in `configs/input_contracts.json`; it is never pooled with
  clearing-state prediction or a different observation budget as one ranking.
- Primary rollout receives `delta(0)`, `omega(0)`, `Pm_equilibrium`,
  `Pm_clear`, equilibrium and declared graph/inertia priors. It evolves with
  `Pm_clear` held constant after clearing and uses
  `Delta Pm = Pm - Pm_equilibrium` in the swing equation. `Pm0` is a
  deprecated compatibility alias for `Pm_clear`, never for equilibrium power.
- Observed `Pm(t)` is a conditional-rollout sensitivity and is never mixed with
  the causal-ZOH main table.
- A fitted governor is a causal ablation. It is trained on train-only observed
  `Pm(t)` and deployed from `Pm_clear`, equilibrium and predicted speed; it
  never reads the future observed schedule.
- A discovered reduced `Pm` ODE follows the same causality rule. Its learned
  modal poles and cross-node response are reduced closed-loop structure, not
  machine-level governor parameters. Accuracy and long-horizon stability are
  gated separately.
- Predictions are future `delta, omega`; trajectory-derived stability remains
  post hoc and exploratory. A separately gated energy-structured model may
  additionally report an integration-free clearing-state margin under fixed
  post-clear input and topology, without changing the trajectory labels or
  training route.
- `test` IDs may not enter fitting, normalization, checkpoint selection,
  calibration, or gate construction.
- In each newly authorized round, freeze the model and evaluation contract,
  then evaluate the complete frozen `train=346`, `val=107`, and `test=76`
  rosters from the first assessment. Do not use two-case or other subset pilots
  unless the user specifically asks for one. Produce complete trajectory and
  stability results for all three splits regardless of whether RMSE passes its
  promotion gate. `test` results are reporting-only and never feed tuning or
  model revision. A quick assessment without added noise still covers all three
  complete rosters; it does not replace the full frozen evaluation. Missing,
  nonfinite, or otherwise invalid results
  are reported as explicit failures with coverage, never as zero or fabricated
  metrics.
- For real TASK3 stability, retain the frozen official truth labels and counts;
  derive predicted labels from the frozen LOS rule and report their agreement
  with the official labels without relabeling them. For synthetic known-truth
  data, derive both labels from the truth and predicted trajectories using the
  frozen thresholds and time window. Treat instability as positive and report
  accuracy, TP/TN/FP/FN, unstable recall, missed-instability rate, and stable
  false-alarm rate separately for `train`, `val`, and `test`. An unresolved
  prediction is not stable: report unresolved/uncovered cases and numerical
  failures separately. If any case is unresolved, uncovered, or numerically
  failed, full-cohort accuracy is `null`; also report accuracy among
  classifiable cases and coverage. Any zero metric denominator is `null`.
  `test` stability results do not tune thresholds or revise the model.
- One supplied trajectory corpus produces one final equation and one parameter
  set. Fit/selection excitation views may support different candidate modules,
  but their union is fitted jointly and pruned as whole blocks. Source,
  excitation regime, fault identity, event time, stability label, `Pe`,
  voltage, masks, valid length, and future summaries are never model features
  or routing keys.
- Prefix tasks use a fixed number of real observations. Short cases and cases
  whose event occurred in the observation window are ineligible for the
  prospective cohort.
- Report mean, median, pooled metrics, failure rate, and fault-line clustered
  uncertainty. Never turn failed predictions or empty populations into zeros.
- M0 and static M2A are single weak-form constrained solves. Multi-horizon
  rollout optimisation is an optional support-preserving candidate and may
  replace that solve only after a predeclared val improvement gate.
