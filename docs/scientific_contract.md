# Scientific contract

Effective 2026-10-23: new real-data experiments use the registered kiki sources
and retain ten generator nodes; spatial clustering is paused. Synthetic truth
checks and explicitly requested historical replay retain their separate roles.
Existing source test has prior exposure and cannot provide fresh confirmation.
The active rules are machine-checked by `configs/research_policy.json`.

The explicitly approved `kiki_a04_development_20261023` exception permits one
registered pre-certification development evaluation inside original source
train, following its frozen internal train/val, causal inputs, candidates and
budget. The empirical input support audit is fitted on train only, is labelled
uncertified, and does not filter the full comparison or risk denominator.
Source val/test, deployment predictions and model promotion remain closed.
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
- Test IDs may not enter fitting, normalization, checkpoint selection,
  calibration, or gate construction.
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
  replace that solve only after a predeclared validation improvement gate.
