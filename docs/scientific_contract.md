# Scientific contract

- Prediction begins at exact clearing (`t=0`).
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
- Predictions are future `delta, omega`; stability is derived post hoc from the
  predicted trajectory.
- Test IDs may not enter fitting, normalization, checkpoint selection,
  calibration, or gate construction.
- Prefix tasks use a fixed number of real observations. Short cases and cases
  whose event occurred in the observation window are ineligible for the
  prospective cohort.
- Report mean, median, pooled metrics, failure rate, and fault-line clustered
  uncertainty. Never turn failed predictions or empty populations into zeros.
- M0 and static M2A are single weak-form constrained solves. Multi-horizon
  rollout optimisation is an optional support-preserving candidate and may
  replace that solve only after a predeclared validation improvement gate.
