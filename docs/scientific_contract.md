# Scientific contract

- Prediction begins at exact clearing (`t=0`).
- Primary rollout receives `delta(0)`, `omega(0)`, equilibrium and declared
  graph/inertia priors, with `Pm(0)` held constant after clearing.
- Observed `Pm(t)` is a conditional-rollout sensitivity and is never mixed with
  the causal-ZOH main table.
- Predictions are future `delta, omega`; stability is derived post hoc from the
  predicted trajectory.
- Test IDs may not enter fitting, normalization, checkpoint selection,
  calibration, or gate construction.
- Prefix tasks use a fixed number of real observations. Short cases and cases
  whose event occurred in the observation window are ineligible for the
  prospective cohort.
- Report mean, median, pooled metrics, failure rate, and fault-line clustered
  uncertainty. Never turn failed predictions or empty populations into zeros.
