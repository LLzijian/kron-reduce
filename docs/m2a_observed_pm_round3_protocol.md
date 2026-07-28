# M2A round 3A: conditional observed-Pm protocol

## Question and claim boundary

Round 3A asks whether the frozen M0 plus a static graph-SINDy residual can
reach the requested reintegration accuracy when the complete measured
mechanical-power schedule is supplied.  This is a conditional upper bound, not
a causal forecast or deployable real-time model.  No ZOH result is generated in
this round.  A governor ODE is out of scope until the validation gate below
passes.

## Registered model and selection

- One seed only: 324, across all five registered fault-line folds.
- Frozen, hash-verified round-2 M0; it is never refitted.
- M2A is fitted once from integral weak-form windows of 1, 5 and 20 samples.
- Observed Pm is used both in the weak force balance and in closed-loop
  validation.  RK4 linearly interpolates Pm at every internal stage.
- Candidate selection uses no stability label.  The exact M0 residual is an
  eligible fallback.
- Identifiable linear residual terms are projected onto dissipative signs:
  node omega coefficients are non-positive and edge-speed coefficients are
  non-negative.
- A candidate must improve mean standardized validation loss by at least 2%,
  have zero rollout failures and not worsen the worst-10% validation CVaR.

## Validation gate before any test access

At H=100, the five-fold validation evidence must satisfy all of:

1. FINAL pooled omega R2 is strictly greater than 0.9.
2. FINAL mean combined trajectory RMSE improves on frozen M0 by at least 2%.
3. FINAL rollout failure rate is zero.
4. The fault-line clustered bootstrap 95% CI upper endpoint for
   FINAL-minus-M0 paired RMSE is below zero.

The validation-only stage writes an immutable selection marker and records an
empty test-ID read set.  If any gate fails, test access remains locked and the
Pm ODE is not attempted.  If every gate passes, a separate explicit approval
is still required before the full test stage.

## Reporting

All tables are labelled `conditional_observed_pm`.  Only events with enough
real samples for a requested horizon enter that horizon; exclusions are
reported rather than padded.  No stability FSR claim is made in round 3A,
because a complete future Pm schedule is not available for a fixed prospective
clock on truncated events.
