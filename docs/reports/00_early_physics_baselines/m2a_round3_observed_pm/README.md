# M2A round 3A: conditional observed-Pm protocol

[返回主题页](../README.md)

## 运行结果与归档

- [归档说明](../../../../results/backup/00_early_physics_baselines/C_m2a_comparison.DESCRIPTION.md)

## 执行协议

<a id="doc-d5d8e846ccc8"></a>

<a id="doc-d5d8e846ccc8-m2a-round-3a-conditional-observed-pm-protocol"></a>

### M2A round 3A: conditional observed-Pm protocol


<a id="doc-d5d8e846ccc8-question-and-claim-boundary"></a>

#### Question and claim boundary

Round 3A asks whether the frozen M0 plus a static graph-SINDy residual can
reach the requested reintegration accuracy when the complete measured
mechanical-power schedule is supplied.  This is a conditional upper bound, not
a causal forecast or deployable real-time model.  No ZOH result is generated in
this round.  A governor ODE is out of scope until the validation gate below
passes.


<a id="doc-d5d8e846ccc8-registered-model-and-selection"></a>

#### Registered model and selection

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


<a id="doc-d5d8e846ccc8-validation-gate-before-any-test-access"></a>

#### Validation gate before any test access

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


<a id="doc-d5d8e846ccc8-reporting"></a>

#### Reporting

All tables are labelled `conditional_observed_pm`.  Only events with enough
real samples for a requested horizon enter that horizon; exclusions are
reported rather than padded.  No stability FSR claim is made in round 3A,
because a complete future Pm schedule is not available for a fixed prospective
clock on truncated events.

<details>
<summary>原件与版本</summary>

- [docs/m2a_observed_pm_round3_protocol.md](_sealed/m2a_observed_pm_round3_protocol.md)

</details>
