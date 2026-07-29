# Round 5: graph-potential M0 protocol

## Scientific question

Round 5 tests whether a physically constrained graph potential and explicit
relative-speed dissipation improve closed-loop reintegration beyond the current
M0.  Parameter estimates are diagnostics of whether the fitted vector field is
credible; the acceptance endpoints remain reintegration accuracy, numerical
failure rate, and stability prediction from a fixed causal rollout.

This round uses seed 324 only.  The accuracy experiment uses observed
mechanical power with linear interpolation at Runge--Kutta stages and is
therefore a conditional, noncausal upper bound.  No governor model is fitted in
this round.  Test trajectories remain inaccessible until the validation gate
passes.

## Fair population and comparators

- Fit and validation selection include only trajectories with at least 100
  recorded samples. Shorter trajectories are listed as exclusions and are not
  used for normalization, fitting, or selection.
- All three methods use the canonical exact-clearing state, the same observed
  `Pm(t)`, the same COI projections, and the same RK4 implementation with a
  maximum internal step of 0.01 s.
- `M0` is refitted rather than loaded from an earlier run.
- `M0-DD` reproduces direct numerical-derivative regression under the current
  contract. It is a diagnostic comparator, not a preferred estimator.
- `M0-GRAPH` is the Round 5 candidate.

The maximum fitting prefix is 101 samples so a 100-interval integral window is
available when the trajectory supports it.  Windows of 1, 5, 20, 40, and 100
intervals are sampled deterministically without overlap.  Each
case-by-window-length stratum contributes unit total least-squares weight, so
neither long records nor the numerous one-step windows dominate the solve.

## Candidate equation

For incidence matrix `B`, edge angle `eta=B.T delta`, and equilibrium edge
angle `eta_eq`, the candidate is

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s[\sin\eta-\sin\eta_{eq}]
+K_c[\cos\eta-\cos\eta_{eq}]+D_eB^T\omega\}.
\]

`D_n`, `D_e`, and `K_s` are nonnegative. `K_c` may be signed, while the local
edge curvature is constrained by

\[
K_s\cos\eta_{eq}-K_c\sin\eta_{eq}\ge 0.
\]

The model is equilibrium anchored, invariant to a common angle shift, and its
internal graph force sums to zero. Parameters are found by one constrained
weak-form ridge solve for each train-only ridge value. Pruning removes an
entire edge triplet and is selected on validation data only. Learned edge
weights are effective reduced parameters and are not claimed to be the true
Kron network.

## Parameter and modal audit

Every fold records node and edge damping, sine and cosine coupling, local edge
curvature, active-edge count, near-zero damping counts, design rank and
condition number, quotient damping and stiffness eigenvalues, and all
linearized modal real and imaginary parts.  A candidate is ineligible if its
constrained coefficients are nonfinite, its local curvature is negative, or a
linearized quotient mode has a positive real part above numerical tolerance.

These diagnostics do not override trajectory evidence: plausible-looking
coefficients cannot rescue a model with worse closed-loop reintegration.

## Validation and test gates

The frozen validation selection uses the standardized frequency and edge-angle
trajectory loss.  The Round 5 accuracy gate additionally requires, at H=100:

1. zero rollout failures and COI drift below the existing numerical contract;
2. pooled frequency R2 above 0.8;
3. at least 20% lower mean RMSE than refitted M0;
4. improvement in at least four of five folds;
5. no degradation in worst-10% CVaR RMSE;
6. a fault-line cluster-bootstrap 95% upper confidence bound below zero;
7. `M0-GRAPH` selected in at least four folds; and
8. all fitted `M0-GRAPH` checkpoints passing their physical audit.

Only if all validation conditions pass may the full stage open test data.  The
final stability experiment then rolls the already frozen model for 5 s using
only the clearing state and `Pm_clear` held constant, independent of recorded
valid length. Stability labels are used only for final evaluation. The round is
accepted only if the accuracy gates still pass on test, classification has zero
numerical failures, and instability FSR is below 0.2.
