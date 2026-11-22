# M0-AFSD v2 module-selection protocol

This workflow is independent of `m0_afsd_v1`: it does not read, resume, or
overwrite a v1 result directory.  Its scope is F1, seed 324, validation-only,
with observed `Pm(t)` as a conditional rollout input and test locked.

Each scientific round is ordered as hypothesis registration, controlled
experiment, residual diagnosis, adversarial review, and acceptance/rejection.
The registered first round is deliberately comprehensive.  If its governance
is valid but the data do not separate the modules, the admissible conclusion is
`INSUFFICIENT_INFORMATION`, not physical absence.  A rejected implementation
must nominate the pre-registered next experiment with maximum expected
information gain before it may be rerun.

## Registered module question

Starting from a train-discovered seven-node port-structured model, the closure
is divided into three physical module families:

1. conservative edge-potential corrections;
2. non-negative instantaneous edge dissipation;
3. stable passive edge memory.

All eight subsets of these families are refit on exactly the support envelope
selected by the full model.  Spatial representation, backbone, feature columns,
time constants, ridge value, coefficient scale, train/validation split, and
rollout integrator remain fixed.  Thus performance differences are module
knockouts, not changes in feature opportunity.

The minimum effective subset is the least complex subset within 2% of the best
10-second validation RMSE.  A module may be declared necessary only when its
removal worsens paired per-event RMSE by at least 2%, the 95% fault-line cluster
bootstrap interval is strictly above zero, and the module appears in at least
80% of train fault-line cluster support refits.  Specific edges are reported as
stable only when their support rate is at least 80%.

Cross-layer evidence evaluates C3-trained models on C1/C2/C3 validation
cohorts, and independently fitted C1/C2 controls on the richer cohorts.  All
predictions are reconstructed to the original ten-machine coordinates before
metrics are calculated.  Residual autocorrelation, residual stable rank,
spectrum, modal metrics, COI drift, failure rate, and RK4 step convergence are
audited separately.

Acceptance is permitted only for either a supported minimum-effective-module
conclusion or a fully evidenced `INSUFFICIENT_INFORMATION` conclusion.  Test
access, additional folds, canonical-data mutation, destructive operations, and
the final merge gate require explicit operator approval.
