# M0-AFSD v3 joint-selection validity protocol

## Scope

This workflow audits whether F1/seed324 contains enough source-balanced,
fault-line-independent Q-A evidence to support joint selection of spatial
resolution, effective dynamic order, closure modules and equivalent graph
support.  It is independent of the v1 and v2 result directories.  Existing
model classes may be reused as software components; prior response metrics,
checkpoints and selections may not be read.

Observed `Pm(t)` is the registered conditional-rollout input.  Test, F2--F5
and canonical-data mutation remain locked.

## Mandatory order

Every round is executed in this order:

1. hypothesis registration;
2. eligibility and source--excitation overlap audit;
3. controlled experiment;
4. residual diagnosis;
5. adversarial review;
6. acceptance or rejection.

Structure, order, partition and support selection may use only Q-A F1-train
trajectories.  Q-B trajectories may be used only after support is frozen for
parameter robustness and rollout extrapolation.  Q-C trajectories are
excluded.  Fault line is the independent grouping and primary weighting unit.

## Fail-closed precondition

For every amplitude--richness cell represented in the Q-A design, each source
must provide at least five independent fault-line clusters, and at least one
cell must be directly comparable.  A source-exclusive excitation cell counts
as zero support for the absent source.  When this condition is not met, the
controlled model lattice is not fitted and the admissible terminal conclusion
is `SOURCE_EXCITATION_CONFOUNDED`.  This is an accepted negative scientific
result, not evidence that a dynamic module is absent.

## Candidate lattice after eligibility PASS

Spatial candidates are train-discovered 5--9-node partitions, a fixed
seven-node control and the raw ten-machine control.  B2 at H=100 screens the
spatial family; candidates within 5% of the best plus both controls proceed to
H=100/500/1000.

Dynamic candidates are B2, the preregistered full M0-3EG alpha/tau grid, BS,
single-time-scale B4, multi-rank/multi-time-scale BM and all eight conservative,
dissipative and memory same-support PSD combinations.  Complexity is the
lexicographic tuple of total state count, active parameters, active edges and
spatial coordinates.  A module count is not a complexity surrogate.

## Statistical contract

The outer five fault-line folds estimate branch performance.  Inner four-fold
splits select partitions, support and hyperparameters.  The primary estimand is
the equal-fault-line mean; event-weighted results are secondary.  Paired
inference uses 5,000 fault-line bootstrap draws, exact sign flips up to 15
clusters and 10,000 random sign flips above that threshold.  Module knockouts
use Holm correction.

The original F1 validation set is confirmation-only after all rules are
frozen.  It cannot revise a threshold, support or branch.

## Claim boundary

Even after a positive outcome, the selected graph is a dynamic equivalent,
not the IEEE39 physical topology.  Learned parameters are effective coordinate-
conditioned parameters.  A parameter-recovery claim additionally requires full
active rank, condition number below `1e8`, 95% sign stability and no more than
20% normalized drift across outer folds.  Failure returns
`PARAMETER_NOT_IDENTIFIABLE` and retains the declared-prior rollout.
