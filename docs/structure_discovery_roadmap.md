# Structure discovery beyond a hand-written swing equation

## Current boundary

`M0-3EG` and `M0-GMD` are grey-box models.  Their conservation laws,
coordinates and broad state equations are supplied by the researcher; the
algorithm selects coefficients, feature-group support, memory order and time
scales.  `M0-GMD-PM` goes one step further by learning the full balanced
cross-node state and speed-response matrices of the mechanical-power port.
None of these models discovers an unrestricted symbolic equation from raw
measurements.

This boundary is deliberate.  With only post-clear COI-relative angle,
relative speed and balanced mechanical power, the full machine model is not
identifiable: common frequency, excitation states, algebraic voltages and
machine-level governor states have been removed.  A method may discover an
equivalent reduced realization, but it must not attach unique physical names
to non-identifiable coordinates or coefficients.

## Proposed hierarchy

### Level 1: discover invariant coordinates and dynamical order

Construct block-Hankel matrices from train-only trajectories after removing the
known COI symmetry.  Select order by singular-value stability across folds and
sources, not by a single POD spectrum.  Compare subspaces by principal angles
and compare latent dynamics by poles; do not compare arbitrary basis columns.

### Level 2: discover graph support

Start from a declared candidate supergraph rather than fixing the final edge
set.  Each candidate edge owns one coefficient group containing conservative,
dissipative and memory terms.  Solve a group-sparse weak regression under:

- zero total internal force;
- common-angle-shift invariance;
- symmetric nonnegative conservative coupling;
- positive-semidefinite dissipation;
- shared support across folds and data sources.

An edge is retained only when its group is selected consistently and removing
it worsens source-balanced validation reintegration.

### Level 3: discover the constitutive law

For each retained edge, use an overcomplete but symmetry-compatible library:

```text
sin(eta)-sin(eta_eq)
cos(eta_eq) (eta-eta_eq)
sin(2 eta)-sin(2 eta_eq)
(B^T omega)
abs(B^T omega) (B^T omega)
(B^T omega) [sin(eta)-sin(eta_eq)]
memory states driven by the same edge variables
```

Use integral/weak equations and rotation-invariant group sparsity.  Candidate
terms are selected once from train/validation data; repeated epochs are not
part of static SINDy.  Optional rollout refinement may change coefficient
values but not the selected support.

### Level 4: discover energy, dissipation and ports

Represent the reduced dynamics as

```text
x_dot = [J(x)-R(x)] grad H(x) + G(x) u
```

and discover coefficients of `H`, skew `J`, positive-semidefinite `R`, and the
input map `G` under exact structural parameterizations.  This replaces a
manually enumerated residual with a discovered constitutive model while
preserving the supply inequality by construction.  Use `M2A-LYAP` only after
the numerical certificate suite passes.

### Level 5: test identifiability before interpretation

For every claimed physical parameter or term, require:

- persistent excitation of its regressor on train data;
- acceptable condition number and profile-likelihood width;
- stable selection frequency across folds and sources;
- recovery in synthetic systems with known ground truth;
- improvement on held-out closed-loop reintegration;
- invariance to sample order, window weights and basis rotation.

Terms that predict well but fail these checks remain effective reduced
coefficients.  They are not reported as recovered machine parameters.

## Recommended next implementation

The next bounded experiment should be `M0-SD1`: a candidate-supergraph,
group-sparse, port-Hamiltonian weak discoverer with order candidates 2--6.  It
should keep seed 324, use all five train/validation folds, and freeze the
selected support before any stability-label read.  The primary comparison is
against `M0-GMD-PM` at H=100; the decisive evidence is cross-fold support and
pole stability plus closed-loop error, not the number of selected terms.

