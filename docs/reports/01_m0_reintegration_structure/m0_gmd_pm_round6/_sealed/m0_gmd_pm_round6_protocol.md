# M0-GMD-PM Round 6 protocol

## Purpose

Round 6 asks whether the post-clear mechanical-power port can be discovered
from train-only observed `Pm(t)` and then reintegrated causally from `Pm_clear`.
The experiment is limited to fold F1 and seed 324.  Test trajectories remain
unread until a later confirmatory run is explicitly approved.

## Discovered model

The model first fits the frozen `M0-GMD` electrical backbone.  Mechanical power
is represented in a deterministic basis `U_p` of the zero-total-power
subspace:

```text
a = U_p^T (Pm - Pm_equilibrium)
v = U_p^T Pi_power(omega)
da/dt = A a + B v
```

`A` and `B` are obtained by one equal-case, equal-horizon integral weak solve.
The discovery library contains the full cross-node linear state and relative
speed response, not a machine-by-machine governor template.  Ridge strength,
relative support threshold and Hurwitz margin are selected on validation
rollouts.  Deployment initializes `a(0)` exactly from `Pm_clear` and does not
consume future observed `Pm(t)`.

## Fixed physical structure and learned structure

Fixed structure consists of power balance, causal clearing initialization,
relative-speed input, the registered graph/electrical backbone, and a Hurwitz
state matrix.  Learned structure consists of the cross-node matrices `A` and
`B`, their sparse support, modal poles and effective time constants.  The
canonical data remove common frequency, so no machine-level droop or governor
parameter recovery claim is permitted.

## Two independent gates

The accuracy gate requires at least 2% lower F1 validation H=100 mean RMSE than
`M0-GMD-ZOH`, zero failures, COI/power-balance drift below `1e-10`, RK4 step
convergence, a Hurwitz state matrix and maximum effective time constant no
larger than 2.5 s.

The stability gate is frozen after the accuracy selection.  It uses a fixed
5 s causal rollout and must not tune on labels.  Failure of this gate does not
invalidate the one-second accuracy result; it prevents `M0-GMD-PM` from
replacing the stability model.

## Interpretation gates

- Learned seven-node damping is reported as effective reduced damping, not as
  the ten-machine `GENROU.D` parameter.
- Agreement of discovered poles with registered GENROU/TGOV1N time bands is
  spectral compatibility, not parameter identification.
- A rank-six residual space is the full seven-node zero-total-force quotient;
  it is not evidence for six individually identifiable physical modes.
- Individual modal directions require cross-fold subspace stability.  Only
  rotation-invariant subspaces, poles, supports and closed-loop predictions
  are used for scientific claims.

## Frozen F1 decision

The 2.5 s-bounded candidate improves H=100 validation mean RMSE from 0.188415
to 0.164101 (12.90%), has zero rollout failures and passes the integration and
balance audits.  It is accepted for causal reintegration accuracy.  The frozen
stability audit has one false stable prediction (FSR 1/9), so the candidate is
not accepted as the stability branch.

