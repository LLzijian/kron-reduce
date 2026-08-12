# M0-AFSD protocol

## Scientific question

M0-AFSD asks which minimum effective temporal order, spatial resolution and
equivalent interaction structure are supported by the dynamics naturally
excited in the existing IEEE39 post-clear trajectories. It does not generate
systems of different physical orders and does not infer that an unselected
module is absent from the original system.

The first registered experiment is restricted to F1, seed 324, validation-only
selection and conditional observed-`Pm(t)` reintegration. Test IDs remain
unopened. Voltage, `Pe`, `Qe`, labels, event time and fault metadata are not
model features or stratification variables.

## Train-only data hierarchy

Every real trajectory is audited independently at H=100, 500 and 1000. The
label-free audit uses mass-weighted kinetic response, edge-angle response,
large-signal sine nonlinearity, block-Hankel stable rank, spectrum entropy and
repeatable oscillatory peaks. Train-only terciles define amplitude and
activation levels. A trajectory receives:

- `Q-A` when its numerical, COI, excitation-rank and Hankel-condition checks
  support structure selection;
- `Q-B` when it supports coefficient fitting but not an order claim;
- `Q-C` when it is ineligible.

The cumulative corpora C1/C2/C3 use frozen train activation thresholds. Data
quality is never a supervised branch label. Insufficient information produces
an explicit refusal rather than a low-order decision.

## Nested branch tree

The spatial hierarchy contains train-discovered 5--9-node average-linkage
partitions, the historical fixed seven-node partition and raw ten-machine
coordinates. Pair costs balance dynamic coherency and accelerating-power
mismatch across data sources. Increasing the number of clusters retains an
additional internal spatial degree of freedom, while raw ten-machine dynamics
is the identity endpoint.

The temporal hierarchy is:

- `B2`: nonnegative integral weak-form swing model;
- `B3`: M0-3EG latent electrical-coupling state;
- `BS`: static graph-modal closure;
- `B4`: one selected stable memory family around B3;
- `BM-r`: multiple selected stable memory families;
- `PSD-BS/PSD-B4`: edge-group-sparse port-structured closure on the C3
  shortlist.

Weak windows are deterministically and uniformly sampled to at most 8 starts
per trajectory and window width. Every trajectory/width block has equal total
least-squares weight; dense overlapping windows therefore cannot make a long
trajectory dominate merely by sample count.

All 5--10 node views first receive the inexpensive B2 solve. Only spatial views
within the predeclared validation tolerance, plus the fixed-seven and raw-ten
controls, proceed to B3. The best B3 spatial view in each corpus proceeds to
GMD, and C3 additionally proceeds to port-structured edge discovery. This is
successive scientific screening, not arbitrary epoch training.

## Selection and claims

Validation reports one-, five- and ten-second trajectory errors, pooled SSE,
worst-decile CVaR, spectral overlap, dominant-frequency error, DMD damping and
MAC where identifiable, finite-time stability and COI drift. Candidates with
failed rollouts or failed physics audits are inadmissible. Among candidates
within 2% of the best mean RMSE at the longest adequately populated horizon,
the lowest-complexity branch is selected.

Observed `Pm(t)` makes these conditional reintegration results, not autonomous
forecasts. Canonical inertia remains a declared coordinate prior because the
stored trajectories are already expressed in its COI quotient. Learned
damping, coupling and latent time scales are effective reduced quantities.
Equivalent selected edges are not physical 39-bus transmission lines.
