# Round 5 method and results: graph-structured M0

## One-sentence argument

On the canonical IEEE39 post-clear dataset, a locally stable graph-potential
model with explicitly learned node and edge damping did not provide reliable
closed-loop reintegration beyond a refitted M0 under observed mechanical-power
forcing; Round 5 was therefore rejected before test access and stability
classification.

## Terminology ledger

| Canonical term | Definition |
|---|---|
| M0 | Equilibrium-centred swing model fitted by an integral weak form |
| M0-DD | Direct numerical-derivative comparator under the current data and RK4 contract |
| M0-GRAPH | Round 5 graph-potential model with node damping, edge damping, and sine/cosine edge forces |
| observed Pm | Recorded post-clear mechanical-power schedule, used as a conditional noncausal input |
| H=100 | The first 100 recorded samples after exact clearing, approximately 1 s |
| FINAL | Fold-specific model frozen by train/validation selection before any test access |

## Method

### Study design and evidence boundary

Round 5 used seed 324 and all five fault-line folds of the registered
`combined_line_5fold_v1` split. Fitting and validation selection excluded every
trajectory with fewer than 100 samples before data adaptation. This removed 31,
35, 32, 33, and 38 training events and 9, 5, 8, 5, and 8 validation events in
F1--F5, respectively. The H=100 validation population contained 177 trajectories
from 33 fault-line clusters. Test data, test labels, and stability outcomes were
not read.

The accuracy experiment supplied the recorded `Pm(t)` schedule to all methods
and linearly interpolated it at Runge--Kutta stages. It is therefore a
conditional upper bound, not a deployable causal forecast. All candidates used
the exact clearing state, the same COI projection, and the same fourth-order
Runge--Kutta integrator with an internal maximum step of 0.01 s.

SINDy identifies coefficients by sparse regression over a prescribed function
library rather than by many neural-network epochs [Brunton et al.](https://arxiv.org/abs/1509.03580).
External inputs can enter that library explicitly, as in SINDYc
[Brunton et al.](https://arxiv.org/abs/1605.06682). The present M0 fit therefore
used one train-only linear solve per ridge value. Its weak integral form avoided
placing numerical derivatives in the target, consistent with the noise-reduction
motivation of weak SINDy [Messenger and Bortz](https://epubs.siam.org/doi/abs/10.1137/20M1343166?journalCode=mmsubt).

### Comparators

The refitted M0 obeyed

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s[\sin(B^T\delta)-\sin(B^T\delta_{eq})]\}.
\]

M0-DD used the same equation, current long-trajectory cohort, observed `Pm(t)`,
and RK4 rollout, but estimated acceleration with a second-order numerical
gradient and solved a bounded ridge problem. It isolated the effect of the old
direct-derivative fitting style without reintroducing old clearing offsets,
integration schemes, or selected plotting populations.

### Graph-potential candidate

M0-GRAPH extended the reduced swing equation to

\[
m\dot\omega=\Delta P_m-D_n\omega-
B\{K_s\Delta\sin\eta+K_c\Delta\cos\eta+D_eB^T\omega\},
\qquad \eta=B^T\delta.
\]

Here `Dn`, `De`, and `Ks` were nonnegative, `Kc` was signed, and each edge was
constrained to have nonnegative local curvature,

\[
\kappa_e=K_{s,e}\cos\eta_{eq,e}-K_{c,e}\sin\eta_{eq,e}\ge0.
\]

This structure anchors the equilibrium, preserves invariance to a common angle
shift, and balances internal force through the incidence matrix. Learned edge
weights are effective reduced coefficients; they are not interpreted as the
true Kron network.

The weak system used deterministic non-overlapping windows of 1, 5, 20, 40,
and 100 intervals within at most 101 samples. Each case-by-window-length stratum
received unit total weight. Four train-only ridge values and four validation-only
edge-pruning thresholds were evaluated. Physical inequality constraints were
encoded as a two-ray convex cone per edge. This replaced an initial SLSQP
implementation that could return a feasible-looking but unconverged solution.
The corrected implementation rejects every candidate for which the optimizer
does not report success. Constrained sparse identification follows the broader
principle that known physics and inequalities should be encoded in the
regression problem rather than checked only after fitting
[Kaptanoglu et al.](https://arxiv.org/abs/2111.08481).

### Prespecified validation gate

The validation gate required zero rollout failures, pooled frequency R2 above
0.8, at least 20% lower H=100 mean RMSE than refitted M0, improvement in at
least four folds, no increase in worst-10% CVaR RMSE, a fault-line cluster
bootstrap 95% upper confidence bound below zero, M0-GRAPH selection in at
least four folds, and passing graph-physics audits. Only a complete pass could
unlock test trajectories and the fixed-causal 5 s stability rollout.

## Results

### Closed-loop reintegration

| Method | H=100 mean RMSE | Median RMSE | Pooled RMSE | Worst-10% CVaR | Pooled frequency R2 | Failures |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.2846 | 0.1242 | 0.4326 | 0.9839 | 0.4366 | 0% |
| M0-DD | 0.2726 | 0.1195 | 0.4112 | 0.9214 | 0.4925 | 0% |
| M0-GRAPH | 0.5135 | 0.1432 | 1.1872 | 3.1893 | -3.2248 | 0% |
| FINAL | 0.2683 | 0.1408 | 0.3996 | 0.9028 | 0.5211 | 0% |

The fold-specific selector chose M0-DD in F1, F3, and F5, and M0-GRAPH in F2
and F4. FINAL improved mean RMSE over M0 by only 5.72%, below the required 20%.
Fold improvements were +4.16%, -1.64%, +11.64%, +7.89%, and +4.07% for
F1--F5. Although four folds moved in the same direction and the clustered
paired effect was negative (difference FINAL minus M0, -0.0163; 95% bootstrap
CI -0.0286 to -0.00186), pooled frequency R2 remained 0.521, and the graph
candidate was selected in only two folds. The validation gate failed.

![Round 5 validation summary](../../../results/figures/m0-graph-round5-solver-fix/fig_round5_validation_summary.png)

### Learned damping, stiffness, and modes

All five corrected graph optimizations converged in 430--857 iterations and
passed equilibrium, force-balance, gauge-invariance, local-curvature, and
linearized-mode audits. Nevertheless, the parameter estimates were close to a
boundary solution. The number of near-zero node damping coefficients was 7,
6, 6, 4, and 6 out of 7; near-zero edge damping counts were 15, 18, 15, 14,
and 15 out of 21. Minimum quotient damping eigenvalues were approximately 0,
0, 0.031, 0.111, and 0 in F1--F5. Maximum modal real parts were all negative,
from -0.0037 to -0.1024, while minimum quotient stiffness eigenvalues remained
positive, from 9.68 to 10.08.

The normalized design condition number remained high in every fold
(7.67e3--9.28e3). More importantly, the weighted weak residual did not predict
closed-loop validation quality: F5 had the smallest weak residual (2.59) but a
mean validation RMSE of 0.741, whereas F2 had a larger weak residual (4.09) but
a mean validation RMSE of 0.149. Thus, locally admissible coefficients and a
small integrated equation residual were insufficient evidence for accurate
rollout.

![Round 5 parameter audit](../../../results/figures/m0-graph-round5-solver-fix/fig_round5_parameter_audit.png)

### Trajectory-level evidence

The trajectory figure used the median H=100 difference between M0-GRAPH and
M0 within F1 and F2; it did not select the best-looking event. Even in F2,
where the graph model won the fold-level standardized selection loss, the
median event showed almost no RMSE advantage. The discrepancy was dominated by
phase and amplitude evolution after the initial transient rather than by an
incorrect clearing state.

![Round 5 trajectory examples](../../../results/figures/m0-graph-round5-solver-fix/fig_round5_trajectory_examples.png)

## Discussion

Round 5 separates physical admissibility from predictive adequacy. The graph
model was equilibrium anchored and locally stable, but these constraints only
control behavior near the operating point. They do not guarantee that a
70-parameter complete-graph representation is identifiable from 1 s of
post-clear data or that its nonlinear phase evolution is correct. The high
condition numbers, fold-dependent boundary damping, and mismatch between weak
residual and rollout error indicate that parameter redundancy is the leading
explanation. This is an inference from the diagnostics, not a demonstrated
ground-truth parameter comparison.

M0-DD produced a modest empirical improvement over M0 under the present clean
simulation data, but it did not reach the acceptance targets and does not solve
the physical-identifiability problem. Its result supports retaining a direct
derivative fit as a diagnostic initializer, not replacing the weak-form method
as the main physical estimator.

Observed `Pm(t)` was already supplied. Therefore, the current failure cannot be
attributed to the ZOH mechanical-power approximation, and introducing a Pm ODE
now would mix a new input-model error with an unresolved state-dynamics error.

## Decision and boundary

**Decision: REJECT Round 5.** Test data remained locked, so no test
reintegration metric and no stability FSR were computed. This absence is a
prespecified consequence of the failed accuracy gate, not a missing analysis.
The result does not show that graph physics is unhelpful in general; it shows
that the present 70-parameter complete-graph sine/cosine/damping library is not
reliably identifiable or predictive under the registered one-second,
single-seed protocol.

## Proposed next round (not yet executed)

The next candidate should reduce graph degrees of freedom before adding any Pm
ODE or trajectory optimizer. A bounded proposal is to learn quotient damping
and stiffness in a low-dimensional modal basis, enforce positive semidefiniteness
there, and then fit only a small nonlinear correction tied to those linearized
operators. The proposal should retain M0 and M0-DD, the same long-trajectory
population, observed `Pm(t)`, and the same validation gate. It should proceed
only after explicit approval.

## Reproducibility

- Corrected source commit: `ac3a3dc867ea96dee1ecfd91a55a5bce21745c3a`
- Corrected immutable run: `paper_m0_graph_round5_solver_fix_validation-validation_run-20260729T104104Z-ac3a3dc8`
- Unit tests: 33 passed
- Project audit: PASS
- Test access: false; test IDs read: none

