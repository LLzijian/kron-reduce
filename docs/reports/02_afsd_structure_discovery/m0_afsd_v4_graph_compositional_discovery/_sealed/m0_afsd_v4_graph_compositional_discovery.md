# M0-AFSD v4 graph-compositional discovery protocol

## Scope

This immutable F1/seed324 workflow identifies the minimum predictable graph
quotient and registered physical-module composition supported by Q-A training
trajectories.  It uses **conditional rollout with observed Pm**.  It is not an
autonomous causal deployment experiment.

The workflow does not read test, F2--F5, or v1--v3 result artifacts and never
mutates canonical data.  Validation is confirmation-only after train-internal
rules are frozen.  Q-B trajectories are post-freeze robustness cases only.

## Ordered evidence chain

Every scientific round is represented by immutable artifacts in this order:

1. hypothesis registration;
2. controlled experiment;
3. residual diagnosis;
4. adversarial review;
5. acceptance, rejection, or a preregistered information-limited terminal state.

The controlled experiment contains governance Round 0, synthetic graph gates
G1--G2, synthetic module/search gates G3--G5, and only then F1 qualification.
An upstream hard gate fails closed: later structure fits are recorded as not
run, never silently replaced by a weaker selector.

## Spatial structure

For partition membership matrix S and declared positive mass M,

\[
P_C=S(S^TMS)^{-1}S^TM,\qquad Q_C=I-P_C.
\]

Every candidate must pass projector algebra, mass conservation, COI
compatibility, input heterogeneity, within-cluster energy, repeatable modal
loss, held-out closure, and equal-capacity ridge-Hankel CPIG checks.  Graph
contraction starts at ten singleton machines and uses certificate-gated Beam
search with widths 3, 5, and 10.  A failed hard cluster may add the smallest
train-derived mass-orthogonal internal-mode basis.  Validation and test never
construct a partition or a basis.

## Dynamic structure

The search space is restricted to ten typed modules whose ports, units,
parameter domains, equilibrium anchors, storage/dissipation conditions,
allowed connections, and description lengths are preregistered.  Residuals
are separated by node/edge location, instantaneous/memory character,
conservative/dissipative role, input-channel alignment, and smooth/switching
behavior.  Device names are prohibited without independent post-freeze
measurements.

Tier 1 uses weak-form screening, Tier 2 uses H=100 closure, and Tier 3 is
reserved for final leaves at H=100, 500, and 1000.  MCTS is disabled unless a
synthetic paired-module audit proves that every preregistered Beam width misses
the exhaustive oracle.  Different state dimensions are never averaged.

## Selection and stopping

Selection is lexicographic: qualification, physical/numerical hard gates,
10-second rollout, 1/5-second rollout, relative modal gates, absolute modal
gates, stability, then description length.  A complex structure requires at
least 2% equal-fault-line improvement, positive 95% cluster CI lower bound,
4/5 directional fold agreement, zero integration failures, COI drift below
1e-10, RK4 convergence, no modal degradation, and at least 80% structural
support.

Permitted scientific terminal states are those listed in the frozen workflow
plan, including `SOURCE_EXCITATION_CONFOUNDED`, `INSUFFICIENT_EXCITATION`,
`GRAPH_SUPPORT_NOT_IDENTIFIABLE`, and `MODEL_CLASS_INADEQUATE`.  A negative
terminal must identify the rejected hypothesis, direct evidence and interval,
alternatives, why more computation cannot repair the design, and the new
observations or excitation needed.  The final merge gate always requires
explicit approval.
