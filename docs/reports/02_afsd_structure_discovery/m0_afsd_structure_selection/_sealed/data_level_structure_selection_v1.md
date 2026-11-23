# Data-level structure selection v1

## Contract

The workflow consumes all eligible F1 training trajectories in original
generator coordinates. Independent fault-line groups are split 60/20/20 into
fit, structure-selection and audit roles. Test IDs and other folds remain
locked. The causal input is the clearing mechanical power held by zero-order
hold.

One run produces exactly one `ClusterDecision` and one frozen
`PrunableEnergyModel`. Amplitude, information-richness and empirical coupling
strength are computed only after freezing and are used only to slice audit
metrics.

## Cluster decision

`selection/clusterability.py` fits one inexpensive B2 weak-form model and
forms the synchronising generalized Laplacian. For each reduced rank it uses
the maximum normalized information, input, worst-mode and spectral-gap loss.
The 95% trajectory-group bootstrap upper bound must be at most one and the
pass support must be at least 80%.

Every soft candidate is converted to a deterministic hard partition and
checked for state reconstruction, input heterogeneity, synchronising closure,
conditional predictive information gain and worst-mode loss. The smallest
passing partition is selected. If none passes, the identity partition is
used. All reduced predictions are lifted to original node coordinates before
evaluation.

## Unified model and deletion

`models/prunable_energy_model.py` retains the B2 swing equation and optionally
adds five whole blocks: nonlinear potential, nonlinear edge damping, node
storage, decaying edge memory and damped oscillating memory. Dynamic blocks
use reciprocal input/output connections, nonnegative operators and explicit
storage, so their supply identity is structural.

Candidates are proposed from B2 residual correlations and poles using
independent trajectory-group support. The full supported model is fitted in
integral weak form, calibrated against closed-loop reintegration, then pruned
one whole block at a time. Every deletion refits all remaining coefficients
and repeats closed-loop calibration. A deletion is accepted only if each of
the 1, 5 and 10 second losses degrades by no more than 2%, frequencies and
damping remain within 5%, modal MAC remains at least 0.8, and physical gates
continue to pass.

After structure freezing, fit and selection groups are merged for final
parameter estimation. Audit trajectories cannot change rank, partition,
blocks, regularization or thresholds.

## Outputs and gates

The controlled stage writes `cluster_decision.json`, `partition.json`,
`candidate_blocks.json`, `pruning_trace.csv`, `selected_model.npz`,
`modal_report.json`, `stability_certificate.json` and
`structure_selection_result.json`, plus artifact hashes and per-case metrics.

The numerical gates cover zero integration failures, COI drift at most
`1e-10`, positive storage curvature, nonnegative dissipation and a numerical
supply-identity error at most `1e-8`. Stability margins use the minimum energy
over adjacent saddles and the allowed angle boundary under fixed post-clear
input. Outcome labels are read only after model freezing to report FSR and its
95% upper bound; they are never model features or selection inputs.
