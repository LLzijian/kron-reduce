# M0-AFSD v5 unified quotient selection protocol

[返回主题页](../README.md)

## 执行协议

<a id="doc-d6802ae74fea"></a>

<a id="doc-d6802ae74fea-m0-afsd-v5-unified-quotient-selection-protocol"></a>

### M0-AFSD v5 unified quotient selection protocol


<a id="doc-d6802ae74fea-scientific-question"></a>

#### Scientific question

Given all qualified F1 training trajectories, choose one spatial representation
that covers their union of observed local excitations.  This is distinct from
estimating the causal effect of excitation amplitude or data source on model
order.  Source-by-excitation overlap is therefore an attribution audit, not a
precondition for global quotient selection.


<a id="doc-d6802ae74fea-unified-quotient-gap"></a>

#### Unified Quotient Gap

For partition C, use the mass-orthogonal decomposition x = P_C x + Q_C x.  For
each held-out fault line and horizon H, compute:

- R: energy fraction in the discarded future state increment;
- G: positive error reduction obtained by adding Q_C x and the discarded
  mechanical-input coordinates to a matched-rank direct-horizon probe.

The local loss is max(R,G).  The Unified Quotient Gap is

```text
UQG(C) = max_H CVaR_10%( max(R_line,H, G_line,H) ).
```

Fault lines are equally weighted.  The 95% cluster-bootstrap upper confidence
bound must not exceed 0.02.  No source label, amplitude label or richness label
is an input to the statistic.  Those fields remain available only for a frozen
post-selection attribution audit.

The max operation is intentional: low reconstruction loss cannot compensate
for a closure failure, and strong average performance cannot hide a repeatedly
failed local excitation.


<a id="doc-d6802ae74fea-how-the-partition-is-selected"></a>

#### How the partition is selected

Start from ten singleton machines.  At every round, test every legal pairwise
merge at H=100.  Only screen-passing candidates receive H=100,500,1000
evaluation.  Among candidates passing every horizon, accept the merge with the
smallest UQG upper bound.  Stop when no additional merge passes.  Thus the
method chooses both whether to aggregate and which nodes to aggregate without a
preselected seven-node target or a source-overlap gate.

The selected partition is valid only for the observed Q-A excitation envelope.
An unselected physical mode may still exist outside that envelope.


<a id="doc-d6802ae74fea-evidence-and-governance"></a>

#### Evidence and governance

- partition construction and selection: Q-A F1 train only;
- F1 validation: frozen confirmation only;
- Q-B: no structure selection;
- test and F2-F5: locked;
- input policy: conditional rollout with observed Pm;
- final merge requires explicit approval.

Every scientific round follows hypothesis registration, controlled experiment,
residual diagnosis, adversarial review and acceptance.

<details>
<summary>原件与版本</summary>

- [docs/protocols/m0_afsd_v5_unified_quotient_selection.md](_sealed/m0_afsd_v5_unified_quotient_selection.md)

</details>
