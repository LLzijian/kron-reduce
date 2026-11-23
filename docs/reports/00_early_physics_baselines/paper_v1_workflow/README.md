# Paper v1 experiment card

[返回主题页](../README.md)

## 运行结果与归档

- [完整流程归档说明](../../../../results/backup/00_early_physics_baselines/A_paper_v1_workflow.DESCRIPTION.md)
- [结果与报告归档说明](../../../../results/backup/00_early_physics_baselines/A_paper_v1_workflow_saved_reports.DESCRIPTION.md)

## 报告与结果

<a id="doc-46816da8f836"></a>

<a id="doc-46816da8f836-paper-v1-experiment-card"></a>

### Paper v1 experiment card

- Task: identification plus exact-clearing closed-loop rollout.
- Dataset: `ieee39_postclear_canonical_v1`.
- Primary view/split: `combined_dynamics/combined_line_5fold_v1`.
- Primary representation: frozen clustered-7.
- Primary Pm policy: post-clear zero-order hold at `Pm(tc)`.
- Sensitivity: observed synchronous `Pm(t)`, reported separately.
- Main estimand: paired M2A minus M0 trajectory error by physical event, with
  fault-line cluster bootstrap.
- Stability: prospective, rollout-derived, exploratory only.
- Stop gates: data checksum, support, real model smoke, source commit, explicit
  full-run approval, energy naming gate, and publication gate.

See the [Paper v1 archive guide](../../../../results/backup/00_early_physics_baselines/A_paper_v1_workflow_saved_reports.DESCRIPTION.md) for the original card and generated reports.

<details>
<summary>原件与版本</summary>

- [docs/reports/00_early_physics_baselines/paper_v1.md](_sealed/paper_v1.md)

</details>
