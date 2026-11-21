# Paper v1 experiment card

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

The original card and generated reports are available in the [Paper v1 archive](../../results/backup/retired_layout_20261121T093836Z/paper_v1.zip).
