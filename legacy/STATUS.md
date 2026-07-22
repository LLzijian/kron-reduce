# Historical 3.10/3.11 snapshot

This directory is provenance only and is excluded from runtime imports and
scientific selection. Historical 3.10/3.11 scores used old datasets, old split
semantics, and an observed-Pm conditional-rollout contract. Task 3.11 optimized
accuracy-first thresholds and retained nonzero false-stable rates; it is an
exploratory appendix, not a safety claim.

The clean implementation follows the later 3.37r1 corrections for actual PINN
inference, prospective cohorts, forecast scoring after the observation prefix,
fault-line validation, real Pm-ZOH sensitivity, honest failure states, and the
three-level energy naming gate. No file under `legacy/` may be imported.

Only reports, run configurations, compact summaries, source snapshots, and the
original artifact-hash indexes are retained. Large raw grids, per-case tables,
checkpoints, and caches are intentionally omitted; their original hashes and
paths remain in the historical artifact indexes for traceability.
