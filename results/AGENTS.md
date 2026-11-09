# AGENTS.md

Every child directory is an immutable run. Never overwrite, merge, or silently
resume a run. Caches are valid only when data, split, source, config, and input
contract fingerprints match exactly.

Keep failed and interrupted attempts with their evidence. Use the run ID and
required artifact hashes when recording stage evidence. Inspect completed
runs through saved artifacts and `loop status`; keep regenerated summaries
under `reports/generated/`.
