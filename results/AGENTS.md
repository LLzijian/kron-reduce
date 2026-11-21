# AGENTS.md

Every child directory is an immutable run. Never overwrite, merge, or silently
resume a run. Caches are valid only when data, split, source, config, and input
contract fingerprints match exactly.

Keep failed and interrupted attempts with their evidence. Use the run ID and
required artifact hashes when recording stage evidence. Inspect completed
runs through saved artifacts and `loop status`. Store generated summaries in
the run directory that produces them.

## Normal operation (default)

NORMAL WORK IS UNCHANGED. Keep generating immutable run directories under
`results/<run-id>/` exactly as the harness does. Do **not** archive, compress,
merge, thin, or delete runs during normal work. Runs accumulate here until the
operator explicitly triggers archival (see below).

## Archival trigger

Archival runs **only when the operator says "归档"** (or an equivalent explicit
instruction). At that moment, archive **the latest completed batch of runs that
has not yet been archived** — nothing older, nothing still running. Do not
archive proactively or on a schedule.

This is a deliberate, operator-approved exception to the immutability rule; it
does not relax immutability for any run that has not been archived.

## Archival procedure (two phases per batch)

### Phase 1 — small-conclusion sub-bundles (not month-based, not one big zip)

Group the batch by **small conclusion / small sub-question**, not by month and
not into one coarse archive. A good sub-bundle = a few runs that together solve
one small problem or establish one small result. Name sub-bundles with an
ordered key and a short semantic label, e.g. `05_damping_corrections`,
`B_afsd_v2_module_selection`.

For each sub-bundle produce:

1. `<sub-bundle>.zip` under `results/backup/<group>_subsets/`. It contains, for
   every run in that sub-bundle, ONLY the conclusion + model artifacts:
   - KEEP: `summary.json`, `acceptance.json`, `stage_evidence.json`,
     `audit.json`, `selection*.json`, `verdict`/`closeout` JSON, model bodies
     (`frozen*.npz`, `selected-*.npz`, initial checkpoints), small result CSVs
     and the run's `DESCRIPTION`-source JSON.
   - DROP (regenerable bulk): `train_predictions.npz`, `val_predictions.npz`,
     all `*predictions.npz`, `operations.jsonl`, and diagnostic duplicate
     dumps (`*_posterior.npz`, `*noise*.npz`, `*_clean.npz`). These are
     recomputable from model + data and are not kept.
2. A per-bundle ledger/description `<sub-bundle>.DESCRIPTION.md` placed
   **OUTSIDE the zip, next to it** so it is readable without unzipping. It must
   state: the sub-question this bundle answers (导言: what was tried, what was
   obtained) and a per-run table `run | conclusion`. A copy may also sit inside
   the zip.

### Phase 2 — batch ledger + research summary + removal

3. A batch ledger `results/backup/<group>_subsets/LEDGER_<batch>.md` that
   records, for every run: what was attempted and what was obtained (status,
   fit ok/fail, key metrics), with the run ID.
4. A research-level summary `results/SUMMARY_*.md` written from the research
   angle: what method, how the algorithm is implemented, what result, what was
   learned, and what validation pitfalls exist. It does not list every run; it
   cites the ledgers/sub-bundles. **One SUMMARY may cover multiple backup
   packages** — it aggregates across sub-bundles (and across batches) into one
   topic story. Keep `results/README.md` as the index of summaries and packages.

### Verification gate before deleting anything

A run directory may be deleted **only after**:
- its sub-bundle zip exists and was verified — for every run in the sub-bundle
  the source file count equals the archive entry count and the uncompressed
  bytes match, with an expand/sample check; and
- the `DESCRIPTION.md` and batch `LEDGER` exist; and
- a `SUMMARY_*.md` references the package.

Never delete an unarchived run. Never archive-and-delete a run whose evidence is
still an active or pending input to a registered workflow, report, or manuscript
gate without operator confirmation. Record the batch identity and verification
result in the ledger so removals stay auditable and reversible. The zip is the
recoverable backup; deletion of the original run directory is the last step, not
the first.

## Layout

```
results/
├── AGENTS.md                      # this file (the only path tracked by Git)
├── README.md                      # index of SUMMARY_* and backup packages
├── SUMMARY_*.md                   # research-level summaries (aggregate several packages)
├── backup/
│   └── <group>_subsets/
│       ├── <sub-bundle>.zip
│       ├── <sub-bundle>.DESCRIPTION.md   # outside the zip, readable
│       └── LEDGER_<batch>.md
└── <run-id>/ ...                  # live runs; unchanged during normal work
```

`results/` is not tracked in Git (`results/*` is ignored except this AGENTS.md);
backups, ledgers and summaries here are local artifacts and are intentionally
not committed.
