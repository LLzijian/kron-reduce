# AGENTS.md

This repository learns and audits physics-structured reduced models of
post-disturbance power-system dynamics. Prioritize physical correctness,
closed-loop trajectory accuracy and leakage-free evidence before classification.

## Where to look

- Read the nearest local `AGENTS.md` before editing a subsystem.
- [Architecture](docs/architecture.md) defines code and artifact ownership.
  The [scientific contract](docs/scientific_contract.md) defines inputs,
  rollout, stability and naming.
- The [project manifest](configs/project_manifest.json) registers datasets,
  experiments and entry points. `configs/workflows/` defines stages, gates
  and evidence contracts. The [research policy](configs/research_policy.json)
  governs scientific execution authority.
- Check the [research index](docs/research/README.md) for follow-up scope and
  prior decisions. The [report index](docs/reports/README.md) lists results
  and evidence receipts.

## Workflow

- Start with `python -m project_harness loop --workflow <name> plan`. Use
  `loop status` for saved history and `list` for the registry.
- Advance one stage at a time through its registered entry point. Use
  `loop run-one` for built-in checks and `loop advance` for verified external
  evidence. Use `loop approve` only after explicit operator approval.
  Scientific execution requires its own registered scope and budget.
- Use `python -m project_harness audit` for metadata and evidence checks.
  Run `git-check` for the clean-checkout gate. Use `data-check --dataset <name>`
  only within the authorized read scope.

## Scientific boundaries

- New real-data work uses registered `kiki_ieee39_v1` at ten-generator
  resolution. Spatial clustering requires an explicit policy change.
  Synthetic known-truth checks support structure and stability verification.
  Canonical replay requires `--historical-reproduction` and matching frozen
  config and workflow hashes.
- New kiki comparisons use the frozen TASK3 cohorts: `train=346`
  for fitting, `val=107` for predeclared selection and `test=76` for final
  evaluation. Use the exact source-qualified IDs, not counts alone. Do not create another
  internal train/val split. Freeze the model before test evaluation and match
  TASK3's 100 ms inputs and scoring. See the
  [split standard](docs/protocols/kiki_official_train_test_v1_zh.md) and frozen
  [source-qualified roster](configs/proposals/kiki_task3_split_reference_v1.json).
  Report this previously exposed test roster as a fixed comparison benchmark.
  Independent confirmation requires independent data. Consumed historical
  protocols retain their original splits.
- For each newly authorized training round, freeze its model and evaluation
  contract, then complete the registered `val` and `test` evaluations whether
  or not RMSE meets the promotion gate. Never use `test` to tune or revise the
  model. Start with a quick assessment without added noise, then finish the
  full frozen evaluation. Missing, nonfinite or otherwise invalid metrics are
  explicit failures with their coverage reported; never replace them with zero
  or an invented value.
- Follow the named causal input contract and its forecast origin. Main-model
  features exclude `Pe`, voltage, labels, event time, valid length, masks,
  fault metadata and future summaries. Fit one model per supplied corpus.
  Use excitation views to audit support and prune unsupported modules.
  Predictions must use the same model across sources and regimes.
- Refuse outside certified excitation envelopes except within a registered
  bounded evaluation. Derive exploratory stability from predicted trajectories.
  Use `M2A-LYAP` after all numerical certificate gates pass. The name
  `stability-certified equivalent` also requires the
  [Level 6 and region/margin gates](docs/structure_discovery_roadmap.md).
- Keep consumed authorizations closed and retain failed and uncovered cases.
  Review saved evidence for follow-up. New comparisons require separate
  registration and scientific approval. Model promotion, deployment-default
  changes and extraction from rejected branches require their own review.

## Engineering conventions

- The coordinating agent owns runtime changes, integration and evidence
  verification. Delegate core algorithm design, scientific acceptance and
  boundary decisions to Astra Ultra. Assign independent, well-scoped work to
  the least costly capable subagent, typically Luna for simple edits and tests.
  Follow the [engineering task allocation rules](docs/architecture.md#engineering-task-allocation).
  Do not invoke Claude Code or CC Orchestrator for this project.
- Invoke the selected Python 3.12 interpreter directly with versions constrained
  by `requirements/constraints-py312.txt`. Keep replacement environments and
  temporary programs in ignored `scratch/`. Use `scratch/one_off/<run-id>/` for
  one-off programs. Do not use `uv run` in the shared checkout. Record the actual
  runtime with the experiment evidence.
- Give every execution a new immutable run ID. Commit and push completed changes
  on `codex/` branches and review before merging to main. Track manifests and
  receipts in Git. Keep raw trajectories, result bodies, data archives,
  credentials and local environments outside Git.
- Keep `AGENTS.md` files for durable instructions and pointers. Record campaign
  work and decisions in research docs, protocols in `docs/protocols/`, and
  curated results in `docs/reports/`.
