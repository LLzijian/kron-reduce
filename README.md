# kron-reduce

Physics-structured IEEE39 post-fault dynamics with reproducible experiments.

From 2026-10-23 all new real-data training and evaluation use `kiki_ieee39_v1`:
591 registered events, originally train 369 / val 117 / test 105. Development
uses registered internal train/val groups inside the original source train.
Existing test exposure prevents fresh confirmatory claims; a new independent
kiki holdout must be registered before confirmation. The original canonical
dataset is retained for explicit historical reproduction.

Spatial clustering is paused: retain ten generator nodes while learning and
pruning the dynamical/latent modules. Synthetic ground-truth checks continue.
See `configs/research_policy.json` and
`docs/reports/research_governance_20261023_zh.md` for the active decision.

The research policy records active and consumed authorizations. See the
[research index](docs/research/README.md) for the follow-up scope, protocols
and saved results.

The current [algorithm design](docs/research/machine_local_structure_plan_20261112_zh.md)
prepares one hidden state per machine and learnable physical modules, with
separate checks for prediction accuracy and structure identifiability. A06
remains the reference; the new design does not authorize an experiment.

## Scientific scope

The primary task is closed-loop prediction of `delta(t), omega(t)` from the
exact clearing state. The default post-clear mechanical-power policy is a
zero-order hold at `Pm(tc)`. A run with the observed synchronous `Pm(t)` is a
separately reported conditional-rollout sensitivity, not a causal forecast.
Stability scores are derived from predicted trajectories and are exploratory.

Main methods are M0, M2A, energy-structured M2A, SINDYc, a shared
physics-informed neural ODE, and an autoregressive LSTM. Operator forecasts and
direct prefix classifiers are supplementary because they use a different
information budget.

## Research workflow

```powershell
./.venv/Scripts/python.exe -m project_harness audit
./.venv/Scripts/python.exe -m project_harness list
./.venv/Scripts/python.exe -m project_harness loop --workflow kiki_a06_refinement_v1 plan
```

1. **Inspect.** `audit` checks repository contracts and saved evidence. `list`
   shows the dataset and experiment registry. `loop plan` shows the current
   and next stages, the evidence contract, pending approvals, and matching
   experiment registrations with their execution modes and authorization status.
   Entries registered at the workflow level appear separately.
   `loop status` shows the full saved history. These inspection commands read
   repository metadata and saved evidence.
2. **Execute one step.** Use `loop run-one` for a built-in check. For an
   authorized experiment, use its registered entry point and stage. Generic
   experiments use `run <experiment> --stage <stage>`. Dedicated audits use the
   module and procedure in their approved protocol. `loop approve <gate>`
   records an explicit operator approval. `loop advance <stage> --evidence-file
   <file>` records externally produced evidence for the next stage.
3. **Review.** The loop advances after the evidence passes validation. A
   scientific verdict can route to acceptance or rollback according to the
   registered workflow. Each attempted experiment keeps its own run directory.

Use `--workflow <name>` before a loop subcommand. The default `paper_v1` is
the historical paper workflow. Workflow definitions live in `configs/workflows/`.
`git-check` checks the clean-checkout gate. `data-check --dataset <name>` checks
registered metadata and source hashes separately from the workflow preview.
Contributor rules are in [AGENTS.md](AGENTS.md) and the
[harness instructions](project_harness/AGENTS.md).

Set `KIKI_DATA_ROOT` to the directory containing `ieee39_postfault` and
`ieee39_expansion`; the local fallback is `../kiki/private_data`. The raw HDF5
files remain external. Metadata, checksums and exposure records are versioned.
An old canonical run additionally requires `--historical-reproduction`; its
workflow still permits only its registered next stage and original approvals.

The private GitHub remote is `https://github.com/LLzijian/kron-reduce`.
GitHub Actions checks policy and synthetic fixtures on pushes and pull requests.
Install with `pip install -e '.[dev]' -c requirements/constraints-py312.txt` on
Python 3.12; the constraint file records tested core versions.

## Artifact locations

- Immutable numerical evidence: `results/<run-id>/`
- Generated summaries: `reports/generated/`
- Curated research reports: `docs/reports/`
- Manuscript sources: `paper/`
- Temporary investigations: `scratch/one_off/<run-id>/`

Do not place report files directly under `reports/`; the harness treats that as an
artifact-layout error. See `docs/reports/README.md` for the curated report index.

Full paper runs remain locked until the source baseline is committed, the data
support audit passes, and `full_run_approved` is explicitly approved.
After approval, the registered `paper_v1_full` runner freezes every fold/seed
selection record before opening test trajectories. Summarization, automated
adversarial checks, LaTeX table generation, and manuscript assembly are separate
one-stage commands in the project manifest; publication remains a human gate.
