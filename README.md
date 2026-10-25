# kron-reduce

Physics-structured IEEE39 post-fault dynamics with reproducible experiments.

From 2026-10-23 all new real-data training and evaluation use `kiki_ieee39_v1`:
591 registered events, originally train 369 / val 117 / test 105. Development
uses registered internal train/val groups inside the original source train.
Existing test exposure prevents fresh confirmatory claims; a new independent
kiki holdout must be registered before confirmation. The original canonical
dataset is retained for explicit historical reproduction.

The [completed A17 comparison](docs/reports/kiki_a17_input_consistency_20261025_zh.md)
retains all 268 allocated train records and zero new fits. Physical, numerical
and independent evidence checks passed; the input-consistency candidate failed
all frozen improvement comparisons. Astra reviewed this as a valid negative
result: REJECT, KEEP_BASELINE, consumed authorization and no new missed-instability
claim. The complete proposal, checker failures and repairs are archived.

Spatial clustering is paused: retain ten generator nodes while learning and
pruning the dynamical/latent modules. Synthetic ground-truth checks continue.
See `configs/research_policy.json` and
`docs/reports/research_governance_20261023_zh.md` for the active decision.

The [next-phase research design](docs/research/kiki_next_phase_20261023_zh.md)
and [A04 proposed protocol](docs/protocols/kiki_a04_short_history_design_20261023_zh.md)
define the next short-history comparison, uncertainty and local-region work.
The fixed real-data comparison is complete; no new real model has been promoted.

The [A04 engineering and mechanism results](docs/reports/kiki_a04_execution_20261023_zh.md)
record implemented short-history tools, accepted synthetic checks and the
earlier real-data envelope prerequisite. The user has now approved the
[bounded A04 development evaluation](docs/protocols/kiki_a04_development_authorization_20261023_zh.md)
on frozen internal train/val inside original source train. The
[completed A04 result](docs/reports/kiki_a04_development_20261023_zh.md) retains
all 12 fits and 960 updates. The registered improvement gates were not met;
the existing model is retained and the one-time authorization is consumed.
Source val/test, prediction deployment and model promotion remain closed.

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

## Commands

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m project_harness list
./.venv/Scripts/python.exe -m project_harness audit
./.venv/Scripts/python.exe -m project_harness data-check
./.venv/Scripts/python.exe -m project_harness git-check
./.venv/Scripts/python.exe -m project_harness data-check --dataset ieee39_postclear_canonical_v1
./.venv/Scripts/python.exe -m project_harness loop status
```

Use `--workflow <name>` before the loop subcommand to inspect a named workflow,
for example:

```powershell
./.venv/Scripts/python.exe -m project_harness loop --workflow m0_afsd_v4_graph_compositional_discovery status
```

The loop is command-driven rather than a resident background process. Its state,
ordered evidence, explicit approvals, data locks, and Git gates persist between
commands and are checked again by `project_harness audit`.

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
