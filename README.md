# kron-reduce

Physics-structured IEEE39 post-fault dynamics with reproducible experiments.

New KIKI comparisons use TASK3's frozen roles: `train=346` for fitting,
`val=107` for predeclared selection, and `test=76` for final evaluation. The
76-event test roster contains 43 expansion and 33 additional holdout events;
it is not derived by screening the official 105-event test allocation. The
original source split allocated train 369 / val 117 / test 105; after the
100 ms length screen those source roles contain 346 / 107 / 99 events. These
source counts describe provenance, while the TASK3 76-event list is the current
main test population. Do not create another internal train/val split. The test
roster has prior exposure and supports comparative reporting on the fixed
benchmark. Independent confirmation uses a separately registered population.
Historical experiments retain their original roles and scopes.

The [KIKI IEEE39 blind v2 reuse guide](data/kiki_ieee39_blind_v2/README.md)
documents the frozen 94-event comparison cohort, its data dictionary, local
HDF5 verification, and a causal Python reader example.

Spatial clustering is paused: retain ten generator nodes while learning and
pruning the dynamical/latent modules. Synthetic ground-truth checks continue.
See `configs/research_policy.json` and
`docs/reports/05_research_governance/governance_decision_20261023/_sealed/research_governance_20261023_zh.md` for the active decision.

The research policy records active and consumed authorizations. See the
[research index](docs/reports/05_research_governance/research_history/README.md#doc-0a640a72f9b4) for the follow-up scope, protocols
and saved results.

New real-data work uses the frozen TASK3-compatible train, val and test lists
under the [benchmark standard](docs/reports/05_research_governance/task3_standard/README.md#doc-6c6486ac9f76).
The exact source-qualified IDs and list hashes are in the
[split reference](configs/proposals/kiki_task3_split_reference_v1.json). The active
[joint initial-state study](docs/reports/08_machine_local_topology/joint_observer/README.md#doc-d1b07a56bdcb)
tests a noise correction before learned-model confirmation and real-data execution.

The current [algorithm design](docs/reports/08_machine_local_topology/synthetic_implementation/README.md#doc-6f0ce03af776)
implements one hidden state per machine, with separate checks for prediction
accuracy and structure identifiability. The [synthetic protocol](docs/reports/08_machine_local_topology/synthetic_implementation/README.md#doc-6888fd388cdf)
defines S0 and twelve S1-base fits under separate gates and budgets. A06 remains
the reference; later physical modules and real-data comparisons have their own gates.
[S0 passed](docs/reports/08_machine_local_topology/synthetic_implementation/README.md#doc-2a087778b7e9); the twelve-fit
[S1-base result](docs/reports/08_machine_local_topology/synthetic_implementation/README.md#doc-c290428fd661) failed its prediction
gate. The current scope is initialization/generalization diagnosis preparation;
T0 is stopped and both synthetic execution authorities are consumed.

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

### Git archives

Commit the final code and evidence receipt on the attempt's `codex/` branch.
Run the harness from that clean worktree and record the outcome and reason.

```powershell
python -m project_harness git-archive --outcome rejected --message "Trajectory RMSE missed the promotion gate" --push
```

Choose `completed`, `rejected`, `failed`, or `abandoned`. The annotated tag
`attempt/<branch>/<full-commit>` records the branch, outcome, summary, and exact
commit. Include the run ID or receipt path in the summary when available.
`--push` publishes that tag to `origin`. Keep the branch until the push
succeeds, and retry with the same outcome and message if publishing fails.
Use `git fetch --tags` to retrieve the archives, `git tag --list 'attempt/*'`
to list them, and `git show <tag>` to inspect a record after branch deletion.

### CUDA for machine-local forecasts

On Windows with a compatible NVIDIA driver, install the pinned CUDA build in
a separate Python 3.12 environment. See the [official PyTorch CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/).

```powershell
py -3.12 -m venv scratch/.venv-cu130
./scratch/.venv-cu130/Scripts/python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
./scratch/.venv-cu130/Scripts/python.exe -m pip install -e '.[dev]' -c requirements/constraints-py312.txt
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
./scratch/.venv-cu130/Scripts/python.exe -m pytest tests/test_nonlinear_history.py -k cuda
```

Move a loaded `FrozenMachineLocalForecaster` to CUDA with `.to('cuda')`.
`predict_batch` returns tensors on the model device, and `predict` returns a
NumPy array. Checkpoints saved on CUDA load on CPU. For a new registered
paired experiment, bind `"device": "cuda:0"` in the exact fit or test plan and
pass the registered ID through `--experiment` to its workflow entry point.
With no `device` in the plan, the entry point uses CPU.

The CUDA forecast groups ten 10 ms output frames in a replayable GPU graph.
RHS and hidden-state solve budgets still count stages and events, respectively.
The CPU and CUDA paths both use float64, projected RK4 and three Gauss–Newton
updates. The rank check and the final trajectory check run outside the graph.

On an i5-13400F and RTX 4060 Ti, the median of five warmed runs on a synthetic
64-event batch was as follows. Training includes a 3 s rollout at 5 ms,
backpropagation and one Adam update. Forecasting includes the initializer and
an 8 s rollout at 2.5 ms. The benchmark measures model work with inputs already
on the selected device. Dataset reads, operation logs and scoring are outside
the timing window.

| Workload | CPU before optimization | CPU after optimization | CUDA after optimization |
|---|---:|---:|---:|
| Training batch | 8.03 s | 4.66 s | 9.52 s |
| Frozen forecast batch | 9.42 s | 5.74 s | 2.11 s |

## Artifact locations

The [research report and archive index](docs/reports/README.md) links experiments
to curated reports, evidence receipts, and archive packages.
Run evidence and generated reports are stored together in `results/<run-id>/`.
Local archives are stored in `results/backup/`.
Manuscript sources are in `paper/`, and temporary work is in `scratch/one_off/`.

Full paper runs remain locked until the source baseline is committed, the data
support audit passes, and `full_run_approved` is explicitly approved.
After approval, the registered `paper_v1_full` runner freezes every fold/seed
selection record before opening test trajectories. Summarization, automated
adversarial checks, LaTeX table generation, and manuscript assembly are separate
one-stage commands in the project manifest; publication remains a human gate.
