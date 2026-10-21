"""A03 registered mechanism comparisons; IEEE39 audit remains locked."""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.data.p2_roles import build_p2_role_rows
from kron_reduce.evaluation.p2_metrics import paired_stratified_fault_line_bootstrap
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    fit_nonlinear_model,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project
from kron_reduce.physics.units import effective_mass
from kron_reduce.workflows.ieee39_history_learning import AuthorizedHistorySamples
from project_harness.data_check import verify_data
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines


def _write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class TrainValNonlinearSamples(AuthorizedHistorySamples):
    def _make_sample(self, *args, equilibrium_delta):
        return EquilibriumLearningSample(*args, equilibrium_delta)

    def _authorize_audit(self):
        raise RuntimeError("A03 train/val stage never authorizes IEEE39 audit reads")


def synthetic_samples(config, role):
    """Generate visible-only learning records, with independently seeded roles."""
    if role not in {"train", "val", "audit"}:
        raise ValueError("unregistered synthetic role")
    n, dt = 3, 0.01
    mass = np.array([1.0, 1.2, 1.5])
    basis = coi_basis(np.ones(n))
    truth = NonlinearPassiveClosure(
        mass,
        np.array([1.3, 0.7, 1.1]),
        0.2 * basis @ basis.T,
        0.9 * basis,
        np.array([[0.0, -1.2], [1.2, 0.0]]),
        np.diag([0.2, 0.4]),
    )
    count = config["history_sample_counts"][0]
    steps = round(config["forecast_duration_s"] / dt)
    rng = np.random.default_rng(config["seed"] + {"train": 1000, "val": 2000, "audit": 3000}[role])
    samples = []
    for line in range(config[f"{role}_lines"]):
        for pair in range(config["pairs_per_line"]):
            eq = coi_project(rng.normal(0, 0.25, n), mass)
            q = coi_project(rng.normal(0, 0.5, n), mass)
            v = coi_project(rng.normal(0, 0.2, n), mass)
            hidden = rng.normal(0, 0.7, 2)
            power = rng.normal(0, 0.05, n)
            power -= power.mean()
            for sign in (-1, 1):
                path = truth.rollout(q, v, sign * hidden, np.arange(count + steps) * dt, power, eq)
                history = CausalStateHistory(
                    np.arange(count) * dt - (count - 1) * dt,
                    path[:count, :n],
                    path[:count, n : 2 * n],
                    power,
                )
                samples.append(
                    EquilibriumLearningSample(
                        f"{role}-{line}-{pair}-{sign}",
                        "synthetic",
                        f"{role}-line-{line}",
                        role,
                        history,
                        np.arange(steps + 1) * dt,
                        path[count - 1 :, :n],
                        path[count - 1 :, n : 2 * n],
                        eq,
                    )
                )
    return mass, tuple(samples)


def _prefix(samples, horizon):
    output = []
    for s in samples:
        count = round(horizon / 0.01) + 1
        if count > len(s.forecast_time) or abs(s.forecast_time[count - 1] - horizon) > 1e-10:
            raise ValueError("sample does not cover registered horizon")
        output.append(
            replace(
                s,
                forecast_time=s.forecast_time[:count],
                target_delta=s.target_delta[:count],
                target_omega=s.target_omega[:count],
            )
        )
    return tuple(output)


def _scores(model, samples, horizons, *, zero_latent=False):
    scores = {str(h): model.score(_prefix(samples, h), zero_latent=zero_latent) for h in horizons}
    values = [s["normalized_rmse"] for s in scores.values()]
    value = None if any(v is None for v in values) else float(np.sqrt(np.mean(np.square(values))))
    return {"multi_horizon_normalized_rmse": value, "horizons": scores}


def execute_study(*, root, config, config_path, result_dir, run_id, provider=None):
    synthetic = config["data_policy"] == "analytic_nonlinear_synthetic_only"
    if synthetic:
        mass, train = synthetic_samples(config, "train")
        _, val = synthetic_samples(config, "val")
    else:
        mass, train = provider(config, "train", 0.2)
        _, val = provider(config, "val", 0.2)
    if {s.event_id for s in train} & {s.event_id for s in val} or {
        (s.source_dataset, s.fault_line) for s in train
    } & {(s.source_dataset, s.fault_line) for s in val}:
        raise ValueError("train and val IDs or source/line groups overlap")
    candidates, records = {}, []
    horizons = config["report_horizons_s"]
    for candidate in config["candidates"]:
        token = candidate["name"]
        settings = NonlinearFitSettings(
            latent_dimension=candidate["latent_dimension"],
            backbone=candidate["backbone"],
            frames=config["encoder_frames"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            encoder_penalty=config["encoder_penalty"],
            history_weight=candidate["history_weight"],
            horizons_s=tuple(candidate["horizons_s"]),
            seed=config["seed"],
        )
        started = time.perf_counter()
        print(f"Fitting {token} on {len(train)} train events", flush=True)
        fit_samples = _prefix(train, max(settings.horizons_s))
        # All candidates share full registered train-horizon normalization,
        # including the short-horizon objective ablation.
        model = fit_nonlinear_model(fit_samples, mass, settings, target_scale_samples=train)
        path = result_dir / f"candidate-{token}.npz"
        model.save(path)
        candidates[token] = LearnedNonlinearModel.load(path)
        train_score = _scores(model, train, horizons)
        score = _scores(model, val, horizons)
        _write(result_dir / f"candidate-{token}-train.json", train_score)
        _write(result_dir / f"candidate-{token}-val.json", score)
        record = {
            "name": token,
            "settings": candidate,
            "train_multi_horizon_normalized_rmse": train_score["multi_horizon_normalized_rmse"],
            "val_multi_horizon_normalized_rmse": score["multi_horizon_normalized_rmse"],
            "parameter_count": model.fit_record["parameter_count"],
            "checkpoint_sha256": sha256_file(path),
            "wall_clock_s": time.perf_counter() - started,
        }
        records.append(record)
        print(f"Completed {token}: val={record['val_multi_horizon_normalized_rmse']}", flush=True)
    selected = min(
        records,
        key=lambda r: (
            float("inf")
            if r["val_multi_horizon_normalized_rmse"] is None
            else r["val_multi_horizon_normalized_rmse"],
            r["parameter_count"],
            r["name"],
        ),
    )
    if selected["val_multi_horizon_normalized_rmse"] is None:
        raise RuntimeError("all candidates failed; no independent population may open")
    _write(
        result_dir / "selection_freeze.json",
        {
            "selection_role": "val",
            "selection_metric": "sqrt(mean(horizon_normalized_rmse_squared))",
            "selected": selected,
            "candidate_table": records,
            "refit_train_val": False,
            "ieee39_audit_authorized": False,
        },
    )
    model = candidates[selected["name"]]
    s = val[0]
    first = model.predict(s.history, s.forecast_time, s.equilibrium_delta)
    loaded = LearnedNonlinearModel.load(result_dir / f"candidate-{selected['name']}.npz")
    if not np.array_equal(first, loaded.predict(s.history, s.forecast_time, s.equilibrium_delta)):
        raise RuntimeError("checkpoint prediction replay changed")
    z = model.initialize(s.history, s.equilibrium_delta)[0]
    fine = model.closure.rollout(
        s.history.delta[-1],
        s.history.omega[-1],
        z,
        s.forecast_time,
        s.history.constant_power,
        s.equilibrium_delta,
        max_step=0.005,
    )
    refinement = float(np.max(np.abs(first - fine)))
    rng, identity = np.random.default_rng(config["seed"]), []
    for candidate_model in candidates.values():
        for _ in range(25):
            state = np.r_[
                coi_project(rng.normal(size=len(mass)), mass),
                coi_project(rng.normal(size=len(mass)), mass),
                rng.normal(size=candidate_model.closure.latent_dimension),
            ]
            power = rng.normal(size=len(mass))
            power -= power.mean()
            identity.append(
                candidate_model.closure.energy_identity(state, power, s.equilibrium_delta)[
                    "identity_error"
                ]
            )
    if (
        max(identity) > config["energy_identity_tolerance"]
        or refinement > config["step_refinement_tolerance"]
    ):
        raise RuntimeError("learned model failed passive-identity or integration refinement gate")
    population = val
    if synthetic:
        # The analytic independent population is generated only after the
        # complete candidate set and selected checkpoint hashes are frozen.
        _, population = synthetic_samples(config, "audit")
        if {s.event_id for s in population} & {s.event_id for s in train + val}:
            raise ValueError("synthetic independent IDs overlap fitting or selection")
    selected_score = _scores(model, population, horizons)
    zero_score = _scores(model, population, horizons, zero_latent=True)
    memoryless = _scores(candidates["sine-memoryless"], population, horizons)
    for name, value in (
        ("selected", selected_score),
        ("selected-zero-latent", zero_score),
        ("sine-memoryless", memoryless),
    ):
        _write(result_dir / f"{'audit' if synthetic else 'val'}-{name}.json", value)
    best, zero, baseline = [
        v["multi_horizon_normalized_rmse"] for v in (selected_score, zero_score, memoryless)
    ]
    checks = {
        "nonzero_memory_selected": model.settings.latent_dimension > 0,
        "absolute_accuracy": best is not None and best <= config["normalized_rmse_limit"],
        "history_vs_zero": best is not None
        and zero is not None
        and zero > 1e-12
        and best <= 0.9 * zero,
        "memory_vs_memoryless": best is not None
        and baseline is not None
        and baseline > 1e-12
        and best <= 0.9 * baseline,
    }
    verdict = "ACCEPT" if all(checks.values()) else "RETRY"

    def paired(score):
        return [
            {
                **r,
                "normalized_joint_rmse": np.sqrt(r["normalized_mse"])
                if r["status"] == "ok"
                else None,
            }
            for r in score["horizons"][str(horizons[-1])]["rows"]
        ]

    uncertainty = paired_stratified_fault_line_bootstrap(
        paired(memoryless), paired(selected_score), "normalized_joint_rmse"
    )
    _write(result_dir / "paired_fault_line_uncertainty.json", uncertainty)
    access = (
        provider.access_audit()
        if provider is not None
        else {
            "ieee39_trajectory_reads": [],
            "canonical_val_ids_read": [],
            "test_ids_read": [],
            "audit_open_count": 0,
        }
    )
    _write(result_dir / "data_access_audit.json", access)
    summary = {
        "process_status": "PASS",
        "scientific_verdict": verdict,
        "promotion_decision": "KEEP_BASELINE",
        "selected": selected["name"],
        "train_event_count": len(train),
        "val_event_count": len(val),
        "independent_synthetic_event_count": len(population) if synthetic else 0,
        "ieee39_audit_event_count": 0,
        "evaluation_role": "independent_synthetic" if synthetic else "val",
        "selected_multi_horizon_normalized_rmse": best,
        "zero_multi_horizon_normalized_rmse": zero,
        "memoryless_multi_horizon_normalized_rmse": baseline,
        "checks": checks,
        "candidate_table": records,
        "energy_identity_max_abs_error": max(identity),
        "step_refinement_max_abs_state_difference": refinement,
        "checkpoint_replay_exact": True,
        "hidden_training_targets_read": False,
        "level_6_claimed": False,
    }
    _write(result_dir / "learning_summary.json", summary)
    sources = [
        "src/kron_reduce/models/nonlinear_latent.py",
        "src/kron_reduce/models/learned_latent.py",
        "src/kron_reduce/models/causal_latent.py",
        "src/kron_reduce/data/history_contract.py",
        "src/kron_reduce/data/causal_history.py",
        "src/kron_reduce/data/p2_roles.py",
        "src/kron_reduce/physics/coordinates.py",
        "src/kron_reduce/physics/units.py",
        "src/kron_reduce/evaluation/p2_metrics.py",
        "src/kron_reduce/workflows/ieee39_history_learning.py",
        "src/kron_reduce/workflows/nonlinear_history_learning.py",
    ]
    _write(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "source_commit": git_head(root),
            "working_tree_status": status_lines(root),
            "source_sha256": {path: sha256_file(root / path) for path in sources},
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": sha256_file(root / config["protocol_path"]),
            "data_policy": config["data_policy"],
        },
    )
    _write(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "source_commit": git_head(root),
            "scientific_verdict": verdict,
            "promotion_decision": "KEEP_BASELINE",
            "data_policy": config["data_policy"],
            "ieee39_audit_event_count": 0,
            "hidden_training_targets_read": False,
            "canonical_val_ids_read": [],
            "test_ids_read": [],
            "level_6_claimed": False,
            "claim_boundary": config["claim_boundary"],
        },
    )
    _write(
        result_dir / "artifact_hashes.json",
        {
            p.relative_to(result_dir).as_posix(): sha256_file(p)
            for p in sorted(result_dir.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    verify_artifact_directory(result_dir, strict=True)


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str):
    if stage != "mechanism_learning":
        raise ValueError("unregistered nonlinear learning stage")
    config_path = root / "configs" / "experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config["data_policy"] != experiment["data_policy"]:
        raise ValueError("experiment/config data policies differ")
    provider = None
    if config["data_policy"] == "canonical_F1_train_val_only":
        if config["fold"] != "F1" or config["seed"] != 324 or verify_data(root)["status"] != "PASS":
            raise RuntimeError("nonlinear F1 policy or data verification failed")
        dataset = root / "data" / config["dataset"]
        rows, allocation = build_p2_role_rows(dataset)
        if allocation["role_manifest_sha256"] != config["role_manifest_sha256"]:
            raise RuntimeError("frozen train/val allocation changed")
        schema = json.loads((dataset / "schema.json").read_text(encoding="utf-8"))
        mass = effective_mass(np.asarray(schema["M_prior"]))
        provider = TrainValNonlinearSamples(dataset, rows, mass, result_dir)
        _write(result_dir / "role_allocation.json", allocation)
    elif config["data_policy"] != "analytic_nonlinear_synthetic_only":
        raise ValueError("unregistered nonlinear data policy")
    execute_study(
        root=root,
        config=config,
        config_path=config_path,
        result_dir=result_dir,
        run_id=run_id,
        provider=provider,
    )
