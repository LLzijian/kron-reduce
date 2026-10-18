"""Visible-only synthetic identification with a frozen independent audit."""

from __future__ import annotations

import json
import time
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm

from kron_reduce.evaluation.p2_metrics import paired_stratified_fault_line_bootstrap
from kron_reduce.models.causal_latent import CausalStateHistory, PassiveLatentClosure
from kron_reduce.models.learned_latent import (
    LatentFitSettings,
    LatentLearningSample,
    LearnedLatentModel,
    fit_latent_model,
    refit_latent_model,
    visible_scales,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def make_visible_samples(
    config: dict[str, Any], role: str, history_duration: float
) -> tuple[np.ndarray, tuple[LatentLearningSample, ...]]:
    """Generate visible observations; truth parameters never enter the learner."""
    if role not in {"train", "val", "audit"}:
        raise ValueError("unregistered synthetic role")
    truth, n = config["truth"], int(config["n_nodes"])
    mass = np.linspace(truth["mass_min"], truth["mass_max"], n)
    laplacian = np.eye(n) - np.ones((n, n)) / n
    angular = truth["latent_angular_rate"]
    model = PassiveLatentClosure(
        mass,
        truth["stiffness_gain"] * laplacian,
        truth["damping_gain"] * laplacian,
        truth["port_gain"] * coi_basis(np.ones(n))[:, :2],
        np.array([[0.0, -angular], [angular, 0.0]]),
        np.diag(truth["latent_decay"]),
    )
    dt = float(config["history_step_s"])
    offset = round(history_duration / dt)
    forecast = (
        np.arange(round(config["forecast_duration_s"] / config["forecast_output_step_s"]) + 1)
        * config["forecast_output_step_s"]
    )
    times = np.arange(offset + 1) * dt
    sample_times = np.r_[times, history_duration + forecast[1:]]
    # Include constant forcing with one affine coordinate.
    size = 2 * n + 2
    affine = np.zeros((size + n, size + n))
    affine[:size, :size] = model.generator()
    affine[n : 2 * n, size:] = np.diag(1.0 / mass)
    transitions = np.stack([expm(affine * t) for t in sample_times])
    role_offset = {"train": 1000, "val": 2000, "audit": 3000}[role]
    line_offset = {"train": 0, "val": 100, "audit": 200}[role]
    rng = np.random.default_rng(int(config["seed"]) + role_offset)
    samples = []
    for source_index, source in enumerate(("synthetic_A", "synthetic_B")):
        amplitude = (0.8, 1.2)[source_index]
        for line in range(int(config[f"{role}_lines_per_source"])):
            for pair in range(int(config[f"{role}_events_per_line"]) // 2):
                q0 = coi_project(rng.normal(0, truth["initial_delta_std_rad"] * amplitude, n), mass)
                v0 = coi_project(
                    rng.normal(0, truth["initial_omega_std_rad_per_s"] * amplitude, n), mass
                )
                hidden = rng.normal(0, truth["initial_latent_std"] * amplitude, 2)
                power = rng.normal(0, truth["constant_power_std"] * amplitude, n)
                power -= power.mean()
                for sign in (-1, 1):
                    state = np.concatenate((q0, v0, sign * hidden, power))
                    visible = np.einsum("tij,j->ti", transitions[:, : 2 * n, :], state)
                    history = CausalStateHistory(
                        times - times[-1],
                        visible[: offset + 1, :n],
                        visible[: offset + 1, n : 2 * n],
                        power,
                    )
                    target = np.vstack((visible[offset], visible[offset + 1 :]))
                    samples.append(
                        LatentLearningSample(
                            f"{role}-{source}-{line}-{pair}-{sign}",
                            source,
                            f"line-{line_offset + line}",
                            role,
                            history,
                            forecast,
                            target[:, :n],
                            target[:, n : 2 * n],
                        )
                    )
    return mass, tuple(samples)


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if stage != "learning_audit" or experiment.get("data_policy") != "analytic_synthetic_only":
        raise ValueError("unregistered learned-latent execution")
    config_path = root / "configs" / "experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    execute_learning_study(
        root=root,
        config_path=config_path,
        config=config,
        result_dir=result_dir,
        run_id=run_id,
        sample_provider=make_visible_samples,
        data_policy="analytic_synthetic_only",
    )


def execute_learning_study(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
    sample_provider: Callable,
    data_policy: str,
    extra_source_paths: tuple[str, ...] = (),
) -> None:
    """Shared scientific lifecycle; callers supply only registered role readers."""
    cache = {}
    for duration in config["history_durations_s"]:
        mass, train = sample_provider(config, "train", float(duration))
        _, val = sample_provider(config, "val", float(duration))
        cache[duration] = (train, val)
    scale = visible_scales(cache[config["scale_reference_history_s"]][0])
    _write_json(
        result_dir / "train_scales.json",
        {
            "values": scale.tolist(),
            "role": "train",
            "reference_history_s": config["scale_reference_history_s"],
        },
    )
    records = []
    candidates = {}
    for duration in config["history_durations_s"]:
        train, val = cache[duration]
        for dimension in config["latent_dimensions"]:
            started = time.perf_counter()
            token = f"history-{duration:.2f}-latent-{dimension}"
            settings = LatentFitSettings(
                latent_dimension=int(dimension),
                frames=int(config["encoder_frames"]),
                epochs=int(config["epochs"]),
                learning_rate=float(config["learning_rate"]),
                encoder_penalty=float(config["encoder_penalty"]),
                seed=int(config["seed"]),
                maximum_step_s=float(config["maximum_integrator_step_s"]),
            )
            print(f"Fitting {token}: {settings.epochs} epochs", flush=True)
            model = fit_latent_model(train, mass, settings, target_scale=scale)
            score = model.score(val)
            model.save(result_dir / f"candidate-{token}.npz")
            row = {
                "token": token,
                "history_duration_s": duration,
                "latent_dimension": dimension,
                "val_normalized_rmse": score["normalized_rmse"],
                "fit_record": model.fit_record,
                "parameter_count": model.fit_record["rhs_parameter_count"]
                + model.fit_record["encoder_parameter_count"],
                "wall_clock_seconds": time.perf_counter() - started,
            }
            records.append(row)
            candidates[token] = model
            _write_json(result_dir / f"candidate-{token}-val.json", score)
            print(f"Completed {token}: val normalized RMSE={score['normalized_rmse']}", flush=True)
    selected = min(
        records,
        key=lambda row: (
            float("inf") if row["val_normalized_rmse"] is None else row["val_normalized_rmse"],
            row["parameter_count"],
            row["history_duration_s"],
        ),
    )
    duration = selected["history_duration_s"]
    if selected["val_normalized_rmse"] is None:
        _write_json(result_dir / "failed_candidate_table.json", records)
        raise RuntimeError("all latent candidates failed on val; audit remains locked")
    train, val = cache[duration]
    settings = candidates[selected["token"]].settings
    final = refit_latent_model(train, val, mass, settings, scale)
    final.save(result_dir / "selected-final.npz")
    baseline_settings = candidates[f"history-{duration:.2f}-latent-0"].settings
    baseline = refit_latent_model(train, val, mass, baseline_settings, scale)
    baseline.save(result_dir / "memoryless-final.npz")
    _write_json(
        result_dir / "selection_freeze.json",
        {
            "selection_role": "val",
            "selection_priority": ["val_normalized_rmse", "parameter_count", "history_duration_s"],
            "selected": selected,
            "candidate_table": records,
            "final_checkpoint_sha256": sha256_file(result_dir / "selected-final.npz"),
            "memoryless_checkpoint_sha256": sha256_file(result_dir / "memoryless-final.npz"),
            "train_scale_sha256": sha256_file(result_dir / "train_scales.json"),
            "audit_generated": False,
        },
    )
    freeze_hash = sha256_file(result_dir / "selection_freeze.json")
    final = LearnedLatentModel.load(result_dir / "selected-final.npz")
    baseline = LearnedLatentModel.load(result_dir / "memoryless-final.npz")
    _, audit = sample_provider(config, "audit", float(duration))
    print("Checkpoints frozen; evaluating independent audit trajectories", flush=True)
    learned_score = final.score(audit)
    zero_score = final.score(audit, zero_latent=True)
    baseline_score = baseline.score(audit)
    for name, value in (
        ("learned_history", learned_score),
        ("zero_initialization", zero_score),
        ("memoryless", baseline_score),
    ):
        _write_json(result_dir / f"audit-{name}.json", value)
    for horizon in config.get("report_horizons_s", []):
        horizon_samples = []
        for sample in audit:
            prefix = int(
                np.searchsorted(sample.forecast_time, float(horizon) + 1e-10, side="right")
            )
            if abs(sample.forecast_time[prefix - 1] - horizon) > 1e-8:
                raise ValueError("audit target clock does not cover a registered horizon")
            horizon_samples.append(
                replace(
                    sample,
                    forecast_time=sample.forecast_time[:prefix],
                    target_delta=sample.target_delta[:prefix],
                    target_omega=sample.target_omega[:prefix],
                )
            )
        _write_json(
            result_dir / f"audit-horizon-{horizon}s.json",
            {
                "horizon_after_forecast_origin_s": horizon,
                "learned_history": final.score(horizon_samples),
                "zero_initialization": final.score(horizon_samples, zero_latent=True),
                "memoryless": baseline.score(horizon_samples),
            },
        )
    replay = LearnedLatentModel.load(result_dir / "selected-final.npz").predict(
        audit[0].history, audit[0].forecast_time[:2]
    )
    original = final.predict(audit[0].history, audit[0].forecast_time[:2])
    if not np.array_equal(replay, original):
        raise RuntimeError("learned checkpoint replay changed predictions")
    rng = np.random.default_rng(int(config["seed"]))
    errors = []
    for _ in range(100):
        state = np.r_[
            coi_project(rng.normal(size=len(mass)), mass),
            coi_project(rng.normal(size=len(mass)), mass),
            rng.normal(size=final.closure.latent_dimension),
        ]
        power = rng.normal(size=len(mass))
        power -= power.mean()
        errors.append(final.closure.energy_identity(state, power)["identity_error"])
    if max(errors) > config["energy_identity_tolerance"]:
        raise RuntimeError("learned model violated its passive identity")
    learned, zero, memoryless = (
        score["normalized_rmse"] for score in (learned_score, zero_score, baseline_score)
    )
    checks = {
        "absolute_accuracy": learned is not None
        and learned <= config["audit_normalized_rmse_limit"],
        "history_initialization_effect": settings.latent_dimension > 0
        and learned is not None
        and zero is not None
        and zero > 1e-12
        and learned <= config["audit_history_to_zero_error_ratio_limit"] * zero,
        "memory_effect": learned is not None
        and memoryless is not None
        and memoryless > 1e-12
        and learned <= config["audit_history_to_memoryless_error_ratio_limit"] * memoryless,
    }
    paired_rows = lambda rows: [
        {
            **row,
            "status": row.get("status", "ok"),
            "normalized_joint_rmse": float(np.sqrt(row["normalized_mse"]))
            if row["normalized_mse"] is not None
            else None,
        }
        for row in rows
    ]
    uncertainty = paired_stratified_fault_line_bootstrap(
        paired_rows(baseline_score["rows"]),
        paired_rows(learned_score["rows"]),
        "normalized_joint_rmse",
    )
    _write_json(result_dir / "paired_fault_line_uncertainty.json", uncertainty)
    verdict = "ACCEPT" if all(checks.values()) else "RETRY"
    _write_json(
        result_dir / "learning_summary.json",
        {
            "process_status": "PASS",
            "scientific_verdict": verdict,
            "promotion_decision": "KEEP_BASELINE",
            "selected_history_duration_s": duration,
            "selected_latent_dimension": settings.latent_dimension,
            "train_event_count": len(train),
            "val_event_count": len(val),
            "audit_event_count": len(audit),
            "forecast_duration_s": config.get(
                "audit_forecast_duration_s", config["forecast_duration_s"]
            ),
            "learned_history_normalized_rmse": learned,
            "zero_initialization_normalized_rmse": zero,
            "memoryless_normalized_rmse": memoryless,
            "checks": checks,
            "energy_identity_max_abs_error": max(errors),
            "checkpoint_replay_exact": True,
            "hidden_training_targets_read": False,
            "audit_opened_after_checkpoint_freeze": True,
            "selection_freeze_sha256": freeze_hash,
            "equation_family": "learned_linear_passive_dynamics_shared_linear_history_encoder",
        },
    )
    source_paths = [
        "src/kron_reduce/models/learned_latent.py",
        "src/kron_reduce/models/causal_latent.py",
        "src/kron_reduce/data/history_contract.py",
        "src/kron_reduce/workflows/learned_causal_latent.py",
        *extra_source_paths,
    ]
    access_audit = (
        sample_provider.access_audit()
        if hasattr(sample_provider, "access_audit")
        else {"ieee39_trajectory_reads": [], "canonical_val_ids_read": [], "test_ids_read": []}
    )
    _write_json(result_dir / "data_access_audit.json", access_audit)
    _write_json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "source_commit": git_head(root),
            "working_tree_status": status_lines(root),
            "source_sha256": {path: sha256_file(root / path) for path in source_paths},
            "config_sha256": sha256_file(config_path),
            "data_policy": data_policy,
            "data_access_audit_sha256": sha256_file(result_dir / "data_access_audit.json"),
            "canonical_val_ids_read": [],
            "test_ids_read": [],
        },
    )
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "scientific_verdict": verdict,
            "promotion_decision": "KEEP_BASELINE",
            "source_commit": git_head(root),
            "data_policy": data_policy,
            "hidden_training_targets_read": False,
            "audit_opened_after_checkpoint_freeze": True,
            "ieee39_trajectory_reads": access_audit["ieee39_trajectory_reads"],
            "canonical_val_ids_read": access_audit.get("canonical_val_ids_read", []),
            "test_ids_read": access_audit.get("test_ids_read", []),
            "other_folds_read": access_audit.get("other_folds_read", []),
            "future_pm_reads": access_audit.get("future_pm_reads", []),
            "claim_boundary": config.get(
                "claim_boundary",
                "Visible-only learning and shared initialization evaluated on independent synthetic trajectories; IEEE39 and nonlinear identification require separate evidence.",
            ),
        },
    )
    _write_json(
        result_dir / "artifact_hashes.json",
        {
            path.relative_to(result_dir).as_posix(): sha256_file(path)
            for path in sorted(result_dir.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )
    verify_artifact_directory(result_dir, strict=True)
