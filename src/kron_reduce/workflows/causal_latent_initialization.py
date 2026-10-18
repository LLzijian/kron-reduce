"""Registered analytic evidence for hidden-state ambiguity and initialization."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm

from kron_reduce.models.causal_latent import (
    CausalStateHistory,
    PassiveLatentClosure,
    UnobservableHistoryError,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _errors(prediction: np.ndarray, truth: np.ndarray, n: int) -> dict[str, float]:
    return {
        "delta_rmse_rad": float(np.sqrt(np.mean((prediction[:, :n] - truth[:, :n]) ** 2))),
        "omega_rmse_rad_per_s": float(
            np.sqrt(np.mean((prediction[:, n : 2 * n] - truth[:, n : 2 * n]) ** 2))
        ),
    }


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if stage != "synthetic_audit" or experiment.get("data_policy") != "analytic_synthetic_only":
        raise ValueError("latent initialization supports only the registered analytic audit")
    config_path = root / "configs" / "experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config["data_policy"] != "analytic_synthetic_only":
        raise ValueError("synthetic data policy mismatch")
    n, r = int(config["n_nodes"]), int(config["latent_dimension"])
    if n != 10 or r != 2:
        raise ValueError("the registered counterexample uses ten nodes and two latent states")
    laplacian = np.eye(n) - np.ones((n, n)) / n
    angular = float(config["latent_angular_rate"])
    model = PassiveLatentClosure(
        np.linspace(config["mass_min"], config["mass_max"], n),
        config["stiffness_gain"] * laplacian,
        config["damping_gain"] * laplacian,
        config["port_gain"] * coi_basis(np.ones(n))[:, :r],
        np.array([[0.0, -angular], [angular, 0.0]]),
        np.diag(config["latent_decay"]),
    )
    model.save(result_dir / "known_closure.npz")
    model = PassiveLatentClosure.load(result_dir / "known_closure.npz")
    delta0 = coi_project(np.linspace(-0.03, 0.03, n), model.mass)
    omega0, power = np.zeros(n), np.zeros(n)
    latent0 = np.asarray(config["latent_initial"])
    max_step = float(config["maximum_rk4_step_s"])
    observer_options = {
        "maximum_condition_number": config["maximum_observation_condition_number"],
        "minimum_singular_value": config["minimum_observation_singular_value"],
        "delta_scale": config["observer_delta_scale_rad"],
        "omega_scale": config["observer_omega_scale_rad_per_s"],
    }
    clock = np.linspace(0.0, config["forecast_duration_s"], config["forecast_samples"])
    initial_plus = np.concatenate((delta0, omega0, latent0))
    initial_minus = np.concatenate((delta0, omega0, -latent0))
    transitions = [expm(model.generator() * t) for t in clock]
    plus = np.stack([matrix @ initial_plus for matrix in transitions])
    minus = np.stack([matrix @ initial_minus for matrix in transitions])
    difference = plus - minus
    zero_at_clearing = model.rollout(delta0, omega0, np.zeros(r), clock, power, max_step=max_step)
    ambiguity = {
        "same_complete_allowed_clearing_inputs": bool(
            np.array_equal(plus[0, : 2 * n], minus[0, : 2 * n])
        ),
        "different_unobserved_initial_states": True,
        "delta_pair_separation_rmse_rad": float(np.sqrt(np.mean(difference[:, :n] ** 2))),
        "omega_pair_separation_rmse_rad_per_s": float(
            np.sqrt(np.mean(difference[:, n : 2 * n] ** 2))
        ),
        "delta_deterministic_pair_rmse_lower_bound_rad": float(
            0.5 * np.sqrt(np.mean(difference[:, :n] ** 2))
        ),
        "omega_deterministic_pair_rmse_lower_bound_rad_per_s": float(
            0.5 * np.sqrt(np.mean(difference[:, n : 2 * n] ** 2))
        ),
        "bound_population": "two_equiprobable_initial_hidden_states_with_identical_allowed_inputs",
        "bound_definition": "minimum_root_mean_squared_error_over_both_events_all_times_and_nodes_per_channel",
        "observation_duration_s": 0.0,
        "forecast_origin_after_clearing_s": 0.0,
        "forecast_duration_s": float(config["forecast_duration_s"]),
        "zero_initialized_memory_pair_midpoint_max_abs_error": float(
            np.max(np.abs(zero_at_clearing - 0.5 * (plus + minus)))
        ),
    }
    observation_clock = np.linspace(
        0.0, config["observation_duration_s"], config["observation_samples"]
    )
    observations = np.stack([expm(model.generator() * t) @ initial_plus for t in observation_clock])
    history = CausalStateHistory(
        observation_clock - observation_clock[-1],
        observations[:, :n],
        observations[:, n : 2 * n],
        power,
    )
    truth_at_origin = observations[-1]
    truth = np.stack([matrix @ truth_at_origin for matrix in transitions])
    prediction, estimate = model.forecast_from_history(
        history, clock, max_step=max_step, **observer_options
    )
    zero_memory = model.rollout(
        history.delta[-1], history.omega[-1], np.zeros(r), clock, power, max_step=max_step
    )
    no_memory = PassiveLatentClosure(
        model.mass,
        model.stiffness,
        model.damping,
        np.zeros((n, 0)),
        np.zeros((0, 0)),
        np.zeros((0, 0)),
    ).rollout(history.delta[-1], history.omega[-1], np.zeros(0), clock, power, max_step=max_step)
    oracle = model.rollout(
        truth_at_origin[:n],
        truth_at_origin[n : 2 * n],
        truth_at_origin[2 * n :],
        clock,
        power,
        max_step=max_step,
    )
    restored = PassiveLatentClosure.load(result_dir / "known_closure.npz")
    replay = restored.forecast_from_history(history, clock, max_step=max_step, **observer_options)[
        0
    ]
    recovery_error = float(np.max(np.abs(estimate.latent_at_origin - truth_at_origin[2 * n :])))
    oracle_error = float(np.max(np.abs(prediction - truth)))
    ablations = {
        "parameters": "known_synthetic_equations_no_parameter_learning",
        "population": "positive_hidden_initial_state_same_history_for_all_candidates",
        "observation_channels": ["delta", "omega"],
        "observation_sample_count": int(config["observation_samples"]),
        "observation_duration_s": float(config["observation_duration_s"]),
        "forecast_origin_after_clearing_s": float(config["observation_duration_s"]),
        "forecast_duration_s": float(config["forecast_duration_s"]),
        "score_includes_forecast_origin": True,
        "memoryless": _errors(no_memory, truth, n),
        "latent_zero_initialized": _errors(zero_memory, truth, n),
        "latent_history_initialized": _errors(prediction, truth, n),
        "true_hidden_state_oracle": _errors(oracle, truth, n),
        "oracle_access": "synthetic_comparator_only_not_observer_input",
        "hidden_recovery_max_abs_error": recovery_error,
        "checkpoint_replay_exact": bool(np.array_equal(prediction, replay)),
        "observer": {**asdict(estimate), "latent_at_origin": estimate.latent_at_origin.tolist()},
    }
    disconnected = model.coupling.copy()
    disconnected[:, 1] = 0.0
    unobservable = PassiveLatentClosure(
        model.mass,
        model.stiffness,
        model.damping,
        disconnected,
        np.zeros((r, r)),
        np.eye(r),
    )
    refusal = False
    refusal_message = ""
    try:
        unobservable.estimate_latent(history, **observer_options)
    except UnobservableHistoryError as error:
        refusal, refusal_message = True, str(error)
    rng = np.random.default_rng(int(config["seed"]))
    noisy = CausalStateHistory(
        history.time,
        history.delta
        + coi_project(rng.normal(0.0, config["noise_delta_rad"], history.delta.shape), model.mass),
        history.omega
        + coi_project(
            rng.normal(0.0, config["noise_omega_rad_per_s"], history.omega.shape), model.mass
        ),
        power,
    )
    noisy_prediction, noisy_estimate = model.forecast_from_history(
        noisy, clock, max_step=max_step, **observer_options
    )
    noise_audit = {
        "seed": int(config["seed"]),
        "delta_noise_std_before_coi_rad": config["noise_delta_rad"],
        "omega_noise_std_before_coi_rad_per_s": config["noise_omega_rad_per_s"],
        "latent_recovery_max_abs_error": float(
            np.max(np.abs(noisy_estimate.latent_at_origin - truth_at_origin[2 * n :]))
        ),
        **_errors(noisy_prediction, truth, n),
        "role": "sensitivity_no_model_selection",
    }
    identity_errors = []
    for _ in range(100):
        state = np.concatenate(
            (
                coi_project(rng.normal(size=n), model.mass),
                coi_project(rng.normal(size=n), model.mass),
                rng.normal(size=r),
            )
        )
        audit_power = rng.normal(size=n)
        audit_power -= audit_power.mean()
        identity_errors.append(model.energy_identity(state, audit_power)["identity_error"])
    energy_error = float(max(identity_errors))
    coi_drift = float(
        max(
            np.max(np.abs(prediction[:, :n] @ model.mass)),
            np.max(np.abs(prediction[:, n : 2 * n] @ model.mass)),
        )
    )
    passed = {
        "ambiguity_demonstrated": bool(
            ambiguity["same_complete_allowed_clearing_inputs"]
            and ambiguity["omega_pair_separation_rmse_rad_per_s"] > 0
        ),
        "latent_recovery_passed": recovery_error < config["recovery_tolerance"],
        "free_forecast_passed": oracle_error < config["forecast_tolerance"],
        "unobservable_refusal_passed": refusal,
        "energy_identity_passed": energy_error < config["energy_identity_tolerance"],
        "coi_passed": coi_drift < config["coi_tolerance"],
        "checkpoint_replay_passed": ablations["checkpoint_replay_exact"],
    }
    _write_json(result_dir / "clearing_state_ambiguity.json", ambiguity)
    _write_json(result_dir / "history_initialization_ablations.json", ablations)
    _write_json(result_dir / "noise_sensitivity.json", noise_audit)
    _write_json(
        result_dir / "structural_audit.json",
        {
            "energy_identity_max_abs_error": energy_error,
            "random_states_checked": 100,
            "maximum_coi_drift": coi_drift,
            "unobservable_refusal": refusal_message,
            "maximum_free_forecast_error_against_matrix_exponential": oracle_error,
            "initialization_audit": "observation_equations_use_known_model_not_true_hidden_state",
            "level_6_claimed": False,
            **passed,
        },
    )
    np.savez_compressed(
        result_dir / "trajectories.npz",
        forecast_time=clock,
        observation_time_after_clearing=observation_clock,
        clearing_plus=plus,
        clearing_minus=minus,
        observations=observations,
        forecast_truth=truth,
        forecast_memoryless=no_memory,
        forecast_latent_zero=zero_memory,
        forecast_history=prediction,
        forecast_oracle=oracle,
    )
    source_paths = [
        "src/kron_reduce/data/history_contract.py",
        "src/kron_reduce/models/causal_latent.py",
        "src/kron_reduce/workflows/causal_latent_initialization.py",
        "src/kron_reduce/physics/coordinates.py",
        "tests/test_causal_latent.py",
        "configs/experiments/causal_latent_initialization_synthetic_v1.toml",
    ]
    _write_json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "source_commit": git_head(root),
            "working_tree_status": status_lines(root),
            "source_sha256": {path: sha256_file(root / path) for path in source_paths},
            "config_sha256": sha256_file(config_path),
            "data_policy": "analytic_synthetic_only",
            "train_ids_read": [],
            "val_ids_read": [],
            "test_ids_read": [],
            "ieee39_trajectory_reads": [],
            "future_state_used_by_observer": False,
            "source_routing": False,
        },
    )
    if not all(passed.values()):
        raise RuntimeError(f"synthetic latent audit failed: {passed}")
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "source_commit": git_head(root),
            "data_policy": "analytic_synthetic_only",
            "ieee39_trajectory_reads": [],
            **passed,
            "claim_boundary": "Known-equation synthetic ambiguity, causal hidden-state recovery, and passive storage identity; learned parameters and IEEE39 accuracy require separate evidence.",
        },
    )
    hashes = {
        path.relative_to(result_dir).as_posix(): sha256_file(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(result_dir / "artifact_hashes.json", hashes)
    verify_artifact_directory(result_dir, strict=True)
