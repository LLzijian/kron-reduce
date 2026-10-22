"""Five frozen A03 comparisons on kiki's original TRAIN, without source writes."""

from __future__ import annotations

import os
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

from kron_reduce.data.kiki_history import KikiHistorySamples, allocate_roles
from kron_reduce.evaluation.p2_metrics import paired_stratified_fault_line_bootstrap
from kron_reduce.models.learned_latent import event_weights, visible_scales
from kron_reduce.models.nonlinear_latent import (
    LearnedNonlinearModel,
    NonlinearFitSettings,
    fit_nonlinear_model,
)
from kron_reduce.physics.coordinates import coi_project
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines


def available_prefix(samples, horizon):
    return tuple(
        replace(
            s,
            forecast_time=s.forecast_time[: min(len(s.forecast_time), round(horizon / 0.01) + 1)],
            target_delta=s.target_delta[: min(len(s.forecast_time), round(horizon / 0.01) + 1)],
            target_omega=s.target_omega[: min(len(s.forecast_time), round(horizon / 0.01) + 1)],
        )
        for s in samples
    )


def predict_population(model, samples, *, zero_latent=False):
    output = {}
    for s in samples:
        try:
            value = model.predict(
                s.history, s.forecast_time, s.equilibrium_delta, zero_latent=zero_latent
            )
            output[s.event_id] = value[:, : 2 * model.closure.n_nodes]
        except (FloatingPointError, OverflowError, ValueError) as error:
            output[s.event_id] = f"{type(error).__name__}: {error}"
    return output


def score_predictions(predictions, samples, horizon, scale, *, require_full=0.0):
    population = [s for s in samples if s.forecast_time[-1] >= require_full - 1e-10]
    rows = []
    for s in population:
        count = min(len(s.forecast_time) - 1, round(horizon / 0.01))
        row = {
            "event_id": s.event_id,
            "source_dataset": s.source_dataset,
            "fault_line": s.fault_line,
            "role": s.role,
            "requested_horizon_s": horizon,
            "actual_duration_s": count * 0.01,
            "scored_real_future_points": count,
            "full_requested_horizon": count == round(horizon / 0.01),
        }
        prediction = predictions[s.event_id]
        if isinstance(prediction, str):
            rows.append({**row, "status": "failed", "failure_reason": prediction})
            continue
        error = (
            prediction[1 : count + 1]
            - np.c_[s.target_delta[1 : count + 1], s.target_omega[1 : count + 1]]
        )
        n = error.shape[1] // 2
        rows.append(
            {
                **row,
                "status": "ok",
                "normalized_mse": float(np.mean((error / scale) ** 2)),
                "delta_mse_rad2": float(np.mean(error[:, :n] ** 2)),
                "omega_mse_rad2_per_s2": float(np.mean(error[:, n:] ** 2)),
            }
        )
    failed = sum(r["status"] != "ok" for r in rows)
    result = {
        "event_count": len(population),
        "full_horizon_event_count": sum(r["full_requested_horizon"] for r in rows),
        "failed_event_count": failed,
        "rows": rows,
        "normalized_rmse": None,
        "score_includes_forecast_origin": False,
        "failures_imputed_as_zero": False,
        "weighting": "source_line_event_real_time_equal",
    }
    if population and not failed:
        weights = event_weights(population)
        result.update(
            normalized_rmse=float(np.sqrt(weights @ [r["normalized_mse"] for r in rows])),
            mean_event_normalized_rmse=float(
                weights @ np.sqrt([r["normalized_mse"] for r in rows])
            ),
            median_event_normalized_rmse=float(
                np.median(np.sqrt([r["normalized_mse"] for r in rows]))
            ),
            pooled_delta_rmse_rad=float(np.sqrt(weights @ [r["delta_mse_rad2"] for r in rows])),
            pooled_omega_rmse_rad_per_s=float(
                np.sqrt(weights @ [r["omega_mse_rad2_per_s2"] for r in rows])
            ),
        )
    return result


def evaluate(model, samples, reference_scale, *, zero_latent=False):
    predictions = predict_population(model, samples, zero_latent=zero_latent)
    sections = {}
    for name, scale, threshold in [
        ("available", model.target_scale, 0.0),
        ("complete3_common", model.target_scale, 3.0),
        ("reference_scale_complete3", reference_scale, 3.0),
    ]:
        scores = {
            str(h): score_predictions(predictions, samples, h, scale, require_full=threshold)
            for h in (1.0, 3.0)
        }
        vals = [s["normalized_rmse"] for s in scores.values()]
        sections[name] = {
            "horizons": scores,
            "multi_horizon_normalized_rmse": None
            if any(v is None for v in vals)
            else float(np.sqrt(np.mean(np.square(vals)))),
        }
    sections["complete1"] = score_predictions(
        predictions, samples, 1.0, model.target_scale, require_full=1.0
    )
    return sections


def execute_study(*, root, config, config_path, result_dir, run_id, provider):
    study_started = time.perf_counter()
    mass, train = provider.load("train")
    _, val = provider.load("val")
    if not train or not val:
        raise ValueError("registered train/val population is empty")
    reference_path = root / config["reference_checkpoint"]
    if sha256_file(reference_path) != config["reference_checkpoint_sha256"]:
        raise ValueError("frozen A03 reference checkpoint hash differs")
    reference = LearnedNonlinearModel.load(reference_path)
    if not np.allclose(reference.closure.mass, mass, rtol=0, atol=1e-12):
        raise ValueError("reference and kiki inertia/channel alignment differs")
    reference_scale = reference.target_scale.copy()
    train_scale = visible_scales(train)
    _write(
        result_dir / "train_scales.json",
        {
            "role": "train",
            "fitted_event_ids": sorted(s.event_id for s in train),
            "target_scale": train_scale.tolist(),
            "frozen_canonical_reference_scale": reference_scale.tolist(),
        },
    )
    source_paths = [
        "src/kron_reduce/models/nonlinear_latent.py",
        "src/kron_reduce/models/learned_latent.py",
        "src/kron_reduce/models/causal_latent.py",
        "src/kron_reduce/data/kiki_history.py",
        "src/kron_reduce/data/history_contract.py",
        "src/kron_reduce/physics/coordinates.py",
        "src/kron_reduce/workflows/kiki_history_learning.py",
        "src/kron_reduce/evaluation/p2_metrics.py",
    ]
    source_hashes = {p: sha256_file(root / p) for p in source_paths}
    _write(
        result_dir / "role_allocation.json",
        {"rows": provider.rows, "source_bindings": provider.bindings},
    )
    _write(
        result_dir / "planning_binding.json",
        {
            "planning_path": config["planning_path"],
            "planning_sha256": sha256_file(root / config["planning_path"]),
            "protocol_path": config["protocol_path"],
            "protocol_sha256": sha256_file(root / config["protocol_path"]),
        },
    )
    _write(result_dir / "data_quality.json", provider.access_audit())
    candidates, records, scores = {}, [], {}
    for candidate in config["candidates"]:
        name = candidate["name"]
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
        print(
            f"Fitting {name}: train={len(train)}, val={len(val)}, {settings.epochs} updates",
            flush=True,
        )
        started = time.perf_counter()
        model = fit_nonlinear_model(
            available_prefix(train, max(settings.horizons_s)),
            mass,
            settings,
            target_scale_samples=train,
            allow_partial_horizons=True,
        )
        if not np.array_equal(model.target_scale, train_scale):
            raise RuntimeError("candidate normalization differs from common train scale")
        checkpoint = result_dir / f"candidate-{name}.npz"
        model.save(checkpoint)
        restored = LearnedNonlinearModel.load(checkpoint)
        first = val[0]
        if not np.array_equal(
            model.predict(first.history, first.forecast_time, first.equilibrium_delta),
            restored.predict(first.history, first.forecast_time, first.equilibrium_delta),
        ):
            raise RuntimeError("checkpoint replay differs")
        train_score = evaluate(model, train, reference_scale)
        val_score = evaluate(model, val, reference_scale)
        _write(result_dir / f"candidate-{name}-train.json", train_score)
        _write(result_dir / f"candidate-{name}-val.json", val_score)
        row = {
            "name": name,
            "settings": candidate,
            "train_available_multi_nrmse": train_score["available"][
                "multi_horizon_normalized_rmse"
            ],
            "val_available_multi_nrmse": val_score["available"]["multi_horizon_normalized_rmse"],
            "val_complete3_multi_nrmse": val_score["complete3_common"][
                "multi_horizon_normalized_rmse"
            ],
            "parameter_count": model.fit_record["parameter_count"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "wall_clock_s": time.perf_counter() - started,
        }
        records.append(row)
        candidates[name] = model
        scores[name] = val_score
        print(
            f"Completed {name}: val available multi={row['val_available_multi_nrmse']}", flush=True
        )
    selected = min(
        records,
        key=lambda r: (
            float("inf")
            if r["val_available_multi_nrmse"] is None
            else r["val_available_multi_nrmse"],
            r["parameter_count"],
            r["name"],
        ),
    )
    if selected["val_available_multi_nrmse"] is None:
        raise RuntimeError("all candidates failed; do not select from failed scores")
    _write(
        result_dir / "selection_freeze.json",
        {
            "selected": selected,
            "candidate_table": records,
            "selection_role": "new_val_inside_source_train",
            "selection_metric": "available_multi_nrmse",
            "source_holdouts_authorized": False,
        },
    )
    chosen = candidates[selected["name"]]
    zero = evaluate(chosen, val, reference_scale, zero_latent=True)
    _write(result_dir / "val-selected-zero-latent.json", zero)
    # Same new-val events and scale, but no update to the old train-fitted model.
    transfer = replace(reference, target_scale=chosen.target_scale.copy())
    _write(
        result_dir / "val-frozen-canonical-transfer.json", evaluate(transfer, val, reference_scale)
    )
    rng = np.random.default_rng(config["seed"])
    identity = []
    refinement = []
    for name, model in candidates.items():
        for _ in range(25):
            state = np.r_[
                coi_project(rng.normal(size=len(mass)), mass),
                coi_project(rng.normal(size=len(mass)), mass),
                rng.normal(size=model.closure.latent_dimension),
            ]
            power = rng.normal(size=len(mass))
            power -= power.mean()
            identity.append(
                model.closure.energy_identity(state, power, val[0].equilibrium_delta)[
                    "identity_error"
                ]
            )
        s = next((s for s in val if s.forecast_time[-1] >= 3 - 1e-10), val[0])
        z = model.initialize(s.history, s.equilibrium_delta)[0]
        a = model.predict(s.history, s.forecast_time, s.equilibrium_delta)
        b = model.closure.rollout(
            s.history.delta[-1],
            s.history.omega[-1],
            z,
            s.forecast_time,
            s.history.constant_power,
            s.equilibrium_delta,
            max_step=0.005,
        )
        refinement.append({"name": name, "max_abs_state_difference": float(np.max(np.abs(a - b)))})
    numerical_pass = (
        max(identity) <= config["energy_identity_tolerance"]
        and max(r["max_abs_state_difference"] for r in refinement)
        <= config["step_refinement_tolerance"]
    )

    def paired(score):
        return [
            {
                **r,
                "normalized_joint_rmse": np.sqrt(r["normalized_mse"])
                if r["status"] == "ok"
                else None,
            }
            for r in score["available"]["horizons"]["3.0"]["rows"]
        ]

    _write(
        result_dir / "paired_fault_line_uncertainty.json",
        paired_stratified_fault_line_bootstrap(
            paired(scores["sine-memoryless"]),
            paired(scores[selected["name"]]),
            "normalized_joint_rmse",
        ),
    )
    best, base, zeros = [
        x["available"]["multi_horizon_normalized_rmse"]
        for x in (scores[selected["name"]], scores["sine-memoryless"], zero)
    ]
    checks = {
        "numerical": numerical_pass,
        "nonzero_memory_selected": chosen.settings.latent_dimension > 0,
        "absolute_accuracy": best <= 0.35,
        "history_vs_zero": zeros is not None and zeros > 1e-12 and best <= 0.9 * zeros,
        "memory_vs_memoryless": base is not None and base > 1e-12 and best <= 0.9 * base,
    }
    verdict = "ACCEPT" if all(checks.values()) else "RETRY"
    access = provider.access_audit()
    _write(result_dir / "data_access_audit.json", access)
    _write(
        result_dir / "learning_summary.json",
        {
            "process_status": "PASS",
            "scientific_verdict": verdict,
            "promotion_decision": "KEEP_BASELINE",
            "selected": selected["name"],
            "train_event_count": len(train),
            "val_event_count": len(val),
            "source_train_total": len(provider.rows),
            "insufficient_history_count": sum(not r["history_eligible"] for r in provider.coverage),
            "checks": checks,
            "selected_available_multi_nrmse": best,
            "zero_available_multi_nrmse": zeros,
            "memoryless_available_multi_nrmse": base,
            "candidate_table": records,
            "energy_identity_max_abs_error": max(identity),
            "step_refinement": refinement,
            "source_val_ids_read": [],
            "test_ids_read": [],
            "checkpoint_replay_exact": True,
            "transfer_scope": "exploratory unadapted reference; not a certified-envelope deployment",
            "cross_dataset_scores_are_not_paired_model_improvements": True,
        },
    )
    # Detect concurrent changes to raw data or numerical source instead of silently binding a new version.
    for source, binding in provider.bindings.items():
        if any(
            sha256_file(provider.source_root / source / key) != value
            for key, value in binding.items()
        ):
            raise RuntimeError("external source changed during experiment")
    if any(sha256_file(root / p) != value for p, value in source_hashes.items()):
        raise RuntimeError("numerical source changed during experiment")
    _write(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "source_commit": git_head(root),
            "working_tree_status": status_lines(root),
            "source_sha256": source_hashes,
            "config_sha256": sha256_file(config_path),
            "source_data_bindings": provider.bindings,
            "reference_checkpoint_sha256": sha256_file(reference_path),
            "source_files_written": False,
            "process_id": os.getpid(),
            "process_cpu_seconds_since_start": time.process_time(),
            "study_wall_seconds": time.perf_counter() - study_started,
            "child_processes_launched": False,
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
            "source_val_ids_read": [],
            "test_ids_read": [],
            "hidden_training_targets_read": False,
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


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != "kiki_rerun" or experiment["data_policy"] != "kiki_original_train_only":
        raise ValueError("unregistered kiki experiment")
    config_path = root / "configs" / "experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config["data_policy"] != experiment["data_policy"]:
        raise ValueError("configuration policy differs")
    source_root = Path(config["source_root"])
    rows, bindings = allocate_roles(source_root, config["sources"], config["seed"])
    provider = KikiHistorySamples(source_root, rows, bindings)
    execute_study(
        root=root,
        config=config,
        config_path=config_path,
        result_dir=result_dir,
        run_id=run_id,
        provider=provider,
    )
