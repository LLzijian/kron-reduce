"""Read-only train/val diagnosis of frozen history models; no audit reads."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data.p2_roles import build_p2_role_rows
from kron_reduce.models.causal_latent import UnobservableHistoryError
from kron_reduce.models.learned_latent import LearnedLatentModel, event_weights, history_features
from kron_reduce.physics.units import effective_mass
from kron_reduce.workflows.ieee39_history_learning import AuthorizedHistorySamples
from project_harness.data_check import verify_data
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def observer_score(model, samples) -> dict[str, Any]:
    losses, failed, conditions = [], [], []
    for sample in samples:
        try:
            prediction, estimate = model.closure.forecast_from_history(
                sample.history, sample.forecast_time
            )
            truth = np.concatenate((sample.target_delta, sample.target_omega), axis=1)
            losses.append(
                float(
                    np.mean(
                        (
                            (prediction[1:, : len(model.target_scale)] - truth[1:])
                            / model.target_scale
                        )
                        ** 2
                    )
                )
            )
            conditions.append(estimate.condition_number)
        except (UnobservableHistoryError, FloatingPointError, ValueError) as error:
            failed.append({"event_id": sample.event_id, "reason": str(error)})
    return {
        "normalized_rmse": None if failed else float(np.sqrt(event_weights(samples) @ losses)),
        "failed_event_count": len(failed),
        "failed_events": failed,
        "maximum_condition_number": max(conditions) if conditions else None,
    }


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if stage != "train_val_diagnosis":
        raise ValueError("unregistered diagnosis stage")
    diagnostic_path = root / "configs" / "experiments" / experiment["config"]
    diagnostic = tomllib.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (
        diagnostic["roles"] != ["train", "val"]
        or diagnostic["fit_calls"] != 0
        or diagnostic["audit_access"] != "forbidden"
    ):
        raise ValueError("diagnosis must remain train/val-only and read-only")
    source = root / "results" / diagnostic["source_run"]
    verify_artifact_directory(
        source, expected_manifest_sha256=diagnostic["source_artifact_manifest_sha256"], strict=True
    )
    config_path = root / "configs" / "experiments" / diagnostic["source_config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    binding = json.loads((source / "protocol_binding.json").read_text(encoding="utf-8"))
    if (
        sha256_file(config_path) != binding["config_sha256"]
        or verify_data(root)["status"] != "PASS"
    ):
        raise RuntimeError("source config or dataset differs from the frozen run")
    dataset = root / "data" / config["dataset"]
    rows, allocation = build_p2_role_rows(dataset)
    if allocation["role_manifest_sha256"] != config["role_manifest_sha256"]:
        raise RuntimeError("diagnosis role allocation changed")
    mass = effective_mass(
        np.asarray(json.loads((dataset / "schema.json").read_text(encoding="utf-8"))["M_prior"])
    )
    # Supplying no audit-role records makes further audit access impossible.
    reader = AuthorizedHistorySamples(
        dataset, [row for row in rows if row.role in {"fit", "selection"}], mass, result_dir
    )
    results = []
    freeze = json.loads((source / "selection_freeze.json").read_text(encoding="utf-8"))
    for duration in config["history_durations_s"]:
        _, train = reader(config, "train", duration)
        _, val = reader(config, "val", duration)
        for dimension in config["latent_dimensions"]:
            token = f"history-{duration:.2f}-latent-{dimension}"
            model = LearnedLatentModel.load(source / f"candidate-{token}.npz")
            train_score, val_score = model.score(train), model.score(val)
            recorded = next(row for row in freeze["candidate_table"] if row["token"] == token)
            if val_score["normalized_rmse"] != recorded["val_normalized_rmse"]:
                raise RuntimeError("frozen candidate val predictions did not replay exactly")
            train_features = np.stack(
                [
                    history_features(s.history, model.settings.frames) / model.feature_scale
                    for s in train
                ]
            )
            val_features = np.stack(
                [
                    history_features(s.history, model.settings.frames) / model.feature_scale
                    for s in val
                ]
            )
            _, singular, vh = np.linalg.svd(train_features, full_matrices=False)
            rank = int(np.sum(singular > diagnostic["feature_svd_relative_cutoff"] * singular[0]))
            supported = vh[:rank]
            residual = val_features - (val_features @ supported.T) @ supported
            outside = float(np.sum(residual**2) / np.sum(val_features**2))
            row = {
                "token": token,
                "history_duration_s": duration,
                "latent_dimension": dimension,
                "train_normalized_rmse": train_score["normalized_rmse"],
                "val_normalized_rmse": val_score["normalized_rmse"],
                "val_zero_initialization": model.score(val, zero_latent=True)["normalized_rmse"],
                "train_feature_rank": rank,
                "feature_dimension": train_features.shape[1],
                "val_feature_energy_outside_train_span_fraction": outside,
            }
            if dimension:
                row["val_model_based_observer"] = observer_score(model, val)
            results.append(row)
    access = reader.access_audit()
    if access["audit_open_count"] != 0 or {row["role"] for row in access["accesses"]} != {
        "train",
        "val",
    }:
        raise RuntimeError("diagnosis read a locked role")
    _write(result_dir / "candidate_diagnosis.json", results)
    _write(result_dir / "data_access_audit.json", access)
    _write(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "source_commit": git_head(root),
            "source_run": source.name,
            "source_artifact_manifest_sha256": diagnostic["source_artifact_manifest_sha256"],
            "diagnostic_source_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(diagnostic_path),
            "fit_calls": 0,
        },
    )
    _write(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "source_run": source.name,
            "fit_calls": 0,
            "audit_open_count": 0,
            "claim_boundary": "Frozen train/val replay and observer/encoder diagnosis; no refit, new model selection, or repeated internal audit.",
        },
    )
    _write(
        result_dir / "artifact_hashes.json",
        {
            path.relative_to(result_dir).as_posix(): sha256_file(path)
            for path in sorted(result_dir.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )
    verify_artifact_directory(result_dir, strict=True)
