"""Saved whole-module interventions and read-only internal-audit evaluation."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest, validate_p1_samples
from kron_reduce.evaluation.p1_paired_statistics import summarize_paired_rows
from kron_reduce.models.base import sample_rollout_inputs
from kron_reduce.models.p1_calibration import P1CalibrationSettings, calibrate_loss_pair
from kron_reduce.models.p1_module_counterfactuals import (
    module_physical_traces,
    summarize_physical_trace,
    zero_module_instances,
)
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel
from kron_reduce.workflows.p1_five_arm_execution import (
    _checked_file,
    _fit_fresh_core,
    _read_json,
    _write_json,
    file_sha256,
    score_trajectory_role,
    verify_five_arm_bundle,
)


def _files(directory: Path) -> dict[str, Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("artifact directory must be regular")
    entries = list(directory.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("symlinks are not allowed in artifact trees")
    return {path.relative_to(directory).as_posix(): path for path in entries if path.is_file()}


def _seal(directory: Path) -> str:
    files = _files(directory)
    if "artifact_hashes.json" in files or "FAILED.json" in files:
        raise ValueError("cannot seal an already frozen or failed directory")
    _write_json(directory / "artifact_hashes.json", {name: file_sha256(path) for name, path in sorted(files.items())})
    return file_sha256(directory / "artifact_hashes.json")


def verify_counterfactual_artifacts(directory: Path, expected_manifest_sha256: str) -> dict[str, str]:
    directory = Path(directory)
    _checked_file(directory / "artifact_hashes.json", expected_manifest_sha256)
    hashes = _read_json(directory / "artifact_hashes.json")
    if not isinstance(hashes, dict) or not hashes or "FAILED.json" in hashes:
        raise ValueError("invalid artifact manifest")
    for name in hashes:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or path.as_posix() != name:
            raise ValueError("artifact path must be canonical and relative")
    if set(_files(directory)) != set(hashes) | {"artifact_hashes.json"}:
        raise ValueError("artifact tree changed or incomplete")
    for name, digest in hashes.items():
        _checked_file(directory / name, digest)
    return hashes


def _new_output(directory: Path, protected: Sequence[Path]) -> None:
    if any(directory.resolve().is_relative_to(path.resolve()) for path in protected):
        raise ValueError("output must not modify a frozen input artifact")
    directory.mkdir(parents=True, exist_ok=False)


def _save_model(model: PrunableEnergyModel, path: Path) -> None:
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    model.save(path)
    _read_json(path.with_suffix(".json"))


def _variants(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = sorted(row["instance_id"] for row in instances)
    return [{"variant_id": f"single_{index:03d}", "kind": "single", "removed_instance_ids": [name]} for index, name in enumerate(ids)] + [{"variant_id": "joint", "kind": "joint", "removed_instance_ids": ids}]


def fit_and_save_counterfactuals(
    base_directory: Path,
    base_manifest_sha256: str,
    manifest: P1RoleManifest,
    fit_samples: Sequence[dict[str, Any]],
    selection_samples: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Every deletion fits all remaining coefficients from scratch on fit only."""
    validate_p1_samples(fit_samples, manifest, "fit")
    validate_p1_samples(selection_samples, manifest, "selection")
    base_directory, output_dir = Path(base_directory), Path(output_dir)
    base = verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
    spec = _read_json(base_directory / "core3.json")
    lineage = _read_json(base_directory / "fit_lineage.json")
    settings = P1CalibrationSettings(**lineage["calibrations"][2]["common_design"]["solver"])
    core = PrunableEnergyModel.load(base_directory / "core3.npz")
    variants = _variants(spec["active_block_instances"])
    _new_output(output_dir, (base_directory,))
    plan = {
        "schema": "kron_reduce.p1_counterfactual_models.v1", "reference_arm": 3,
        "base_manifest_sha256": base_manifest_sha256, "base_core_sha256": base["arms"][2]["checkpoint_hash"],
        "role_manifest_sha256": manifest.source_sha256, "common_contract_hash": base["common_contract_hash"],
        "fit_scales": base["fit_scales"], "variants": variants,
        "joint_refit_name": "B2_structure_balanced_loss_not_arm1",
        "scientific_verdict": None, "promotion_decision": "KEEP_BASELINE",
    }
    _write_json(output_dir / "plan.json", plan)
    active = None
    try:
        for variant in variants:
            active = variant["variant_id"]
            removed = variant["removed_instance_ids"]
            zero = zero_module_instances(core, removed)
            _save_model(zero, output_dir / f"{active}_zero.npz")
            reduced = copy.deepcopy(spec)
            reduced["active_block_instances"] = [row for row in spec["active_block_instances"] if row["instance_id"] not in removed]
            fitted, fresh_lineage = _fit_fresh_core(reduced, fit_samples, selection_samples, manifest, b2=False)
            if fitted.validation_scales != base["fit_scales"]:
                raise ValueError("counterfactual fit scales differ from frozen reference")
            ((refit, calibration),) = calibrate_loss_pair(fitted, fit_samples, manifest, settings=settings, loss_modes=(True,))
            if [row.instance_id for row in refit.active_instances] != [row["instance_id"] for row in reduced["active_block_instances"]]:
                raise ValueError("delete/refit changed more than the registered module set")
            _save_model(refit, output_dir / f"{active}_refit.npz")
            _write_json(output_dir / f"{active}_lineage.json", {
                **variant, "fresh_fit": fresh_lineage, "calibration": calibration,
                "remaining_structure": reduced["active_block_instances"],
                "reference_arrays_used_to_initialize_refit": False,
                "remaining_coefficients_fitted_on": "fit", "hyperparameter_selection_on": "selection",
                "independent_support_verdict": "NOT_EVALUATED",
            })
        verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
        digest = _seal(output_dir)
        return {"directory": str(output_dir.resolve()), "artifact_manifest_sha256": digest, "plan": plan}
    except Exception as error:
        _write_json(output_dir / "FAILED.json", {"variant_id": active, "error_type": type(error).__name__, "error": str(error), "scientific_verdict": None, "promotion_decision": "KEEP_BASELINE"})
        raise


def _evaluate_model(model: PrunableEnergyModel, samples: Sequence[dict[str, Any]], output: Path, *, step: float = .01) -> dict[str, tuple[np.ndarray, ...] | str]:
    output.mkdir(exist_ok=False)
    predictions: dict[str, tuple[np.ndarray, ...] | str] = {}
    index_rows = []
    for index, sample in enumerate(sorted(samples, key=lambda row: row["event_id"])):
        event = sample["event_id"]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                prediction = model.rollout_augmented(*sample_rollout_inputs(sample), max_step=step)
                if any(not np.isfinite(value).all() for value in prediction) or prediction[0].shape != sample["training_context"]["targets"]["delta"].shape or prediction[1].shape != prediction[0].shape:
                    raise ValueError("incomplete or nonfinite augmented prediction")
            predictions[event] = prediction
            path = output / f"event_{index:04d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, delta=prediction[0], omega=prediction[1], hidden=prediction[2])
            index_rows.append({"event_id": event, "status": "completed", "path": path.name})
        except (FloatingPointError, ValueError) as error:
            predictions[event] = str(error)
            index_rows.append({"event_id": event, "status": "failed", "error": str(error)})
    _write_json(output / "index.json", {"max_step": step, "events": index_rows})
    return predictions


def _score(samples: Sequence[dict], manifest: P1RoleManifest, predictions: dict, scales: dict) -> dict:
    def predict(sample: dict) -> tuple[np.ndarray, np.ndarray]:
        value = predictions[sample["event_id"]]
        if isinstance(value, str):
            raise FloatingPointError(f"saved rollout failed: {value}")
        return value[:2]

    return score_trajectory_role(samples, manifest, "internal_audit", predict, scales)


def paired_score_rows(reference: dict, candidate: dict) -> list[dict]:
    """Keep all cells for strict role authorization by the statistics kernel."""
    if (reference["role"], reference["role_manifest_sha256"], reference["event_ids"]) != (candidate["role"], candidate["role_manifest_sha256"], candidate["event_ids"]):
        raise ValueError("paired scores require the same complete role")
    lookup = {(row["event_id"], row["horizon_seconds"]): row for row in candidate["rows"]}
    if len(lookup) != len(candidate["rows"]) or set(lookup) != {(row["event_id"], row["horizon_seconds"]) for row in reference["rows"]}:
        raise ValueError("paired metric cells differ or duplicate")
    output = []
    for original in reference["rows"]:
        other = lookup[(original["event_id"], original["horizon_seconds"])]
        if any(original[key] != other[key] for key in ("source_dataset", "fault_line")):
            raise ValueError("paired metric metadata mismatch")
        good = original["status"] == other["status"] == "completed"
        for name in ("delta", "omega"):
            output.append({key: original[key] for key in ("event_id", "source_dataset", "fault_line", "horizon_seconds")} | {
                "output": name, "status": "completed" if good else "failed",
                "reference_rmse": original["outputs"][name]["rmse"] if good else None,
                "candidate_rmse": other["outputs"][name]["rmse"] if good else None,
            })
    return output


def _change(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...], sample: dict, stop: int, scales: dict) -> dict[str, float]:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        d = (left[0][:stop] - right[0][:stop]) @ sample["model_inputs"]["static"]["incidence"] / scales["edge_angle"]
        w = (left[1][:stop] - right[1][:stop]) / scales["omega"]
        values = {"delta_change": float(np.sqrt(np.mean(d**2))), "omega_change": float(np.sqrt(np.mean(w**2)))}
        values["combined_change"] = float(np.sqrt((values["delta_change"]**2 + values["omega_change"]**2)/2))
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError("nonfinite prediction-change metric")
    return values


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved_counterfactuals(
    base_directory: Path,
    base_manifest_sha256: str,
    counterfactual_directory: Path,
    counterfactual_manifest_sha256: str,
    manifest: P1RoleManifest,
    internal_samples: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    base_directory, counterfactual_directory, output_dir = map(Path, (base_directory, counterfactual_directory, output_dir))
    base = verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
    hashes = verify_counterfactual_artifacts(counterfactual_directory, counterfactual_manifest_sha256)
    validate_p1_samples(internal_samples, manifest, "internal_audit")
    plan = _read_json(counterfactual_directory / "plan.json")
    spec = _read_json(base_directory / "core3.json")
    variants = _variants(spec["active_block_instances"])
    expected_files = {"plan.json"} | {f"{row['variant_id']}_{suffix}" for row in variants for suffix in ("zero.npz", "zero.json", "refit.npz", "refit.json", "lineage.json")}
    if set(hashes) != expected_files or plan["schema"] != "kron_reduce.p1_counterfactual_models.v1" or plan["base_manifest_sha256"] != base_manifest_sha256 or plan["base_core_sha256"] != base["arms"][2]["checkpoint_hash"] or plan["role_manifest_sha256"] != manifest.source_sha256 or plan["variants"] != variants or plan["fit_scales"] != base["fit_scales"] or plan["common_contract_hash"] != base["common_contract_hash"] or plan["reference_arm"] != 3 or plan["scientific_verdict"] is not None or plan["promotion_decision"] != "KEEP_BASELINE":
        raise ValueError("counterfactual bundle binding or variant set mismatch")
    core = PrunableEnergyModel.load(base_directory / "core3.npz")
    scales = base["fit_scales"]
    _new_output(output_dir, (base_directory, counterfactual_directory))
    try:
        reference = _evaluate_model(core, internal_samples, output_dir / "reference")
        half = _evaluate_model(core, internal_samples, output_dir / "reference_half", step=.005)
        quarter = _evaluate_model(core, internal_samples, output_dir / "reference_quarter", step=.0025)
        score_reference = _score(internal_samples, manifest, reference, scales)
        _write_json(output_dir / "reference_metrics.json", score_reference)
        numerical, physical_cache = [], {}
        for sample in sorted(internal_samples, key=lambda row: row["event_id"]):
            event, static = sample["event_id"], sample["model_inputs"]["static"]
            metadata = sample["training_context"]["metadata"]
            predictions = reference[event], half[event], quarter[event]
            if not isinstance(reference[event], str):
                try:
                    physical_cache[event] = module_physical_traces(core, static, *reference[event])
                except (FloatingPointError, ValueError) as error:
                    physical_cache[event] = str(error)
            else:
                physical_cache[event] = reference[event]
            for horizon in (1, 5, 10):
                row = {**metadata, "horizon_seconds": horizon}
                if any(isinstance(value, str) for value in predictions):
                    numerical.append({**row, "status": "failed", "reason": "reference_or_refined_rollout_failed"})
                    continue
                change = _change(predictions[0], predictions[1], sample, horizon*100 + 1, scales)
                fine = _change(predictions[1], predictions[2], sample, horizon*100 + 1, scales)
                ratio = change["combined_change"] / fine["combined_change"] if fine["combined_change"] > 1e-12 else None
                numerical.append({**row, "status": "completed", "step_01_to_005_change": change["combined_change"], "step_005_to_0025_change": fine["combined_change"], "convergence_ratio": ratio, "ratio_unavailable_reason": "near_roundoff" if ratio is None else None})
        _write_json(output_dir / "numerical_error.json", {"rows": numerical, "numerical_gate_verdict": "NOT_EVALUATED"})
        counter_rows, physical_rows, statistics = [], [], []
        for variant in variants:
            name, ids = variant["variant_id"], variant["removed_instance_ids"]
            for mode in ("zero", "refit"):
                model = PrunableEnergyModel.load(counterfactual_directory / f"{name}_{mode}.npz")
                expected_ids = [row["instance_id"] for row in spec["active_block_instances"] if mode == "zero" or row["instance_id"] not in ids]
                if [row.instance_id for row in model.active_instances] != expected_ids:
                    raise ValueError("saved intervention structure mismatch")
                predictions = _evaluate_model(model, internal_samples, output_dir / f"{name}_{mode}")
                metrics = _score(internal_samples, manifest, predictions, scales)
                _write_json(output_dir / f"{name}_{mode}_metrics.json", metrics)
                pairs = paired_score_rows(score_reference, metrics)
                inference = summarize_paired_rows(pairs, manifest, "internal_audit")
                statistics.append({**variant, "mode": mode, "inference": inference})
                by_cell = {(row["event_id"], row["horizon_seconds"], row["output"]): row for row in pairs}
                for sample in sorted(internal_samples, key=lambda row: row["event_id"]):
                    event = sample["event_id"]
                    metadata = sample["training_context"]["metadata"]
                    control_trace = predictions[event]
                    if not isinstance(control_trace, str):
                        try:
                            control_trace = module_physical_traces(model, sample["model_inputs"]["static"], *control_trace)
                        except (FloatingPointError, ValueError) as error:
                            control_trace = str(error)
                    for horizon in (1, 5, 10):
                        header = {**metadata, "variant_id": name, "kind": variant["kind"], "removed_instance_ids": json.dumps(ids), "mode": mode, "horizon_seconds": horizon}
                        physical_header = {**header, "state_source": f"{mode}_prediction"}
                        if isinstance(control_trace, str):
                            physical_rows.append({**physical_header, "status": "failed", "failure": control_trace})
                        else:
                            energy_scale = .5*float(np.sum(sample["model_inputs"]["static"]["mass"]))*scales["omega"]**2
                            present = [item for item in ids if item in control_trace]
                            remaining = [item for item in control_trace if item not in ids]
                            selected_summary = summarize_physical_trace(control_trace, present, horizon*100 + 1, energy_scale)
                            remaining_summary = summarize_physical_trace(control_trace, remaining, horizon*100 + 1, energy_scale)
                            physical_rows.append({**physical_header, "status": "completed", **selected_summary, "selected_modules_present": len(present), "remaining_storage_mean": remaining_summary["storage_mean"], "remaining_port_abs_mean": remaining_summary["port_abs_mean"], "remaining_parameter_norm": remaining_summary["parameter_norm"], "absent_module_semantics": "structurally_deleted_module_has_zero_storage_and_port" if len(present) < len(ids) else "all_selected_instances_present"})
                        if isinstance(reference[event], str) or isinstance(predictions[event], str):
                            counter_rows.append({**header, "status": "failed", "failure": "reference_or_intervention_rollout_failed"})
                            continue
                        change = _change(reference[event], predictions[event], sample, horizon*100 + 1, scales)
                        errors = {f"{side}_{output}_rmse": by_cell[(event, horizon, output)][f"{side}_rmse"] for output in ("delta", "omega") for side in ("reference", "candidate")}
                        counter_rows.append({**header, "status": "completed", **change, **errors})
            # Attribute stored energy to the original module on the original
            # trajectory, not to a difference between two refitted systems.
            for sample in internal_samples:
                event, static = sample["event_id"], sample["model_inputs"]["static"]
                for horizon in (1, 5, 10):
                    header = {**sample["training_context"]["metadata"], "variant_id": name, "kind": variant["kind"], "removed_instance_ids": json.dumps(ids), "horizon_seconds": horizon, "mode": "reference", "state_source": "full_reference_prediction"}
                    trace = physical_cache[event]
                    if isinstance(trace, str):
                        physical_rows.append({**header, "status": "failed", "failure": trace})
                    else:
                        energy_scale = .5*float(np.sum(static["mass"]))*scales["omega"]**2
                        physical_rows.append({**header, "status": "completed", **summarize_physical_trace(trace, ids, horizon*100 + 1, energy_scale)})
        _write_csv(output_dir / "module_counterfactual_metrics.csv", counter_rows)
        _write_csv(output_dir / "module_energy_port_audit.csv", physical_rows)
        _write_json(output_dir / "module_paired_statistics.json", {"variants": statistics})
        verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
        verify_counterfactual_artifacts(counterfactual_directory, counterfactual_manifest_sha256)
        result = {"base_manifest_sha256": base_manifest_sha256, "counterfactual_manifest_sha256": counterfactual_manifest_sha256, "role_manifest_sha256": manifest.source_sha256, "role": "internal_audit", "variant_count": len(variants), "module_counterfactual_rows": len(counter_rows), "module_physical_rows": len(physical_rows), "failed_counterfactual_rows": sum(row["status"] == "failed" for row in counter_rows), "failed_physical_rows": sum(row["status"] == "failed" for row in physical_rows), "coefficient_identifiability_evaluated": False, "profile_audit_evaluated": False, "independent_support_verdict": "NOT_EVALUATED", "scientific_verdict": None, "model_unit_gate_passed": False, "promotion_decision": "KEEP_BASELINE"}
        _write_json(output_dir / "summary.json", result)
        digest = _seal(output_dir)
        return {**result, "directory": str(output_dir.resolve()), "artifact_manifest_sha256": digest}
    except Exception as error:
        _write_json(output_dir / "FAILED.json", {"error_type": type(error).__name__, "error": str(error), "scientific_verdict": None, "promotion_decision": "KEEP_BASELINE"})
        raise
