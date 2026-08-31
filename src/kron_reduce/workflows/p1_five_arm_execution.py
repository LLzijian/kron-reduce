"""Fit-only P1 five-arm computation, not the complete scientific experiment.

Fitting and saved-model evaluation are separate entry points. Neither advances
the harness or promotes a model; bootstrap, refit deletion and modal audits are
still required by the parent protocol.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest, validate_p1_samples
from kron_reduce.data.p1_pm_targets import PmObservationBatch, PmReadPurpose
from kron_reduce.evaluation.pm_closure_diagnostics import (
    CONTROLLED_ARMS,
    FiveArmRecord,
    PMClosureArm,
    SharedF1EvaluationContract,
    validate_five_arm_contract,
)
from kron_reduce.models.base import RunContext, sample_rollout_inputs
from kron_reduce.models.p1_calibration import (
    CalibrationRolloutError,
    P1CalibrationSettings,
    calibrate_loss_pair,
    core_fingerprint,
)
from kron_reduce.models.passive_pm_closure import (
    ClosureAugmentedModel,
    PassivePmClosure,
    fit_passive_pm_candidates,
)
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel

_BUNDLE_SCHEMA = "kron_reduce.p1_five_arm_bundle.v1"
_BUNDLE_FILES = {"bundle.json", "fit_lineage.json", "closure_selection.json", "closure.json", *(f"core{arm}.{suffix}" for arm in (1, 2, 3) for suffix in ("npz", "json"))}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _write_json(path: Path, value: Any) -> None:
    # Serialize before opening, so malformed evidence does not look completed.
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _checked_file(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"file missing, nonregular or hash mismatch: {path.name}")


def _event_weights(samples: Sequence[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        metadata = sample["training_context"]["metadata"]
        groups[metadata["source_dataset"]][metadata["fault_line"]].append(sample["event_id"])
    return {event: 1 / len(groups) / len(lines) / len(events) for lines in groups.values() for events in lines.values() for event in events}


def score_trajectory_role(
    samples: Sequence[dict[str, Any]],
    manifest: P1RoleManifest,
    role: str,
    predict: Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]],
    fit_scales: dict[str, float],
) -> dict[str, Any]:
    """All events remain present; a failed event makes the aggregate unavailable."""
    validate_p1_samples(samples, manifest, role)
    if set(fit_scales) != {"edge_angle", "omega"} or any(not np.isfinite(value) or value < 1e-8 for value in fit_scales.values()):
        raise ValueError("finite positive fit-only scales required")
    weights, rows, failed, loss = _event_weights(samples), [], [], 0.0
    for sample in sorted(samples, key=lambda item: item["event_id"]):
        event = sample["event_id"]
        target, static = sample["training_context"]["targets"], sample["model_inputs"]["static"]
        metadata = sample["training_context"]["metadata"]
        local_rows, local_loss = [], 0.0
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                delta, omega = predict(sample)
                if any(np.shape(value) != np.shape(target[name]) or not np.isfinite(value).all() for name, value in (("delta", delta), ("omega", omega))):
                    raise ValueError("incomplete or nonfinite prediction")
                for horizon in (1, 5, 10):
                    stop = 100 * horizon + 1
                    row: dict[str, Any] = {**metadata, "horizon_seconds": horizon, "status": "completed", "outputs": {}}
                    for name, value in (("delta", delta), ("omega", omega)):
                        observed, predicted = target[name][:stop], value[:stop]
                        error = predicted - observed
                        sse = np.sum(error**2, axis=0)
                        sst = float(np.sum((observed - observed.mean(axis=0))**2))
                        row["outputs"][name] = {
                            "count_per_node": stop, "sse_by_node": sse.tolist(),
                            "target_sum_by_node": observed.sum(axis=0).tolist(),
                            "target_sum_squares_by_node": np.sum(observed**2, axis=0).tolist(),
                            "rmse": float(np.sqrt(sse.sum() / observed.size)),
                            "r2": float(1 - sse.sum() / sst) if sst > 0 else None,
                            "r2_unavailable_reason": None if sst > 0 else "zero_temporal_variance",
                            "coi_drift": float(np.max(np.abs(predicted @ static["mass"] / np.sum(static["mass"])))),
                        }
                        scaled = error @ static["incidence"] / fit_scales["edge_angle"] if name == "delta" else error / fit_scales["omega"]
                        local_loss += weights[event] / 3 / 2 * float(np.mean(scaled**2))
                    local_rows.append(row)
                # Check derived metrics too, not just finite trajectory elements.
                json.dumps(local_rows, allow_nan=False)
                if not np.isfinite(local_loss):
                    raise FloatingPointError("nonfinite aggregate loss")
            rows.extend(local_rows)
            loss += local_loss
        except (FloatingPointError, ValueError) as error:
            failed.append(event)
            rows.extend({**metadata, "horizon_seconds": horizon, "status": "failed", "failure": str(error), "outputs": None} for horizon in (1, 5, 10))
    if not np.isfinite(loss):
        raise FloatingPointError("nonfinite whole-role loss")
    return {
        "role": role, "role_manifest_sha256": manifest.source_sha256,
        "event_ids": list(manifest.ids(role)), "fit_scales": dict(fit_scales),
        "event_weights": weights, "rows": rows, "failed_event_ids": failed,
        "balanced_loss": None if failed else loss,
        "r2_definition": "per_event_output_node_mean_centering; retain sums for later pooled R2",
        "bootstrap_executed": False, "scientific_acceptance": False,
    }


class _FreshP1Core(PrunableEnergyModel):
    """Use the existing weak estimator, but never its failure-dropping selector."""

    def _solve(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        coefficients, diagnostics = super()._solve(*args, **kwargs)
        if diagnostics.get("solver_success") is not True or not np.isfinite(coefficients).all():
            raise CalibrationRolloutError("weak parameter solver failed")
        return coefficients, diagnostics

    def validation_summary(self, samples: Sequence[dict[str, Any]], context: RunContext) -> dict[str, Any]:
        if context.max_cases < len(samples) or context.horizon_steps != 1001:
            raise ValueError("P1 selection cannot truncate events or horizons")
        result = score_trajectory_role(samples, self.p1_manifest, "selection", lambda sample: self.rollout(*sample_rollout_inputs(sample)), self.validation_scales)
        if result["failed_event_ids"]:
            raise CalibrationRolloutError(f"ridge selection rollout failed: {result['failed_event_ids']}")
        return {"standardized_loss": result["balanced_loss"], "failure_rate": 0.0, "complete_selection": result}


def _fit_fresh_core(spec: dict[str, Any], fit: Sequence[dict], selection: Sequence[dict], manifest: P1RoleManifest, *, b2: bool) -> tuple[PrunableEnergyModel, dict[str, Any]]:
    core = _FreshP1Core(
        block_instances=() if b2 else spec["active_block_instances"],
        node_tau=spec["node_tau"], potential_scale=spec["potential_scale"],
        memory_taus=spec["memory_taus"], resonances=spec["resonances"],
        max_internal_step=0.01, enforce_identifiability=True,
        minimum_equation_parameter_ratio=3.0, maximum_condition_number=1e8,
    )
    core.p1_manifest = manifest
    if core.node_damping is not None or core.coupling is not None:
        raise RuntimeError("fresh fitting must start without numerical coefficients")
    context = RunContext(seed=324, horizon_steps=1001, fit_max_steps=1001, max_cases=max(len(fit), len(selection)), max_internal_step=0.01, integral_windows=(1, 5, 20, 40, 80), ridge_values=(1e-10, 1e-8, 1e-6, 1e-4, 1e-2), weak_max_windows_per_case_horizon=8)
    outcome = core.fit(copy.deepcopy(list(fit)), copy.deepcopy(list(selection)), context)
    if set(outcome.train_case_ids) != set(manifest.ids("fit")) or set(outcome.val_case_ids) != set(manifest.ids("selection")) or outcome.status != "completed" or not np.isfinite(outcome.best_validation_loss):
        raise ValueError("fresh fit did not cover the frozen roles")
    record = {
        "fresh_empty_coefficient_model": True, "old_parameter_arrays_read": False,
        "parameter_role": "fit", "ridge_selection_role": "selection",
        "structure_is_newly_discovered": False, "initial_core_fingerprint": core_fingerprint(core),
        "context": asdict(context), "result": asdict(outcome),
    }
    return core, record


def _arm_records(contract: SharedF1EvaluationContract, hashes: dict[str, str]) -> tuple[FiveArmRecord, ...]:
    records = []
    for index, arm in enumerate(PMClosureArm):
        checkpoint = PMClosureArm(min(int(arm), 3))
        records.append(FiveArmRecord(arm, CONTROLLED_ARMS[arm], (None, PMClosureArm(1), PMClosureArm(2), PMClosureArm(3), PMClosureArm(3))[index], contract.contract_hash, checkpoint, hashes[f"core{int(checkpoint)}.npz"], arm == 4, arm == 4, arm != 4, False))
    audit = validate_five_arm_contract(records, contract=contract)
    if not audit.valid:
        raise ValueError(f"invalid five-arm binding: {audit.violations}")
    return tuple(records)


def fit_and_save_five_arms(
    output_dir: Path,
    structure_path: Path,
    expected_structure_sha256: str,
    fit_samples: Sequence[dict[str, Any]],
    selection_samples: Sequence[dict[str, Any]],
    fit_pm: PmObservationBatch,
    manifest: P1RoleManifest,
    *,
    settings: P1CalibrationSettings | None = None,
) -> dict[str, Any]:
    """A new directory per attempt. No internal-audit samples can enter here."""
    validate_p1_samples(fit_samples, manifest, "fit")
    validate_p1_samples(selection_samples, manifest, "selection")
    _checked_file(Path(structure_path), expected_structure_sha256)
    spec = _read_json(Path(structure_path))
    if spec.get("format_version") != 3 or not isinstance(spec.get("active_block_instances"), list) or (spec.get("node_tau"), spec.get("potential_scale"), spec.get("max_internal_step")) != (0.2, 0.25, 0.01):
        raise ValueError("structure file does not match the registered P1 prior contract")
    # Fit-target role checks happen before expensive core fitting, too.
    candidates = fit_passive_pm_candidates(fit_samples, fit_pm, manifest)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        b2, b2_lineage = _fit_fresh_core(spec, fit_samples, selection_samples, manifest, b2=True)
        current, current_lineage = _fit_fresh_core(spec, fit_samples, selection_samples, manifest, b2=False)
        b2_result = calibrate_loss_pair(b2, fit_samples, manifest, settings=settings, loss_modes=(False,))
        current_results = calibrate_loss_pair(current, fit_samples, manifest, settings=settings)
        calibrated = (*b2_result, *current_results)
        fit_scales = dict(current.validation_scales)
        if fit_scales != b2.validation_scales:
            raise ValueError("core arms must share fit-only scales")
        closure_records, eligible = [], []
        for closure, record in candidates:
            augmented = ClosureAugmentedModel(calibrated[2][0], closure)
            score = score_trajectory_role(selection_samples, manifest, "selection", lambda sample, augmented=augmented: augmented.rollout(*sample_rollout_inputs(sample)), fit_scales)
            closure_records.append({"fit": record, "selection": score})
            if score["balanced_loss"] is not None:
                eligible.append((score["balanced_loss"], closure.decay, closure))
        if not eligible:
            raise CalibrationRolloutError("all registered causal closure candidates failed")
        selected_loss, selected_decay, selected_closure = min(eligible, key=lambda row: (row[0], row[1]))
        for arm, (model, _record) in enumerate(calibrated, 1):
            path = directory / f"core{arm}.npz"
            if path.exists() or path.with_suffix(".json").exists():
                raise FileExistsError(path)
            model.save(path)
            # Existing serializer allows NaN; new evidence explicitly does not.
            _read_json(path.with_suffix(".json"))
        selected_closure.save(directory / "closure.json")
        _write_json(directory / "fit_lineage.json", {
            "structure_source_path": str(Path(structure_path).resolve()), "structure_source_sha256": expected_structure_sha256,
            "inherited_structure": {name: spec[name] for name in ("active_block_instances", "node_tau", "potential_scale", "memory_taus", "resonances")},
            "b2": b2_lineage, "current": current_lineage,
            "calibrations": [record for _model, record in calibrated],
            "fit_pm_reads": list(fit_pm.accesses),
            "fit_only_parameter_lineage_proven_by_this_execution": True,
            "scientific_acceptance": False,
        })
        _write_json(directory / "closure_selection.json", {"candidates": closure_records, "selected_decay": selected_decay, "selected_balanced_loss": selected_loss, "nonzero_states": selected_closure.n_states, "selection_observed_pm_read": False, "module_independent_support_proven": False})
        hashes = {path.name: file_sha256(path) for path in directory.iterdir()}
        contract = SharedF1EvaluationContract(manifest.role_id_hashes)
        bundle = {
            "schema": _BUNDLE_SCHEMA, "role_manifest_sha256": manifest.source_sha256,
            "contract": asdict(contract), "common_contract_hash": contract.contract_hash,
            "fit_scales": fit_scales, "arms": [asdict(record) for record in _arm_records(contract, hashes)],
            "core_bindings": {str(arm): f"core{min(arm, 3)}.npz" for arm in range(1, 6)},
            "closure_file": "closure.json", "bootstrap_executed": False,
            "model_unit_gate_passed": False, "scientific_verdict": None,
            "promotion_decision": "KEEP_BASELINE", "scientific_model_count_claimed": False,
        }
        _write_json(directory / "bundle.json", bundle)
        hashes["bundle.json"] = file_sha256(directory / "bundle.json")
        _write_json(directory / "artifact_hashes.json", hashes)
        return {"directory": str(directory.resolve()), "artifact_manifest_sha256": file_sha256(directory / "artifact_hashes.json"), "bundle": bundle}
    except Exception as error:
        _write_json(directory / "FAILED.json", {"status": "implementation_attempt_failed", "error_type": type(error).__name__, "error": str(error), "scientific_verdict": None, "promotion_decision": "KEEP_BASELINE"})
        raise


def verify_five_arm_bundle(directory: Path, expected_manifest_sha256: str, manifest: P1RoleManifest) -> dict[str, Any]:
    directory = Path(directory)
    _checked_file(directory / "artifact_hashes.json", expected_manifest_sha256)
    hashes = _read_json(directory / "artifact_hashes.json")
    if set(hashes) != _BUNDLE_FILES or {path.name for path in directory.iterdir()} != _BUNDLE_FILES | {"artifact_hashes.json"}:
        raise ValueError("bundle file set changed or incomplete")
    for name, digest in hashes.items():
        _checked_file(directory / name, digest)
    bundle = _read_json(directory / "bundle.json")
    contract = SharedF1EvaluationContract(manifest.role_id_hashes)
    # JSON converts tuples to lists; compare its canonical form explicitly.
    if bundle["schema"] != _BUNDLE_SCHEMA or bundle["role_manifest_sha256"] != manifest.source_sha256 or bundle["common_contract_hash"] != contract.contract_hash or bundle["contract"] != json.loads(json.dumps(asdict(contract))):
        raise ValueError("bundle role/contract mismatch")
    if bundle["arms"] != [asdict(row) for row in _arm_records(contract, hashes)] or bundle["core_bindings"] != {str(arm): f"core{min(arm, 3)}.npz" for arm in range(1, 6)} or bundle["closure_file"] != "closure.json":
        raise ValueError("bundle five-arm checkpoint binding changed")
    if bundle["model_unit_gate_passed"] is not False or bundle["scientific_verdict"] is not None or bundle["promotion_decision"] != "KEEP_BASELINE":
        raise ValueError("computation bundle cannot grant scientific acceptance")
    return bundle


def evaluate_saved_five_arms(
    directory: Path,
    expected_manifest_sha256: str,
    manifest: P1RoleManifest,
    internal_samples: Sequence[dict[str, Any]],
    observed_pm: PmObservationBatch,
    output_dir: Path,
) -> dict[str, Any]:
    """Read-only frozen models; external targets never reach any fitting call."""
    directory, output_dir = Path(directory), Path(output_dir)
    bundle = verify_five_arm_bundle(directory, expected_manifest_sha256, manifest)
    validate_p1_samples(internal_samples, manifest, "internal_audit")
    checkpoint_hash = bundle["arms"][2]["checkpoint_hash"]
    if observed_pm.role != "internal_audit" or observed_pm.purpose != PmReadPurpose.CONDITIONAL_DIAGNOSTIC or observed_pm.role_manifest_sha256 != manifest.source_sha256 or observed_pm.event_ids != manifest.ids("internal_audit") or set(observed_pm.pm_by_event) != set(manifest.ids("internal_audit")) or observed_pm.core_checkpoint_sha256 != checkpoint_hash or observed_pm.main_model_input_allowed is not False:
        raise ValueError("conditional observations do not bind the frozen core and role")
    # Reject malformed observations before any evaluation output is created.
    for sample in internal_samples:
        pm = np.asarray(observed_pm.pm_by_event[sample["event_id"]])
        clear = sample["model_inputs"]["static"]["Pm_clear"].reshape(-1)
        if pm.shape != (1001, len(clear)) or not np.isfinite(pm).all() or not np.allclose(pm[0], clear, rtol=0, atol=1e-10) or np.max(np.abs(pm.sum(axis=1))) > 1e-10:
            raise ValueError("invalid conditional observation")
    if output_dir.resolve().is_relative_to(directory.resolve()):
        raise ValueError("evaluation outputs must not modify the frozen bundle")
    models = {arm: PrunableEnergyModel.load(directory / f"core{arm}.npz") for arm in (1, 2, 3)}
    augmented = ClosureAugmentedModel(models[3], PassivePmClosure.load(directory / "closure.json"))
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        arms = []
        for arm in range(1, 6):
            def predict(sample: dict[str, Any], *, arm: int = arm) -> tuple[np.ndarray, np.ndarray]:
                static, time, exogenous = sample_rollout_inputs(sample)
                if arm == 4:
                    exogenous = {"Pm": np.asarray(observed_pm.pm_by_event[sample["event_id"]]), "interpolation": "linear"}
                model = augmented if arm == 5 else models[min(arm, 3)]
                return model.rollout(static, time, exogenous)

            score = score_trajectory_role(internal_samples, manifest, "internal_audit", predict, bundle["fit_scales"])
            arms.append({"arm": arm, "binding": bundle["arms"][arm - 1], "metrics": score})
        verify_five_arm_bundle(directory, expected_manifest_sha256, manifest)
        result = {"bundle_manifest_sha256": expected_manifest_sha256, "role": "internal_audit", "arms": arms, "observed_pm_reads": list(observed_pm.accesses), "bootstrap_executed": False, "module_refit_deletion_executed": False, "modal_audit_executed": False, "model_unit_gate_passed": False, "scientific_verdict": None, "promotion_decision": "KEEP_BASELINE"}
        _write_json(output_dir / "raw_metrics.json", result)
        _write_json(output_dir / "artifact_hashes.json", {"raw_metrics.json": file_sha256(output_dir / "raw_metrics.json")})
        return result
    except Exception as error:
        _write_json(output_dir / "FAILED.json", {"status": "evaluation_attempt_failed", "error_type": type(error).__name__, "error": str(error), "scientific_verdict": None, "promotion_decision": "KEEP_BASELINE"})
        raise
