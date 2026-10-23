"""A04 fixed mechanisms and a fail-closed real-data evaluation prerequisite.

The current registration has no certified 100 ms excitation envelope. The
entrypoint therefore records the blocked plan BEFORE opening any trajectory or
fitting a model. Numerical helpers are independently testable on synthetic truth;
they are not a bypass for the real-data evaluation gate.
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass, replace

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    _features,
    fixed_quadratic_basis,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project
from kron_reduce.selection.envelopes import ExcitationEnvelopeRouter
from kron_reduce.workflows.kiki_history_learning import score_predictions
from kron_reduce.workflows.kiki_stability_comparison import SCALES, los_classification
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines

SEEDS = (324, 325, 326)
CANDIDATES = {
    "A": {"latent_dimension": 0, "frames": 3, "encoder_mode": "frames", "parameter_count": 54},
    "B": {"latent_dimension": 2, "frames": 3, "encoder_mode": "frames", "parameter_count": 238},
    "C": {"latent_dimension": 2, "frames": 11, "encoder_mode": "frames", "parameter_count": 558},
    "D": {"latent_dimension": 2, "frames": 3, "encoder_mode": "fixed_quadratic", "parameter_count": 238},
}


@dataclass(frozen=True)
class ShortHistoryContract:
    history_count: int = 11
    power_index: int = 10
    dt_s: float = 0.01
    input_contract: str = "measured_history_100ms_zoh_v1"

    def __post_init__(self):
        if (self.history_count, self.power_index, self.dt_s, self.input_contract) != (
            11, 10, 0.01, "measured_history_100ms_zoh_v1"
        ):
            raise ValueError("unregistered A04 input contract")

    @property
    def origin_s(self):
        return (self.history_count - 1) * self.dt_s

    @property
    def splice_offset(self):
        return self.history_count - 1

    def clock(self, horizon_s):
        count = round(horizon_s / self.dt_s)
        if count < 1 or not np.isclose(count * self.dt_s, horizon_s, rtol=0, atol=1e-12):
            raise ValueError("horizon must lie on the registered clock")
        return np.arange(count + 1) * self.dt_s

    def validate_history(self, history):
        if history.delta.shape != (self.history_count, 10) or not np.allclose(
            history.time, self.clock(self.origin_s) - self.origin_s, rtol=0, atol=1e-12
        ):
            raise ValueError("history differs from the 100 ms ten-node contract")


def registered_contract(root, config):
    registry = json.loads((root / "configs/input_contracts.json").read_text(encoding="utf-8"))
    name = config["input_contract"]
    row = registry[name]
    if row["power_policy"] != "zoh_from_declared_origin_power":
        raise ValueError("A04 needs observed origin power")
    count = row["real_observation_count"]
    contract = ShortHistoryContract(count, count - 1, 0.01, name)
    if config["history_count"] != count or config["power_index"] != contract.power_index:
        raise ValueError("config and registered input contract differ")
    return contract


def candidate_settings(candidate, seed):
    if candidate not in CANDIDATES or seed not in SEEDS:
        raise ValueError("candidate or seed is outside the frozen 12-fit plan")
    spec = {k: v for k, v in CANDIDATES[candidate].items() if k != "parameter_count"}
    return NonlinearFitSettings(**spec, backbone="sine", epochs=80, learning_rate=0.01,
                                encoder_penalty=1e-4, history_weight=0.0,
                                horizons_s=(1.0, 3.0), seed=seed)


def select_structure(records):
    """Never remove a failed seed, select an epoch, or use classification scores."""
    identities = [(r["candidate"], r["seed"]) for r in records]
    expected = {(c, s) for c in CANDIDATES for s in SEEDS}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("selection requires exactly the 12 registered fits")
    table = []
    for candidate, spec in CANDIDATES.items():
        rows = sorted((r for r in records if r["candidate"] == candidate), key=lambda r: r["seed"])
        eligible = all(
            r["status"] == "completed" and r.get("actual_optimizer_steps") == 80
            and r.get("parameter_count") == spec["parameter_count"]
            and r.get("val_main_nrmse") is not None
            and np.isfinite(r["val_main_nrmse"]) and r["val_main_nrmse"] >= 0
            for r in rows
        )
        table.append({"candidate": candidate, "eligible": bool(eligible),
                      "parameter_count": spec["parameter_count"],
                      "median_val_main_nrmse": float(np.median([r["val_main_nrmse"] for r in rows]))
                      if eligible else None, "seeds": rows})
    eligible = [r for r in table if r["eligible"]]
    selected = min(eligible, key=lambda r: (r["median_val_main_nrmse"], r["parameter_count"],
                                           r["candidate"])) if eligible else None
    baselines = all(r["eligible"] for r in table if r["candidate"] in {"A", "B"})
    return {"selected": None if selected is None else selected["candidate"],
            "candidate_table": table, "baseline_comparison_complete": baselines,
            "all_fits_completed": all(r["eligible"] for r in table),
            "diagnostic_checkpoint_seed": 324, "classification_used_for_selection": False,
            "failed_seeds_removed": False}


def score_fixed_predictions(predictions, samples, scale):
    """Reuse A03 source/line/event/time weights; reject NaN paths explicitly."""
    checked = {}
    for sample in samples:
        value = predictions[sample.event_id]
        if not isinstance(value, str):
            value = np.asarray(value)
            if value.shape != (len(sample.forecast_time), 20) or not np.isfinite(value).all():
                value = "nonfinite or misaligned visible forecast"
        checked[sample.event_id] = value
    output = {}
    for name, minimum in (("available", 0.0), ("complete3_common", 3.0)):
        scores = {str(h): score_predictions(checked, samples, h, scale, require_full=minimum)
                  for h in (1.0, 3.0)}
        values = [row["normalized_rmse"] for row in scores.values()]
        output[name] = {"horizons": scores, "multi_horizon_normalized_rmse":
                        None if any(v is None for v in values)
                        else float(np.sqrt(np.mean(np.square(values))))}
    output["complete1"] = score_predictions(checked, samples, 1.0, scale, require_full=1.0)
    return output


def classify_los(history, prediction, equilibrium, mass, omega_s, contract):
    """Exactly A03 AND/five-sample semantics, with the declared new origin."""
    contract.validate_history(history)
    result = los_classification(history, prediction, equilibrium, mass, omega_s)
    hit = result["trigger_index"]
    result["predicted_los_time_forecast_s"] = (
        None if hit is None else (hit - contract.splice_offset) * contract.dt_s
    )
    return result


def stability_refinement(fine, coarse, fine_class, coarse_class, contract):
    """Require complete finite paths even when comparison ends at an LOS trigger."""
    fine, coarse = np.asarray(fine), np.asarray(coarse)
    if fine.shape != (801, 20) or coarse.shape != fine.shape:
        raise ValueError("stability refinement needs complete 8 s visible paths")
    same = fine_class["predicted_los"] == coarse_class["predicted_los"]
    endpoint, time_ok = len(fine) - 1, True
    if same and fine_class["predicted_los"]:
        left, right = fine_class["trigger_index"], coarse_class["trigger_index"]
        endpoint = max(0, max(left, right) - contract.splice_offset)
        time_ok = abs(left - right) * contract.dt_s <= 0.02 + 1e-12
    finite = bool(np.isfinite(fine).all() and np.isfinite(coarse).all())
    error = fine[:endpoint + 1] - coarse[:endpoint + 1]
    value = float(np.sqrt(0.5 * np.mean((error[:, :10] / SCALES[0])**2
                                      + (error[:, 10:] / SCALES[1])**2))) if finite else None
    return {"qualified": bool(finite and same and time_ok and value <= 1e-3),
            "finite_full_paths": finite, "same_los_decision": same,
            "los_time_difference_within_20ms": bool(time_ok),
            "comparison_future_points_including_origin": endpoint + 1,
            "normalized_joint_state_difference": value,
            "history_initialization_refined_too": True,
            "splice_offset": contract.splice_offset}


def physical_path_checks(model, path, history, equilibrium):
    n, mass = model.closure.n_nodes, model.closure.mass
    if not np.isfinite(path).all():
        return {"qualified": False, "finite_full_path": False}
    identity = max(model.closure.energy_identity(row, history.constant_power, equilibrium)
                   ["identity_error"] for row in path)
    coi = max(float(np.max(np.abs(path[:, :n] @ mass))),
              float(np.max(np.abs(path[:, n:2*n] @ mass))))
    return {"qualified": identity <= 1e-8 and coi <= 1e-10,
            "finite_full_path": True, "energy_identity_max_error": identity,
            "coi_weighted_sum_max_error": coi}


def trajectory_refinement(model, sample, contract, scale):
    """Synthetic-testable primitive; real evaluation requires the envelope gate."""
    contract.validate_history(sample.history)
    paths, checks = [], []
    for step in (0.01, 0.005):
        _, history_path = model.initialize(sample.history, sample.equilibrium_delta, max_step=step)
        path = model.predict(sample.history, sample.forecast_time, sample.equilibrium_delta,
                             max_step=step)
        paths.append(path)
        checks.append({"step_s": step, "history": physical_path_checks(
            model, history_path, sample.history, sample.equilibrium_delta),
            "forecast": physical_path_checks(model, path, sample.history, sample.equilibrium_delta)})
    difference = paths[0][:, :20] - paths[1][:, :20]
    value = float(np.max(np.abs(difference / scale)))
    return {"event_id": sample.event_id, "max_normalized_difference": value,
            "max_delta_difference_rad": float(np.max(np.abs(difference[:, :10]))),
            "max_omega_difference_rad_per_s": float(np.max(np.abs(difference[:, 10:]))),
            "qualified": bool(np.isfinite(value) and value <= 1e-5
                              and all(c[k]["qualified"] for c in checks for k in ("history", "forecast"))),
            "history_initialization_refined_too": True, "physical_checks": checks}


def real_evaluation_prerequisite():
    # No A04 causal envelope with earned evidence is present in this registration.
    # Do not convert train extrema, identifiers, or a user-supplied boolean to a certificate.
    route = ExcitationEnvelopeRouter([]).route({}).as_dict()
    return {**route, "status": "BLOCKED_NO_CERTIFIED_ENVELOPE",
            "router_status": route["status"], "certified_envelopes": [],
            "real_data_fitting_allowed_in_this_run": False,
            "real_data_evaluation_allowed": False,
            "out_of_envelope_diagnostics_allowed": False,
            "reason": "No registered evidence certifies the A04 100ms/Pm[10] evaluation envelope; "
                      "AGENTS requires refusal before out-of-envelope evaluation."}


def require_real_evaluation():
    gate = real_evaluation_prerequisite()
    raise PermissionError(gate["reason"])


def synthetic_truth(*, backbone="linear", observable=True):
    """Declared ten-node analytic fixture; never presented as a kiki surrogate."""
    mass = np.linspace(1.0, 2.0, 10)
    basis = coi_basis(np.ones(10))
    return NonlinearPassiveClosure(
        mass, np.linspace(0.05, 0.12, 45), 0.3 * basis @ basis.T,
        0.8 * basis[:, :2] if observable else np.zeros((10, 2)),
        np.array([[0.0, -0.7], [0.7, 0.0]]), np.diag([0.2, 0.4]), backbone,
    )


def synthetic_observability(*, observable, realization_count=32):
    """Exact exponential measurement map, with known visible initial condition.

    The norm is Euclidean after declared unit scales of 1 rad and 1 rad/s.
    Reconstruction uses known equations, not a trained A04 encoder.
    """
    from scipy.linalg import expm

    closure = synthetic_truth(observable=observable)
    zero = np.zeros(10)
    system = np.column_stack([closure.rhs(e, zero, zero) for e in np.eye(22)])
    maps = np.stack([expm(system * t)[:20, 20:] for t in np.arange(11) * 0.01])
    rng = np.random.default_rng(324)
    latent = rng.normal(size=(realization_count, 2))
    output, arrays = [], {"system_matrix": system, "history_maps": maps,
                          "latent_initial_states": latent}
    for candidate in ("B", "C", "D"):
        if candidate == "D":
            summary = np.einsum("jt,jkr->tkr", fixed_quadratic_basis(), maps)
        else:
            summary = maps[[0, 5, 10]] if candidate == "B" else maps
        matrix = np.concatenate((summary[:, :10].reshape(-1, 2),
                                 summary[:, 10:].reshape(-1, 2)), axis=0)
        left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.linalg.matrix_rank(matrix))
        record = {"candidate": candidate, "system": "observable" if observable else "zero_port",
                  "rank": rank, "latent_dimension": 2, "singular_values": singular.tolist(),
                  "known_visible_initial_state": True, "visible_channel_unit_scales": [1.0, 1.0],
                  "encoder_fitted": False, "reconstruction_max_error": None,
                  "noise_gain_l2": None, "worst_noise_identity_error": None}
        if rank == 2:
            inverse = np.linalg.pinv(matrix)
            recovered = (inverse @ (matrix @ latent.T)).T
            worst_noise = left[:, -1]
            gain = 1.0 / singular[-1]
            record.update(status="IDENTIFIABLE_FOR_THIS_KNOWN_SYNTHETIC_SYSTEM",
                          reconstruction_max_error=float(np.max(np.abs(recovered - latent))),
                          noise_gain_l2=float(gain),
                          worst_noise_identity_error=abs(float(np.linalg.norm(inverse @ worst_noise)) - gain))
            arrays[f"{candidate}_reconstructed"] = recovered
        else:
            record.update(status="REFUSE_HIDDEN_STATE_IDENTIFICATION",
                          reason="zero reciprocal port carries no information about hidden initial state",
                          visible_future_uncertainty_proved=False,
                          unsupported_module_action="remove memory module; do not claim identifiable z")
        arrays[f"{candidate}_measurement_matrix"] = matrix
        output.append(record)
    return output, arrays


def run_synthetic_prerequisite(*, root, experiment, stage, result_dir, run_id):
    """Fixed, no-optimizer mechanism evidence; separately registered from kiki."""
    from project_harness.data_check import _verify_checksum_manifest

    started = time.perf_counter()
    if stage != "synthetic_audit" or experiment["data_policy"] != "analytic_nonlinear_synthetic_only":
        raise ValueError("unregistered synthetic prerequisite")
    config_path = root / "configs/experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if (config["seed"], config["realization_count"], config["history_count"]) != (324, 32, 11):
        raise ValueError("synthetic mechanism budget changed")
    manifest = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["datasets"]["synthetic_benchmark_registry_v1"]
    findings = _verify_checksum_manifest(root / entry["path"], entry["checksum_manifest"],
                                         dataset_name="synthetic_benchmark_registry_v1")
    registry_check = {"status": "PASS" if findings and all(r["pass"] for r in findings) else "FAIL",
                      "findings": findings, "other_datasets_opened": False}
    if registry_check["status"] != "PASS":
        raise ValueError("checksum-locked synthetic registry verification failed")
    _write(result_dir / "registry_check.json", registry_check)
    _write(result_dir / "pre_execution_binding.json", {
        "scope": "analytic_fixture_supplement_to_checksum_locked_registry",
        "registry_modified": False, "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(root / config["protocol_path"]),
        "fixture_source_sha256": sha256_file(root / "src/kron_reduce/workflows/kiki_short_history.py"),
        "seed": 324, "realization_count": 32, "optimizer_updates": 0,
        "real_data_used": False, "candidate_selection_executed": False})
    rows, arrays = [], {}
    for observable in (True, False):
        records, values = synthetic_observability(observable=observable)
        rows.extend(records)
        prefix = "observable" if observable else "zero_port"
        arrays.update({f"{prefix}_{key}": value for key, value in values.items()})
    _write(result_dir / "observability.json", {"rows": rows,
           "noise_gain_definition": "sup ||hidden_initial_error||_2 / ||represented_history_noise||_2",
           "known_dynamics_and_visible_initial_state": True,
           "not_an_unknown_system_identifiability_certificate": True})
    with (result_dir / "analytic_maps.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    truth, contract = synthetic_truth(backbone="sine"), ShortHistoryContract()
    rng = np.random.default_rng(324)
    eq = coi_project(rng.normal(0.0, 0.1, 10), truth.mass)
    q = coi_project(rng.normal(0.0, 0.2, 10), truth.mass)
    v = coi_project(rng.normal(0.0, 0.1, 10), truth.mass)
    power = rng.normal(0.0, 0.02, 10)
    power -= power.mean()
    true_path = truth.rollout(q, v, np.array([0.3, -0.2]), np.arange(311) * 0.01,
                              power, eq, max_step=0.005)
    history = CausalStateHistory(contract.clock(0.1) - 0.1, true_path[:11, :10],
                                 true_path[:11, 10:20], power)
    sample = EquilibriumLearningSample("synthetic-known-sine", "analytic_fixture", "fixture",
                                       "val", history, contract.clock(3.0), true_path[10:, :10],
                                       true_path[10:, 10:20], eq)
    numerical, replays = [], []
    for candidate in CANDIDATES:
        settings = candidate_settings(candidate, 324)
        closure = truth if candidate != "A" else replace(
            truth, coupling=np.zeros((10, 0)), latent_skew=np.zeros((0, 0)),
            latent_dissipation=np.zeros((0, 0)))
        features = _features(history, eq, settings.frames, settings.encoder_mode)
        model = LearnedNonlinearModel(
            closure, rng.normal(0.0, 0.01, (settings.latent_dimension, len(features))),
            np.ones(len(features)), np.ones(20), settings,
            {"fitted_event_ids": [], "actual_optimizer_steps": 0,
             "construction": "known_synthetic_dynamics_with_fixed_random_encoder",
             "not_a_trained_candidate": True})
        check = trajectory_refinement(model, sample, contract, np.ones(20))
        fine = model.predict(history, contract.clock(8.0), eq, max_step=0.005)
        coarse = model.predict(history, contract.clock(8.0), eq, max_step=0.01)
        fc = classify_los(history, fine[:, :20], eq, truth.mass, 2*np.pi*60, contract)
        cc = classify_los(history, coarse[:, :20], eq, truth.mass, 2*np.pi*60, contract)
        stability = stability_refinement(fine[:, :20], coarse[:, :20], fc, cc, contract)
        physical8 = [physical_path_checks(model, path, history, eq) for path in (fine, coarse)]
        checkpoint = result_dir / f"synthetic-mechanism-{candidate}.npz"
        model.save(checkpoint)
        restored = LearnedNonlinearModel.load(checkpoint)
        replay = np.array_equal(fine, restored.predict(history, contract.clock(8.0), eq, max_step=0.005))
        numerical.append({"candidate": candidate, "trajectory": check, "stability": stability,
                          "physical_8s": physical8})
        replays.append({"candidate": candidate, "exact": replay,
                        "checkpoint_sha256": sha256_file(checkpoint)})
    _write(result_dir / "numerical_mechanisms.json", {"rows": numerical, "replays": replays,
           "numeric_unit_scale_is_declared_not_train_fitted": True})
    checks = {
        "three_representations_observable": all(r["rank"] == 2 and r["reconstruction_max_error"] < 1e-10
                                               for r in rows if r["system"] == "observable"),
        "unobservable_counterexample_refused": all(r["rank"] == 0 and r["status"] ==
             "REFUSE_HIDDEN_STATE_IDENTIFICATION" for r in rows if r["system"] == "zero_port"),
        "trajectory_and_8s_refinement": all(r["trajectory"]["qualified"] and r["stability"]["qualified"]
                                           for r in numerical),
        "complete_path_energy_coi": all(c["qualified"] for r in numerical for c in r["physical_8s"]),
        "checkpoint_replay_exact": all(r["exact"] for r in replays),
        "real_evaluation_remains_blocked": real_evaluation_prerequisite()["status"] == "BLOCKED_NO_CERTIFIED_ENVELOPE",
    }
    summary = {"process_status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "scientific_verdict": "ACCEPT" if all(checks.values()) else "REJECT",
               "mechanism_status": "MECHANISM_VERIFIED" if all(checks.values()) else "MECHANISM_FAILED",
               "promotion_decision": "KEEP_BASELINE", "optimizer_updates": 0,
               "real_data_performance_evaluated": False, "real_data_improvement_demonstrated": False,
               "conclusion": "Known-system history maps can identify two coupled hidden coordinates; "
               "zero-port hidden coordinates cannot be identified and provide no supported memory module. "
               "This supplies mechanism evidence, not a kiki excitation-envelope certificate."}
    _write(result_dir / "synthetic_summary.json", summary)
    source_paths = ["src/kron_reduce/workflows/kiki_short_history.py",
                    "src/kron_reduce/models/nonlinear_latent.py",
                    "src/kron_reduce/workflows/kiki_stability_comparison.py",
                    "src/kron_reduce/physics/coordinates.py", config["protocol_path"]]
    _write(result_dir / "lineage.json", {"run_id": run_id, "source_commit": git_head(root),
           "config_sha256": sha256_file(config_path), "source_sha256": {
               p: sha256_file(root / p) for p in source_paths}, "wall_seconds": time.perf_counter() - started,
           "requested_core_agent_model": "gpt-6-astra", "requested_core_agent_effort": "ultra",
           "runtime_identity_independently_verified": False})
    _write(result_dir / "stage_evidence.json", {**summary, "source_commit": git_head(root),
           "data_policy": config["data_policy"], "source_val_ids_read": [], "test_ids_read": [],
           "hidden_training_targets_read": False, "level_6_claimed": False,
           "claim_boundary": config["claim_boundary"]})
    _write(result_dir / "artifact_hashes.json", {p.name: sha256_file(p) for p in sorted(result_dir.iterdir())
           if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
    if not all(checks.values()):
        raise RuntimeError("synthetic mechanism gate failed; evidence preserved")


def run(*, root, experiment, stage, result_dir, run_id):
    """Produce immutable prerequisite evidence; never spend a blocked fit budget."""
    if experiment["data_policy"] == "analytic_nonlinear_synthetic_only":
        return run_synthetic_prerequisite(root=root, experiment=experiment, stage=stage,
                                          result_dir=result_dir, run_id=run_id)
    started = time.perf_counter()
    if stage != "short_history_learning" or experiment["data_policy"] != "kiki_original_train_only":
        raise ValueError("unregistered A04 stage or data policy")
    config_path = root / "configs/experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if (config["data_policy"] != experiment["data_policy"] or config["split_seed"] != 324
        or tuple(config["seeds"]) != SEEDS or config["epochs"] != 80
        or config["learning_rate"] != 0.01 or config["encoder_penalty"] != 1e-4):
        raise ValueError("fixed scientific budget was changed")
    contract = registered_contract(root, config)
    gate = real_evaluation_prerequisite()
    allocation_path = root / config["frozen_allocation_path"]
    allocation_hash = sha256_file(allocation_path)
    if allocation_hash != config["frozen_allocation_sha256"]:
        raise ValueError("A03 frozen allocation changed")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    if any(row["source_role"] != "train" for row in allocation["rows"]):
        raise ValueError("frozen allocation contains source holdouts")
    plan = [{"candidate": c, "seed": s, "status": "not_executed_prerequisite_blocked",
             "planned_optimizer_steps": 80, "actual_optimizer_steps": 0,
             "expected_parameter_count": CANDIDATES[c]["parameter_count"],
             "checkpoint": None, "val_main_nrmse": None}
            for c in CANDIDATES for s in SEEDS]
    _write(result_dir / "evaluation_prerequisite.json", gate)
    _write(result_dir / "training_plan.json", {"records": plan, "planned_fits": 12,
           "completed_fits": 0, "planned_optimizer_steps": 960, "actual_optimizer_steps": 0,
           "seeds_changed": False, "retries": 0})
    _write(result_dir / "role_allocation_binding.json", {
        "path": config["frozen_allocation_path"], "sha256": allocation_hash,
        "allocated_train": sum(r["role"] == "train" for r in allocation["rows"]),
        "allocated_val": sum(r["role"] == "val" for r in allocation["rows"]),
        "allocation_resampled": False, "eligibility_not_recomputed_without_payloads": True})
    _write(result_dir / "data_access_audit.json", {
        "accesses": [], "source_val_ids_read": [], "test_ids_read": [],
        "source_train_payload_ids_read": [], "future_pm_reads": [], "pe_voltage_reads": [],
        "source_metadata_opened": False, "frozen_allocation_metadata_only": True,
        "input_contract": {"name": contract.input_contract, "history_count": contract.history_count,
                           "power_index": contract.power_index, "forecast_origin_s": contract.origin_s,
                           "splice_offset": contract.splice_offset}})
    summary = {"status": gate["status"], "process_status": "BLOCKED",
               "scientific_verdict": "NOT_EVALUATED", "promotion_decision": "KEEP_BASELINE",
               "selected": None, "training_executed": False, "stability_evaluated": False,
               "numerical_diagnostics_are_acceptance_evidence": False,
               "real_prediction_count": 0, "eligible_denominator": None,
               "fsr": None, "fur": None, "all_eligible_predictions_valid": False,
               "reason": gate["reason"]}
    _write(result_dir / "learning_summary.json", summary)
    sources = ["src/kron_reduce/models/nonlinear_latent.py",
               "src/kron_reduce/models/learned_latent.py",
               "src/kron_reduce/models/causal_latent.py",
               "src/kron_reduce/data/kiki_history.py", "src/kron_reduce/data/history_contract.py",
               "src/kron_reduce/physics/coordinates.py", "src/kron_reduce/selection/envelopes.py",
               "src/kron_reduce/workflows/kiki_short_history.py",
               "src/kron_reduce/workflows/kiki_history_learning.py",
               "src/kron_reduce/workflows/kiki_stability_comparison.py",
               "configs/input_contracts.json", "configs/research_policy.json", "AGENTS.md",
               config["protocol_path"], config["planning_path"]]
    _write(result_dir / "lineage.json", {"run_id": run_id, "source_commit": git_head(root),
           "working_tree_status": status_lines(root), "config_sha256": sha256_file(config_path),
           "source_sha256": {p: sha256_file(root / p) for p in sources},
           "allocation_sha256": allocation_hash, "study_wall_seconds": time.perf_counter() - started,
           "source_files_written": False, "requested_core_agent_model": "gpt-6-astra",
           "requested_core_agent_effort": "ultra", "runtime_identity_independently_verified": False})
    _write(result_dir / "stage_evidence.json", {**summary, "source_commit": git_head(root),
           "data_policy": config["data_policy"], "source_val_ids_read": [], "test_ids_read": [],
           "hidden_training_targets_read": False, "level_6_claimed": False,
           "claim_boundary": config["claim_boundary"]})
    _write(result_dir / "artifact_hashes.json", {
        p.relative_to(result_dir).as_posix(): sha256_file(p) for p in sorted(result_dir.rglob("*"))
        if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
    # Do not advance the registered scientific state on a prerequisite failure.
    raise PermissionError(f"{gate['status']}: {gate['reason']}")
