"""Fixed-budget A04 development with separate, fail-closed delivery eligibility."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.data.kiki_history import KikiHistorySamples, allocate_roles, resolve_source_root
from kron_reduce.evaluation.p2_metrics import paired_stratified_fault_line_bootstrap
from kron_reduce.models.learned_latent import event_weights, visible_scales
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    _features,
    complete_incidence,
    fit_nonlinear_model,
    fixed_quadratic_basis,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project
from kron_reduce.selection.envelopes import ExcitationEnvelopeRouter
from kron_reduce.workflows.kiki_history_learning import score_predictions
from kron_reduce.workflows.kiki_stability_comparison import SCALES, los_classification, summarize
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines

SEEDS = (324, 325, 326)
SUPPORT_DEFINITION = "causal_full_history_cross_line_nearest_train_max_v1"
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
            "scope": "default_entry_without_explicit_development_authorization",
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


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def development_authorization(root, experiment, config):
    """An explicit, hash-bound development approval never creates a delivery certificate."""
    identity = config.get("development_authorization_id")
    if identity is None:
        return None
    policy = json.loads((root / "configs/research_policy.json").read_text(encoding="utf-8"))
    approval = policy.get("development_evaluations", {}).get(identity)
    if not isinstance(approval, dict) or not re.fullmatch(r"[a-zA-Z0-9_-]+", identity):
        raise PermissionError("unregistered development authorization")
    expected = {
        "status": "user_approved", "authority": "explicit_user_reply",
        "experiment": "paper_kiki_a04_short_history", "workflow": "kiki_a04_short_history_v1",
        "stage": "short_history_learning", "dataset": "kiki_ieee39_v1",
        "data_policy": "kiki_original_train_only", "input_contract": "measured_history_100ms_zoh_v1",
        "split_seed": 324, "frozen_allocation_sha256": config["frozen_allocation_sha256"],
        "candidates": list(CANDIDATES), "seeds": list(SEEDS), "optimizer_steps_per_fit": 80,
        "max_fit_attempts": 12, "max_optimizer_steps": 960,
        "source_val_payload_allowed": False, "source_test_payload_allowed": False,
        "deployment_predictions_allowed": False, "model_promotion_allowed": False,
        "independent_confirmation_required": True, "support_definition": SUPPORT_DEFINITION,
    }
    if any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()):
        raise PermissionError("development approval does not bind the frozen A04 scientific plan")
    manifest = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    if (manifest["experiments"].get(expected["experiment"]) != experiment
        or [name for name, spec in manifest["experiments"].items() if spec == experiment] != [expected["experiment"]]
        or experiment.get("workflow") != expected["workflow"]
        or experiment.get("dataset") != expected["dataset"]
        or config.get("sources") != ["ieee39_postfault", "ieee39_expansion"]):
        raise PermissionError("development approval belongs to another experiment")
    path = approval.get("approval_protocol_path")
    if (not isinstance(path, str) or path != config.get("development_protocol_path") or not approval.get("approved_date")
        or config.get("support_definition") != SUPPORT_DEFINITION
        or config.get("approval_protocol_sha256") != approval.get("approval_protocol_sha256")
        or sha256_file(root / path) != approval.get("approval_protocol_sha256")):
        raise PermissionError("approved protocol or support-construction binding differs")
    return {"authorization_id": identity, "approval": approval,
            "status": "APPROVED_PRECERTIFICATION_DEVELOPMENT_ONLY",
            "deployment_predictions_allowed": False, "model_promotion_allowed": False}


def construct_support(train):
    if not train or {s.role for s in train} != {"train"}:
        raise ValueError("support construction accepts train only")
    features = np.stack([_features(s.history, s.equilibrium_delta, 11) for s in train])
    if features.shape[1] != 240:
        raise ValueError("support needs the complete 11-point causal representation")
    scale = np.maximum(np.sqrt(event_weights(train) @ features**2), 1e-5)
    centers = features / scale
    distances = np.sqrt(np.mean((centers[:, None] - centers[None, :])**2, axis=2))
    cross = np.array([[s.fault_line != t.fault_line for t in train] for s in train])
    if not cross.any(axis=1).all():
        raise ValueError("support audit requires independent physical train fault lines")
    leave_line_out = np.min(np.where(cross, distances, np.inf), axis=1)
    return {"definition": SUPPORT_DEFINITION, "scale": scale.tolist(),
            "centers": centers.tolist(), "radius": float(np.max(leave_line_out)),
            "train_cross_line_nearest_distances": leave_line_out.tolist(),
            "train_event_ids": [s.event_id for s in train], "train_only": True,
            "certified": False, "prediction_routing": False,
            "scope": "empirical causal support reference, not an accuracy or stability certificate"}


def support_membership(support, history, equilibrium):
    features = _features(history, equilibrium, 11) / np.asarray(support["scale"])
    distance = float(np.min(np.sqrt(np.mean((np.asarray(support["centers"]) - features)**2, axis=1))))
    return {"nearest_train_distance": distance, "inside_empirical_support": distance <= support["radius"],
            "certified": False, "delivery_status": "REFUSE_NO_CERTIFIED_ENVELOPE"}


class FitBudgetLedger:
    """Reserve each scientific fit exactly once; crashed slots are never retrained."""

    def __init__(self, root, authorization_id, plan, run_id, result_dir, resume_from_run=None):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", authorization_id):
            raise ValueError("unsafe authorization identity")
        self.root, self.run_id, self.result_dir = root, run_id, result_dir
        self.plan_hash = _digest(plan)
        self.directory = shared_budget_directory(root) / authorization_id
        self.directory.mkdir(parents=True, exist_ok=True)
        binding = self.directory / "plan.json"
        if binding.exists():
            previous = json.loads(binding.read_text(encoding="utf-8"))
            if previous["execution_root"] != str(root.resolve()):
                raise PermissionError("resume must use the original registered execution checkout")
            if previous["scientific_plan_sha256"] != self.plan_hash:
                raise PermissionError("scientific plan changed within a consumed approval budget")
            if not resume_from_run or not re.fullmatch(r"[a-zA-Z0-9_-]+", resume_from_run):
                raise PermissionError("existing fit budget requires explicit resume_from_run")
            parent = root / "results" / resume_from_run / "budget_binding.json"
            parent_binding = json.loads(parent.read_text(encoding="utf-8")) if parent.exists() else {}
            if (parent_binding.get("scientific_plan_sha256") != self.plan_hash
                or parent_binding.get("authorization_id") != authorization_id
                or parent_binding.get("ledger_plan_sha256") != sha256_file(binding)):
                raise PermissionError("resume source does not bind this scientific plan")
        else:
            if resume_from_run:
                raise PermissionError("cannot resume a nonexistent campaign")
            _write(binding, {"scientific_plan_sha256": self.plan_hash, "plan": plan,
                             "initial_run": run_id, "execution_root": str(root.resolve())})
        _write(result_dir / "budget_binding.json", {
            "scientific_plan_sha256": self.plan_hash, "authorization_id": authorization_id,
            "ledger_plan_sha256": sha256_file(binding), "resume_from_run": resume_from_run,
            "max_fit_attempts": 12, "max_optimizer_steps": 960})

    def slot(self, candidate, seed):
        if candidate not in CANDIDATES or seed not in SEEDS:
            raise ValueError("unregistered fit slot")
        return self.directory / f"{candidate}-{seed}"

    def claim(self, candidate, seed):
        slot = self.slot(candidate, seed)
        try:
            slot.mkdir()
        except FileExistsError:
            return False
        _write(slot / "started.json", {"run_id": self.run_id,
               "scientific_plan_sha256": self.plan_hash, "reserved_optimizer_steps": 80})
        return True

    def progress(self, candidate, seed, epoch):
        if type(epoch) is not int or not 1 <= epoch <= 80:
            raise ValueError("optimizer progress exceeds fixed budget")
        slot = self.slot(candidate, seed)
        self._assert_owner(slot)
        if epoch > 1 and not (slot / f"step-{epoch-1:03d}.json").exists():
            raise ValueError("nonsequential optimizer progress")
        _write(slot / f"step-{epoch:03d}.json", {"completed_optimizer_steps": epoch})

    def finish(self, candidate, seed, record_path, checkpoint=None):
        slot = self.slot(candidate, seed)
        self._assert_owner(slot)
        value = {"record_path": record_path.relative_to(self.root).as_posix(),
                 "record_sha256": sha256_file(record_path), "checkpoint_path": None,
                 "checkpoint_sha256": None}
        if checkpoint is not None:
            value.update(checkpoint_path=checkpoint.relative_to(self.root).as_posix(),
                         checkpoint_sha256=sha256_file(checkpoint))
        _write(slot / "finished.json", value)

    def _assert_owner(self, slot):
        owner = json.loads((slot / "started.json").read_text(encoding="utf-8"))
        if (owner["run_id"] != self.run_id or owner["scientific_plan_sha256"] != self.plan_hash
            or (slot / "finished.json").exists()):
            raise PermissionError("fit slot is not an unfinished reservation owned by this run")

    def recover(self, candidate, seed):
        slot = self.slot(candidate, seed)
        if (slot / "started.json").exists():
            owner = json.loads((slot / "started.json").read_text(encoding="utf-8"))
            if owner["scientific_plan_sha256"] != self.plan_hash:
                raise PermissionError("recovered slot belongs to a different scientific plan")
        final = slot / "finished.json"
        if not final.exists():
            return {"candidate": candidate, "seed": seed, "status": "interrupted",
                    "actual_optimizer_steps": None,
                    "confirmed_optimizer_steps": len(list(slot.glob("step-*.json"))),
                    "reserved_optimizer_steps": 80,
                    "reason": "claimed slot has no completed artifact; no automatic retraining"}, None
        if not (slot / "started.json").exists():
            raise ValueError("finished slot lacks an ownership record")
        binding = json.loads(final.read_text(encoding="utf-8"))
        if binding.get("record_path") is None:
            raise ValueError("finished slot lacks a fit receipt")
        paths = {}
        for kind in ("record", "checkpoint"):
            relative = binding[f"{kind}_path"]
            if relative is None:
                paths[kind] = None
                continue
            path = (self.root / relative).resolve()
            if not path.is_relative_to((self.root / "results" / owner["run_id"]).resolve()) or path.is_symlink():
                raise ValueError("unsafe reused fit artifact")
            if sha256_file(path) != binding[f"{kind}_sha256"]:
                raise ValueError("reused fit artifact hash mismatch")
            paths[kind] = path
        record = json.loads(paths["record"].read_text(encoding="utf-8"))
        if (record["candidate"], record["seed"]) != (candidate, seed):
            raise ValueError("fit artifact identity mismatch")
        if record["status"] == "completed" and (
            paths["checkpoint"] is None or record.get("actual_optimizer_steps") != 80
            or record.get("parameter_count") != CANDIDATES[candidate]["parameter_count"]
            or len(list(slot.glob("step-*.json"))) != 80
        ):
            raise ValueError("completed slot lacks its full budget, parameter count or checkpoint")
        return {**record, "reused_record": binding,
                "source_run_failed": (paths["record"].parent.parent/"RUN_FAILED.json").exists(),
                "reused_slot_artifacts_verified_only": True}, paths["checkpoint"]

    def snapshot(self):
        rows = []
        for c in CANDIDATES:
            for s in SEEDS:
                slot = self.slot(c, s)
                if slot.exists():
                    rows.append({"candidate": c, "seed": s,
                                 "confirmed_optimizer_steps": len(list(slot.glob("step-*.json"))),
                                 "finished": (slot / "finished.json").exists(),
                                 "reserved_optimizer_steps": 80})
        return {"slots": rows, "fit_attempts_reserved": len(rows),
                "optimizer_steps_reserved": 80 * len(rows),
                "confirmed_optimizer_steps": sum(r["confirmed_optimizer_steps"] for r in rows),
                "scientific_plan_sha256": self.plan_hash}


def shared_budget_directory(root):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute",
                             "--git-common-dir"], check=True, capture_output=True, text=True)
    common = Path(result.stdout.strip()).resolve()
    if not common.is_dir():
        raise ValueError("shared Git budget directory is unavailable")
    return common / "a04_fit_budgets"


def forecast_population(model, histories, equilibria, horizon_s, *, max_step=0.01, zero_latent=False):
    """Same RK4 equations batched over events; no labels, lengths or identities."""
    contract = ShortHistoryContract()
    for history in histories:
        contract.validate_history(history)
    if max_step not in (0.01, 0.005):
        raise ValueError("unregistered refinement step")
    closure, n = model.closure, model.closure.n_nodes
    mass, incidence = closure.mass, complete_incidence(n)
    eq = np.stack(equilibria)
    power = np.stack([h.constant_power for h in histories])
    features = np.stack([_features(h, e, model.settings.frames, model.settings.encoder_mode)
                         for h, e in zip(histories, equilibria, strict=True)])
    seed = features / model.feature_scale @ model.encoder.T
    initial = np.stack([np.r_[h.delta[0], h.omega[0]] for h in histories])
    state = np.c_[initial, seed]

    def rhs(value):
        q, v, z = value[:, :n], value[:, n:2*n], value[:, 2*n:]
        edge, reference = q @ incidence, eq @ incidence
        force = edge if closure.backbone == "linear" else np.sin(edge+reference)-np.sin(reference)
        f = power - (force*closure.edge_stiffness) @ incidence.T - v @ closure.damping.T - z @ closure.coupling.T
        f -= f.sum(axis=1, keepdims=True)*mass/mass.sum()
        return np.c_[v, f/mass, v @ closure.coupling + z @ (closure.latent_skew-closure.latent_dissipation).T]

    def advance(value):
        for _ in range(round(0.01/max_step)):
            a = rhs(value)
            b = rhs(value+max_step*a/2)
            c = rhs(value+max_step*b/2)
            d = rhs(value+max_step*c)
            value = value+max_step*(a+2*b+2*c+d)/6
            value[:, :n] -= (value[:, :n] @ mass)[:, None]/mass.sum()
            value[:, n:2*n] -= (value[:, n:2*n] @ mass)[:, None]/mass.sum()
            bad = ~np.isfinite(value).all(axis=1) | (np.max(np.abs(value), axis=1) > 1e12)
            value[bad] = np.nan
        return value

    with np.errstate(over="ignore", invalid="ignore"):
        hp = [state.copy()]
        for _ in range(contract.splice_offset):
            state = advance(state)
            hp.append(state.copy())
        visible = np.stack([np.r_[h.delta[-1], h.omega[-1]] for h in histories])
        state = np.c_[visible, np.zeros_like(seed) if zero_latent else state[:, 2*n:]]
        future = [state.copy()]
        for _ in range(len(contract.clock(horizon_s))-1):
            state = advance(state)
            future.append(state.copy())
    return np.stack(future, axis=1), np.stack(hp, axis=1)


def population_scores(model, samples, scale, *, zero_latent=False):
    paths, _ = forecast_population(model, [s.history for s in samples],
                                    [s.equilibrium_delta for s in samples], 3.0,
                                    zero_latent=zero_latent)
    predictions = {s.event_id: path[:len(s.forecast_time), :20]
                   for s, path in zip(samples, paths, strict=True)}
    return score_fixed_predictions(predictions, samples, scale), predictions


def numerical_population(model, samples, scale, contract):
    fine, fine_history = forecast_population(model, [s.history for s in samples],
                                            [s.equilibrium_delta for s in samples], 3.0, max_step=0.005)
    coarse, coarse_history = forecast_population(model, [s.history for s in samples],
                                                [s.equilibrium_delta for s in samples], 3.0)
    output = []
    for i, s in enumerate(samples):
        count = len(s.forecast_time)
        checks = [physical_path_checks(model, p, s.history, s.equilibrium_delta)
                  for p in (fine[i, :count], coarse[i, :count], fine_history[i], coarse_history[i])]
        difference = fine[i, :count, :20] - coarse[i, :count, :20]
        finite = bool(np.isfinite(difference).all())
        maximum = float(np.max(np.abs(difference/scale))) if finite else None
        output.append({"event_id": s.event_id, "finite_available_paths": finite,
                       "max_normalized_difference": maximum,
                       "max_delta_difference_rad": float(np.max(abs(difference[:, :10]))) if finite else None,
                       "max_omega_difference_rad_per_s": float(np.max(abs(difference[:, 10:]))) if finite else None,
                       "physical_checks": checks, "history_initialization_refined_too": True,
                       "qualified": bool(finite and maximum <= 1e-5 and all(c["qualified"] for c in checks))})
    return output


def truth_metadata(provider):
    """Evaluator-only labels/times; never passed to fitting or forecasting."""
    result = {}
    for source in {r["source_dataset"] for r in provider.rows}:
        with (provider.source_root / source / "dataset_index.csv").open(encoding="utf-8-sig", newline="") as stream:
            index = {r["event_id"]: r for r in csv.DictReader(stream)}
        for row in provider.rows:
            if row["source_dataset"] == source:
                meta = index[row["raw_id"]]
                result[row["event_id"]] = {"label": meta["label"],
                    "true_event_time_record_s": float(meta["event_time_s"]) if meta["event_time_s"] else None,
                    "previous_200ms_eligible": int(meta["valid_length"]) >= 22 and
                    (not meta["event_time_s"] or float(meta["event_time_s"]) > 0.2+1e-10)}
    return result


def stability_population(model, samples, metadata, static, contract):
    started = time.perf_counter()
    fine, fh = forecast_population(model, [s.history for s in samples],
                                   [s.equilibrium_delta for s in samples], 8.0, max_step=0.005)
    fine_seconds = time.perf_counter()-started
    coarse, ch = forecast_population(model, [s.history for s in samples],
                                     [s.equilibrium_delta for s in samples], 8.0)
    rows = []
    for i, s in enumerate(samples):
        truth = metadata[s.event_id]
        row = {"event_id": s.event_id, "source_dataset": s.source_dataset, "fault_line": s.fault_line,
               **truth, "prospective_eligible": truth["true_event_time_record_s"] is None or
               truth["true_event_time_record_s"] > contract.origin_s+1e-10,
               "fine_batch_latency_s": fine_seconds, "batch_size": len(samples),
               "amortized_compute_s": fine_seconds/len(samples),
               "prediction_delivery_status": "REFUSE_NO_CERTIFIED_ENVELOPE",
               "development_evaluation_refused": False}
        args = (s.equilibrium_delta, model.closure.mass, static[s.source_dataset]["omega_s"], contract)
        fc = classify_los(s.history, fine[i, :, :20], *args)
        cc = classify_los(s.history, coarse[i, :, :20], *args)
        numeric = stability_refinement(fine[i, :, :20], coarse[i, :, :20], fc, cc, contract)
        physical = [physical_path_checks(model, path, s.history, s.equilibrium_delta)
                    for path in (fine[i], coarse[i], fh[i], ch[i])]
        row.update(fc, **numeric, physical_checks=physical)
        row["qualified"] = bool(row["qualified"] and all(c["qualified"] for c in physical))
        row["actual_warning_lead_s_using_batch_latency"] = (
            None if truth["true_event_time_record_s"] is None else
            truth["true_event_time_record_s"]-contract.origin_s-fine_seconds)
        row["trigger_error_s"] = (
            fc["predicted_los_time_record_s"]-truth["true_event_time_record_s"]
            if row["qualified"] and fc["predicted_los"] and truth["true_event_time_record_s"] is not None else None)
        rows.append(row)
    errors = [r["trigger_error_s"] for r in rows if r["prospective_eligible"] and r["trigger_error_s"] is not None]
    summary = {**summarize(rows), "development_evaluation_refusals": 0,
               "delivery_refusals": sum(r["prospective_eligible"] for r in rows),
               "median_absolute_trigger_error_s": float(np.median(np.abs(errors))) if errors else None,
               "maximum_absolute_trigger_error_s": float(np.max(np.abs(errors))) if errors else None,
               "mean_signed_trigger_error_s": float(np.mean(errors)) if errors else None}
    summary.update(decided_subset_fsr=summary["fn"]/(summary["tp"]+summary["fn"])
                   if summary["tp"]+summary["fn"] else None,
                   decided_subset_fur=summary["fp"]/(summary["tn"]+summary["fp"])
                   if summary["tn"]+summary["fp"] else None,
                   risk_population="common_prospective_eligible_internal_val",
                   numerical_audit_population="all_history_eligible_internal_val")
    cohorts = {"shared_with_200ms_eligibility": summarize([r for r in rows if r["previous_200ms_eligible"]]),
               "new_100ms_eligibility": summarize([r for r in rows if not r["previous_200ms_eligible"]])}
    return {"summary": summary, "rows": rows, "eligibility_cohort_summaries": cohorts}, {"fine": fine, "coarse": coarse}


def evaluate_improvement(selection, records, numerics, stability):
    selected = selection["selected"]
    table = {(r["candidate"], r["seed"]): r for r in records}
    ratios = []
    for seed in SEEDS:
        row = {"seed": seed}
        for label, baseline, metric in (("versus_B", "B", "val_main_nrmse"),
                                         ("versus_A", "A", "val_main_nrmse"),
                                         ("complete3_versus_B", "B", "val_complete3_nrmse")):
            a = table.get((selected, seed), {}).get(metric)
            b = table.get((baseline, seed), {}).get(metric)
            row[label] = a/b if a is not None and b is not None and b > 0 and np.isfinite([a, b]).all() else None
        ratios.append(row)

    def values(name):
        return [r[name] for r in ratios]

    valid = all(v is not None for r in ratios for k, v in r.items() if k != "seed")
    numerically_valid = len(numerics) == 12 and all(
        result["rows"] and all(r["qualified"] for r in result["rows"]) for result in numerics.values())
    los_valid = len(stability) == 12 and all(
        all(r["qualified"] for r in result["rows"]) for result in stability.values())
    nonworse = all(
        (selected, s) in stability and ("B", s) in stability and
        all(stability[(selected, s)]["summary"][k] <= stability[("B", s)]["summary"][k] for k in ("fn", "fp"))
        for s in SEEDS)
    checks = {"C_or_D_selected": selected in {"C", "D"},
              "baseline_comparison_complete": selection["baseline_comparison_complete"],
              "all_12_fits_completed": selection["all_fits_completed"],
              "paired_main_improvement": bool(valid and np.median(values("versus_B")) <= 0.90
                    and max(values("versus_B")) <= 1.0 and np.median(values("versus_A")) <= 0.90),
              "complete3_no_more_than_5pct_worse": bool(valid and max(values("complete3_versus_B")) <= 1.05),
              "all_event_trajectory_numerics": numerically_valid,
              "all_event_8s_numerics": los_valid,
              "all_seed_fn_fp_nonworse_than_B": nonworse}
    return {"checks": checks, "paired_ratios": ratios,
            "development_improvement_demonstrated": all(checks.values()),
            "certified_comprehensive_acceptance": False, "delivery_certificate_obtained": False,
            "promotion_decision": "KEEP_BASELINE"}


def _verify_fitted_model(model, candidate, seed, train, scale, mass):
    if (asdict(model.settings) != asdict(candidate_settings(candidate, seed))
        or model.fit_record["actual_optimizer_steps"] != 80
        or model.fit_record["parameter_count"] != CANDIDATES[candidate]["parameter_count"]
        or model.fit_record["fitted_event_ids"] != sorted(s.event_id for s in train)
        or not np.array_equal(model.target_scale, scale)
        or not np.array_equal(model.closure.mass, mass)):
        raise ValueError("fitted checkpoint differs from frozen settings, train population, priors or scales")


def execute_development(*, root, experiment, config, config_path, result_dir, run_id,
                        approval, contract, allocation):
    started = time.perf_counter()
    initial_commit, initial_status = git_head(root), status_lines(root)
    if initial_status:
        raise RuntimeError("real A04 requires a committed clean checkout before any fitting")
    fit_sources = ["src/kron_reduce/models/nonlinear_latent.py", "src/kron_reduce/models/learned_latent.py",
                   "src/kron_reduce/models/causal_latent.py", "src/kron_reduce/data/kiki_history.py",
                   "src/kron_reduce/data/history_contract.py", "src/kron_reduce/physics/coordinates.py"]
    guard_paths = [*fit_sources, "src/kron_reduce/workflows/kiki_short_history.py",
                   "src/kron_reduce/workflows/kiki_history_learning.py",
                   "src/kron_reduce/workflows/kiki_stability_comparison.py",
                   config_path.relative_to(root).as_posix(), "configs/research_policy.json",
                   "configs/input_contracts.json", "AGENTS.md", config["development_protocol_path"],
                   config["protocol_path"]]
    frozen_code = {p: sha256_file(root / p) for p in guard_paths}
    _write(result_dir / "pre_execution_binding.json", {"source_commit": initial_commit,
           "working_tree_status": initial_status, "source_sha256": frozen_code,
           "development_authorization_id": approval["authorization_id"]})
    source_root = resolve_source_root(root, config["source_root"])
    rows, bindings = allocate_roles(source_root, config["sources"], config["split_seed"])
    if rows != allocation["rows"] or bindings != allocation["source_bindings"]:
        raise ValueError("source bindings or line-group allocation differ from frozen A03")
    provider = KikiHistorySamples(source_root, rows, bindings, history_count=contract.history_count,
                                  power_index=contract.power_index)
    mass, train = provider.load("train")
    if not train:
        raise ValueError("empty eligible train population")
    scale = visible_scales(train)
    support = construct_support(train)
    _write(result_dir / "support_construction_freeze.json", support)
    _, val = provider.load("val")
    if not val or {s.event_id for s in train} & {s.event_id for s in val}:
        raise ValueError("empty or overlapping internal val population")
    _write(result_dir / "development_authorization.json", approval)
    _write(result_dir / "delivery_prerequisite.json", real_evaluation_prerequisite())
    _write(result_dir / "role_allocation.json", {"rows": rows, "source_bindings": bindings})
    _write(result_dir / "train_scales.json", {"target_scale": scale.tolist(),
           "fitted_event_ids": [s.event_id for s in train], "weighting": "source_line_event_time_equal"})
    support_rows = [{"event_id": s.event_id, **support_membership(support, s.history, s.equilibrium_delta)} for s in val]
    _write(result_dir / "empirical_support_audit.json", {"rows": support_rows,
           "outside_count": sum(not r["inside_empirical_support"] for r in support_rows),
           "all_val_retained_in_development_scores": True, "certified": False})
    plan = {"authorization": approval, "input_contract": asdict(contract), "source_bindings": bindings,
            "allocation_sha256": config["frozen_allocation_sha256"],
            "train_event_ids": [s.event_id for s in train], "val_event_ids": [s.event_id for s in val],
            "train_scale": scale.tolist(), "support_sha256": _digest(support),
            "candidate_settings": {f"{c}-{s}": asdict(candidate_settings(c, s)) for c in CANDIDATES for s in SEEDS},
            "weighting": "source_line_event_time_equal", "checkpoint_rule": "last_update_80",
            "fit_source_sha256": {p: frozen_code[p] for p in fit_sources},
            "protocol_sha256": frozen_code[config["protocol_path"]]}
    _write(result_dir / "scientific_plan.json", plan)
    ledger = FitBudgetLedger(root, approval["authorization_id"], plan, run_id, result_dir,
                             config.get("resume_from_run"))
    records, models, scores, numerics, stability = [], {}, {}, {}, {}
    for candidate in CANDIDATES:
        for seed in SEEDS:
            key, name = (candidate, seed), f"{candidate}-{seed}"
            directory = result_dir / f"candidate-{name}"
            directory.mkdir()
            checkpoint = directory / "model.npz"
            if ledger.claim(candidate, seed):
                print(f"A04 fit {name}: {len(train)} train, 80 fixed updates", flush=True)
                fit_started = time.perf_counter()
                try:
                    model = fit_nonlinear_model(train, mass, candidate_settings(candidate, seed),
                              target_scale_samples=train, allow_partial_horizons=True,
                              progress_callback=lambda epoch, c=candidate, s=seed: ledger.progress(c, s, epoch))
                    _verify_fitted_model(model, candidate, seed, train, scale, mass)
                    model.save(checkpoint)
                    fit_record = {"candidate": candidate, "seed": seed, "status": "completed",
                                  "actual_optimizer_steps": 80, "parameter_count": model.fit_record["parameter_count"],
                                  "fit_wall_seconds": time.perf_counter()-fit_started,
                                  "fit_record": model.fit_record, "checkpoint_sha256": sha256_file(checkpoint)}
                    _write(directory / "fit_record.json", fit_record)
                    ledger.finish(candidate, seed, directory / "fit_record.json", checkpoint)
                except (ValueError, FloatingPointError, OverflowError, RuntimeError, OSError) as error:
                    if (directory / "fit_record.json").exists():
                        raise RuntimeError("saved fit has an incomplete budget receipt; preserve and audit") from error
                    fit_record = {"candidate": candidate, "seed": seed, "status": "failed",
                                  "actual_optimizer_steps": None,
                                  "confirmed_optimizer_steps": len(list(ledger.slot(candidate, seed).glob("step-*.json"))),
                                  "reserved_optimizer_steps": 80, "failure_type": type(error).__name__,
                                  "reason": str(error), "fit_wall_seconds": time.perf_counter()-fit_started}
                    # A partially saved model never counts as a completed fit.
                    if not (directory / "fit_record.json").exists():
                        _write(directory / "fit_record.json", fit_record)
                    if not (ledger.slot(candidate, seed) / "finished.json").exists():
                        ledger.finish(candidate, seed, directory / "fit_record.json")
                    records.append({**fit_record, "val_main_nrmse": None})
                    continue
            else:
                fit_record, original = ledger.recover(candidate, seed)
                _write(directory / "fit_record.json", fit_record)
                if original is None or fit_record["status"] != "completed":
                    records.append({**fit_record, "val_main_nrmse": None})
                    continue
                shutil.copyfile(original, checkpoint)
                model = LearnedNonlinearModel.load(checkpoint)
                _verify_fitted_model(model, candidate, seed, train, scale, mass)
            restored = LearnedNonlinearModel.load(checkpoint)
            _verify_fitted_model(restored, candidate, seed, train, scale, mass)
            train_score, _ = population_scores(restored, train, scale)
            val_score, predictions = population_scores(restored, val, scale)
            # One direct replay checks both serialization and the batched equation path.
            first = val[0]
            score_failure = None
            try:
                direct = restored.predict(first.history, first.forecast_time, first.equilibrium_delta)
                original_prediction = model.predict(first.history, first.forecast_time, first.equilibrium_delta)
                replay = np.array_equal(direct, original_prediction)
                batch_error = float(np.max(np.abs(direct[:, :20]-predictions[first.event_id])/scale))
                if not np.isfinite(batch_error):
                    raise FloatingPointError("nonfinite batched forecast replay")
                if not replay or batch_error > 1e-10:
                    raise RuntimeError("checkpoint or batched forecast replay differs")
            except (FloatingPointError, OverflowError) as error:
                replay, batch_error = False, None
                score_failure = str(error)
            _write(directory / "train_scores.json", train_score)
            _write(directory / "val_scores.json", val_score)
            subset_scores = {}
            for inside in (True, False):
                ids = {r["event_id"] for r in support_rows if r["inside_empirical_support"] == inside}
                population = [s for s in val if s.event_id in ids]
                subset_scores["inside" if inside else "outside"] = (
                    score_fixed_predictions(predictions, population, scale) if population else None)
            _write(directory / "support_stratified_scores.json", subset_scores)
            numerical = {"candidate": candidate, "seed": seed,
                         "rows": numerical_population(restored, val, scale, contract),
                         "checkpoint_replay_exact": replay, "batched_direct_max_normalized_difference": batch_error}
            _write(directory / "trajectory_numerics.json", numerical)
            record = {**fit_record, "val_main_nrmse": val_score["available"]["multi_horizon_normalized_rmse"],
                      "val_complete3_nrmse": val_score["complete3_common"]["multi_horizon_normalized_rmse"],
                      "checkpoint_sha256": sha256_file(checkpoint)}
            if score_failure is not None:
                record.update(status="score_failed", val_main_nrmse=None, score_failure=score_failure)
            records.append(record)
            models[key], scores[key], numerics[key] = restored, val_score, numerical
            print(f"A04 {name} val={record['val_main_nrmse']}", flush=True)
    _write(result_dir / "budget_snapshot.json", ledger.snapshot())
    selection = select_structure(records)
    freeze = {**selection, "records": records, "scientific_plan_sha256": ledger.plan_hash,
              "checkpoint_hashes": {f"{c}-{s}": sha256_file(result_dir/f"candidate-{c}-{s}/model.npz")
                                    for c, s in models},
              "scoring_source_sha256": frozen_code,
              "stability_evaluation_started": False, "selection_metric": "three_seed_median_val_main_nrmse"}
    _write(result_dir / "selection_freeze.json", freeze)
    freeze_hash = sha256_file(result_dir / "selection_freeze.json")
    # Evaluator-only truth is opened after structural selection and all model hashes freeze.
    metadata = truth_metadata(provider)
    coverage = []
    for row in provider.coverage:
        truth = metadata[row["event_id"]]
        already = truth["true_event_time_record_s"] is not None and truth["true_event_time_record_s"] <= contract.origin_s+1e-10
        coverage.append({**row, **truth, "prospective_eligible": row["history_eligible"] and not already,
                         "exclusion_reason": "insufficient_real_history_plus_future" if not row["history_eligible"]
                         else "already_los_within_observation" if already else None})
    _write(result_dir / "coverage.json", {"rows": coverage})
    for (candidate, seed), model in models.items():
        directory = result_dir / f"candidate-{candidate}-{seed}"
        if sha256_file(directory/"model.npz") != freeze["checkpoint_hashes"][f"{candidate}-{seed}"]:
            raise RuntimeError("checkpoint changed after selection freeze")
        result, arrays = stability_population(model, val, metadata, provider.static, contract)
        allocated_val = [r for r in coverage if r["role"] == "val"]
        allocated_unstable = sum(r["label"] == "unstable" for r in allocated_val)
        result["summary"].update(
            allocated_events=len(allocated_val), allocated_unstable=allocated_unstable,
            allocated_stable=len(allocated_val)-allocated_unstable,
            history_eligible_events=len(val),
            unstable_eligibility_coverage=result["summary"].get("eligible_unstable", 0)/allocated_unstable
            if allocated_unstable else None,
            assigned_unstable_detection_fraction=result["summary"].get("tp", 0)/allocated_unstable
            if allocated_unstable else None,
        )
        result.update(selection_freeze_sha256=freeze_hash, candidate=candidate, seed=seed)
        _write(directory / "stability.json", result)
        with (directory / "stability_paths.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        stability[(candidate, seed)] = result
        print(f"A04 {candidate}-{seed} development LOS: {result['summary']}", flush=True)
    selected = selection["selected"]
    if (selected, 324) in models:
        zero, _ = population_scores(models[(selected, 324)], val, scale, zero_latent=True)
        _write(result_dir / "selected-seed324-zero-latent.json", zero)
    bootstrap = []
    for seed in SEEDS:
        if (selected, seed) not in scores or ("B", seed) not in scores:
            bootstrap.append({"seed": seed, "status": "UNDETERMINED", "reason": "failed_fit_retained"})
            continue
        def paired(score):
            return [{**r, "normalized_joint_rmse": np.sqrt(r["normalized_mse"]) if r["status"] == "ok" else None}
                    for r in score["available"]["horizons"]["3.0"]["rows"]]
        bootstrap.append({"seed": seed, **paired_stratified_fault_line_bootstrap(
            paired(scores[("B", seed)]), paired(scores[(selected, seed)]), "normalized_joint_rmse",
            draws=5000, seed=324)})
    _write(result_dir / "paired_uncertainty.json", {"rows": bootstrap})
    improvement = evaluate_improvement(selection, records, numerics, stability)
    complete = selection["all_fits_completed"]
    summary = {**improvement, "process_status": "PASS" if complete else "FAIL",
               "scientific_verdict": "RETRY", "selected": selected,
               "development_result": "IMPROVEMENT_SUPPORTED" if improvement["development_improvement_demonstrated"] else "KEEP_BASELINE",
               "train_event_count": len(train), "val_event_count": len(val),
               "allocated_train": sum(r["role"] == "train" for r in rows),
               "allocated_val": sum(r["role"] == "val" for r in rows),
               "real_data_performance_evaluated": True, "certified_envelopes": [],
               "delivery_predictions_issued": 0, "all_val_retained": True,
               "budget": ledger.snapshot(), "candidate_table": records}
    _write(result_dir / "learning_summary.json", summary)
    _write(result_dir / "data_access_audit.json", {**provider.access_audit(),
           "labels_and_event_times_read_only_after_selection_freeze": True,
           "development_authorization_id": approval["authorization_id"]})
    for source, binding in bindings.items():
        if any(sha256_file(source_root/source/p) != digest for p, digest in binding.items()):
            raise RuntimeError("source changed during development run")
    if any(sha256_file(root/p) != digest for p, digest in frozen_code.items()):
        raise RuntimeError("implementation changed during development run")
    if git_head(root) != initial_commit:
        raise RuntimeError("Git commit changed during development run")
    if sha256_file(result_dir / "selection_freeze.json") != freeze_hash:
        raise RuntimeError("selection freeze changed during stability diagnostics")
    _write(result_dir / "lineage.json", {"run_id": run_id, "source_commit": initial_commit,
           "source_sha256": frozen_code, "config_sha256": sha256_file(config_path),
           "scientific_plan_sha256": ledger.plan_hash, "source_bindings": bindings,
           "wall_seconds": time.perf_counter()-started, "working_tree_status": status_lines(root)})
    _write(result_dir / "stage_evidence.json", {"process_status": summary["process_status"],
           "scientific_verdict": "RETRY", "promotion_decision": "KEEP_BASELINE",
           "source_commit": initial_commit, "data_policy": config["data_policy"],
           "source_val_ids_read": [], "test_ids_read": [], "hidden_training_targets_read": False,
           "level_6_claimed": False, "claim_boundary": config["claim_boundary"],
           "development_improvement_demonstrated": improvement["development_improvement_demonstrated"],
           "certified_comprehensive_acceptance": False,
           "development_authorization_id": approval["authorization_id"],
           "evidence_kind": "uncertified_development_evaluation",
           "certified_envelope_established": False,
           "deployment_predictions_allowed": False, "model_promoted": False})
    _write(result_dir / "artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
           for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
    if not complete:
        raise RuntimeError("A04 fit/score failure retained; comparison incomplete and no extra attempts permitted")


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
    approval = development_authorization(root, experiment, config)
    if approval is not None:
        return execute_development(root=root, experiment=experiment, config=config,
                   config_path=config_path, result_dir=result_dir, run_id=run_id,
                   approval=approval, contract=contract, allocation=allocation)
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
