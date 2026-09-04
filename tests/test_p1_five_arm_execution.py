from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_p1_causal_calibration import _fit_samples, _manifest, _ready_model, _sample

import kron_reduce.workflows.p1_five_arm_execution as execution
from kron_reduce.data.p1_causal import validate_p1_samples
from kron_reduce.data.p1_pm_targets import PmObservationBatch, PmReadPurpose
from kron_reduce.models.base import RunContext, sample_rollout_inputs
from kron_reduce.models.p1_calibration import (
    CalibrationRolloutError,
    P1CalibrationObjective,
    P1CalibrationSettings,
)
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel
from kron_reduce.workflows.p1_five_arm_execution import (
    evaluate_saved_five_arms,
    file_sha256,
    fit_and_save_five_arms,
    score_trajectory_role,
    verify_five_arm_bundle,
)


def _arrays(value):
    if isinstance(value, np.ndarray):
        return [(value.shape, value.dtype.str, value.tobytes())]
    if isinstance(value, dict):
        return [array for item in value.values() for array in _arrays(item)]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in _arrays(item)]
    return []


@pytest.fixture(scope="module")
def saved_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("five")
    manifest = _manifest((('a', 's1', 'l1'), ('b', 's2', 'l2')))
    teacher = _ready_model()
    samples = {}
    # Distinct independent initial states give the six backbone coefficients
    # enough excitation. No role's arrays are passed to another role's fit.
    for index, row in enumerate(manifest.records):
        sample = _sample(row, manifest)
        static = sample["model_inputs"]["static"]
        static["delta0"] = np.array([[0.12 + 0.01*index, -0.04*index, -0.12 + 0.03*index]])
        static["omega0"] = np.array([[0.02*index, 0.05 - 0.01*index, -0.05 - 0.01*index]])
        delta, omega = teacher.rollout(*sample_rollout_inputs(sample))
        sample["training_context"]["targets"].update(delta=delta, omega=omega)
        static["delta0"], static["omega0"] = delta[[0]], omega[[0]]
        samples.setdefault(row.role, []).append(sample)
    for role, values in samples.items():
        validate_p1_samples(values, manifest, role)
    # Deliberately impossible old coefficient payload. The executor may read
    # the structure JSON, but no companion old NPZ even exists.
    spec = directory / "structure.json"
    spec.write_text(json.dumps({"format_version": 3, "active_block_instances": [], "node_tau": 0.2, "potential_scale": 0.25, "max_internal_step": 0.01, "memory_taus": [0.05, 0.2, 1.0], "resonances": [[1.0, 0.1]], "old_coefficients_must_not_be_used": [-9999]}), encoding="utf-8")
    observations = PmObservationBatch("fit", PmReadPurpose.FIT_CLOSURE_TARGET, manifest.source_sha256, manifest.ids("fit"), {event: np.zeros((1001, 3)) for event in manifest.ids("fit")}, (), None)
    before = _arrays(samples)
    assert before
    saved = fit_and_save_five_arms(directory / "bundle", spec, file_sha256(spec), samples["fit"], samples["selection"], observations, manifest, settings=P1CalibrationSettings(max_nfev=1))
    assert _arrays(samples) == before
    return directory, saved, manifest, samples, observations


def test_real_from_scratch_fit_saves_only_three_cores_and_binds_five_arms(saved_bundle):
    directory, saved, manifest, _samples, _observations = saved_bundle
    bundle_dir = directory / "bundle"
    bundle = verify_five_arm_bundle(bundle_dir, saved["artifact_manifest_sha256"], manifest)
    lineage = json.loads((bundle_dir / "fit_lineage.json").read_text())
    assert lineage["fit_only_parameter_lineage_proven_by_this_execution"] is True
    for name in ("b2", "current"):
        assert lineage[name]["fresh_empty_coefficient_model"] is True
        assert lineage[name]["result"]["train_case_ids"] == ["a", "b"]
        assert lineage[name]["result"]["val_case_ids"] == ["select"]
        assert lineage[name]["result"]["details"]["solver_success"] is True
        assert lineage[name]["result"]["details"]["identifiability_pass"] is True
    calibrations = lineage["calibrations"]
    assert [row["loss"] for row in calibrations] == ["original_all_case", "original_all_case", "balanced"]
    assert calibrations[1]["initial_core_fingerprint"] == calibrations[2]["initial_core_fingerprint"]
    assert calibrations[1]["common_design_sha256"] == calibrations[2]["common_design_sha256"]
    assert bundle["core_bindings"] == {"1": "core1.npz", "2": "core2.npz", "3": "core3.npz", "4": "core3.npz", "5": "core3.npz"}
    assert bundle["arms"][2]["checkpoint_hash"] == bundle["arms"][3]["checkpoint_hash"] == bundle["arms"][4]["checkpoint_hash"]
    assert len(list(bundle_dir.glob("*.npz"))) == 3
    assert PrunableEnergyModel.load(bundle_dir / "core3.npz").coupling == pytest.approx(_ready_model().coupling, rel=0.02)
    assert bundle["model_unit_gate_passed"] is False
    assert bundle["scientific_verdict"] is None


def _conditional(saved, manifest):
    return PmObservationBatch("internal_audit", PmReadPurpose.CONDITIONAL_DIAGNOSTIC, manifest.source_sha256, manifest.ids("internal_audit"), {event: np.zeros((1001, 3)) for event in manifest.ids("internal_audit")}, (), saved["bundle"]["arms"][2]["checkpoint_hash"])


def test_saved_evaluator_never_fits_and_preserves_bundle(saved_bundle, monkeypatch, tmp_path):
    directory, saved, manifest, samples, _ = saved_bundle
    def forbidden(*_args, **_kwargs):
        raise AssertionError("saved evaluation must not fit or estimate scales")

    monkeypatch.setattr(PrunableEnergyModel, "fit", forbidden)
    monkeypatch.setattr(PrunableEnergyModel, "_training_scales", forbidden)
    result = evaluate_saved_five_arms(directory / "bundle", saved["artifact_manifest_sha256"], manifest, samples["internal_audit"], _conditional(saved, manifest), tmp_path / "evaluation")
    verify_five_arm_bundle(directory / "bundle", saved["artifact_manifest_sha256"], manifest)
    assert [row["arm"] for row in result["arms"]] == [1, 2, 3, 4, 5]
    assert all(len(row["metrics"]["rows"]) == 3 and not row["metrics"]["failed_event_ids"] for row in result["arms"])
    losses = [row["metrics"]["balanced_loss"] for row in result["arms"]]
    assert losses[2] == losses[3] == losses[4]  # Zero target honestly collapses to ZOH.
    assert result["arms"][3]["binding"]["conditional_only"] is True
    assert result["arms"][3]["binding"]["main_table"] is False
    assert result["scientific_verdict"] is None
    with pytest.raises(FileExistsError):
        evaluate_saved_five_arms(directory / "bundle", saved["artifact_manifest_sha256"], manifest, samples["internal_audit"], _conditional(saved, manifest), tmp_path / "evaluation")


@pytest.mark.parametrize("attack", ["manifest_hash", "core_hash", "wrong_role", "future_main_flag", "inside_bundle"])
def test_frozen_evaluation_rejects_role_hash_and_output_attacks(saved_bundle, tmp_path, attack):
    directory, saved, manifest, samples, _ = saved_bundle
    digest = saved["artifact_manifest_sha256"]
    observation, inputs, output = _conditional(saved, manifest), samples["internal_audit"], tmp_path / "new"
    if attack == "manifest_hash":
        digest = "0" * 64
    elif attack == "core_hash":
        observation = replace(observation, core_checkpoint_sha256="0" * 64)
    elif attack == "wrong_role":
        inputs = samples["selection"]
    elif attack == "future_main_flag":
        observation = replace(observation, main_model_input_allowed=True)
    else:
        output = directory / "bundle" / "inside"
    with pytest.raises(ValueError):
        evaluate_saved_five_arms(directory / "bundle", digest, manifest, inputs, observation, output)
    assert not output.exists()


def test_changed_core_is_rejected_before_any_model_load(saved_bundle, tmp_path, monkeypatch):
    import shutil

    directory, saved, manifest, *_ = saved_bundle
    altered = tmp_path / "altered"
    shutil.copytree(directory / "bundle", altered)
    (altered / "core3.npz").write_bytes(b"tampered")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("tampered core must not load")

    monkeypatch.setattr(PrunableEnergyModel, "load", forbidden)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_five_arm_bundle(altered, saved["artifact_manifest_sha256"], manifest)


def test_attempt_cannot_overwrite_and_solver_failure_is_preserved(saved_bundle, monkeypatch, tmp_path):
    directory, _saved, manifest, samples, observations = saved_bundle
    spec = directory / "structure.json"
    args = (spec, file_sha256(spec), samples["fit"], samples["selection"], observations, manifest)
    with pytest.raises(FileExistsError):
        fit_and_save_five_arms(directory / "bundle", *args)
    def fail(*_args, **_kwargs):
        raise CalibrationRolloutError("deliberate solver failure")

    monkeypatch.setattr(execution, "_fit_fresh_core", fail)
    destination = tmp_path / "failed"
    with pytest.raises(CalibrationRolloutError, match="deliberate solver failure"):
        fit_and_save_five_arms(destination, *args)
    record = json.loads((destination / "FAILED.json").read_text())
    assert record["promotion_decision"] == "KEEP_BASELINE"
    assert record["scientific_verdict"] is None
    assert not (destination / "artifact_hashes.json").exists()


def test_scoring_matches_calibration_and_never_deletes_failed_events():
    manifest = _manifest()
    samples = _fit_samples(manifest)
    objective = P1CalibrationObjective(samples, manifest, balanced=True)
    predictions = {sample["event_id"]: tuple(sample["training_context"]["targets"][name] * 1.1 for name in ("delta", "omega")) for sample in samples}
    result = score_trajectory_role(samples, manifest, "fit", lambda sample: predictions[sample["event_id"]], objective.scales)
    residual = objective.assemble(predictions)
    assert result["balanced_loss"] == pytest.approx(float(residual @ residual), rel=1e-12)
    predictions["a"][0][12, 0] = np.nan
    result = score_trajectory_role(samples, manifest, "fit", lambda sample: predictions[sample["event_id"]], objective.scales)
    assert result["balanced_loss"] is None
    assert result["failed_event_ids"] == ["a"]
    assert len(result["rows"]) == 6
    assert sum(row["status"] == "failed" for row in result["rows"]) == 3


def test_zero_variance_and_overflow_are_not_fabricated_scores():
    manifest = _manifest((('a', 's', 'l'),))
    samples = _fit_samples(manifest)
    for sample in samples:
        for name in ("delta", "omega"):
            sample["training_context"]["targets"][name][:] = 0
            sample["model_inputs"]["static"][f"{name}0"][:] = 0
    predict = lambda _sample: (np.zeros((1001, 3)), np.zeros((1001, 3)))
    result = score_trajectory_role(samples, manifest, "fit", predict, {"edge_angle": 1e-8, "omega": 1e-8})
    assert result["rows"][0]["outputs"]["omega"]["r2"] is None
    assert result["rows"][0]["outputs"]["omega"]["r2_unavailable_reason"] == "zero_temporal_variance"
    result = score_trajectory_role(samples, manifest, "fit", lambda _: (np.full((1001, 3), 1e200), np.zeros((1001, 3))), {"edge_angle": 1., "omega": 1.})
    assert result["failed_event_ids"] == ["a"]
    assert result["balanced_loss"] is None


def test_new_ridge_selector_does_not_skip_failed_events(monkeypatch):
    manifest = _manifest()
    samples = [_sample(row, manifest) for row in manifest.records if row.role == "selection"]
    core = execution._FreshP1Core(active_blocks=())
    core.p1_manifest = manifest
    core.validation_scales = {"edge_angle": 1., "omega": 1.}
    def failed(*_args, **_kwargs):
        raise FloatingPointError("synthetic failure")

    monkeypatch.setattr(core, "rollout", failed)
    with pytest.raises(CalibrationRolloutError, match="ridge selection"):
        core.validation_summary(samples, RunContext(horizon_steps=1001))


def test_internal_role_is_rejected_before_structure_read_or_fitting(saved_bundle, tmp_path):
    _, _, manifest, samples, observations = saved_bundle
    with pytest.raises(ValueError, match="complete registered role"):
        fit_and_save_five_arms(tmp_path / "no_output", tmp_path / "missing.json", "0" * 64, samples["internal_audit"], samples["selection"], observations, manifest)
    assert not (tmp_path / "no_output").exists()


def test_unsuccessful_weak_solver_cannot_return_a_successful_core(monkeypatch):
    monkeypatch.setattr(PrunableEnergyModel, "_solve", lambda *_args, **_kwargs: (np.ones(6), {"solver_success": False}))
    with pytest.raises(CalibrationRolloutError, match="weak parameter solver failed"):
        execution._FreshP1Core(active_blocks=())._solve(None, None, 0.01)
