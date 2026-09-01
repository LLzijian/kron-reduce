from __future__ import annotations

import csv
import hashlib
import json
import shutil

import numpy as np
import pytest
from test_p1_causal_calibration import _sample

import kron_reduce.workflows.p1_counterfactual_execution as execution
from kron_reduce.data.p1_causal import P1EventRole, P1RoleManifest
from kron_reduce.data.p1_pm_targets import PmObservationBatch, PmReadPurpose
from kron_reduce.models import EDGE_MEMORY, CandidateBlockInstance, PrunableEnergyModel
from kron_reduce.models.base import sample_rollout_inputs
from kron_reduce.models.p1_calibration import P1CalibrationSettings
from kron_reduce.workflows.p1_counterfactual_execution import (
    evaluate_saved_counterfactuals,
    fit_and_save_counterfactuals,
    verify_counterfactual_artifacts,
)
from kron_reduce.workflows.p1_five_arm_execution import (
    file_sha256,
    fit_and_save_five_arms,
    verify_five_arm_bundle,
)


@pytest.fixture(scope="module")
def saved_nonzero_counterfactual(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cf")
    records = tuple(P1EventRole(f"{role}_{index}", "synthetic", f"{role}_line_{index}", role, 1001) for role, count in (("fit", 3), ("selection", 1), ("internal_audit", 3)) for index in range(count))
    manifest = P1RoleManifest(records, "F1", hashlib.sha256(b"nonzero counterfactual fixture").hexdigest())
    instance = CandidateBlockInstance("memory:known", EDGE_MEMORY, time_constant=.2, factor=((.5,), (.2,), (-.3,)), port_rank=1)
    teacher = PrunableEnergyModel(block_instances=(instance,))
    teacher._assign_parameters(np.array([.2, .25, .3, 1., .8, .9, 1.2]), 3, 3)
    teacher.save(directory / "teacher.npz")
    samples = {}
    for index, row in enumerate(records):
        sample = _sample(row, manifest)
        static = sample["model_inputs"]["static"]
        static["delta0"] = np.array([[.12 + .005*index, -.012*index, -.12 + .007*index]])
        static["omega0"] = np.array([[.012*index, .05 - .003*index, -.05 - .009*index]])
        d, w = teacher.rollout(*sample_rollout_inputs(sample))
        sample["training_context"]["targets"].update(delta=d, omega=w)
        static["delta0"], static["omega0"] = d[[0]], w[[0]]
        samples.setdefault(row.role, []).append(sample)
    observations = PmObservationBatch("fit", PmReadPurpose.FIT_CLOSURE_TARGET, manifest.source_sha256, manifest.ids("fit"), {event: np.zeros((1001, 3)) for event in manifest.ids("fit")}, (), None)
    base = fit_and_save_five_arms(directory / "base", directory / "teacher.json", file_sha256(directory / "teacher.json"), samples["fit"], samples["selection"], observations, manifest, settings=P1CalibrationSettings(max_nfev=1))
    reference = PrunableEnergyModel.load(directory / "base/core3.npz")
    assert np.linalg.norm(reference.memory_factors[0]) > .1  # Never substitute an empty-core fixture.
    counter = fit_and_save_counterfactuals(directory / "base", base["artifact_manifest_sha256"], manifest, samples["fit"], samples["selection"], directory / "counter")
    return directory, manifest, samples, base, counter


def test_nonzero_module_zero_and_complete_delete_refit_are_different_operations(saved_nonzero_counterfactual):
    directory, manifest, _, base, counter = saved_nonzero_counterfactual
    original = PrunableEnergyModel.load(directory / "base/core3.npz")
    for variant in ("single_000", "joint"):
        zero = PrunableEnergyModel.load(directory / f"counter/{variant}_zero.npz")
        refit = PrunableEnergyModel.load(directory / f"counter/{variant}_refit.npz")
        lineage = json.loads((directory / f"counter/{variant}_lineage.json").read_text())
        assert np.array_equal(zero.node_damping, original.node_damping)
        assert np.array_equal(zero.coupling, original.coupling)
        assert zero._memory_factor(0, 3).shape == (3, 0)
        assert zero.active_instances == original.active_instances
        assert refit.active_instances == ()
        assert np.linalg.norm(refit.node_damping - original.node_damping) + np.linalg.norm(refit.coupling - original.coupling) > 1e-3
        assert lineage["reference_arrays_used_to_initialize_refit"] is False
        assert lineage["fresh_fit"]["fresh_empty_coefficient_model"] is True
        assert set(lineage["fresh_fit"]["result"]["train_case_ids"]) == set(manifest.ids("fit"))
        assert lineage["calibration"]["loss"] == "balanced"
        assert lineage["independent_support_verdict"] == "NOT_EVALUATED"
    verify_counterfactual_artifacts(directory / "counter", counter["artifact_manifest_sha256"])
    verify_five_arm_bundle(directory / "base", base["artifact_manifest_sha256"], manifest)


def test_saved_nonzero_evaluation_computes_predictions_physics_and_paired_statistics(saved_nonzero_counterfactual, tmp_path, monkeypatch):
    directory, manifest, samples, base, counter = saved_nonzero_counterfactual
    def forbidden(*_args, **_kwargs):
        raise AssertionError("internal audit must not fit, select, or rescale")

    monkeypatch.setattr(PrunableEnergyModel, "fit", forbidden)
    monkeypatch.setattr(PrunableEnergyModel, "_training_scales", forbidden)
    result = evaluate_saved_counterfactuals(directory / "base", base["artifact_manifest_sha256"], directory / "counter", counter["artifact_manifest_sha256"], manifest, samples["internal_audit"], tmp_path / "eval")
    assert result["variant_count"] == 2
    assert result["module_counterfactual_rows"] == 36 and result["module_physical_rows"] == 54
    assert result["failed_counterfactual_rows"] == result["failed_physical_rows"] == 0
    assert result["scientific_verdict"] is None and result["independent_support_verdict"] == "NOT_EVALUATED"
    with (tmp_path / "eval/module_counterfactual_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert min(float(row["combined_change"]) for row in rows if row["mode"] == "zero") > 1e-3
    with (tmp_path / "eval/module_energy_port_audit.csv").open(encoding="utf-8", newline="") as stream:
        physical = list(csv.DictReader(stream))
    assert min(float(row["storage_mean"]) for row in physical if row["mode"] == "reference") > 0
    assert min(float(row["port_abs_mean"]) for row in physical if row["mode"] == "reference") > 0
    assert all(float(row["storage_mean"]) == float(row["port_abs_mean"]) == float(row["parameter_norm"]) == 0 for row in physical if row["mode"] in {"zero", "refit"})
    assert max(float(row["identity_error_max"]) for row in physical) < 1e-8
    stats = json.loads((tmp_path / "eval/module_paired_statistics.json").read_text())
    assert len(stats["variants"]) == 4
    assert all(any(view["bootstrap_executed"] for view in row["inference"]["views"].values()) for row in stats["variants"])
    assert len(list((tmp_path / "eval").rglob("*.npz"))) == 21  # Three references plus four interventions × three events.
    verify_counterfactual_artifacts(tmp_path / "eval", result["artifact_manifest_sha256"])
    verify_counterfactual_artifacts(directory / "counter", counter["artifact_manifest_sha256"])
    verify_five_arm_bundle(directory / "base", base["artifact_manifest_sha256"], manifest)


def test_overwrite_and_internal_fit_are_rejected_before_computation(saved_nonzero_counterfactual, tmp_path):
    directory, manifest, samples, base, _ = saved_nonzero_counterfactual
    with pytest.raises(FileExistsError):
        fit_and_save_counterfactuals(directory / "base", base["artifact_manifest_sha256"], manifest, samples["fit"], samples["selection"], directory / "counter")
    with pytest.raises(ValueError, match="complete registered role"):
        fit_and_save_counterfactuals(tmp_path / "missing", "0"*64, manifest, samples["internal_audit"], samples["selection"], tmp_path / "no_output")
    assert not (tmp_path / "no_output").exists()


def test_refit_failure_preserves_registered_intervention_and_partial_result(saved_nonzero_counterfactual, tmp_path, monkeypatch):
    directory, manifest, samples, base, _ = saved_nonzero_counterfactual
    def failed(*_args, **_kwargs):
        raise ValueError("deliberate full-refit failure")

    monkeypatch.setattr(execution, "_fit_fresh_core", failed)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="deliberate full-refit"):
        fit_and_save_counterfactuals(directory / "base", base["artifact_manifest_sha256"], manifest, samples["fit"], samples["selection"], output)
    assert (output / "plan.json").is_file()
    assert (output / "single_000_zero.npz").is_file()
    record = json.loads((output / "FAILED.json").read_text())
    assert record["variant_id"] == "single_000" and record["promotion_decision"] == "KEEP_BASELINE"
    assert not (output / "artifact_hashes.json").exists()


@pytest.mark.parametrize("attack", ["changed", "added", "missing"])
def test_frozen_counterfactual_files_reject_mutation(saved_nonzero_counterfactual, tmp_path, attack):
    directory, _, _, _, counter = saved_nonzero_counterfactual
    changed = tmp_path / "changed"
    shutil.copytree(directory / "counter", changed)
    if attack == "changed":
        (changed / "single_000_refit.npz").write_bytes(b"changed")
    elif attack == "added":
        (changed / "extra.txt").write_text("not registered")
    else:
        (changed / "single_000_lineage.json").unlink()
    with pytest.raises(ValueError):
        verify_counterfactual_artifacts(changed, counter["artifact_manifest_sha256"])


def test_reference_failure_retains_every_pair_in_statistics():
    def metric(status):
        return {"role": "internal_audit", "role_manifest_sha256": "0"*64, "event_ids": ["e"], "rows": [{"event_id": "e", "source_dataset": "s", "fault_line": "l", "horizon_seconds": h, "status": status, "outputs": {"delta": {"rmse": 1.}, "omega": {"rmse": 2.}} if status == "completed" else None} for h in (1, 5, 10)]}

    pairs = execution.paired_score_rows(metric("failed"), metric("completed"))
    assert len(pairs) == 6
    assert all(row["status"] == "failed" and row["candidate_rmse"] is None for row in pairs)


def test_saved_evaluation_retains_every_failed_event_and_seals_diagnostics(
    saved_nonzero_counterfactual, tmp_path, monkeypatch
):
    directory, manifest, samples, base, counter = saved_nonzero_counterfactual

    def failed_rollout(*_args, **_kwargs):
        raise ValueError("deliberate saved-rollout failure")

    monkeypatch.setattr(PrunableEnergyModel, "rollout_augmented", failed_rollout)
    output = tmp_path / "failed_eval"
    result = evaluate_saved_counterfactuals(
        directory / "base",
        base["artifact_manifest_sha256"],
        directory / "counter",
        counter["artifact_manifest_sha256"],
        manifest,
        samples["internal_audit"],
        output,
    )
    assert result["module_counterfactual_rows"] == result["failed_counterfactual_rows"] == 36
    assert result["module_physical_rows"] == result["failed_physical_rows"] == 54
    statistics = json.loads((output / "module_paired_statistics.json").read_text())
    assert all(
        view["status"] == "UNAVAILABLE_FAILED"
        and view["bootstrap_executed"] is False
        and len(view["failed_event_ids"]) == 3
        for variant in statistics["variants"]
        for view in variant["inference"]["views"].values()
    )
    assert not list(output.rglob("event_*.npz"))
    verify_counterfactual_artifacts(output, result["artifact_manifest_sha256"])


def test_empty_optional_structure_still_registers_a_distinct_joint_intervention():
    assert execution._variants([]) == [
        {"variant_id": "joint", "kind": "joint", "removed_instance_ids": []}
    ]
